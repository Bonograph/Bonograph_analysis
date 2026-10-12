"""Explain an outcome with three or more categories that have no order - a
diagnosis, a cell type, a choice - from several predictors.

X and each data set are the variables, one subject on each row; one of them
holds the outcome as codes (1, 2, 3 ...). Each category is compared with a
reference category: an odds ratio says how a predictor shifts the odds of
that category rather than the reference one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg


@dataclass
class MultinomialRegressionOptions:
    outcome_index: int = -1            # among X and the data sets; -1 = the last
    reference: str = "first"           # the category the others are compared with: "first" | "last" | "largest"
    reference_levels: dict = field(default_factory=dict)
    auto_reference: dict = field(default_factory=dict)
    test_each_term: bool = True        # each predictor tested by refitting without it
    show_fit: bool = True
    show_classification: bool = True
    show_predictions: bool = False     # each row's chances on a new sheet
    graph_probabilities: bool = True
    graph_shares: bool = False
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def probabilities(X: np.ndarray, B: np.ndarray, others: list[int], k: int) -> np.ndarray:
    """Rows x categories. ``B`` holds one row of coefficients for each
    category in ``others``; the remaining category is the reference."""
    eta = np.zeros((X.shape[0], k))
    eta[:, others] = X @ B.T
    eta -= eta.max(axis=1, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=1, keepdims=True)


def fit_multinomial(X: np.ndarray, codes: np.ndarray, k: int, reference: int, rounds: int = 100):
    """Baseline-category logit model by Newton's method.
    Returns (B: one row per non-reference category, covariance of B read row
    by row, log-likelihood, settled?)."""
    n, p = X.shape
    others = [j for j in range(k) if j != reference]
    m = len(others)
    Y = np.zeros((n, k))
    Y[np.arange(n), codes] = 1.0
    counts = Y.sum(axis=0)
    B = np.zeros((m, p))
    if np.allclose(X[:, 0], 1.0):
        B[:, 0] = np.log(np.clip(counts[others], 0.5, None) / max(counts[reference], 0.5))

    def loglik(P):
        return float(np.sum(np.log(np.clip(P[np.arange(n), codes], 1e-300, None))))

    def information(P):
        H = np.zeros((m * p, m * p))
        for a in range(m):
            for b in range(a, m):
                w = P[:, others[a]] * ((1.0 if a == b else 0.0) - P[:, others[b]])
                block = (X.T * w) @ X
                H[a * p:(a + 1) * p, b * p:(b + 1) * p] = block
                H[b * p:(b + 1) * p, a * p:(a + 1) * p] = block.T
        return H

    P = probabilities(X, B, others, k)
    now = loglik(P)
    converged = False
    for _ in range(rounds):
        gradient = (X.T @ (Y[:, others] - P[:, others])).T.ravel()
        step = rg.solve_spd(information(P), gradient).reshape(m, p)
        trial = B + step
        for _half in range(30):
            P_new = probabilities(X, trial, others, k)
            new = loglik(P_new)
            if np.isfinite(new) and new >= now - 1e-10 * (abs(now) + 1):
                break
            step = step / 2
            trial = B + step
        gained = new - now
        B, P, now = trial, P_new, new
        if abs(gained) < 1e-10 * (abs(now) + 1) and np.max(np.abs(step)) < 1e-7:
            converged = True
            break
    cov = rg.inverse(information(P))
    return B, cov, now, converged


def _reference_of(choice: str, counts: np.ndarray) -> int:
    if choice == "last":
        return counts.size - 1
    if choice == "largest":
        return int(np.argmax(counts))
    return 0


@register("multinomial_regression")
def run(data: SheetData, options: MultinomialRegressionOptions | None = None) -> AnalysisResult:
    opt = options or MultinomialRegressionOptions()
    design = rg.read(data, opt.outcome_index, opt.reference_levels, opt.auto_reference, what="A regression on categories")
    name = design.outcome
    levels, codes = rg.categories_of(design.y, name, least=3)
    names = rg.level_names(levels)
    k = levels.size
    if not design.names:
        raise AnalysisError("None of the predictors varies on the rows that are filled in.")
    X = rg.with_constant(design.Z)
    terms = ["Constant"] + design.names
    n, p = X.shape
    if n < (k - 1) * p + 5:
        raise AnalysisError(f"There are {n} complete rows for {(k - 1) * p} coefficients ({p} model terms for each of {k - 1} "
                            "categories): too few to fit. Add rows, merge categories, or leave predictors out.")
    rg.check_rank(X, design.names)
    counts = np.bincount(codes, minlength=k)
    reference = _reference_of(opt.reference, counts)
    others = [j for j in range(k) if j != reference]
    m = len(others)
    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"

    def num(v) -> str:
        return rg.clean_zero(f"{v:.{sig}g}") if np.isfinite(v) else "-"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    B, cov, loglik, converged = fit_multinomial(X, codes, k, reference)
    se = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(m, p)
    notes: list[str] = []
    adrift = {(a, j + 1) for a in range(m) for j in rg.runaway(B[a, 1:], se[a, 1:], design.Z.std(axis=0))}
    if adrift:
        notes.append(rg.runaway_note([f"{terms[t]} (for {names[others[a]]})" for a, t in sorted(adrift)],
                                     "that term separates one category from the reference completely"))
    elif not converged:
        notes.append("The fit did not settle: some estimates may be unreliable. This usually means a category holds very few rows.")
    loglik0 = float(np.sum(counts * np.log(counts / n)))
    df_model = m * (p - 1)
    whole = max(2 * (loglik - loglik0), 0.0)
    whole_p = float(stats.chi2.sf(whole, df_model))

    tables = []
    best = None                      # (P, category, term) of the strongest single effect
    for a, j in enumerate(others):
        stat, p_values, lower, upper = rg.wald(B[a], se[a], level)
        rows = []
        for t, term in enumerate(terms):
            rows.append([term, num(B[a, t]), num(se[a, t]), f"{stat[t]:.3f}" if np.isfinite(stat[t]) else "-", p_text(p_values[t]),
                         rg.ratio_text(rg.ratio(B[a, t]), sig), rg.ratio_span(rg.ratio(lower[t]), rg.ratio(upper[t]), sig)])
            if t > 0 and (a, t) not in adrift and np.isfinite(p_values[t]) and (best is None or p_values[t] < best[0]):
                best = (float(p_values[t]), a, t, float(rg.ratio(B[a, t])), float(rg.ratio(lower[t])), float(rg.ratio(upper[t])))
        tables.append(ResultTable(f"{name} = {names[j]}, rather than {names[reference]}",
                                  ["Term", "Coefficient", "SE", "z", "P value", "Odds ratio", f"{pct} CI"], rows))

    each_text = ""
    if opt.test_each_term:
        each, low = [], None
        for predictor in design.predictors:
            keep = [0] + [c + 1 for c in range(p - 1) if design.sources[c] != predictor]
            lost = (p - len(keep)) * m
            llr = fit_multinomial(X[:, keep], codes, k, reference)[2]
            value = max(2 * (loglik - llr), 0.0)
            pv = float(stats.chi2.sf(value, lost))
            each.append([predictor, f"{value:.3f}", str(lost), p_text(pv)])
            if low is None or pv < low[0]:
                low = (pv, predictor, value, lost)
        tables.insert(0, ResultTable("Does each predictor matter? (fitting the model without it)",
                                     ["Predictor", "Chi-square", "df", "P value"], each))
        if low is not None:
            each_text = f" Strongest predictor: {low[1]}, chi-square({low[3]}) = {low[2]:.3f}, P = {low[0]:.4g}."

    if opt.show_fit:
        params = m * p
        r2 = rg.pseudo_r2(loglik, loglik0, n)
        tables.append(ResultTable("How well the model fits", ["Measure", "Value"], [
            ["Rows used", str(n)], ["Categories", str(k)], ["Coefficients", str(params)],
            ["−2 × log-likelihood, with no predictors", f"{-2 * loglik0:.3f}"], ["−2 × log-likelihood, this model", f"{-2 * loglik:.3f}"],
            ["All predictors together, against none", f"chi-square({df_model}) = {whole:.3f}, P = {p_text(whole_p)}"],
            ["AIC", f"{2 * params - 2 * loglik:.3f}"], ["BIC", f"{params * np.log(n) - 2 * loglik:.3f}"],
            ["McFadden's R²", f"{r2['mcfadden']:.4f}"], ["Cox and Snell's R²", f"{r2['cox_snell']:.4f}"],
            ["Nagelkerke's R²", f"{r2['nagelkerke']:.4f}"]]))

    def probabilities_at(rows_of_terms):
        return probabilities(rg.with_constant(rows_of_terms), B, others, k)

    probs = probabilities(X, B, others, k)
    seen, predicted = counts / n, probs.mean(axis=0)
    if opt.show_classification:
        table, right = rg.classification(codes, probs.argmax(axis=1), names, name)
        tables.append(table)
        notes.append(f"The model's pick is the category with the highest chance; it is right for {right:.1%} of the rows, against "
                     f"{counts.max() / n:.1%} for always picking the commonest category. The same rows were used to fit the model, "
                     "so on new subjects it will do a little worse.")
    tables.append(ResultTable("Each category: rows and share", ["Category", "Rows", "Share"],
                              [[names[j] + (" (reference)" if j == reference else ""), str(int(counts[j])), f"{seen[j]:.1%}"] for j in range(k)]))

    charts = []
    if opt.graph_probabilities:
        charts += rg.probability_charts(design, names, probabilities_at, "multinomial_curves")
    if opt.graph_shares:
        charts.append(rg.share_chart(names, seen, predicted, name))

    summary = (f"{name}, with {k} categories, was modelled on {len(design.predictors)} predictor"
               f"{'s' if len(design.predictors) != 1 else ''} in {n} rows; each category is compared with {names[reference]}. "
               f"Together the predictors {'explain' if whole_p < opt.alpha else 'do not clearly explain'} the category: "
               f"chi-square({df_model}) = {whole:.3f}, P = {whole_p:.4g}.")
    if each_text:
        summary += each_text
    elif best is not None:
        summary += (f" Strongest single effect: {terms[best[2]]} on {names[others[best[1]]]} rather than {names[reference]}, odds ratio "
                    f"{best[3]:.{sig}g} ({pct} CI {best[4]:.{sig}g} to {best[5]:.{sig}g}), P = {best[0]:.4g}.")

    if design.dropped:
        notes.insert(0, f"{design.dropped} row{'s' if design.dropped != 1 else ''} with a blank in the outcome or a predictor "
                        f"{'were' if design.dropped != 1 else 'was'} left out.")
    notes.append(f"An odds ratio is what the odds of being in that category rather than in {names[reference]} are multiplied by for "
                 "one more unit of the predictor, with the others held fixed. It is not the change in the chance of the category "
                 "itself: that depends on every category at once, and is what the graphs show.")
    notes.append("A predictor has one coefficient for every category but the reference, so judge whether it matters from the test "
                 "that leaves it out of the model altogether, not from any single row.")
    if design.categories:
        notes.append("Predictors read as categories (two to six distinct values): " + "; ".join(
            f"{key} - levels {', '.join(f'{v:g}' for v in lv)}, reference {ref:g}" for key, (lv, ref) in design.categories.items()) + ".")
    if counts.min() < 10:
        notes.append(f"The smallest category holds {int(counts.min())} row{'s' if counts.min() != 1 else ''}. Its coefficients rest on "
                     "little and may be unstable; consider merging it with a similar category.")
    notes.append("If the categories have a natural order, a model for ordered categories uses that and needs fewer coefficients.")
    notes.append("Multinomial (baseline-category) logistic regression, fitted by maximum likelihood; Wald intervals. In the graphs "
                 "of chances, every other term is held at its mean.")

    result = AnalysisResult(title="Several predictors, unordered categories", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.show_predictions:
        columns = {f"Chance of {names[j]}": probs[:, j].tolist() for j in range(k)}
        columns["Most likely category"] = levels[probs.argmax(axis=1)].tolist()
        columns[f"{name} (seen)"] = design.y.tolist()
        result.new_sheet = NewSheetData("Category chances", "Row", (design.rows + 1).astype(float).tolist(), columns)
        notes.append("Each row's chance of each category is on a new sheet, 'Category chances'.")
    return result
