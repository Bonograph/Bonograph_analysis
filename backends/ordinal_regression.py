"""Explain an outcome whose categories come in an order - a grade, a stage, a
score from "none" to "severe" - from several predictors.

X and each data set are the variables, one subject on each row; one of them
holds the outcome as codes that rise with the order (1, 2, 3 ...). The model
is the proportional-odds one: a predictor shifts the odds of being in a
higher category rather than a lower one by the same factor wherever the line
between "higher" and "lower" is drawn. That assumption is tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg

LINKS = {"logit": "proportional-odds (cumulative logit)", "probit": "cumulative probit"}


@dataclass
class OrdinalRegressionOptions:
    outcome_index: int = -1            # among X and the data sets; -1 = the last
    link: str = "logit"                # "logit" | "probit"
    reference_levels: dict = field(default_factory=dict)
    auto_reference: dict = field(default_factory=dict)
    test_each_term: bool = False       # each predictor tested by refitting without it
    show_cut_points: bool = True
    show_fit: bool = True
    show_parallel_check: bool = True   # do the predictors act alike at every cut point? (logit only)
    show_classification: bool = False
    show_predictions: bool = False     # each row's chances on a new sheet
    graph_probabilities: bool = True
    graph_shares: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def _cdf(link: str, v):
    return special.ndtr(v) if link == "probit" else special.expit(v)


def _pdf(link: str, v):
    if link == "probit":
        return np.exp(-0.5 * np.square(v)) / np.sqrt(2 * np.pi)
    e = special.expit(v)
    return e * (1 - e)


def cumulative_probabilities(cuts: np.ndarray, eta: np.ndarray, link: str) -> np.ndarray:
    """Rows x categories: each row's chance of each category."""
    upper = _cdf(link, cuts[None, :] - eta[:, None])
    edges = np.column_stack([np.zeros(eta.size), upper, np.ones(eta.size)])
    return np.clip(np.diff(edges, axis=1), 0.0, 1.0)


def loglik_and_gradient(cuts: np.ndarray, beta: np.ndarray, Z: np.ndarray, codes: np.ndarray, link: str):
    """Log-likelihood of the cumulative-link model and its slope in the cut
    points and the coefficients (in that order)."""
    k = cuts.size + 1
    eta = Z @ beta if beta.size else np.zeros(codes.size)
    padded = np.concatenate([[-np.inf], cuts, [np.inf]])
    upper, lower = padded[codes + 1] - eta, padded[codes] - eta
    F_up = np.where(np.isinf(upper), 1.0, _cdf(link, upper))
    F_lo = np.where(np.isinf(lower), 0.0, _cdf(link, lower))
    f_up = np.where(np.isinf(upper), 0.0, _pdf(link, np.where(np.isinf(upper), 0.0, upper)))
    f_lo = np.where(np.isinf(lower), 0.0, _pdf(link, np.where(np.isinf(lower), 0.0, lower)))
    prob = np.clip(F_up - F_lo, 1e-300, None)
    value = float(np.sum(np.log(prob)))
    g_cuts = (np.bincount(codes, weights=f_up / prob, minlength=k)[: k - 1]
              - np.bincount(codes, weights=f_lo / prob, minlength=k)[1:])
    g_beta = -Z.T @ ((f_up - f_lo) / prob) if beta.size else np.zeros(0)
    return value, np.concatenate([g_cuts, g_beta])


def fit_cumulative(Z: np.ndarray, codes: np.ndarray, k: int, link: str = "logit"):
    """Cut points and coefficients by maximum likelihood. The predictors are
    put on a common scale for the search and the answer is taken back to the
    scale they were typed in. Returns (cuts, beta, covariance of both in that
    order, log-likelihood, converged)."""
    n, p = Z.shape
    centre = Z.mean(axis=0) if p else np.zeros(0)
    scale = Z.std(axis=0) if p else np.ones(0)
    scale = np.where(scale > 0, scale, 1.0)
    S = (Z - centre) / scale if p else Z
    shares = np.cumsum(np.bincount(codes, minlength=k))[:-1] / n
    inverse = stats.norm.ppf if link == "probit" else special.logit
    start_cuts = inverse(np.clip(shares, 1e-6, 1 - 1e-6))

    # search over the first cut point and the logs of the gaps between cut points, so they stay in order
    def unpack(v):
        return np.concatenate([[v[0]], v[0] + np.cumsum(np.exp(v[1: k - 1]))]), v[k - 1:]

    def objective(v):
        cuts, beta = unpack(v)
        value, g = loglik_and_gradient(cuts, beta, S, codes, link)
        g_cuts = g[: k - 1]
        tail = np.cumsum(g_cuts[::-1])[::-1]                 # the slope in a gap moves every cut point above it
        g_free = np.concatenate([[tail[0]], np.exp(v[1: k - 1]) * tail[1:], g[k - 1:]])
        return -value, -g_free

    gaps = np.clip(np.diff(start_cuts), 1e-3, None)
    v0 = np.concatenate([[start_cuts[0]], np.log(gaps), np.zeros(p)])
    best = optimize.minimize(objective, v0, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
    cuts, beta = unpack(best.x)
    full = np.concatenate([cuts, beta])

    def gradient(v):
        return loglik_and_gradient(v[: k - 1], v[k - 1:], S, codes, link)[1]

    converged = bool(best.success)
    for _ in range(8):                                        # a few exact Newton steps to finish the job
        g = gradient(full)
        if np.max(np.abs(g)) < 1e-9 * n:
            converged = True
            break
        H = rg.hessian_of(gradient, full)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        trial = full - step
        if np.any(np.diff(trial[: k - 1]) <= 0):
            break
        if loglik_and_gradient(trial[: k - 1], trial[k - 1:], S, codes, link)[0] < loglik_and_gradient(full[: k - 1], full[k - 1:], S, codes, link)[0] - 1e-9:
            break
        full = trial
    converged = converged or bool(np.max(np.abs(gradient(full))) < 1e-5 * n)
    information = -rg.hessian_of(gradient, full)
    cov_scaled = rg.inverse(information)
    loglik = loglik_and_gradient(full[: k - 1], full[k - 1:], S, codes, link)[0]
    # back to the typed scale: beta = beta_s / scale, cut = cut_s + sum(beta_s * centre / scale)
    A = np.zeros((k - 1 + p, k - 1 + p))
    A[: k - 1, : k - 1] = np.eye(k - 1)
    if p:
        A[: k - 1, k - 1:] = centre / scale
        A[k - 1:, k - 1:] = np.diag(1 / scale)
    out = A @ full
    return out[: k - 1], out[k - 1:], A @ cov_scaled @ A.T, float(loglik), converged


def _binary_logit(X: np.ndarray, z: np.ndarray, rounds: int = 60):
    """Plain logistic regression: (coefficients, fitted chances, settled?)."""
    beta = np.zeros(X.shape[1])
    share = float(np.clip(z.mean(), 1e-6, 1 - 1e-6))
    beta[0] = np.log(share / (1 - share))
    for _ in range(rounds):
        fitted = special.expit(np.clip(X @ beta, -35, 35))
        weight = fitted * (1 - fitted)
        try:
            step = np.linalg.solve((X.T * weight) @ X, X.T @ (z - fitted))
        except np.linalg.LinAlgError:
            return beta, fitted, False
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            return beta, special.expit(np.clip(X @ beta, -35, 35)), bool(np.max(np.abs(beta)) < 25)
    return beta, fitted, False


def brant_test(Z: np.ndarray, codes: np.ndarray, k: int, groups: dict[str, list[int]]):
    """Brant's (1990) test that each predictor has the same coefficient at
    every cut point. A logistic regression is fitted at each cut ("above it"
    against "at or below it"); their coefficients are compared with a Wald
    test that allows for the fits sharing the same subjects.

    Returns None when a cut point cannot be fitted on its own, else
    {"all": (chi2, df, P), "each": {name: (chi2, df, P)}, "betas": cuts x p}."""
    n, p = Z.shape
    if k < 3 or p == 0:
        return None
    centre, scale = Z.mean(axis=0), Z.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    X = np.column_stack([np.ones(n), (Z - centre) / scale])
    m = k - 1
    betas, fitted, inverses = [], [], []
    for cut in range(m):
        z = (codes > cut).astype(float)
        if z.min() == z.max():
            return None
        b, f, ok = _binary_logit(X, z)
        if not ok:
            return None
        betas.append(b)
        fitted.append(f)
        inverses.append(rg.inverse((X.T * (f * (1 - f))) @ X))
    V = np.zeros((m * p, m * p))
    for a in range(m):
        for b in range(a, m):
            # being above cut b implies being above cut a, so the two indicators share exactly fitted[b]
            middle = (X.T * (fitted[b] - fitted[a] * fitted[b])) @ X
            block = (inverses[a] @ middle @ inverses[b])[1:, 1:]
            V[a * p:(a + 1) * p, b * p:(b + 1) * p] = block
            V[b * p:(b + 1) * p, a * p:(a + 1) * p] = block.T
    stacked = np.concatenate([b[1:] for b in betas])

    def test(columns: list[int]):
        q = len(columns)
        D = np.zeros(((m - 1) * q, m * p))
        for r in range(1, m):
            for i, c in enumerate(columns):
                D[(r - 1) * q + i, c] = 1.0
                D[(r - 1) * q + i, r * p + c] = -1.0
        gap = D @ stacked
        middle = D @ V @ D.T
        try:
            chi2 = float(gap @ np.linalg.solve(middle, gap))
        except np.linalg.LinAlgError:
            return float("nan"), (m - 1) * q, float("nan")
        df = (m - 1) * q
        return chi2, df, float(stats.chi2.sf(chi2, df))

    slopes = np.array([b[1:] / scale for b in betas])
    return {"all": test(list(range(p))), "each": {name: test(cols) for name, cols in groups.items()}, "betas": slopes}


@register("ordinal_regression")
def run(data: SheetData, options: OrdinalRegressionOptions | None = None) -> AnalysisResult:
    opt = options or OrdinalRegressionOptions()
    link = opt.link if opt.link in LINKS else "logit"
    design = rg.read(data, opt.outcome_index, opt.reference_levels, opt.auto_reference, what="A regression on ordered categories")
    name = design.outcome
    levels, codes = rg.categories_of(design.y, name, least=3)
    names = rg.level_names(levels)
    k = levels.size
    if not design.names:
        raise AnalysisError("None of the predictors varies on the rows that are filled in.")
    Z = design.Z
    n, p = Z.shape
    if n < p + k + 5:
        raise AnalysisError(f"There are {n} complete rows for {p} model terms and {k} categories: too few to fit.")
    rg.check_rank(rg.with_constant(Z), design.names)
    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"

    def num(v) -> str:
        return rg.clean_zero(f"{v:.{sig}g}") if np.isfinite(v) else "-"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    cuts, beta, cov, loglik, converged = fit_cumulative(Z, codes, k, link)
    se_all = np.sqrt(np.clip(np.diag(cov), 0, None))
    se_cuts, se = se_all[: k - 1], se_all[k - 1:]
    stat, p_values, lower, upper = rg.wald(beta, se, level)
    counts = np.bincount(codes, minlength=k)
    loglik0 = float(np.sum(special.xlogy(counts, counts / n)))
    whole = max(2 * (loglik - loglik0), 0.0)
    whole_p = float(stats.chi2.sf(whole, p))
    notes: list[str] = []
    adrift = rg.runaway(beta, se, Z.std(axis=0))
    if adrift:
        notes.append(rg.runaway_note([design.names[j] for j in adrift], "that term separates the categories completely"))
    elif not converged:
        notes.append("The fit did not settle: the estimates may be unreliable. This usually means a category holds very few rows.")

    odds = link == "logit"
    headers = ["Term", "Coefficient", "SE", "z", "P value"] + (["Odds ratio (higher category)", f"{pct} CI"] if odds else [f"{pct} CI"])
    rows = []
    for j, term in enumerate(design.names):
        row = [term, num(beta[j]), num(se[j]), f"{stat[j]:.3f}" if np.isfinite(stat[j]) else "-", p_text(p_values[j])]
        row += ([rg.ratio_text(rg.ratio(beta[j]), sig), rg.ratio_span(rg.ratio(lower[j]), rg.ratio(upper[j]), sig)] if odds
                else [f"{num(lower[j])} to {num(upper[j])}"])
        rows.append(row)
    tables = [ResultTable(f"What each predictor does to the {name} category", headers, rows)]

    if opt.test_each_term:
        each = []
        for predictor in design.predictors:
            keep = [c for c in range(p) if design.sources[c] != predictor]
            lost = p - len(keep)
            if keep:
                llr = fit_cumulative(Z[:, keep], codes, k, link)[3]
            else:
                llr = loglik0
            value = max(2 * (loglik - llr), 0.0)
            each.append([predictor, f"{value:.3f}", str(lost), p_text(float(stats.chi2.sf(value, lost)))])
        tables.append(ResultTable("Each predictor, tested by fitting the model without it", ["Predictor", "Chi-square", "df", "P value"], each))

    if opt.show_cut_points:
        zc = stats.norm.ppf(0.5 + level / 2)
        tables.append(ResultTable("Cut points between the categories", ["Between", "Estimate", "SE", f"{pct} CI"],
                                  [[f"{names[j]} and {names[j + 1]}", num(cuts[j]), num(se_cuts[j]),
                                    f"{num(cuts[j] - zc * se_cuts[j])} to {num(cuts[j] + zc * se_cuts[j])}"] for j in range(k - 1)]))

    if opt.show_fit:
        params = p + k - 1
        r2 = rg.pseudo_r2(loglik, loglik0, n)
        tables.append(ResultTable("How well the model fits", ["Measure", "Value"], [
            ["Rows used", str(n)], ["Categories", str(k)],
            ["−2 × log-likelihood, with no predictors", f"{-2 * loglik0:.3f}"], ["−2 × log-likelihood, this model", f"{-2 * loglik:.3f}"],
            ["All predictors together, against none", f"chi-square({p}) = {whole:.3f}, P = {p_text(whole_p)}"],
            ["AIC", f"{2 * params - 2 * loglik:.3f}"], ["BIC", f"{params * np.log(n) - 2 * loglik:.3f}"],
            ["McFadden's R²", f"{r2['mcfadden']:.4f}"], ["Cox and Snell's R²", f"{r2['cox_snell']:.4f}"],
            ["Nagelkerke's R²", f"{r2['nagelkerke']:.4f}"]]))

    parallel_text = ""
    if opt.show_parallel_check and odds:
        groups = {predictor: design.columns_of(predictor) for predictor in design.predictors}
        brant = brant_test(Z, codes, k, groups)
        if brant is None:
            notes.append("The check that each predictor acts alike at every cut point could not be run: at least one cut point "
                         "cannot be fitted on its own (too few rows on one side of it, or a predictor that splits them perfectly).")
        else:
            chi2, df, pv = brant["all"]
            check = [["All predictors", f"{chi2:.3f}", str(df), p_text(pv)]]
            check += [[predictor, f"{c:.3f}", str(d), p_text(q)] for predictor, (c, d, q) in brant["each"].items()]
            tables.append(ResultTable("Does each predictor act alike at every cut point? (Brant test)",
                                      ["Predictor", "Chi-square", "df", "P value"], check))
            tables.append(ResultTable("Odds ratio at each cut point, fitted on its own",
                                      ["Term"] + [f"Above {names[j]}" for j in range(k - 1)],
                                      [[term] + [rg.ratio_text(rg.ratio(brant["betas"][c, j]), sig) for c in range(k - 1)] for j, term in enumerate(design.names)]))
            parallel_text = (" The odds ratios hold steady across the cut points (Brant P = " + f"{pv:.4g})." if not pv < opt.alpha else
                             f" Caution: the predictors do not act alike at every cut point (Brant P = {pv:.4g}), so one odds ratio "
                             "per predictor is too simple a summary.")
            if pv < opt.alpha:
                notes.append("The proportional-odds assumption is in doubt. Look at the odds ratios fitted at each cut point: a "
                             "predictor whose ratio drifts from one cut to the next is the one that breaks it. A model for "
                             "categories without an order makes no such assumption, at the cost of more coefficients.")

    def probabilities_at(rows_of_terms):
        return cumulative_probabilities(cuts, rows_of_terms @ beta, link)

    probs = probabilities_at(Z)
    seen, predicted = counts / n, probs.mean(axis=0)
    tables.append(ResultTable("Each category: seen and predicted", ["Category", "Rows", "Share seen", "Share predicted"],
                              [[names[j], str(int(counts[j])), f"{seen[j]:.1%}", f"{predicted[j]:.1%}"] for j in range(k)]))
    if opt.show_classification:
        table, right = rg.classification(codes, probs.argmax(axis=1), names, name)
        tables.append(table)
        notes.append(f"The model's pick is the category with the highest chance; it is right for {right:.1%} of the rows, against "
                     f"{counts.max() / n:.1%} for always picking the commonest category.")

    charts = []
    if opt.graph_probabilities:
        charts += rg.probability_charts(design, names, probabilities_at, "ordinal_curves")
    if opt.graph_shares:
        charts.append(rg.share_chart(names, seen, predicted, name))

    order = [j for j in range(p) if np.isfinite(p_values[j]) and j not in adrift]
    lead = min(order, key=lambda j: p_values[j]) if order else None
    summary = (f"{name}, with {k} ordered categories, was modelled on {len(design.predictors)} predictor"
               f"{'s' if len(design.predictors) != 1 else ''} in {n} rows ({LINKS[link]} model). Together they "
               f"{'explain' if whole_p < opt.alpha else 'do not clearly explain'} the category: chi-square({p}) = {whole:.3f}, "
               f"P = {whole_p:.4g}.")
    if lead is not None:
        if odds:
            summary += (f" Strongest: {design.names[lead]}, odds ratio {rg.ratio(beta[lead]):.{sig}g} ({pct} CI "
                        f"{rg.ratio(lower[lead]):.{sig}g} to {rg.ratio(upper[lead]):.{sig}g}), P = {p_values[lead]:.4g}.")
        else:
            summary += f" Strongest: {design.names[lead]}, coefficient {beta[lead]:.{sig}g}, P = {p_values[lead]:.4g}."
    summary += parallel_text

    if design.dropped:
        notes.insert(0, f"{design.dropped} row{'s' if design.dropped != 1 else ''} with a blank in the outcome or a predictor "
                        f"{'were' if design.dropped != 1 else 'was'} left out.")
    notes.append(f"The categories of {name} are taken in the order of their codes: {', '.join(names)}. Recode them if that is "
                 "not the order you mean.")
    if odds:
        notes.append("An odds ratio above 1 means that higher values of the predictor go with higher categories: it is what the "
                     "odds of being above any cut point, rather than at or below it, are multiplied by for one more unit of the "
                     "predictor, with the others held fixed. For a predictor read as categories it compares that level with the "
                     "reference level.")
    else:
        notes.append("A positive coefficient means that higher values of the predictor go with higher categories. Probit "
                     "coefficients are in standard deviations of an underlying normal scale and have no odds-ratio reading.")
    if design.categories:
        notes.append("Read as categories (two to six distinct values): " + "; ".join(
            f"{key} - levels {', '.join(f'{v:g}' for v in lv)}, reference {ref:g}" for key, (lv, ref) in design.categories.items()) + ".")
    if counts.min() < 5:
        notes.append(f"The smallest category holds {int(counts.min())} row{'s' if counts.min() != 1 else ''}. Cut points next to a "
                     "thin category are poorly determined; consider merging it with its neighbour.")
    notes.append(f"Fitted by maximum likelihood ({LINKS[link]} model); standard errors from the observed information; Wald "
                 "intervals. In the graphs of chances, every other term is held at its mean.")
    notes.append("References: McCullagh 1980; Brant 1990.")

    result = AnalysisResult(title="Several predictors, ordered categories", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.show_predictions:
        columns = {f"Chance of {names[j]}": probs[:, j].tolist() for j in range(k)}
        columns["Most likely category"] = levels[probs.argmax(axis=1)].tolist()
        columns[f"{name} (seen)"] = design.y.tolist()
        result.new_sheet = NewSheetData("Category chances", "Row", (design.rows + 1).astype(float).tolist(), columns)
        notes.append("Each row's chance of each category is on a new sheet, 'Category chances'.")
    return result
