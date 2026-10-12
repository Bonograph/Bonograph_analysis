"""Multiple logistic regression: a yes/no outcome from several predictors.

Fits by iteratively reweighted least squares (Newton's method on the
binomial log-likelihood) with no penalty on the coefficients: a penalised
fit would pull every coefficient toward zero, away from the maximum-
likelihood estimate that the odds ratios and their tests assume. Reports odds
ratios with profile-likelihood or Wald intervals, model comparison, the
goodness-of-fit measures and graphs chosen in the settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class MultipleLogisticRegressionOptions:
    # the model
    outcome_index: int = -1  # index into full_variable_matrix()'s label list; -1 = last variable
    negative_value: float = 0.0
    positive_value: float = 1.0
    include_intercept: bool = True
    include_two_way: bool = False
    include_three_way: bool = False
    # reference levels
    reference_levels: dict[str, str] = field(default_factory=dict)
    auto_reference: dict[str, bool] = field(default_factory=dict)
    # against a simpler model
    compare_models: bool = False
    dropped_predictors: list[str] = field(default_factory=list)
    # what to report
    ci_approach: str = "profile"  # "profile" | "wald"
    show_se: bool = False
    show_ci: bool = True  # a confidence interval for every odds ratio
    run_significance_test: bool = False
    show_multicollinearity: bool = False
    show_correlation_matrix: bool = False
    show_aicc: bool = True
    show_loglik_both: bool = False
    show_deviance_both: bool = False
    negative_label: str = "0"
    positive_label: str = "1"
    # how well it fits
    gof_roc: bool = True
    gof_classification_table: bool = True
    gof_predicted_prob: bool = True
    classification_cutoff: float = 0.5
    pseudo_r2_tjur: bool = True
    pseudo_r2_mcfadden: bool = False
    pseudo_r2_coxsnell: bool = False
    pseudo_r2_nagelkerke: bool = False
    gof_hosmer_lemeshow: bool = True
    gof_lrt: bool = False
    # graphs (each adds a chart plus its data table)
    graph_predicted_vs_observed: bool = True
    graph_roc: bool = True
    graph_logistic_plot: bool = True
    graph_proportion_correct: bool = False
    plot_confidence_bands: bool = False
    confidence_level: float = 0.95
    significant_digits: int = 4
    # how the model is fitted: "auto" (Firth's fit only when a predictor splits the outcomes) | "ml" | "firth"
    fit_method: str = "auto"
    # can the predicted chances be trusted
    gof_brier: bool = False
    graph_calibration: bool = False
    # which rows pull the fit
    show_influence: bool = False
    graph_influence: bool = False


def _build_design(predictor_labels: list[str], predictor_columns: list[np.ndarray], opt: MultipleLogisticRegressionOptions):
    columns, names, cat_info, sources = [], [], {}, []
    for label, values in zip(predictor_labels, predictor_columns):
        finite = values[~np.isnan(values)]
        levels = np.unique(finite)
        if levels.size >= 2 and levels.size <= 6:
            auto = opt.auto_reference.get(label, True)
            chosen = opt.reference_levels.get(label)
            if auto or chosen is None:
                ref_level = float(levels[0])
            else:
                try:
                    ref_level = float(chosen)
                except ValueError:
                    ref_level = float(levels[0])
                if ref_level not in levels:
                    ref_level = float(levels[0])
            cat_info[label] = (levels.tolist(), ref_level)
            for lvl in levels:
                if lvl == ref_level:
                    continue
                dummy = (values == lvl).astype(float)
                dummy[np.isnan(values)] = np.nan
                columns.append(dummy)
                names.append(f"{label} = {lvl:g} (vs {ref_level:g})")
                sources.append({label})
        else:
            columns.append(values)
            names.append(label)
            sources.append({label})

    if opt.include_two_way and len(columns) >= 2 and len(columns) <= 6:
        base_n = len(columns)
        for i, j in combinations(range(base_n), 2):
            columns.append(columns[i] * columns[j])
            names.append(f"{names[i]} × {names[j]}")
            sources.append(sources[i] | sources[j])
    if opt.include_three_way and len(predictor_labels) >= 3 and len(columns) <= 6:
        base_cols = list(zip(names, columns, sources))
        for (n1, c1, s1), (n2, c2, s2), (n3, c3, s3) in combinations(base_cols[: len(predictor_labels)], 3):
            columns.append(c1 * c2 * c3)
            names.append(f"{n1} × {n2} × {n3}")
            sources.append(s1 | s2 | s3)

    n = len(predictor_columns[0]) if predictor_columns else 0
    Z = np.column_stack(columns) if columns else np.empty((n, 0))
    return Z, names, cat_info, sources


def _fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Unregularized IRLS/Newton fit of the binomial log-likelihood."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-8, None)
        z = eta + (y - mu) / W
        XtWX = X.T @ (X * W[:, None])
        XtWz = X.T @ (W * z)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-8, None)
    cov = np.linalg.pinv(X.T @ (X * W[:, None]))
    eps = 1e-12
    log_lik = float(np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps)))
    return beta, cov, log_lik


def _null_log_lik(y: np.ndarray) -> float:
    p0 = float(np.mean(y))
    eps = 1e-12
    return float(np.sum(y * np.log(p0 + eps) + (1 - y) * np.log(1 - p0 + eps)))


def _logistic_negll_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    eps = 1e-12
    negll = -float(np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps)))
    grad = -(X.T @ (y - mu))
    return negll, grad


def _profile_ci_logistic(idx: int, beta: np.ndarray, se: np.ndarray, X: np.ndarray, y: np.ndarray, full_negll: float, confidence_level: float) -> tuple[float, float] | None:
    p = beta.size
    if p > 8:
        return None
    target_drop = stats.chi2.ppf(confidence_level, 1) / 2
    target_ll = -full_negll - target_drop
    other_idx = [i for i in range(p) if i != idx]

    def profile_negll(beta_k: float) -> float:
        if not other_idx:
            return _logistic_negll_grad(np.array([beta_k]), X, y)[0]

        def obj(b_other):
            b = np.empty(p)
            b[idx] = beta_k
            b[other_idx] = b_other
            return _logistic_negll_grad(b, X, y)[0]

        def obj_jac(b_other):
            b = np.empty(p)
            b[idx] = beta_k
            b[other_idx] = b_other
            return _logistic_negll_grad(b, X, y)[1][other_idx]

        start = np.delete(beta, idx)
        res = minimize(obj, start, jac=obj_jac, method="BFGS")
        return res.fun

    def g(beta_k: float) -> float:
        return -profile_negll(beta_k) - target_ll

    step = max(se[idx], 1e-3)
    lo = hi = None
    b0 = beta[idx]
    prev_offset, offset = 0.0, step
    for _ in range(10):
        if g(b0 - offset) < 0:
            lo = brentq(g, b0 - offset, b0 - prev_offset)
            break
        prev_offset, offset = offset, offset * 2
    prev_offset, offset = 0.0, step
    for _ in range(10):
        if g(b0 + offset) < 0:
            hi = brentq(g, b0 + prev_offset, b0 + offset)
            break
        prev_offset, offset = offset, offset * 2
    if lo is None or hi is None:
        return None
    return lo, hi


def _hosmer_lemeshow(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> tuple[float, int, float]:
    order = np.argsort(probs)
    y_sorted, p_sorted = y[order], probs[order]
    bins = np.array_split(np.arange(y.size), min(n_bins, y.size))
    chi2 = 0.0
    used_bins = 0
    for idx in bins:
        if idx.size == 0:
            continue
        obs = float(np.sum(y_sorted[idx]))
        exp = float(np.sum(p_sorted[idx]))
        n_bin = idx.size
        var = exp * (1 - exp / n_bin) if n_bin else 0.0
        if var > 1e-9:
            chi2 += (obs - exp) ** 2 / var
        used_bins += 1
    df = max(used_bins - 2, 1)
    p_value = float(stats.chi2.sf(chi2, df))
    return chi2, df, p_value


def _encode_scenario(scenario: dict[str, float], predictor_labels: list[str], cat_info: dict, names: list[str]) -> np.ndarray:
    base, base_names = [], []
    for label in predictor_labels:
        val = scenario.get(label, 0.0)
        if label in cat_info:
            levels, ref = cat_info[label]
            for lvl in levels:
                if lvl == ref:
                    continue
                base.append(1.0 if val == lvl else 0.0)
                base_names.append(f"{label} = {lvl:g} (vs {ref:g})")
        else:
            base.append(val)
            base_names.append(label)
    vec = dict(zip(base_names, base))

    def lookup(name: str) -> float:
        if name in vec:
            return vec[name]
        parts = [p.strip() for p in name.split("×")]
        if all(p in vec for p in parts):
            result = 1.0
            for p in parts:
                result *= vec[p]
            return result
        return 0.0

    return np.array([lookup(nm) for nm in names])


@register("multiple_logistic_regression")
def run(data: SheetData, options: MultipleLogisticRegressionOptions | None = None) -> AnalysisResult:
    from ..ml_utils import roc_auc_score, roc_curve

    opt = options or MultipleLogisticRegressionOptions()
    all_labels, full_matrix = data.full_variable_matrix()
    if len(all_labels) < 3:
        raise AnalysisError(f"Multiple logistic regression needs at least 2 predictors and a 0/1 outcome; found {len(all_labels)} variable(s).")

    outcome_idx = opt.outcome_index if opt.outcome_index >= 0 else len(all_labels) - 1
    outcome_idx = max(0, min(outcome_idx, len(all_labels) - 1))
    outcome_label = all_labels[outcome_idx]
    # a variable with no values at all (typically an unused X column, as on a
    # Many variables sheet) is left out rather than making every row incomplete
    has_values = ~np.all(np.isnan(full_matrix), axis=0)
    predictor_labels = [lbl for i, lbl in enumerate(all_labels) if i != outcome_idx and has_values[i]]
    predictor_idx = [i for i in range(len(all_labels)) if i != outcome_idx and has_values[i]]
    if len(predictor_idx) < 1:
        raise AnalysisError("There are no predictor columns with values - add at least one besides the outcome.")

    used = [outcome_idx] + predictor_idx
    complete_mask = ~np.any(np.isnan(full_matrix[:, used]), axis=1)
    complete = full_matrix[complete_mask]
    if complete.shape[0] < 3 * len(predictor_labels) + 5:
        raise AnalysisError(f"Need more complete rows for {len(predictor_labels)} predictor(s); found {complete.shape[0]}.")

    y_raw = complete[:, outcome_idx]
    valid = np.isin(y_raw, (opt.negative_value, opt.positive_value))
    if not np.all(valid):
        complete = complete[valid]
        y_raw = y_raw[valid]
    y = (y_raw == opt.positive_value).astype(float)
    if len(np.unique(y)) < 2:
        raise AnalysisError(f"'{outcome_label}' needs both outcome values present (negative = {opt.negative_value:g}, positive = {opt.positive_value:g}), not just one.")

    predictor_columns = [complete[:, i] for i in predictor_idx]
    Z_all, names_all, cat_info, sources_all = _build_design(predictor_labels, predictor_columns, opt)
    n = y.size
    sig = opt.significant_digits

    design = np.column_stack([np.ones(n), Z_all]) if opt.include_intercept else Z_all
    term_names = (["Intercept"] if opt.include_intercept else []) + names_all
    p_terms = design.shape[1]
    if n < 3 * p_terms + 5:
        raise AnalysisError(f"Need more complete rows for {p_terms} model term(s); found {n}.")

    from . import _regress as rg
    from . import logistic_regression as simple

    spread = np.where(np.std(design, axis=0) > 0, np.std(design, axis=0), 1.0)
    beta, cov, _top, settled = simple.fit(design, y)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    lost = rg.runaway(beta, se, spread)
    split = bool(lost) or not settled      # a term (or a mix of them) predicts the outcome perfectly: no finite fit
    penalised = opt.fit_method == "firth" or (opt.fit_method == "auto" and split)
    if penalised:
        beta, cov, top, settled = simple.fit(design, y, True)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    else:
        top = simple._loglik(beta, design, y, False)
    log_lik = simple._loglik(beta, design, y, False)   # the plain log-likelihood, for deviance, AIC and the R² measures
    z_scores = beta / np.where(se > 1e-12, se, np.nan)
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    odds_ratio = np.exp(beta)
    z_crit = stats.norm.ppf(0.5 + opt.confidence_level / 2)
    pct = int(round(opt.confidence_level * 100))
    pos_label, neg_label = opt.positive_label or f"{opt.positive_value:g}", opt.negative_label or f"{opt.negative_value:g}"

    if opt.ci_approach == "profile" and opt.show_ci and penalised and p_terms <= 8:
        # under Firth's fit the profile is of the penalised likelihood (Heinze & Schemper 2002)
        cis = [simple.profile_limits(i, beta, se, design, y, top, opt.confidence_level, True) for i in range(p_terms)]
        beta_lo = np.array([c[0] if c[0] is not None else -np.inf for c in cis])
        beta_hi = np.array([c[1] if c[1] is not None else np.inf for c in cis])
        ci_method_note = "profile of the penalised likelihood"
    elif opt.ci_approach == "profile" and opt.show_ci and not (split and not penalised):  # profile CIs refit the model: only when shown
        cis = [_profile_ci_logistic(i, beta, se, design, y, -log_lik, opt.confidence_level) for i in range(p_terms)]
        beta_lo = np.array([(c[0] if c else beta[i] - z_crit * se[i]) for i, c in enumerate(cis)])
        beta_hi = np.array([(c[1] if c else beta[i] + z_crit * se[i]) for i, c in enumerate(cis)])
        ci_method_note = "profile-likelihood" if p_terms <= 8 else "Wald (profile-likelihood skipped - more than 8 terms)"
    else:
        beta_lo, beta_hi = beta - z_crit * se, beta + z_crit * se
        ci_method_note = "Wald"
    with np.errstate(over="ignore"):   # a limit that runs off the scale is shown as 0 or inf
        or_lo, or_hi = np.exp(beta_lo), np.exp(beta_hi)

    if opt.run_significance_test:
        lrt_p_values = []
        for i in range(p_terms):
            if penalised:   # the term held at zero inside the same penalised likelihood
                lrt_stat_i = 2 * (top - simple.fit(design, y, True, {i: 0.0}, start=beta)[2])
                lrt_p_values.append(float(stats.chi2.sf(max(lrt_stat_i, 0), 1)))
                continue
            keep = [j for j in range(p_terms) if j != i]
            design_r = design[:, keep] if keep else np.empty((n, 0))
            if design_r.shape[1]:
                _b_r, _c_r, log_lik_r = _fit_logistic(design_r, y)
            else:
                log_lik_r = _null_log_lik(y)
            lrt_stat_i = 2 * (log_lik - log_lik_r)
            lrt_p_values.append(float(stats.chi2.sf(max(lrt_stat_i, 0), 1)))
        p_values = np.array(lrt_p_values)
        p_value_label = "P value (LRT)"
    else:
        p_value_label = "P value (Wald)"

    rows = [[nm, f"{b:.{sig}g}", f"{o:.{sig}g}", f"{lo:.{sig}g} to {hi:.{sig}g}", f"{pv:.4g}" if np.isfinite(pv) else "-"] for nm, b, o, lo, hi, pv in zip(term_names, beta, odds_ratio, or_lo, or_hi, p_values)]
    headers = ["Term", "Coefficient"] + (["SE"] if opt.show_se and opt.ci_approach == "wald" else []) + ["Odds ratio", f"{pct}% CI ({ci_method_note})", p_value_label]
    if opt.show_se and opt.ci_approach == "wald":
        for row, s in zip(rows, se):
            row.insert(2, f"{s:.{sig}g}")
    if not opt.show_ci:
        ci_col = headers.index(f"{pct}% CI ({ci_method_note})")
        headers.pop(ci_col)
        for row in rows:
            row.pop(ci_col)
    tables = [ResultTable(title=f"How each predictor changes the odds that {outcome_label} = {pos_label}", headers=headers, rows=rows)]
    charts = []

    accuracy = float(np.mean((1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30))) >= opt.classification_cutoff) == y))
    null_ll = _null_log_lik(y)
    deviance = -2 * log_lik
    lr_stat = 2 * (log_lik - null_ll)
    lr_df = p_terms - (1 if opt.include_intercept else 0)
    if penalised and opt.include_intercept and lr_df > 0:   # every predictor held at zero, inside the penalised likelihood
        lr_stat = 2 * (top - simple.fit(design, y, True, {j: 0.0 for j in range(1, p_terms)})[2])
    lr_p = float(stats.chi2.sf(max(lr_stat, 0), lr_df))
    tables.append(ResultTable(
        title="The model as a whole",
        headers=["Log-likelihood", "Deviance", "Test against no predictors" + (" (penalised)" if penalised else ""),
                 f"Rows called right at {opt.classification_cutoff:g}"],
        rows=[[f"{log_lik:.{sig}g}", f"{deviance:.{sig}g}", f"chi-square = {lr_stat:.4f}, df = {lr_df}, P = {lr_p:.4g}",
               f"{accuracy*100:.1f}%"]],
    ))

    if opt.show_aicc or opt.show_loglik_both or opt.show_deviance_both:
        diag_rows = []
        if opt.show_aicc:
            k = p_terms
            aic = 2 * k - 2 * log_lik
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("nan")
            diag_rows.append(["AICc", f"{aicc:.{sig}g}" if aicc == aicc else "n/a"])
        if opt.show_loglik_both:
            diag_rows.append(["Log-likelihood, this model", f"{log_lik:.{sig}g}"])
            diag_rows.append(["Log-likelihood, no predictors", f"{null_ll:.{sig}g}"])
        if opt.show_deviance_both:
            diag_rows.append(["Deviance, this model", f"{deviance:.{sig}g}"])
            diag_rows.append(["Deviance, no predictors", f"{-2*null_ll:.{sig}g}"])
        tables.append(ResultTable(title="For comparing with other models of these data", headers=["Measure", "Value"], rows=diag_rows))

    if opt.show_multicollinearity and Z_all.shape[1] >= 2:
        corr = np.corrcoef(Z_all, rowvar=False)
        try:
            vif = np.diag(np.linalg.inv(corr))
        except np.linalg.LinAlgError:
            vif = np.full(Z_all.shape[1], float("nan"))
        tables.append(ResultTable(title="How far each predictor is explained by the others", headers=["Predictor", "Variance inflation (VIF)"], rows=[[nm, f"{v:.{sig}g}" if v == v else "n/a"] for nm, v in zip(names_all, vif)]))
    if opt.show_correlation_matrix and Z_all.shape[1] >= 2:
        corr = np.corrcoef(Z_all, rowvar=False)
        tables.append(ResultTable(title="Correlations between the predictors", headers=["Predictor"] + names_all, rows=[[names_all[i]] + [f"{corr[i,j]:.{sig}g}" for j in range(len(names_all))] for i in range(len(names_all))]))

    summary = (f"{len(predictor_labels)} predictor(s) of '{outcome_label}'; against no predictors P = {lr_p:.4g}; "
               f"{accuracy*100:.1f}% of rows called right.")
    notes = [
        f"'{outcome_label}' is the outcome (yes = {pos_label}, no = {neg_label}); every other variable is a predictor. "
        f"{n} complete rows, {int(np.sum(y))} with the outcome.",
        "A predictor with six distinct values or fewer is treated as a category and compared with a reference level.",
        "Fitted by " + ("Firth's penalised likelihood (Firth 1993; Heinze & Schemper 2002)." if penalised
                        else "maximum likelihood, with no penalty."),
    ]
    if penalised and opt.fit_method == "auto":
        notes.insert(0, "In these data the predictors split yes from no completely, so the ordinary fit has no finite answer: "
                        "its coefficients run off towards infinity. The fit shown is by Firth's penalised likelihood, which "
                        "always has one. Its intervals are wide because the data cannot say how strong the effect is.")
    elif split and not penalised:
        notes.insert(0, rg.runaway_note([term_names[j] for j in lost] or ["one or more terms"],
                                        "a term predicts the outcome perfectly") + " Firth's fit gives a finite answer.")
    rarer = int(min(np.sum(y), n - np.sum(y)))
    if rarer < 10 * max(1, p_terms - 1) and not penalised:
        notes.append(f"The rarer outcome has {rarer} rows for {max(1, p_terms - 1)} model term(s) - fewer than ten each. "
                     "Estimates from so few lean away from zero; Firth's fit is less biased.")

    # ---- how well it fits ----
    probs = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
    if (opt.gof_roc or opt.graph_roc) and len(np.unique(y)) != 2:
        from ..chart_helpers import unavailable_chart

        charts.append(unavailable_chart(
            "ROC curve (fitted model)",
            f"A ROC curve needs both outcomes (0 and 1) of '{outcome_label}' among the complete rows; only one was found.",
        ))
    # the area (a table) and the curve (a graph) are separate choices
    if (opt.gof_roc or opt.graph_roc) and len(np.unique(y)) == 2:
        try:
            auc = float(roc_auc_score(y, probs))
            fpr, tpr, thresh = roc_curve(y, probs)
            step = max(1, len(thresh) // 20)
            from ..chart_helpers import roc_chart

            if opt.graph_roc:
                charts.append(roc_chart("ROC curve (fitted model)", fpr, tpr, auc))
            if opt.gof_roc:
                tables.append(ResultTable(title=f"How well the model tells yes from no (area under the ROC curve = {auc:.4f})", headers=["Threshold", "False positive rate", "True positive rate"], rows=[[f"{thresh[i]:.{sig}g}" if np.isfinite(thresh[i]) else "-", f"{fpr[i]:.{sig}g}", f"{tpr[i]:.{sig}g}"] for i in range(0, len(thresh), step)]))
        except ValueError:
            pass
    if opt.gof_classification_table:
        pred = (probs >= opt.classification_cutoff).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tables.append(ResultTable(
            title=f"Rows called {pos_label} at a predicted chance of {opt.classification_cutoff:g} or more",
            headers=["", f"Called {neg_label}", f"Called {pos_label}"],
            rows=[[f"Was {neg_label}", str(tn), str(fp)], [f"Was {pos_label}", str(fn), str(tp)]],
        ))
    if opt.gof_predicted_prob:
        tables.append(ResultTable(title="Predicted chance for each row", headers=["Outcome", "Predicted chance"], rows=[[f"{yi:g}", f"{pi:.{sig}g}"] for yi, pi in zip(y, probs)][:100]))

    if any((opt.pseudo_r2_tjur, opt.pseudo_r2_mcfadden, opt.pseudo_r2_coxsnell, opt.pseudo_r2_nagelkerke)):
        r2_rows = []
        if opt.pseudo_r2_tjur and np.any(y == 1) and np.any(y == 0):
            tjur = float(np.mean(probs[y == 1]) - np.mean(probs[y == 0]))
            r2_rows.append(["Tjur's R²", f"{tjur:.{sig}g}"])
        if opt.pseudo_r2_mcfadden:
            mcfadden = 1 - log_lik / null_ll if null_ll else float("nan")
            r2_rows.append(["McFadden's R²", f"{mcfadden:.{sig}g}"])
        cox_snell = 1 - np.exp((2.0 / n) * (null_ll - log_lik))
        if opt.pseudo_r2_coxsnell:
            r2_rows.append(["Cox and Snell's R²", f"{cox_snell:.{sig}g}"])
        if opt.pseudo_r2_nagelkerke:
            max_cox_snell = 1 - np.exp((2.0 / n) * null_ll)
            nagelkerke = cox_snell / max_cox_snell if max_cox_snell else float("nan")
            r2_rows.append(["Nagelkerke's R²", f"{nagelkerke:.{sig}g}" if nagelkerke == nagelkerke else "n/a"])
        tables.append(ResultTable(title="R²-like measures, for comparing models", headers=["Measure", "Value"], rows=r2_rows))

    if opt.gof_hosmer_lemeshow or opt.gof_lrt:
        hl_rows = []
        if opt.gof_hosmer_lemeshow and n >= 20:
            hl_chi2, hl_df, hl_p = _hosmer_lemeshow(y, probs)
            hl_rows.append(["Hosmer-Lemeshow test of calibration", f"chi-square = {hl_chi2:.4f}, df = {hl_df}", f"{hl_p:.4g}"])
        elif opt.gof_hosmer_lemeshow:
            hl_rows.append(["Hosmer-Lemeshow test of calibration", "n/a", "needs 20 rows or more"])
        if opt.gof_lrt:
            hl_rows.append(["Likelihood-ratio test against no predictors", f"chi-square = {lr_stat:.4f}, df = {lr_df}", f"{lr_p:.4g}"])
        tables.append(ResultTable(title="Tests of the model", headers=["Test", "Statistic", "P value"], rows=hl_rows))

    # ---- can the predicted chances be trusted; which rows pull the fit ----
    if opt.gof_brier:
        score = float(np.mean((probs - y) ** 2))
        base = float(np.mean(y) * (1 - np.mean(y)))
        tables.append(ResultTable(title="Can the predicted chances be trusted?", headers=["Measure", "Value"],
                                  rows=[["Brier score (0 is perfect)", f"{score:.{sig}g}"],
                                        ["Brier score with no predictors", f"{base:.{sig}g}"],
                                        ["Share of that error removed by the model", f"{(1 - score / base) * 100:.3g}%"]]))
    if opt.graph_calibration:
        from ..backend_types import ChartSeries as _Series, ChartSpec as _Chart

        grouped = simple.hosmer_lemeshow(y, probs)[3]
        charts.append(_Chart(title="Predicted chance against what happened",
                             series=[_Series("Groups of rows", [g[0] for g in grouped], [g[1] for g in grouped], kind="scatter")],
                             x_label="Mean predicted chance in the group", y_label=f"Share that were {pos_label}",
                             x_limits=(-0.03, 1.03), y_limits=(-0.03, 1.03),
                             reference_lines=[{"kind": "line", "points": [[0, 0], [1, 1]], "style": ":"}]))
    if opt.show_influence or opt.graph_influence:
        from ..backend_types import ChartSeries as _Series, ChartSpec as _Chart

        leverage, standard, _deviance_res, cook = simple.influence_measures(design, y, probs, cov)
        sheet_rows = np.where(complete_mask)[0][valid] + 1 if not np.all(valid) else np.where(complete_mask)[0] + 1
        if opt.show_influence:
            order = np.argsort(-cook)[:10]
            flagged = int(np.sum(cook > 4 / n))
            tables.append(ResultTable(
                title=f"The rows that pull the fit hardest ({flagged} with Cook's distance above 4/n = {4 / n:.3g})",
                headers=["Sheet row", "Outcome", "Predicted chance", "Standardised residual", "Leverage", "Cook's distance"],
                rows=[[str(int(sheet_rows[i])), pos_label if y[i] == 1 else neg_label, f"{probs[i]:.{sig}g}",
                       f"{standard[i]:.{sig}g}", f"{leverage[i]:.{sig}g}", f"{cook[i]:.{sig}g}"] for i in order]))
        if opt.graph_influence:
            charts.append(_Chart(title="How hard each row pulls the fit",
                                 series=[_Series("Rows", [float(v) for v in sheet_rows], cook.tolist(), kind="scatter")],
                                 x_label="Sheet row", y_label="Cook's distance",
                                 reference_lines=[{"kind": "h", "value": 4 / n, "style": ":"}]))

    # ---- graphs ----
    from ..backend_types import ChartSeries, ChartSpec
    from ..chart_helpers import xy_chart

    if opt.graph_predicted_vs_observed:
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, y.size)  # separate the stacked 0/1 columns
        charts.append(ChartSpec(title="Predicted chance, by what happened",
                                series=[ChartSeries("", [float(v) for v in y + jitter], [float(p_) for p_ in probs])],
                                x_categories=[f"Was {neg_label}", f"Was {pos_label}"], y_label="Predicted chance",
                                reference_lines=[{"kind": "h", "value": float(opt.classification_cutoff)}], y_limits=(-0.03, 1.03)))
        tables.append(ResultTable(title="Predicted chance, by what happened", headers=["Outcome", "Predicted chance"], rows=[[f"{yi:g}", f"{pi:.{sig}g}"] for yi, pi in zip(y, probs)][:100]))
    if opt.graph_logistic_plot and len(predictor_labels) == 1 and not cat_info:
        x_single = predictor_columns[0]
        order = np.argsort(x_single)
        rows_lp = []
        for xi, pi in zip(x_single[order], probs[order]):
            row = [f"{xi:.{sig}g}", f"{pi:.{sig}g}"]
            if opt.plot_confidence_bands:
                design_vec = np.array([1.0, xi]) if opt.include_intercept else np.array([xi])
                var_eta = float(design_vec @ cov @ design_vec)
                se_eta = np.sqrt(max(var_eta, 0.0))
                eta_i = float(design_vec @ beta)
                lo_eta, hi_eta = eta_i - z_crit * se_eta, eta_i + z_crit * se_eta
                row.append(f"{1/(1+np.exp(-lo_eta)):.{sig}g} to {1/(1+np.exp(-hi_eta)):.{sig}g}")
            rows_lp.append(row)
        headers_lp = [predictor_labels[0], "Predicted chance"] + ([f"{pct}% CI"] if opt.plot_confidence_bands else [])
        charts.append(ChartSpec(
            title="Logistic curve",
            series=[
                ChartSeries("Observed", [float(v) for v in x_single], [float(v) for v in y]),
                ChartSeries("Fitted chance", [float(v) for v in x_single[order]], [float(v) for v in probs[order]], kind="line"),
            ],
            x_label=predictor_labels[0], y_label=f"P({outcome_label} = {pos_label})", y_limits=(-0.05, 1.05),
        ))
        tables.append(ResultTable(title="The fitted curve, point by point", headers=headers_lp, rows=rows_lp[:200]))
    elif opt.graph_logistic_plot and (len(predictor_labels) != 1 or cat_info):
        notes.append("The S-curve can be drawn only for a model with one measured predictor; this model has more, or a category, so it was not drawn.")
        from ..chart_helpers import unavailable_chart

        charts.append(unavailable_chart(
            "Logistic curve",
            f"The logistic curve plots P({outcome_label} = {pos_label}) against a single continuous predictor. "
            f"This model has {len(predictor_labels)} predictor(s)"
            + (" including a categorical one" if cat_info else "")
            + ". Run the analysis with just one continuous predictor (for example X and the outcome) to see it.",
        ))
    if opt.graph_proportion_correct:
        cutoffs = np.linspace(0.05, 0.95, 19)
        pc_rows = [[f"{c:.2f}", f"{float(np.mean((probs >= c).astype(int) == y))*100:.{sig}g}%"] for c in cutoffs]
        charts.append(xy_chart("Share called right at every cutoff", cutoffs,
                               {"": [float(np.mean((probs >= c).astype(int) == y)) * 100 for c in cutoffs]},
                               x_label="Cutoff", y_label="Rows called right (%)"))
        tables.append(ResultTable(title="Share called right at every cutoff", headers=["Cutoff", "Called right"], rows=pc_rows))

    if opt.compare_models and opt.dropped_predictors:
        dropped_set = set(opt.dropped_predictors)
        keep_idx = [i for i, srcs in enumerate(sources_all) if not (srcs & dropped_set)]
        if len(keep_idx) < len(names_all):
            Z_reduced = Z_all[:, keep_idx] if keep_idx else np.empty((n, 0))
            design_r = np.column_stack([np.ones(n), Z_reduced]) if opt.include_intercept else Z_reduced
            _beta_r, _cov_r, log_lik_r = _fit_logistic(design_r, y) if design_r.shape[1] else (np.zeros(0), np.zeros((0, 0)), null_ll)
            df_diff = design.shape[1] - design_r.shape[1]
            lrt_stat = 2 * (log_lik - log_lik_r)
            p_lrt = float(stats.chi2.sf(max(lrt_stat, 0), df_diff))
            tables.append(ResultTable(
                title="Would a simpler model do as well? Without " + ", ".join(opt.dropped_predictors),
                headers=["Model", "Log-likelihood", "Deviance"],
                rows=[["This model", f"{log_lik:.{sig}g}", f"{-2*log_lik:.{sig}g}"], ["The simpler one", f"{log_lik_r:.{sig}g}", f"{-2*log_lik_r:.{sig}g}"]],
            ))
            tables.append(ResultTable(title="Likelihood-ratio test of this model against the simpler one", headers=["Chi-square", "df", "P value"], rows=[[f"{lrt_stat:.4f}", str(df_diff), f"{p_lrt:.4g}"]]))

    return AnalysisResult(title="Several predictors, yes/no", summary=summary, tables=tables, notes=notes, charts=charts)
