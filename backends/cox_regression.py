"""What changes the risk of an event (Cox proportional hazards regression) - a from-scratch Newton-type fit
of the Cox partial likelihood (Efron or Breslow tie handling), written
here so that the program needs no further library. Reads the sheet
the same way Kaplan-Meier does (X = time, one group = event indicator),
with every other group treated as a covariate - detected-categorical
covariates (<=6 distinct values) get dummy-coded against a reference
level, the rest are used as continuous covariates.

Each residual check and each predicted curve asked for adds a graph to
the results, with the numbers behind it as a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class CoxRegressionOptions:
    # the model
    event_index: int = 0  # which group column is the event/censor indicator
    censored_value: float = 0.0
    event_value: float = 1.0
    ties_method: str = "efron"  # "efron" | "breslow"
    # reference levels: covariate label -> chosen reference level (string of the numeric level); absent/auto = first level
    reference_levels: dict[str, str] = field(default_factory=dict)
    auto_reference: dict[str, bool] = field(default_factory=dict)
    # predictions
    predict_missing_rows: bool = True
    scenarios: list[dict[str, float]] = field(default_factory=list)  # each: {covariate_label: value}
    # against a simpler model
    compare_models: bool = False
    dropped_covariates: list[str] = field(default_factory=list)
    # what to report
    ci_approach: str = "profile"  # "profile" | "wald"
    show_se: bool = False
    show_ci: bool = True  # a confidence interval for every hazard ratio
    run_significance_test: bool = False
    show_multicollinearity: bool = False
    show_covariance_matrix: bool = False
    show_aic: bool = True
    show_partial_loglik: bool = False
    show_neg2ll: bool = False
    show_pseudo_r2: bool = False
    confidence_level: float = 0.95
    significant_digits: int = 4
    # how well it fits
    gof_lrt: bool = False
    gof_wald: bool = False
    gof_score: bool = False
    gof_concordance: bool = False
    # checks on the residuals
    check_schoenfeld_ph: bool = True
    check_loglog_survival: bool = False
    check_deviance_vs_linear: bool = True
    check_martingale_vs_linear: bool = False
    check_schoenfeld_vs_time: bool = False
    check_deviance_vs_covariate: bool = True
    check_martingale_vs_covariate: bool = False
    check_coxsnell: bool = False
    # graphs: each entry is {"title": str, "include_baseline": bool, "covariate_values": dict[str, float]}
    graph_defs: list[dict] = field(default_factory=list)


def _risk_set_sums(beta: np.ndarray, Z: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Everything both tie methods need, for every distinct event time at
    once: the risk-set sums come from cumulative sums over the subjects
    sorted by time (a subject is at risk at t when its time >= t), and the
    sums over the deaths at each time from grouped additions - no loop over
    event times."""
    linear_pred = Z @ beta
    risk_scores = np.exp(linear_pred)
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    r_sorted = risk_scores[order]
    zr_sorted = Z[order] * r_sorted[:, None]
    # suffix sums: at position i, the sum over subjects i..n-1 (time >= t_sorted[i])
    suffix_r = np.cumsum(r_sorted[::-1])[::-1]
    suffix_zr = np.cumsum(zr_sorted[::-1], axis=0)[::-1]

    dead = event == 1
    event_times = np.unique(time[dead])
    first = np.searchsorted(t_sorted, event_times, side="left")
    sum_risk = suffix_r[first]
    sum_z_risk = suffix_zr[first]

    group = np.searchsorted(event_times, time[dead])  # which event time each death belongs to
    k = event_times.size
    d_k = np.bincount(group, minlength=k).astype(float)
    sum_lp_deaths = np.bincount(group, weights=linear_pred[dead], minlength=k)
    sum_risk_deaths = np.bincount(group, weights=risk_scores[dead], minlength=k)
    p = Z.shape[1]
    sum_z_deaths_plain = np.zeros((k, p))
    sum_z_deaths_risk = np.zeros((k, p))
    np.add.at(sum_z_deaths_plain, group, Z[dead])
    np.add.at(sum_z_deaths_risk, group, Z[dead] * risk_scores[dead, None])
    return (d_k, sum_risk, sum_z_risk, sum_lp_deaths, sum_risk_deaths, sum_z_deaths_plain, sum_z_deaths_risk)


def _neg_log_partial_likelihood_and_grad(beta: np.ndarray, Z: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Breslow's partial likelihood (ties share one risk-set denominator)."""
    d_k, sum_risk, sum_z_risk, sum_lp, _srd, sum_z_plain, _szr = _risk_set_sums(beta, Z, time, event)
    log_lik = float(np.sum(sum_lp - d_k * np.log(sum_risk)))
    grad = np.sum(sum_z_plain - d_k[:, None] * (sum_z_risk / sum_risk[:, None]), axis=0)
    return -log_lik, -grad


def _efron_neg_log_partial_likelihood_and_grad(beta: np.ndarray, Z: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Efron's approximation for tied event times - more accurate than
    Breslow's when there are many ties, at the same computational cost."""
    d_k, sum_risk, sum_z_risk, sum_lp, sum_risk_deaths, sum_z_plain, sum_z_deaths = _risk_set_sums(beta, Z, time, event)
    # the l-th of d tied deaths (l = 0..d-1) uses denominator S - (l/d)·S_deaths
    counts = d_k.astype(int)
    idx = np.repeat(np.arange(counts.size), counts)
    within = np.arange(idx.size) - np.repeat(np.cumsum(counts) - counts, counts)
    frac = within / np.repeat(d_k, counts)
    denom = sum_risk[idx] - frac * sum_risk_deaths[idx]
    log_lik = float(np.sum(sum_lp) - np.sum(np.log(denom)))
    numer = sum_z_risk[idx] - frac[:, None] * sum_z_deaths[idx]
    grad = np.sum(sum_z_plain, axis=0) - np.sum(numer / denom[:, None], axis=0)
    return -log_lik, -grad


def _neg_log_partial_likelihood_and_grad_loop(beta, Z, time, event):
    """Reference (event time by event time) - kept for the tests."""
    linear_pred = Z @ beta
    risk_scores = np.exp(linear_pred)
    log_lik, grad = 0.0, np.zeros(Z.shape[1])
    for t in np.unique(time[event == 1]):
        risk_set = time >= t
        event_set = (time == t) & (event == 1)
        d = int(np.sum(event_set))
        s_r = np.sum(risk_scores[risk_set])
        s_zr = np.sum(Z[risk_set] * risk_scores[risk_set, None], axis=0)
        log_lik += np.sum(linear_pred[event_set]) - d * np.log(s_r)
        grad += np.sum(Z[event_set], axis=0) - d * (s_zr / s_r)
    return -log_lik, -grad


def _efron_loop(beta, Z, time, event):
    """Reference (event time by event time) - kept for the tests."""
    linear_pred = Z @ beta
    risk_scores = np.exp(linear_pred)
    log_lik, grad = 0.0, np.zeros(Z.shape[1])
    for t in np.unique(time[event == 1]):
        risk_set = time >= t
        event_set = (time == t) & (event == 1)
        d = int(np.sum(event_set))
        s_r = np.sum(risk_scores[risk_set])
        s_zr = np.sum(Z[risk_set] * risk_scores[risk_set, None], axis=0)
        s_rd = np.sum(risk_scores[event_set])
        s_zd = np.sum(Z[event_set] * risk_scores[event_set, None], axis=0)
        log_lik += np.sum(linear_pred[event_set])
        grad += np.sum(Z[event_set], axis=0)
        for l in range(d):
            frac = l / d
            denom = s_r - frac * s_rd
            log_lik -= np.log(denom)
            grad -= (s_zr - frac * s_zd) / denom
    return -log_lik, -grad


def _numerical_hessian(fn_grad, beta: np.ndarray, Z, time, event, eps: float = 1e-5) -> np.ndarray:
    p = beta.size
    hessian = np.zeros((p, p))
    for i in range(p):
        step = np.zeros(p)
        step[i] = eps
        _, grad_plus = fn_grad(beta + step, Z, time, event)
        _, grad_minus = fn_grad(beta - step, Z, time, event)
        hessian[:, i] = (grad_plus - grad_minus) / (2 * eps)
    return hessian


def _fit_cox(Z: np.ndarray, time: np.ndarray, event: np.ndarray, ties_method: str):
    """Fit a Cox model on a (possibly empty, p=0) design matrix. Returns
    beta_std, cov_std, means, sds, neg_log_lik, fit_fn - all on the
    standardized covariate scale, since that's what the optimizer and
    Hessian are computed against."""
    fit_fn = _efron_neg_log_partial_likelihood_and_grad if ties_method == "efron" else _neg_log_partial_likelihood_and_grad
    n, p = Z.shape
    if p == 0:
        neg_ll, _ = fit_fn(np.zeros(0), Z, time, event)
        return np.zeros(0), np.zeros((0, 0)), np.zeros(0), np.ones(0), neg_ll, fit_fn

    means, sds = Z.mean(axis=0), Z.std(axis=0, ddof=1)
    sds = np.where(sds == 0, 1.0, sds)
    Z_std = (Z - means) / sds

    result = minimize(
        lambda b: fit_fn(b, Z_std, time, event)[0],
        np.zeros(p),
        jac=lambda b: fit_fn(b, Z_std, time, event)[1],
        method="BFGS",
    )
    if not result.success:
        raise AnalysisError(f"The Cox model didn't converge: {result.message}")

    hessian = _numerical_hessian(fit_fn, result.x, Z_std, time, event)
    try:
        cov_std = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        raise AnalysisError("The information matrix is singular - covariates may be collinear.") from None

    return result.x, cov_std, means, sds, result.fun, fit_fn


def _build_design(cov_columns: list[np.ndarray], covariate_labels: list[str], opt: CoxRegressionOptions):
    """Continuous covariates pass through as-is; a covariate with <=6
    distinct values is treated as categorical and dummy-coded against a
    reference level (auto = its first observed level, the usual
    own default)."""
    columns, names, cat_info = [], [], {}
    for label, values in zip(covariate_labels, cov_columns):
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
        else:
            columns.append(values)
            names.append(label)
    n = len(cov_columns[0]) if cov_columns else 0
    Z = np.column_stack(columns) if columns else np.empty((n, 0))
    return Z, names, cat_info


def _profile_ci(idx: int, beta_std: np.ndarray, se_std: np.ndarray, Z_std: np.ndarray, time: np.ndarray, event: np.ndarray, fit_fn, full_neg_ll: float, confidence_level: float) -> tuple[float, float] | None:
    """Profile-likelihood CI for one standardized coefficient: the range
    of that coefficient for which, re-optimizing every other coefficient,
    the log-likelihood drops by at most half the chi-square(1) critical
    value - the conventional definition."""
    p = beta_std.size
    if p > 8:
        return None  # re-optimizing p-1 params per evaluation gets slow; fall back to Wald above this size
    target_drop = stats.chi2.ppf(confidence_level, 1) / 2
    target_ll = -full_neg_ll - target_drop
    other_idx = [i for i in range(p) if i != idx]

    def profile_neg_ll(beta_k: float) -> float:
        if not other_idx:
            return fit_fn(np.array([beta_k]), Z_std, time, event)[0]

        def obj(b_other):
            b = np.empty(p)
            b[idx] = beta_k
            b[other_idx] = b_other
            return fit_fn(b, Z_std, time, event)[0]

        def obj_jac(b_other):
            b = np.empty(p)
            b[idx] = beta_k
            b[other_idx] = b_other
            return fit_fn(b, Z_std, time, event)[1][other_idx]

        start = np.delete(beta_std, idx)
        res = minimize(obj, start, jac=obj_jac, method="BFGS")
        return res.fun

    def g(beta_k: float) -> float:
        return -profile_neg_ll(beta_k) - target_ll

    # Exponentially growing bracket (few evaluations - each one refits
    # p-1 parameters, so linear stepping would be far too slow) starting
    # from the Wald SE, which is normally within a factor of ~2-3 of the
    # true profile bound.
    step = max(se_std[idx], 1e-3)
    lo = hi = None
    b0 = beta_std[idx]
    prev_offset = 0.0
    offset = step
    for _ in range(10):
        if g(b0 - offset) < 0:
            lo = brentq(g, b0 - offset, b0 - prev_offset)
            break
        prev_offset, offset = offset, offset * 2
    prev_offset = 0.0
    offset = step
    for _ in range(10):
        if g(b0 + offset) < 0:
            hi = brentq(g, b0 + prev_offset, b0 + offset)
            break
        prev_offset, offset = offset, offset * 2
    if lo is None or hi is None:
        return None
    return lo, hi


def _encode_scenario(scenario: dict[str, float], covariate_labels: list[str], cat_info: dict) -> np.ndarray:
    """Build one design-matrix row for a hypothetical covariate scenario,
    using the SAME reference levels the fitted model actually used - so
    a categorical covariate's dummy columns line up with ``beta``."""
    vec = []
    for label in covariate_labels:
        val = scenario.get(label, 0.0)
        if label in cat_info:
            levels, ref = cat_info[label]
            for lvl in levels:
                if lvl == ref:
                    continue
                vec.append(1.0 if val == lvl else 0.0)
        else:
            vec.append(val)
    return np.array(vec)


def _harrell_c(Z: np.ndarray, beta: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    risk = Z @ beta
    n = time.size
    ti, tj = time[:, None], time[None, :]
    ei = event[:, None]
    comparable = (ei == 1) & (ti < tj)
    ri, rj = risk[:, None], risk[None, :]
    concordant = comparable & (ri > rj)
    tied = comparable & (ri == rj)
    denom = comparable.sum()
    if denom == 0:
        return float("nan")
    return float((concordant.sum() + 0.5 * tied.sum()) / denom)


def _breslow_baseline_hazard(Z: np.ndarray, beta: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Breslow's estimator of the cumulative baseline hazard H0(t) - the
    standard, tie-method-agnostic baseline used for martingale/deviance/
    Cox-Snell residuals and for predicted survival curves, regardless of
    whether the coefficients themselves were fit with Efron or Breslow."""
    risk_scores = np.exp(Z @ beta) if beta.size else np.ones(time.size)
    unique_times = np.unique(time[event == 1])
    H0 = np.zeros(unique_times.size)
    running = 0.0
    for i, t in enumerate(unique_times):
        risk_set = time >= t
        d_t = int(np.sum((time == t) & (event == 1)))
        sum_risk = np.sum(risk_scores[risk_set])
        running += d_t / sum_risk if sum_risk else 0.0
        H0[i] = running
    return unique_times, H0


def _cumulative_hazard_at(unique_times: np.ndarray, H0: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(unique_times, t, side="right") - 1
    result = np.zeros(t.size)
    valid = idx >= 0
    result[valid] = H0[idx[valid]]
    return result


def _martingale_residuals(Z: np.ndarray, beta: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_times, H0 = _breslow_baseline_hazard(Z, beta, time, event)
    linear_pred = Z @ beta if beta.size else np.zeros(time.size)
    H_i = _cumulative_hazard_at(unique_times, H0, time) * np.exp(linear_pred)
    martingale = event - H_i
    return martingale, H_i, linear_pred


def _deviance_residuals(martingale: np.ndarray, event: np.ndarray) -> np.ndarray:
    inner = event - martingale  # equals H_i for events, and for censored equals -martingale = H_i too
    inner = np.clip(inner, 1e-12, None)
    term = -2 * (martingale + event * np.log(inner))
    return np.sign(martingale) * np.sqrt(np.clip(term, 0, None))


def _schoenfeld_residuals(Z: np.ndarray, beta: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw Schoenfeld residuals (one row per event time, one column per
    covariate): the covariate's observed value at that event minus the
    risk-set's exposure-weighted mean covariate value at that time."""
    linear_pred = Z @ beta if beta.size else np.zeros(time.size)
    risk_scores = np.exp(linear_pred)
    event_times = np.unique(time[event == 1])
    p = Z.shape[1]
    residuals = np.zeros((event_times.size, p))
    for i, t in enumerate(event_times):
        risk_set = time >= t
        event_here = (time == t) & (event == 1)
        weights = risk_scores[risk_set]
        z_bar = np.sum(Z[risk_set] * weights[:, None], axis=0) / np.sum(weights)
        residuals[i] = np.mean(Z[event_here], axis=0) - z_bar
    return event_times, residuals


def _nelson_aalen(values: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    v, e = values[order], event[order]
    unique_v = np.unique(v)
    n_at_risk = v.size
    H = np.zeros(unique_v.size)
    running = 0.0
    for i, val in enumerate(unique_v):
        at_this = v == val
        d = int(np.sum(e[at_this] == 1))
        running += d / n_at_risk if n_at_risk else 0.0
        H[i] = running
        n_at_risk -= int(np.sum(at_this))
    return unique_v, H


@register("cox_regression")
def run(data: SheetData, options: CoxRegressionOptions | None = None) -> AnalysisResult:
    opt = options or CoxRegressionOptions()
    labels = data.group_labels()
    if len(labels) < 2:
        raise AnalysisError("Cox regression needs the event/censor column plus at least one covariate.")

    event_idx = max(0, min(opt.event_index, len(labels) - 1))
    event_label = labels[event_idx]
    covariate_labels = [lbl for i, lbl in enumerate(labels) if i != event_idx]
    if not covariate_labels:
        raise AnalysisError("Cox regression needs at least one covariate besides the event/censor column.")

    # Row by row, blanks kept: a subject's time, event code and covariates must stay on one row.
    # (Reading each column with its blanks dropped would move every value below a blank up a row,
    # pairing it with another subject's time.)
    time_full = np.asarray(data.x, dtype=float)
    event_raw_full = data.group_rows(event_label)
    cov_columns_full = [data.group_rows(label) for label in covariate_labels]

    n_all = min([time_full.size, event_raw_full.size] + [c.size for c in cov_columns_full])
    time_full = time_full[:n_all]
    event_raw_full = event_raw_full[:n_all]
    cov_columns_full = [c[:n_all] for c in cov_columns_full]

    valid_code = np.isin(event_raw_full, (opt.censored_value, opt.event_value))
    covs_filled = ~np.any(np.isnan(np.column_stack(cov_columns_full)), axis=1)
    complete_mask = ~np.isnan(time_full) & valid_code & covs_filled
    # rows with something typed on them that cannot take part in the fit, and why
    typed = ~np.isnan(time_full) | ~np.isnan(event_raw_full) | np.any(~np.isnan(np.column_stack(cov_columns_full)), axis=1)
    left_out = {"a blank covariate": int(np.sum(typed & ~covs_filled)),
                "no time in X": int(np.sum(typed & covs_filled & np.isnan(time_full))),
                "no recognised event code": int(np.sum(typed & covs_filled & ~np.isnan(time_full) & ~valid_code))}

    time, event_raw = time_full[complete_mask], event_raw_full[complete_mask]
    cov_columns = [c[complete_mask] for c in cov_columns_full]
    event = (event_raw == opt.event_value).astype(int)

    Z_all, names_all, cat_info = _build_design(cov_columns, covariate_labels, opt)
    if Z_all.size and np.any(np.isnan(Z_all)):
        row_ok = ~np.any(np.isnan(Z_all), axis=1)
        time, event, Z_all = time[row_ok], event[row_ok], Z_all[row_ok]
        cov_columns = [c[row_ok] for c in cov_columns]

    if time.size < 5 * Z_all.shape[1] + 5:
        raise AnalysisError(f"Need more complete subjects for {Z_all.shape[1]} covariate(s); found {time.size}.")
    if np.sum(event) < 3:
        raise AnalysisError("Need at least 3 events (not censorings) to fit a Cox model.")

    beta_std, cov_std, means, sds, neg_ll, fit_fn = _fit_cox(Z_all, time, event, opt.ties_method)
    Z_std = (Z_all - means) / sds if Z_all.shape[1] else Z_all
    p = beta_std.size

    beta = beta_std / sds
    se_std = np.sqrt(np.clip(np.diag(cov_std), 0, None))
    se = se_std / sds
    z_crit = stats.norm.ppf(0.5 + opt.confidence_level / 2)
    hr = np.exp(beta)
    sig = opt.significant_digits
    pct = int(round(opt.confidence_level * 100))

    if opt.ci_approach == "profile" and opt.show_ci:  # profile CIs refit the model: only when shown
        cis = [_profile_ci(i, beta_std, se_std, Z_std, time, event, fit_fn, neg_ll, opt.confidence_level) for i in range(p)]
        ci_lo = np.array([(c[0] if c else beta_std[i] - z_crit * se_std[i]) for i, c in enumerate(cis)])
        ci_hi = np.array([(c[1] if c else beta_std[i] + z_crit * se_std[i]) for i, c in enumerate(cis)])
        ci_lo, ci_hi = ci_lo / sds, ci_hi / sds
        ci_method_note = "profile-likelihood" if p <= 8 else "Wald (profile-likelihood skipped - more than 8 covariates)"
    else:
        ci_lo, ci_hi = beta - z_crit * se, beta + z_crit * se
        ci_method_note = "Wald"

    hr_lo, hr_hi = np.exp(ci_lo), np.exp(ci_hi)
    z_scores = beta / se
    p_values_wald = 2 * stats.norm.sf(np.abs(z_scores))

    p_values = p_values_wald
    if opt.run_significance_test:
        p_values = []
        for i in range(p):
            keep = [j for j in range(p) if j != i]
            _b_r, _c_r, _m_r, _s_r, neg_ll_reduced, _f_r = _fit_cox(Z_all[:, keep], time, event, opt.ties_method) if keep else _fit_cox(np.empty((time.size, 0)), time, event, opt.ties_method)
            lrt_stat = 2 * (neg_ll_reduced - neg_ll)
            p_values.append(float(stats.chi2.sf(max(lrt_stat, 0), 1)))
        p_values = np.array(p_values)

    rows = []
    for i, label in enumerate(names_all):
        row = [label, f"{beta[i]:.{sig}g}", f"{hr[i]:.{sig}g}", f"{hr_lo[i]:.{sig}g} to {hr_hi[i]:.{sig}g}", f"{p_values[i]:.4g}"]
        if opt.show_se:
            row.insert(2, f"{se[i]:.{sig}g}")
        rows.append(row)
    headers = ["Covariate", "Coefficient"] + (["SE"] if opt.show_se else []) + ["Hazard ratio", f"{pct}% CI ({ci_method_note})", "P value" + (" (LRT)" if opt.run_significance_test else " (Wald)")]
    if not opt.show_ci:
        ci_col = headers.index(f"{pct}% CI ({ci_method_note})")
        headers.pop(ci_col)
        for row in rows:
            row.pop(ci_col)
    table = ResultTable(title="How each covariate changes the risk", headers=headers, rows=rows)

    tables = [table]
    notes = [
        f"Time is read from X; '{event_label}' is the event/censor column "
        f"(event = {opt.event_value:g}, censored = {opt.censored_value:g}). Covariates with <=6 distinct values are dummy-coded against a reference level; the rest are used as continuous covariates. Main effects only - interactions and transforms aren't implemented.",
        f"Uses {'Efron' if opt.ties_method == 'efron' else 'Breslow'}'s method for tied event times.",
    ]
    if cat_info:
        notes.append("Reference levels: " + "; ".join(f"{lbl} vs {ref:g}" for lbl, (_lv, ref) in cat_info.items()) + ".")
    if any(left_out.values()):
        total_out = sum(left_out.values())
        notes.insert(0, f"{total_out} row{'s' if total_out != 1 else ''} with something typed {'were' if total_out != 1 else 'was'} "
                        "left out of the fit: " + ", ".join(f"{count} with {why}" for why, count in left_out.items() if count)
                        + ". A subject needs a time, an event code and every covariate; the other rows are used as they stand.")

    # ---- what to report: model diagnostics ----
    diag_rows = []
    null_neg_ll = None
    if opt.show_aic or opt.show_pseudo_r2:
        null_neg_ll, _ = fit_fn(np.zeros(0), np.empty((time.size, 0)), time, event)
    if opt.show_aic:
        aic = 2 * p - 2 * (-neg_ll)
        diag_rows.append(["Akaike's Information Criterion (AIC)", f"{aic:.{sig}g}"])
    if opt.show_partial_loglik:
        diag_rows.append(["Partial log-likelihood (LL)", f"{-neg_ll:.{sig}g}"])
    if opt.show_neg2ll:
        diag_rows.append(["-2 x partial log-likelihood (-2LL)", f"{2*neg_ll:.{sig}g}"])
    if opt.show_pseudo_r2:
        r2 = 1 - np.exp(-(2.0 / time.size) * ((-neg_ll) - (-null_neg_ll)))
        diag_rows.append(["Pseudo R^2 (Cox-Snell/Nagelkerke-style)", f"{max(0.0, r2):.{sig}g}"])
    if diag_rows:
        tables.append(ResultTable(title="For comparing with other models of these data", headers=["Metric", "Value"], rows=diag_rows))

    if opt.show_multicollinearity and p >= 2:
        corr = np.corrcoef(Z_std, rowvar=False)
        try:
            vif = np.diag(np.linalg.inv(corr))
        except np.linalg.LinAlgError:
            vif = np.full(p, float("nan"))
        tables.append(ResultTable(title="How far each term is explained by the others (VIF)", headers=["Covariate", "VIF"], rows=[[n, f"{v:.{sig}g}" if v == v else "n/a"] for n, v in zip(names_all, vif)]))

    if opt.show_covariance_matrix and p >= 1:
        cov_orig = cov_std / np.outer(sds, sds)
        tables.append(ResultTable(title="Covariance of the coefficients", headers=["Covariate"] + names_all, rows=[[names_all[i]] + [f"{cov_orig[i, j]:.{sig}g}" for j in range(p)] for i in range(p)]))

    # ---- against a simpler model ----
    if opt.compare_models and opt.dropped_covariates:
        keep_idx = [i for i, n in enumerate(names_all) if n not in opt.dropped_covariates]
        if len(keep_idx) < p:
            _b_r, _c_r, _m_r, _s_r, neg_ll_reduced, _f_r = _fit_cox(Z_all[:, keep_idx], time, event, opt.ties_method) if keep_idx else _fit_cox(np.empty((time.size, 0)), time, event, opt.ties_method)
            df_diff = p - len(keep_idx)
            lrt_stat = 2 * (neg_ll_reduced - neg_ll)
            p_lrt = float(stats.chi2.sf(max(lrt_stat, 0), df_diff))
            aic_full, aic_reduced = 2 * p - 2 * (-neg_ll), 2 * len(keep_idx) - 2 * (-neg_ll_reduced)
            preferred = "full" if aic_full < aic_reduced else "simpler"
            tables.append(ResultTable(
                title="Compare: full model vs. simpler model (dropping " + ", ".join(opt.dropped_covariates) + ")",
                headers=["Model", "AIC", "Log-likelihood"],
                rows=[["Full", f"{aic_full:.{sig}g}", f"{-neg_ll:.{sig}g}"], ["Simpler", f"{aic_reduced:.{sig}g}", f"{-neg_ll_reduced:.{sig}g}"]],
            ))
            tables.append(ResultTable(title="Likelihood-ratio test of this model against the simpler one", headers=["Chi-square", "df", "P value"], rows=[[f"{lrt_stat:.4f}", str(df_diff), f"{p_lrt:.4g}"]]))
            notes.append(f"AICc/AIC favors the {preferred} model.")

    # ---- predictions ----
    if opt.predict_missing_rows:
        # subjects with every covariate but no usable time or event code: their
        # relative hazard can still be worked out from the fitted model
        covs_full = np.column_stack(cov_columns_full)
        has_covs = ~np.any(np.isnan(covs_full), axis=1)
        to_predict = np.flatnonzero(has_covs & ~complete_mask)
        pred_rows = []
        for r in to_predict[:5000]:
            z_vec = _encode_scenario({lbl: float(covs_full[r, j]) for j, lbl in enumerate(covariate_labels)},
                                     covariate_labels, cat_info)
            if z_vec.size == beta.size:
                pred_rows.append([str(r + 1), f"{np.exp(float(z_vec @ beta)):.{sig}g}"])
        if pred_rows:
            tables.append(ResultTable(title="Relative hazard for subjects left out of the fit (no time or event code)",
                                      headers=["Row", "Relative hazard"], rows=pred_rows))
        skipped = left_out["a blank covariate"]
        if skipped and pred_rows:
            notes.append(f"{skipped} row(s) miss a covariate value, so they could not be given a relative hazard either.")
    if opt.scenarios:
        scen_rows = []
        for i, scenario in enumerate(opt.scenarios, start=1):
            z_vec = _encode_scenario(scenario, covariate_labels, cat_info)
            lin_pred = float(z_vec @ beta) if z_vec.size == beta.size else 0.0
            scen_rows.append([str(i), f"{np.exp(lin_pred):.{sig}g}"])
        tables.append(ResultTable(title="Predicted relative hazard for each scenario (vs. a subject at all-zero/reference covariate values)", headers=["Scenario", "Relative hazard"], rows=scen_rows))

    # ---- how well it fits ----
    if any((opt.gof_lrt, opt.gof_wald, opt.gof_score, opt.gof_concordance)):
        gof_rows = []
        if opt.gof_lrt:
            if null_neg_ll is None:
                null_neg_ll, _ = fit_fn(np.zeros(0), np.empty((time.size, 0)), time, event)
            lrt_stat = 2 * (null_neg_ll - neg_ll)
            gof_rows.append(["Partial likelihood ratio test (vs. null model)", f"chi2 = {lrt_stat:.4f}, df = {p}", f"{stats.chi2.sf(max(lrt_stat,0), p):.4g}"])
        if opt.gof_wald and p:
            wald_stat = float(beta_std @ np.linalg.pinv(cov_std) @ beta_std)
            gof_rows.append(["Wald test (all covariates jointly)", f"chi2 = {wald_stat:.4f}, df = {p}", f"{stats.chi2.sf(wald_stat, p):.4g}"])
        if opt.gof_score and p:
            _neg_ll0, grad0 = fit_fn(np.zeros(p), Z_std, time, event)
            info0 = _numerical_hessian(fit_fn, np.zeros(p), Z_std, time, event)
            try:
                score_stat = float(grad0 @ np.linalg.pinv(info0) @ grad0)
            except np.linalg.LinAlgError:
                score_stat = float("nan")
            gof_rows.append(["Score test (all covariates jointly)", f"chi2 = {score_stat:.4f}, df = {p}" if score_stat == score_stat else "n/a", f"{stats.chi2.sf(score_stat, p):.4g}" if score_stat == score_stat else "n/a"])
        if gof_rows:
            tables.append(ResultTable(title="Does the model explain anything?", headers=["Test", "Statistic", "P value"], rows=gof_rows))
        if opt.gof_concordance:
            c_index = _harrell_c(Z_all, beta, time, event)
            tables.append(ResultTable(title="How well the model ranks who has the event first", headers=["Harrell's C-statistic"], rows=[[f"{c_index:.{sig}g}" if c_index == c_index else "n/a"]]))

    # ---- checks on the residuals (each checked item adds a chart plus its data table) ----
    from ..chart_helpers import xy_chart

    charts = []
    zero_line = [{"kind": "h", "value": 0.0}]
    need_martingale = opt.check_martingale_vs_linear or opt.check_martingale_vs_covariate or opt.check_deviance_vs_linear or opt.check_deviance_vs_covariate or opt.check_coxsnell
    if need_martingale:
        martingale, H_i, linear_pred = _martingale_residuals(Z_all, beta, time, event)
        if opt.check_martingale_vs_linear:
            charts.append(xy_chart("Martingale residuals against risk score", linear_pred, {"": martingale}, kind="scatter",
                                   x_label="Linear predictor (log HR)", y_label="Martingale residual", reference_lines=zero_line))
            tables.append(ResultTable(title="Martingale residuals against risk score, point by point", headers=["Linear predictor", "HR = exp(linear predictor)", "Martingale residual"], rows=[[f"{lp:.{sig}g}", f"{np.exp(lp):.{sig}g}", f"{m:.{sig}g}"] for lp, m in zip(linear_pred, martingale)][:200]))
        if opt.check_deviance_vs_linear or opt.check_deviance_vs_covariate:
            deviance = _deviance_residuals(martingale, event)
        if opt.check_deviance_vs_linear:
            charts.append(xy_chart("Deviance residuals against risk score", linear_pred, {"": deviance}, kind="scatter",
                                   x_label="Linear predictor (log HR)", y_label="Deviance residual", reference_lines=zero_line))
            tables.append(ResultTable(title="Deviance residuals against risk score, point by point", headers=["Linear predictor", "HR", "Deviance residual"], rows=[[f"{lp:.{sig}g}", f"{np.exp(lp):.{sig}g}", f"{d:.{sig}g}"] for lp, d in zip(linear_pred, deviance)][:200]))
        if opt.check_martingale_vs_covariate:
            for j, name in enumerate(names_all):
                charts.append(xy_chart(f"Martingale residuals against '{name}'", Z_all[:, j], {"": martingale}, kind="scatter",
                                       x_label=name, y_label="Martingale residual", reference_lines=zero_line))
                tables.append(ResultTable(title=f"Martingale residuals against the covariate '{name}'", headers=[name, "Martingale residual"], rows=[[f"{z:.{sig}g}", f"{m:.{sig}g}"] for z, m in zip(Z_all[:, j], martingale)][:200]))
        if opt.check_deviance_vs_covariate:
            for j, name in enumerate(names_all):
                charts.append(xy_chart(f"Deviance residuals against '{name}'", Z_all[:, j], {"": deviance}, kind="scatter",
                                       x_label=name, y_label="Deviance residual", reference_lines=zero_line))
                tables.append(ResultTable(title=f"Deviance residuals against the covariate '{name}'", headers=[name, "Deviance residual"], rows=[[f"{z:.{sig}g}", f"{d:.{sig}g}"] for z, d in zip(Z_all[:, j], deviance)][:200]))
        if opt.check_coxsnell:
            na_x, na_h = _nelson_aalen(H_i, event)
            charts.append(xy_chart("Cox-Snell residuals against the cumulative hazard they should follow", na_x, {"": na_h}, kind="scatter",
                                   x_label="Cox-Snell residual", y_label="Nelson-Aalen cumulative hazard",
                                   reference_lines=[{"kind": "line", "points": [[0.0, 0.0], [1.0, 1.0]]}]))
            tables.append(ResultTable(
                title="Cox-Snell residuals against the cumulative hazard they should follow (should track the diagonal if the model fits well)",
                headers=["Cox-Snell residual", "Nelson-Aalen cumulative hazard"],
                rows=[[f"{x:.{sig}g}", f"{h:.{sig}g}"] for x, h in zip(na_x, na_h)][:200],
            ))

    if opt.check_schoenfeld_ph or opt.check_schoenfeld_vs_time:
        event_times, schoenfeld = _schoenfeld_residuals(Z_all, beta, time, event)
        if schoenfeld.shape[0] >= 3:
            cov_orig_for_scale = (cov_std / np.outer(sds, sds)) if p else np.zeros((0, 0))
            n_events_total = event_times.size
            scaled = n_events_total * (schoenfeld @ cov_orig_for_scale) if p else schoenfeld
            if opt.check_schoenfeld_ph:
                ph_rows = []
                for j, name in enumerate(names_all):
                    rho, p_corr = stats.spearmanr(event_times, scaled[:, j])
                    ph_rows.append([name, f"rho = {rho:.4f}", f"{p_corr:.4g}", "PH assumption may be violated" if p_corr < 0.05 else "consistent with PH"])
                tables.append(ResultTable(title="Does each effect hold steady over time? (scaled Schoenfeld residuals against time)", headers=["Covariate", "Correlation with time", "P value", "Verdict"], rows=ph_rows))
            if opt.check_schoenfeld_vs_time:
                for j, name in enumerate(names_all):
                    charts.append(xy_chart(f"Schoenfeld residuals against time - '{name}'", event_times, {"": schoenfeld[:, j]},
                                           kind="scatter", x_label="Event time", y_label="Schoenfeld residual",
                                           reference_lines=zero_line))
                    tables.append(ResultTable(title=f"Schoenfeld residuals in time order - '{name}'", headers=["Event time", "Schoenfeld residual"], rows=[[f"{t:.{sig}g}", f"{r:.{sig}g}"] for t, r in zip(event_times, schoenfeld[:, j])][:200]))
        else:
            notes.append("Too few distinct event times to compute Schoenfeld residuals.")
            if opt.check_schoenfeld_vs_time:
                from ..chart_helpers import unavailable_chart

                charts.append(unavailable_chart(
                    "Schoenfeld residuals vs time",
                    f"Schoenfeld residuals need at least 3 distinct event times; this data has {schoenfeld.shape[0]}.",
                ))

    if opt.check_loglog_survival:
        cat_labels = list(cat_info.keys())
        if not cat_labels:
            notes.append("Log-minus-log survival plot needs at least one categorical covariate (<=6 distinct values); none was found.")
            from ..chart_helpers import unavailable_chart

            charts.append(unavailable_chart(
                "Log-minus-log survival",
                "This plot draws one curve per level of a categorical covariate (a column with 6 or fewer distinct "
                "values, such as 0/1 for sex or treatment). None of the covariates here is categorical.",
            ))
        else:
            for label in cat_labels:
                col_idx = covariate_labels.index(label)
                raw_values = cov_columns[col_idx]
                loglog_series = {}  # level -> (time, ln(-ln S)); parallel curves support proportional hazards
                for level in cat_info[label][0]:
                    level_mask = raw_values == level
                    if np.sum(level_mask) < 2:
                        continue
                    t_level, e_level = time[level_mask], event[level_mask]
                    order = np.argsort(t_level)
                    t_sorted, e_sorted = t_level[order], e_level[order]
                    n_risk = t_sorted.size
                    survival = 1.0
                    rows = []
                    curve_t, curve_y = [], []
                    for t in np.unique(t_sorted):
                        d = int(np.sum(e_sorted[t_sorted == t] == 1))
                        if d > 0:
                            survival *= 1 - d / n_risk
                        if 0 < survival < 1:
                            rows.append([f"{t:.{sig}g}", f"{np.log(-np.log(survival)):.{sig}g}"])
                            curve_t.append(float(t))
                            curve_y.append(float(np.log(-np.log(survival))))
                        n_risk -= int(np.sum(t_sorted == t))
                    tables.append(ResultTable(title=f"log(−log survival) - '{label}' = {level:g}", headers=["Time", "ln(-ln(S(t)))"], rows=rows[:200]))
                    if curve_t:
                        loglog_series[f"{label} = {level:g}"] = (curve_t, curve_y)
                if loglog_series:
                    from ..backend_types import ChartSeries, ChartSpec

                    charts.append(ChartSpec(
                        title=f"Log-minus-log survival by '{label}'",
                        series=[ChartSeries(name, ts, ys, kind="step") for name, (ts, ys) in loglog_series.items()],
                        x_label="Time", y_label="ln(-ln S(t))",
                    ))

    # ---- graphs (predicted survival curves) ----
    if opt.graph_defs:
        from ..chart_helpers import survival_curves_chart

        unique_times_g, H0_g = _breslow_baseline_hazard(Z_all, beta, time, event)
        for graph in opt.graph_defs:
            z_vec = _encode_scenario(graph.get("covariate_values", {}), covariate_labels, cat_info)
            lin_pred_g = float(z_vec @ beta) if z_vec.size == beta.size else 0.0
            s_curve = np.exp(-H0_g * np.exp(lin_pred_g))
            headers = ["Time", graph.get("title") or "Predicted survival"]
            rows = [[f"{t:.{sig}g}", f"{s*100:.{sig}g}%"] for t, s in zip(unique_times_g, s_curve)]
            if graph.get("include_baseline"):
                s0_curve = np.exp(-H0_g)
                headers.append("Baseline survival (all covariates = 0/reference)")
                rows = [row + [f"{s0*100:.{sig}g}%"] for row, s0 in zip(rows, s0_curve)]
            curves = {graph.get("title") or "Predicted survival": s_curve}
            if graph.get("include_baseline"):
                curves["Baseline (covariates = 0/reference)"] = np.exp(-H0_g)
            charts.append(survival_curves_chart(f"Predicted survival - {graph.get('title') or 'graph'}", unique_times_g, curves))
            tables.append(ResultTable(title=f"Predicted survival curve - {graph.get('title') or 'graph'}", headers=headers, rows=rows[:200]))

    n_events = int(np.sum(event))
    n_cov = len(covariate_labels)
    terms = f" ({p} model terms)" if p != n_cov else ""
    summary = (f"n = {time.size} subjects, {n_events} events, {n_cov} covariate{'s' if n_cov != 1 else ''}{terms}; "
               f"log-likelihood = {-neg_ll:.4g}.")

    return AnalysisResult(title="What changes the risk", summary=summary, tables=tables, notes=notes, charts=charts)
