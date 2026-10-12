"""A measured outcome explained by several predictors (multiple linear regression).

Reads the sheet via ``full_variable_matrix()`` (X plus every group, one
column per variable); the dependent (outcome) variable is chosen from
that list rather than being fixed to "the last column", and every other
variable is a predictor. A predictor with <=6 distinct values is treated
as categorical and dummy-coded against a reference level, as in the
Cox regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class MultipleLinearRegressionOptions:
    # the model
    regression_type: str = "least_squares"  # "least_squares" | "poisson"
    outcome_index: int = -1  # index into full_variable_matrix()'s label list; -1 = last variable
    include_intercept: bool = True
    include_two_way: bool = False
    include_three_way: bool = False
    # reference levels
    reference_levels: dict[str, str] = field(default_factory=dict)
    auto_reference: dict[str, bool] = field(default_factory=dict)
    # predictions
    interpolate_missing_rows: bool = True
    scenarios: list[dict[str, float]] = field(default_factory=list)
    report_ci_for_interpolated: bool = False
    # against a simpler model
    compare_models: bool = False
    dropped_predictors: list[str] = field(default_factory=list)
    # weights
    weighting: str = "none"  # "none" | "inv_y" | "inv_y2" | "inv_yk"
    weighting_k: float = 2.0
    # measures of fit
    ci_approach: str = "profile"  # "profile" | "wald" - coincide exactly for least-squares; differ for Poisson
    show_se: bool = False
    show_ci: bool = True  # a confidence interval for every coefficient
    run_significance_test: bool = False
    show_multicollinearity: bool = False
    show_correlation_matrix: bool = False
    show_multiple_r: bool = False
    show_r2: bool = True
    show_adj_r2: bool = False
    show_syx: bool = False
    show_rmse: bool = False
    show_sum_of_square: bool = False
    show_aicc: bool = False
    normality_dagostino: bool = False
    normality_anderson: bool = False
    normality_shapiro: bool = False
    normality_ks: bool = False
    # Output
    confidence_level: float = 0.95
    significant_digits: int = 4
    # checks on the residuals
    show_residual_plot: bool = True
    show_homoscedasticity_plot: bool = False
    show_qq_plot: bool = False
    show_residual_vs_order: bool = False


def _build_design(predictor_labels: list[str], predictor_columns: list[np.ndarray], opt: MultipleLinearRegressionOptions):
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


def _encode_scenario(scenario: dict[str, float], predictor_labels: list[str], cat_info: dict, has_interactions: bool, names: list[str]) -> np.ndarray:
    base = []
    base_names = []
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
    return np.array([vec.get(name, 0.0) if name in vec else _product_lookup(name, vec) for name in names])


def _product_lookup(name: str, vec: dict[str, float]) -> float:
    parts = [p.strip() for p in name.split("×")]
    if all(p in vec for p in parts):
        result = 1.0
        for p in parts:
            result *= vec[p]
        return result
    return 0.0


def _dallal_wilkinson_lilliefors_p(d_stat: float, n: int) -> float:
    if n < 5:
        return float("nan")
    if d_stat <= 0:
        return 1.0
    kd = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * d_stat
    if kd <= 0.775:
        p = np.exp(-1.2337141 / kd**2) * (2.00012 + 0.247105 / kd - 0.0649821 / kd**2)
        return float(np.clip(p, 0.0, 1.0))
    p = 1 - 2 * np.exp(-2 * kd**2) * (1 - 2 / 3 * kd) if kd < 5 else 0.0
    return float(np.clip(p, 0.0, 1.0))


def _durbin_watson(resid: np.ndarray) -> float:
    diffs = np.diff(resid)
    denom = np.sum(resid**2)
    return float(np.sum(diffs**2) / denom) if denom else float("nan")


def _fit_wls(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        W = weights
        XtX = X.T @ (X * W[:, None])
        Xty = X.T @ (y * W)
    coeffs = np.linalg.pinv(XtX) @ Xty
    return coeffs, np.linalg.pinv(XtX)


def _fit_poisson(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    p = X.shape[1]
    beta = np.zeros(p)
    beta[0] = np.log(max(np.mean(y), 0.1)) if p else 0.0
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        W = np.clip(mu, 1e-8, None)
        z = eta + (y - mu) / W
        XtWX = X.T @ (X * W[:, None])
        XtWz = X.T @ (W * z)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    mu = np.exp(eta)
    XtWX = X.T @ (X * np.clip(mu, 1e-8, None)[:, None])
    return beta, np.linalg.pinv(XtWX)


@register("multiple_linear_regression")
def run(data: SheetData, options: MultipleLinearRegressionOptions | None = None) -> AnalysisResult:
    opt = options or MultipleLinearRegressionOptions()
    all_labels, full_matrix = data.full_variable_matrix()
    if len(all_labels) < 3:
        raise AnalysisError(f"Multiple regression needs at least 2 predictors and 1 outcome; found {len(all_labels)} variable(s).")

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
    if complete.shape[0] < len(predictor_labels) + 2:
        raise AnalysisError(f"Need at least {len(predictor_labels) + 2} complete rows for {len(predictor_labels)} predictor(s); found {complete.shape[0]}.")

    y = complete[:, outcome_idx]
    predictor_columns = [complete[:, i] for i in predictor_idx]

    if opt.regression_type == "poisson" and (np.any(y < 0) or np.any(y != np.round(y))):
        raise AnalysisError("Poisson regression needs Y to be non-negative whole-number counts.")

    Z_all, names_all, cat_info, sources_all = _build_design(predictor_labels, predictor_columns, opt)
    n = y.size
    sig = opt.significant_digits

    design = np.column_stack([np.ones(n), Z_all]) if opt.include_intercept else Z_all
    term_names = (["Intercept"] if opt.include_intercept else []) + names_all
    p_terms = design.shape[1]
    if n < p_terms + 2:
        raise AnalysisError(f"Need at least {p_terms + 2} complete rows for {p_terms} model term(s); found {n}.")

    weights = None
    if opt.weighting != "none":
        if np.any(y == 0) and opt.weighting != "none":
            raise AnalysisError("Weighting by a power of Y needs every Y value to be nonzero.")
        if opt.weighting == "inv_y":
            weights = 1.0 / np.abs(y)
        elif opt.weighting == "inv_y2":
            weights = 1.0 / y**2
        else:
            weights = 1.0 / np.abs(y) ** opt.weighting_k

    tables = []
    notes = [f"'{outcome_label}' was used as the outcome; every other variable (X plus every other group) is a predictor.", "Predictors with <=6 distinct values are dummy-coded against a reference level."]

    if opt.regression_type == "poisson":
        coeffs, cov = _fit_poisson(design, y)
        fitted = np.exp(np.clip(design @ coeffs, -30, 30))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        z_stats = coeffs / np.where(se > 1e-12, se, np.nan)
        p_values = 2 * stats.norm.sf(np.abs(z_stats))
        deviance = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y / np.clip(fitted, 1e-12, None), 1)), 0) - (y - fitted))
        fit_rows = [["Deviance", f"{deviance:.{sig}g}"], ["N", str(n)]]
        fit_title = "Model fit (Poisson)"
    else:
        coeffs, cov = _fit_wls(design, y, weights)
        fitted = design @ coeffs
        resid = y - fitted
        w = weights if weights is not None else np.ones(n)
        ss_res = float(np.sum(w * resid**2))
        y_wmean = np.average(y, weights=w)
        ss_tot = float(np.sum(w * (y - y_wmean) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
        df_model = p_terms - (1 if opt.include_intercept else 0)
        df_resid = n - p_terms
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else float("nan")
        mse = ss_res / df_resid if df_resid > 0 else float("nan")
        se = np.sqrt(np.clip(np.diag(cov) * mse, 0, None)) if df_resid > 0 else np.full(p_terms, float("nan"))
        with np.errstate(invalid="ignore", divide="ignore"):
            t_stats = np.where(se > 1e-12, coeffs / np.where(se > 1e-12, se, 1), np.nan)
            p_values = np.where(np.isfinite(t_stats), 2 * stats.t.sf(np.abs(t_stats), df_resid), np.nan) if df_resid > 0 else np.full(p_terms, np.nan)
        f_stat = ((ss_tot - ss_res) / df_model / mse) if (df_model and mse) else float("nan")
        f_p = float(stats.f.sf(f_stat, df_model, df_resid)) if np.isfinite(f_stat) and df_resid > 0 else float("nan")
        fit_rows = [[f"{r_squared:.4f}", f"{adj_r_squared:.4f}", f"{f_stat:.4g}" if f_stat == f_stat else "-", f"{f_p:.4g}" if f_p == f_p else "-", str(n)]]
        fit_title = "Model fit"

    if opt.regression_type == "least_squares" and opt.run_significance_test and df_resid > 0:
        lrt_p_values = []
        w_full = weights if weights is not None else np.ones(n)
        for i in range(p_terms):
            keep = [j for j in range(p_terms) if j != i]
            design_r = design[:, keep] if keep else np.empty((n, 0))
            coeffs_r, _cov_r = _fit_wls(design_r, y, weights) if design_r.shape[1] else (np.zeros(0), np.zeros((0, 0)))
            resid_r = y - (design_r @ coeffs_r if design_r.shape[1] else np.zeros(n))
            ss_res_r = float(np.sum(w_full * resid_r**2))
            f_i = ((ss_res_r - ss_res) / 1) / mse if mse else float("nan")
            lrt_p_values.append(float(stats.f.sf(f_i, 1, df_resid)) if f_i == f_i else float("nan"))
        p_values = np.array(lrt_p_values)
        p_value_label = "P value (extra SS F test)"
    else:
        p_value_label = "P value"

    # the coefficient table: SE and CI as asked for. Least squares uses t
    # (profile likelihood and Wald coincide exactly there); Poisson uses Wald z.
    pct = int(round(opt.confidence_level * 100))
    if opt.regression_type == "least_squares":
        crit = float(stats.t.ppf(0.5 + opt.confidence_level / 2, df_resid)) if df_resid > 0 else float("nan")
        ci_note = "t"
    else:
        crit = float(stats.norm.ppf(0.5 + opt.confidence_level / 2))
        ci_note = "Wald"
    rows = []
    for name, c, s, pv in zip(term_names, coeffs, se, p_values):
        row = [name, f"{c:.{sig}g}"]
        if opt.show_se:
            row.append(f"{s:.{sig}g}" if np.isfinite(s) else "-")
        if opt.show_ci:
            row.append(f"{c - crit * s:.{sig}g} to {c + crit * s:.{sig}g}" if np.isfinite(s) and np.isfinite(crit) else "-")
        row.append(f"{pv:.4g}" if np.isfinite(pv) else "-")
        rows.append(row)
    headers = ["Term", "Coefficient"] + (["Std. error"] if opt.show_se else []) + \
        ([f"{pct}% CI ({ci_note})"] if opt.show_ci else []) + [p_value_label]
    tables.append(ResultTable(title=f"Predicting '{outcome_label}'" + (" (Poisson)" if opt.regression_type == "poisson" else ""), headers=headers, rows=rows))
    if opt.regression_type == "poisson":
        tables.append(ResultTable(title=fit_title, headers=["Metric", "Value"], rows=fit_rows))
        summary = f"Poisson regression predicting '{outcome_label}' from {len(predictor_labels)} predictor(s); deviance = {deviance:.{sig}g}."
    else:
        tables.append(ResultTable(title=fit_title, headers=["R²", "Adjusted R²", "F", "P value", "N"], rows=fit_rows))
        summary = f"R² = {r_squared:.4f} predicting '{outcome_label}' from {len(predictor_labels)} predictor(s) (F p = {f_p:.4g})." if f_p == f_p else f"R² = {r_squared:.4f} predicting '{outcome_label}'."

    # ---- against a simpler model ----
    if opt.regression_type == "least_squares" and opt.compare_models and opt.dropped_predictors:
        dropped_set = set(opt.dropped_predictors)
        keep_idx = [i for i, srcs in enumerate(sources_all) if not (srcs & dropped_set)]
        if len(keep_idx) < len(names_all):
            Z_reduced = Z_all[:, keep_idx] if keep_idx else np.empty((n, 0))
            design_r = np.column_stack([np.ones(n), Z_reduced]) if opt.include_intercept else Z_reduced
            coeffs_r, _cov_r = _fit_wls(design_r, y, weights)
            resid_r = y - design_r @ coeffs_r
            w = weights if weights is not None else np.ones(n)
            ss_res_r = float(np.sum(w * resid_r**2))
            df_diff = design.shape[1] - design_r.shape[1]
            df_resid_full = n - design.shape[1]
            f_stat_cmp = ((ss_res_r - ss_res) / df_diff) / (ss_res / df_resid_full) if df_diff and df_resid_full > 0 and ss_res else float("nan")
            p_cmp = float(stats.f.sf(f_stat_cmp, df_diff, df_resid_full)) if f_stat_cmp == f_stat_cmp else float("nan")
            tables.append(ResultTable(
                title="Compare: full model vs. simpler model (dropping " + ", ".join(opt.dropped_predictors) + ")",
                headers=["Model", "SS residual", "df"],
                rows=[["Full", f"{ss_res:.{sig}g}", str(df_resid_full)], ["Simpler", f"{ss_res_r:.{sig}g}", str(n - design_r.shape[1])]],
            ))
            tables.append(ResultTable(title="F test of this model against the simpler one", headers=["F", "df1", "df2", "P value"], rows=[[f"{f_stat_cmp:.4f}" if f_stat_cmp == f_stat_cmp else "n/a", str(df_diff), str(df_resid_full), f"{p_cmp:.4g}" if p_cmp == p_cmp else "n/a"]]))

    # ---- Predictions: rows with a blank outcome, and typed-in scenarios ----
    has_int = opt.include_two_way or opt.include_three_way
    pct = int(round(opt.confidence_level * 100))
    least_squares = opt.regression_type == "least_squares"
    xtx_inv = np.linalg.pinv(design.T @ (design * (weights if weights is not None else np.ones(n))[:, None])) \
        if least_squares else None
    crit = float(stats.t.ppf(0.5 + opt.confidence_level / 2, df_resid)) if least_squares and df_resid > 0 \
        else float(stats.norm.ppf(0.5 + opt.confidence_level / 2))

    def predict(scenario: dict) -> list[str] | None:
        """[prediction, CI] for one set of predictor values (CI when asked for)."""
        z_vec = _encode_scenario(scenario, predictor_labels, cat_info, has_int, names_all)
        vec = np.concatenate([[1.0], z_vec]) if opt.include_intercept else z_vec
        if vec.size != coeffs.size:
            return None
        eta = float(vec @ coeffs)
        if least_squares:
            out = [f"{eta:.{sig}g}"]
            if opt.report_ci_for_interpolated:
                half = crit * np.sqrt(max(mse * float(vec @ xtx_inv @ vec), 0.0)) if df_resid > 0 else float("nan")
                out.append(f"{eta - half:.{sig}g} to {eta + half:.{sig}g}" if np.isfinite(half) else "-")
            return out
        out = [f"{np.exp(eta):.{sig}g}"]
        if opt.report_ci_for_interpolated:
            half = crit * np.sqrt(max(float(vec @ cov @ vec), 0.0))
            out.append(f"{np.exp(eta - half):.{sig}g} to {np.exp(eta + half):.{sig}g}")
        return out

    value_head = "Predicted " + ("count" if not least_squares else f"'{outcome_label}'")
    ci_head = [f"{pct}% CI of the mean" if least_squares else f"{pct}% CI (Wald)"] if opt.report_ci_for_interpolated else []
    if opt.interpolate_missing_rows:
        outcome_blank = np.isnan(full_matrix[:, outcome_idx])
        predictors_there = ~np.any(np.isnan(full_matrix[:, predictor_idx]), axis=1)
        rows_to_predict = np.flatnonzero(outcome_blank & predictors_there)
        pred_rows = []
        for r in rows_to_predict[:5000]:
            got = predict({lbl: float(full_matrix[r, i]) for lbl, i in zip(predictor_labels, predictor_idx)})
            if got is not None:
                pred_rows.append([str(r + 1)] + got)
        if pred_rows:
            tables.append(ResultTable(title=f"Predicted '{outcome_label}' for rows where it is blank",
                                      headers=["Row", value_head] + ci_head, rows=pred_rows))
        skipped = int(np.sum(outcome_blank & ~predictors_there))
        if skipped:
            notes.append(f"{skipped} row(s) with a blank outcome also miss a predictor, so no prediction was made for them.")
    if opt.scenarios:
        scen_rows = []
        for i, scenario in enumerate(opt.scenarios, start=1):
            got = predict(scenario)
            scen_rows.append([str(i)] + (got if got is not None else ["n/a"]))
        tables.append(ResultTable(title="Predicted outcome for the values typed in",
                                  headers=["Scenario", value_head] + ci_head, rows=scen_rows))

    # ---- measures of fit ----
    if opt.regression_type != "least_squares":
        needs_ols_diag = any((opt.show_multicollinearity, opt.show_correlation_matrix, opt.show_multiple_r, opt.show_r2, opt.show_adj_r2, opt.show_syx, opt.show_rmse, opt.show_sum_of_square, opt.show_aicc, opt.normality_dagostino, opt.normality_anderson, opt.normality_shapiro, opt.normality_ks))
        if needs_ols_diag:
            notes.append("The measures of fit below are worked out for least-squares regression only, not for a Poisson model.")
    else:
        if opt.show_multicollinearity and Z_all.shape[1] >= 2:
            corr = np.corrcoef(Z_all, rowvar=False)
            try:
                vif = np.diag(np.linalg.inv(corr))
            except np.linalg.LinAlgError:
                vif = np.full(Z_all.shape[1], float("nan"))
            tables.append(ResultTable(title="How far each term is explained by the others (VIF)", headers=["Predictor", "VIF"], rows=[[nm, f"{v:.{sig}g}" if v == v else "n/a"] for nm, v in zip(names_all, vif)]))
        if opt.show_correlation_matrix and Z_all.shape[1] >= 2:
            corr = np.corrcoef(Z_all, rowvar=False)
            tables.append(ResultTable(title="Correlations between the predictors", headers=["Predictor"] + names_all, rows=[[names_all[i]] + [f"{corr[i,j]:.{sig}g}" for j in range(len(names_all))] for i in range(len(names_all))]))

        fit_metric_rows = []
        if opt.show_multiple_r:
            fit_metric_rows.append(["Multiple R", f"{np.sqrt(max(r_squared, 0.0)):.{sig}g}"])
        if opt.show_r2:
            fit_metric_rows.append(["R squared", f"{r_squared:.{sig}g}"])
        if opt.show_adj_r2:
            fit_metric_rows.append(["Adjusted R squared", f"{adj_r_squared:.{sig}g}" if adj_r_squared == adj_r_squared else "n/a"])
        if opt.show_syx:
            fit_metric_rows.append(["Residual SD", f"{np.sqrt(mse):.{sig}g}" if mse == mse else "n/a"])
        if opt.show_rmse:
            fit_metric_rows.append(["RMSE", f"{np.sqrt(ss_res / n):.{sig}g}"])
        if opt.show_sum_of_square:
            fit_metric_rows.append(["Sum-of-Squares (residual)", f"{ss_res:.{sig}g}"])
        if opt.show_aicc:
            k = p_terms + 1
            aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else float("nan")
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 and aic == aic else float("nan")
            fit_metric_rows.append(["AICc", f"{aicc:.{sig}g}" if aicc == aicc else "n/a"])
        if fit_metric_rows:
            tables.append(ResultTable(title="How well the model fits", headers=["Metric", "Value"], rows=fit_metric_rows))

        if any((opt.normality_dagostino, opt.normality_anderson, opt.normality_shapiro, opt.normality_ks)):
            norm_rows = []
            if opt.normality_dagostino:
                if n >= 8:
                    stat, p_n = stats.normaltest(resid)
                    norm_rows.append(["D'Agostino-Pearson omnibus", f"K2 = {stat:.4f}", f"{p_n:.4g}"])
                else:
                    norm_rows.append(["D'Agostino-Pearson omnibus", "n/a", "needs n >= 8"])
            if opt.normality_anderson:
                result_a = stats.anderson(resid, dist="norm", method="interpolate")
                norm_rows.append(["Anderson-Darling", f"A2 = {result_a.statistic:.4f}", f"{result_a.pvalue:.4g}"])
            if opt.normality_shapiro and 3 <= n <= 5000:
                stat_s, p_s = stats.shapiro(resid)
                norm_rows.append(["Shapiro-Wilk", f"W = {stat_s:.4f}", f"{p_s:.4g}"])
            if opt.normality_ks:
                resid_sd = np.std(resid, ddof=1) or 1.0
                standardized = (resid - np.mean(resid)) / resid_sd
                d_stat, _ = stats.kstest(standardized, "norm")
                p_ks = _dallal_wilkinson_lilliefors_p(d_stat, n)
                norm_rows.append(["Kolmogorov-Smirnov (Dallal-Wilkinson-Lilliefors P)", f"D = {d_stat:.4f}", f"{p_ks:.4g}" if p_ks == p_ks else "n/a"])
            tables.append(ResultTable(title="Are the residuals normal?", headers=["Test", "Statistic", "P value"], rows=norm_rows))

    # ---- checks on the residuals ----
    from ..chart_helpers import qq_chart, residual_chart, xy_chart

    charts = []
    if opt.regression_type == "least_squares" and (opt.show_residual_plot or opt.show_homoscedasticity_plot or opt.show_qq_plot or opt.show_residual_vs_order):
        if opt.show_residual_plot:
            charts.append(residual_chart("Residuals against fitted values", fitted, resid))
            tables.append(ResultTable(title="Residuals against fitted values, point by point", headers=["Predicted Y", "Residual"], rows=[[f"{f:.{sig}g}", f"{r:.{sig}g}"] for f, r in zip(fitted, resid)][:200]))
        if opt.show_homoscedasticity_plot:
            charts.append(residual_chart("Size of the residuals against fitted values", fitted, resid, absolute=True))
            tables.append(ResultTable(title="Size of each residual against its fitted value", headers=["Predicted Y", "Abs(Residual)"], rows=[[f"{f:.{sig}g}", f"{abs(r):.{sig}g}"] for f, r in zip(fitted, resid)][:200]))
        if opt.show_qq_plot and n >= 3:
            order = np.argsort(resid)
            sorted_resid = resid[order]
            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            charts.append(qq_chart("Q–Q plot of the residuals", theoretical, sorted_resid, "Actual residual"))
            tables.append(ResultTable(title="Residuals against the normal quantiles they should match", headers=["Theoretical quantile", "Actual residual"], rows=[[f"{t:.{sig}g}", f"{r:.{sig}g}"] for t, r in zip(theoretical, sorted_resid)][:200]))
        if opt.show_residual_vs_order:
            dw = _durbin_watson(resid)
            charts.append(xy_chart(f"Residuals in row order (Durbin-Watson = {dw:.3f})", np.arange(1, len(resid) + 1),
                                   {"": resid}, x_label="Row order", y_label="Residual",
                                   reference_lines=[{"kind": "h", "value": 0.0}]))
            tables.append(ResultTable(
                title="Residual vs Order plot data - Durbin-Watson = %.4f" % dw,
                headers=["Row order", "Residual"],
                rows=[[str(i + 1), f"{r:.{sig}g}"] for i, r in enumerate(resid)][:200],
            ))
            notes.append("Durbin-Watson statistic near 2 indicates no serial correlation; well below 2 suggests positive autocorrelation, well above suggests negative.")

    return AnalysisResult(title="Several predictors, a number", summary=summary, tables=tables, notes=notes, charts=charts)
