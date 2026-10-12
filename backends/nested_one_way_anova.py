"""Several groups with subsamples inside each subject (nested one-way ANOVA).
Tests the group effect using the correct error term for a nested design
(subjects-within-groups variability), rather than pooling every
replicate as if it were an independent subject."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from ._nested_common import decompose


@dataclass
class NestedOneWayAnovaOptions:
    zero_variance_mode: str = "keep"  # "keep" | "simplify" | "full"
    graph_grand_mean_ci: bool = False
    report_goodness_of_fit: bool = False
    significant_digits: int = 4
    # pairs to compare
    post_hoc: str = "none"  # "none" | "all_pairs" | "vs_control"
    control_index: int = 0
    correction: str = "holm"  # "holm" | "fdr" | "none"
    swap_direction: bool = False
    report_adjusted_p: bool = True
    graph_comparison_ci: bool = False  # each difference with its confidence interval
    # checks on the residuals
    show_residual_plot: bool = False
    show_homoscedasticity_plot: bool = False
    show_qq_plot: bool = False


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)[::-1]
    m = len(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


@register("nested_one_way_anova")
def run(data: SheetData, options: NestedOneWayAnovaOptions | None = None) -> AnalysisResult:
    opt = options or NestedOneWayAnovaOptions()
    d = decompose(data, min_groups=3)
    sig = opt.significant_digits
    notes = [
        "Replicate (sub)columns were averaged into one subject mean per row first; the group effect is tested against "
        "subject-to-subject variability within each group, not raw replicate variability - the statistically correct "
        "error term for data with technical replicates nested under each subject.",
    ]

    ms_subjects = d.ms_subjects
    zero_variance = ms_subjects == 0

    if zero_variance and opt.zero_variance_mode == "simplify":
        arrays = [data.group_values(label, mode="pooled") for label in d.labels]
        f_stat, p = stats.f_oneway(*arrays)
        df_between, df_error = len(arrays) - 1, sum(a.size for a in arrays) - len(arrays)
        pooled_var = sum((a.size - 1) * np.var(a, ddof=1) for a in arrays) / df_error if df_error else float("nan")
        ns = [a.size for a in arrays]
        means = [float(np.mean(a)) for a in arrays]
        notes.append("Every random effect had zero variance, so the model was simplified to an ordinary one-way ANOVA on the raw replicate values (nesting no longer applies).")
    else:
        f_stat = d.f_statistic
        p = float(stats.f.sf(f_stat, d.df_between, d.df_subjects))
        df_between, df_error = d.df_between, d.df_subjects
        pooled_var = ms_subjects
        ns = d.group_n_subjects
        means = d.group_means
        if zero_variance:
            notes.append("The variation between subjects came out as zero. As asked, the model was fitted all the same, letting that variance go below zero if the data call for it.")

    group_rows = [[label, str(n), f"{mean:.{sig}g}"] for label, n, mean in zip(d.labels, ns, means)]
    tables = [ResultTable(title="Each group's mean of its subjects' means", headers=["Group", "Subjects", "Mean"], rows=group_rows)]
    tables.append(ResultTable(
        title="Nested ANOVA (subjects-within-groups error term)" if not (zero_variance and opt.zero_variance_mode == "simplify") else "Simplified ordinary one-way ANOVA",
        headers=["Source", "SS", "df", "MS", "F", "P value"],
        rows=[["Between groups", f"{d.ss_between:.{sig}g}" if not (zero_variance and opt.zero_variance_mode == 'simplify') else '-', str(df_between), "-", f"{f_stat:.4f}", f"{p:.4g}"]],
    ))

    if opt.graph_grand_mean_ci:
        grand_se = np.sqrt(pooled_var / sum(ns)) if pooled_var and pooled_var == pooled_var else float("nan")
        half = stats.t.ppf(0.975, df_error) * grand_se if grand_se == grand_se else float("nan")
        gm = float(np.average(means, weights=ns))
        tables.append(ResultTable(title="Grand mean with 95% CI", headers=["Grand mean", "95% CI"], rows=[[f"{gm:.{sig}g}", f"{gm - half:.{sig}g} to {gm + half:.{sig}g}" if half == half else "n/a"]]))

    if opt.report_goodness_of_fit and not (zero_variance and opt.zero_variance_mode == "simplify"):
        r2 = d.ss_between / (d.ss_between + d.ss_subjects) if (d.ss_between + d.ss_subjects) else float("nan")
        tables.append(ResultTable(title="Goodness of fit", headers=["R^2"], rows=[[f"{r2:.{sig}g}" if r2 == r2 else "n/a"]]))

    if opt.post_hoc != "none":
        labels = d.labels
        if opt.post_hoc == "vs_control":
            ctrl = max(0, min(opt.control_index, len(labels) - 1))
            pairs = [(ctrl, j) for j in range(len(labels)) if j != ctrl]
        else:
            pairs = list(combinations(range(len(labels)), 2))
        if opt.swap_direction:
            pairs = [(j, i) for i, j in pairs]
        raw_p, rows, diffs, halves, names = [], [], [], [], []
        t_crit = float(stats.t.ppf(0.975, df_error)) if df_error else float("nan")
        for i, j in pairs:
            se = np.sqrt(pooled_var * (1 / ns[i] + 1 / ns[j])) if pooled_var else float("nan")
            t_stat = (means[i] - means[j]) / se if se and se == se else 0.0
            p_ij = float(2 * stats.t.sf(abs(t_stat), df_error)) if df_error else float("nan")
            raw_p.append(p_ij)
            diff = float(means[i] - means[j])
            half = t_crit * se if se == se else float("nan")
            diffs.append(diff)
            halves.append(half)
            names.append(f"{labels[i]} vs {labels[j]}")
            rows.append([names[-1], f"{diff:.{sig}g}",
                         f"{diff - half:.{sig}g} to {diff + half:.{sig}g}" if half == half else "n/a",
                         f"{p_ij:.4g}" if p_ij == p_ij else "n/a"])
        if opt.correction == "fdr":
            adjusted, method_label = _benjamini_hochberg(np.array(raw_p)), "Benjamini-Hochberg FDR"
        elif opt.correction == "none":
            adjusted, method_label = raw_p, "uncorrected"
        else:
            adjusted, method_label = _holm(np.array(raw_p)), "Holm"
        headers = ["Comparison", "Mean difference", "95% CI (unadjusted)", "P value (raw)"]
        if opt.report_adjusted_p:
            headers += ["P value (adjusted)", "Significant (α = 0.05)"]
            for row, adj in zip(rows, adjusted):
                adj = float(adj)
                row.append(f"{adj:.4g}")
                row.append("Yes" if adj < 0.05 else "No")
        tables.append(ResultTable(title=f"Pairs of groups compared ({method_label})", headers=headers, rows=rows))
        if opt.graph_comparison_ci:
            from ..backend_types import ChartSeries, ChartSpec

            comparison_chart = ChartSpec(
                title="Each difference, with its 95% CI", x_categories=names, y_label="Difference between means",
                series=[ChartSeries("Differences", [float(k) for k in range(len(names))], diffs, kind="scatter",
                                    y_err=[h if h == h else 0.0 for h in halves])],
                reference_lines=[{"kind": "h", "value": 0.0}])

    charts = [comparison_chart] if opt.post_hoc != "none" and opt.graph_comparison_ci else []
    if opt.show_residual_plot or opt.show_homoscedasticity_plot or opt.show_qq_plot:
        from ..chart_helpers import qq_chart, residual_chart

        predicted_list, values_list = [], []
        for label, mean in zip(d.labels, means):
            _reps, matrix = data.group_replicate_matrix(label)
            subj_means = np.nanmean(matrix, axis=1)
            subj_means = subj_means[~np.isnan(subj_means)]
            predicted_list.append(np.full(subj_means.size, mean))
            values_list.append(subj_means)
        predicted = np.concatenate(predicted_list)
        values = np.concatenate(values_list)
        residuals = values - predicted
        if opt.show_residual_plot:
            charts.append(residual_chart("Residuals against fitted values (subject means)", predicted, residuals))
            tables.append(ResultTable(title="Residuals against fitted values, point by point", headers=["Predicted Y", "Residual"], rows=[[f"{p_:.{sig}g}", f"{r_:.{sig}g}"] for p_, r_ in zip(predicted, residuals)]))
        if opt.show_homoscedasticity_plot:
            charts.append(residual_chart("Size of the residuals against fitted values (subject means)", predicted, residuals, absolute=True))
            tables.append(ResultTable(title="Size of each residual against its fitted value", headers=["Predicted Y", "Abs(Residual)"], rows=[[f"{p_:.{sig}g}", f"{abs(r_):.{sig}g}"] for p_, r_ in zip(predicted, residuals)]))
        if opt.show_qq_plot and residuals.size >= 3:
            order = np.argsort(residuals)
            sorted_resid = residuals[order]
            nn = sorted_resid.size
            theoretical = stats.norm.ppf((np.arange(1, nn + 1) - 0.5) / nn)
            charts.append(qq_chart("Q–Q plot of the residuals", theoretical, sorted_resid, "Actual residual"))
            tables.append(ResultTable(title="Residuals against the normal quantiles they should match", headers=["Theoretical quantile", "Actual residual"], rows=[[f"{t_:.{sig}g}", f"{r_:.{sig}g}"] for t_, r_ in zip(theoretical, sorted_resid)]))

    n_subjects = sum(ns)
    summary = f"{len(d.labels)} groups, {n_subjects} subjects ({d.n_total} replicate measurements total); F({df_between}, {df_error}) = {f_stat:.4f}, p = {p:.4g}."

    return AnalysisResult(title="Several groups, subsamples inside", summary=summary, tables=tables, notes=notes, charts=charts)
