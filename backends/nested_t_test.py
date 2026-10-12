"""Nested t-Test. Compares exactly two groups using
the correct error term for a nested design (subjects-within-groups
variability), rather than pooling every replicate as if it were an
independent subject."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from ._nested_common import decompose


@dataclass
class NestedTTestOptions:
    report_diff_as: str = "a_minus_b"  # "a_minus_b" | "b_minus_a"
    confidence_level: float = 0.95
    zero_variance_mode: str = "keep"  # "keep" | "simplify" | "full"
    graph_ci: bool = False
    report_goodness_of_fit: bool = False
    significant_digits: int = 4
    show_residual_plot: bool = False
    show_homoscedasticity_plot: bool = False
    show_qq_plot: bool = False


@register("nested_t_test")
def run(data: SheetData, options: NestedTTestOptions | None = None) -> AnalysisResult:
    opt = options or NestedTTestOptions()
    d = decompose(data, min_groups=2)
    if len(d.labels) != 2:
        raise AnalysisError(f"A nested t-test compares exactly 2 groups; this sheet has {len(d.labels)} usable groups. Use nested one-way ANOVA for 3 or more.")

    sig = opt.significant_digits
    pct = int(round(opt.confidence_level * 100))
    ms_subjects = d.ms_subjects
    zero_variance = ms_subjects == 0

    notes = ["Replicate (sub)columns were averaged into one subject mean per row first; the comparison is tested against subject-to-subject variability within each group, not raw replicate variability."]

    if zero_variance and opt.zero_variance_mode == "simplify":
        # Fall back to treating every replicate value as an independent
        # observation (collapsing the nested structure) - the only
        # option left once there is no subject-to-subject variance at
        # all to test against.
        arrays = [data.group_values(label, mode="pooled") for label in d.labels]
        t_stat, p = stats.ttest_ind(arrays[0], arrays[1])
        df = arrays[0].size + arrays[1].size - 2
        notes.append("Every random effect had zero variance, so the model was simplified to an ordinary two-sample t-test on the raw replicate values (nesting no longer applies).")
    else:
        f_stat = d.f_statistic
        p = float(stats.f.sf(f_stat, 1, d.df_subjects))
        t_stat = float(np.sqrt(f_stat)) * (1 if d.group_means[0] >= d.group_means[1] else -1)
        df = d.df_subjects
        if zero_variance:
            notes.append("The variation between subjects came out as zero. As asked, the model was fitted all the same, letting that variance go below zero if the data call for it.")

    a_label, b_label = d.labels
    if opt.report_diff_as == "a_minus_b":
        diff_label, mean_diff = f"{a_label} - {b_label}", d.group_means[0] - d.group_means[1]
    else:
        diff_label, mean_diff = f"{b_label} - {a_label}", d.group_means[1] - d.group_means[0]

    group_rows = [[label, str(n), f"{mean:.{sig}g}"] for label, n, mean in zip(d.labels, d.group_n_subjects, d.group_means)]
    tables = [ResultTable(title="Each group's mean of its subjects' means", headers=["Group", "Subjects", "Mean"], rows=group_rows)]

    # the interval of the difference is the main result, so it is always reported
    half = float("nan")
    if ms_subjects:
        se_diff = np.sqrt(ms_subjects * (1 / d.group_n_subjects[0] + 1 / d.group_n_subjects[1]))
        half = float(stats.t.ppf(0.5 + opt.confidence_level / 2, d.df_subjects) * se_diff)
    ci_text = f"{mean_diff - half:.{sig}g} to {mean_diff + half:.{sig}g}" if np.isfinite(half) else "n/a"
    test_row = [diff_label, f"{mean_diff:.{sig}g}", ci_text,
                f"t({df:.4g}) = {t_stat:.4f}" if not (zero_variance and opt.zero_variance_mode == "simplify") else f"t({df}) = {t_stat:.4f}",
                f"{p:.4g}", "Yes" if p < 0.05 else "No"]
    tables.append(ResultTable(title="The two groups compared, subject against subject", headers=["Comparison", "Mean difference", f"{pct}% CI", "Statistic", "P value", "Significant (α = 0.05)"], rows=[test_row]))

    ci_chart = None
    if opt.graph_ci and np.isfinite(half):
        from ..backend_types import ChartSeries, ChartSpec

        ci_chart = ChartSpec(title=f"Difference between means, with its {pct}% CI",
                             series=[ChartSeries(diff_label, [0.0], [float(mean_diff)], kind="scatter", y_err=[half])],
                             x_categories=[diff_label], y_label="Difference", reference_lines=[{"kind": "h", "value": 0.0}])

    if opt.report_goodness_of_fit:
        r2 = d.ss_between / (d.ss_between + d.ss_subjects) if (d.ss_between + d.ss_subjects) else float("nan")
        tables.append(ResultTable(title="Goodness of fit", headers=["R^2"], rows=[[f"{r2:.{sig}g}" if r2 == r2 else "n/a"]]))

    charts = [ci_chart] if ci_chart is not None else []
    if opt.show_residual_plot or opt.show_homoscedasticity_plot or opt.show_qq_plot:
        from ..chart_helpers import qq_chart, residual_chart

        _reps_a, matrix_a = data.group_replicate_matrix(a_label)
        _reps_b, matrix_b = data.group_replicate_matrix(b_label)
        subj_means_a = np.nanmean(matrix_a, axis=1)
        subj_means_b = np.nanmean(matrix_b, axis=1)
        subj_means_a = subj_means_a[~np.isnan(subj_means_a)]
        subj_means_b = subj_means_b[~np.isnan(subj_means_b)]
        predicted = np.concatenate([np.full(subj_means_a.size, d.group_means[0]), np.full(subj_means_b.size, d.group_means[1])])
        values = np.concatenate([subj_means_a, subj_means_b])
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

    summary = f"{diff_label}: mean difference = {mean_diff:.{sig}g}, p = {p:.4g}."

    return AnalysisResult(title="Two groups, subsamples inside", summary=summary, tables=tables, notes=notes, charts=charts)
