"""Summarise each column: summary statistics for every group on the
sheet, with the exact set of statistics and how replicate subcolumns are
combined left up to the user (see DescriptiveStatsOptions / the options
dialog registered at the bottom of this file)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class DescriptiveStatsOptions:
    """Which statistics to compute, and how to combine replicate
    subcolumns, for one run of Descriptive Statistics."""

    # Basics
    min_max_range: bool = True
    mean_sd_sem: bool = True
    quartiles: bool = False
    column_sum: bool = False
    # Advanced
    coefficient_of_variation: bool = False
    skewness_kurtosis: bool = False
    percentile: bool = False
    percentile_value: float = 90.0
    geometric_mean: bool = False
    harmonic_mean: bool = False
    quadratic_mean: bool = False
    # Confidence intervals
    ci_mean: bool = False
    ci_geometric_mean: bool = False
    ci_harmonic_mean: bool = False
    ci_median: bool = False
    confidence_level: float = 0.95
    # Sturdy against odd values, and the spread of values that multiply
    trimmed_mean: bool = False          # the mean with a share cut from each end
    trim_percent: float = 10.0
    robust_spread: bool = False         # interquartile range, median absolute deviation, and the SD it implies
    geometric_sd: bool = False          # the factor the geometric mean is multiplied and divided by
    ci_sd: bool = False                 # an interval for the SD (trusts a normal shape)
    # Subcolumns: "average" | "separate" | "pooled"
    subcolumn_mode: str = "average"
    # Output
    significant_digits: int = 4


def _fmt(value: float, sig: int) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{max(1, sig)}g}"


def _t_ci(mean: float, sem: float, n: int, conf: float) -> tuple[float, float]:
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * sem
    return mean - half, mean + half


def _median_ci(values: np.ndarray, conf: float) -> tuple[float, float] | None:
    """Distribution-free confidence interval for the median, from the
    binomial distribution of order statistics (the standard nonparametric
    approach) - needs a handful of points per tail to say anything."""
    n = values.size
    if n < 6:
        return None
    ordered = np.sort(values)
    alpha = 1 - conf
    lo_rank = int(stats.binom.ppf(alpha / 2, n, 0.5))
    hi_rank = int(stats.binom.ppf(1 - alpha / 2, n, 0.5))
    lo_idx = max(lo_rank - 1, 0)
    hi_idx = min(hi_rank, n - 1)
    if lo_idx >= hi_idx:
        return None
    return float(ordered[lo_idx]), float(ordered[hi_idx])


def _compute_row(label: str, values: np.ndarray, opt: DescriptiveStatsOptions) -> dict[str, str]:
    """One output row's worth of formatted stats for ``values``, keyed by
    the column headers ``_active_columns`` will request."""
    sig = opt.significant_digits
    n = int(values.size)
    row: dict[str, str] = {"Group": label, "N": str(n)}

    if n == 0:
        return row

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")

    if opt.min_max_range:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        row["Minimum"] = _fmt(vmin, sig)
        row["Maximum"] = _fmt(vmax, sig)
        row["Range"] = _fmt(vmax - vmin, sig)

    if opt.mean_sd_sem:
        row["Mean"] = _fmt(mean, sig)
        row["SD"] = _fmt(sd, sig)
        row["SEM"] = _fmt(sem, sig)

    if opt.quartiles:
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        row["Q1 (25%)"] = _fmt(float(q1), sig)
        row["Median"] = _fmt(float(med), sig)
        row["Q3 (75%)"] = _fmt(float(q3), sig)

    if opt.column_sum:
        row["Sum"] = _fmt(float(np.sum(values)), sig)

    if opt.coefficient_of_variation:
        cv = (sd / mean * 100.0) if mean else float("nan")
        row["CV%"] = _fmt(cv, sig)

    if opt.skewness_kurtosis:
        row["Skewness"] = _fmt(float(stats.skew(values)), sig) if n > 2 else "-"
        row["Kurtosis"] = _fmt(float(stats.kurtosis(values)), sig) if n > 3 else "-"

    if opt.percentile:
        pct = np.clip(opt.percentile_value, 0, 100)
        row[f"P{pct:g}"] = _fmt(float(np.percentile(values, pct)), sig)

    positive = values[values > 0]
    all_positive = positive.size == n and n > 0

    if opt.geometric_mean:
        row["Geo. Mean"] = _fmt(float(stats.gmean(positive)), sig) if all_positive else "n/a"

    if opt.harmonic_mean:
        row["Harm. Mean"] = _fmt(float(stats.hmean(positive)), sig) if all_positive else "n/a"

    if opt.quadratic_mean:
        row["Quad. Mean"] = _fmt(float(np.sqrt(np.mean(np.square(values)))), sig)

    if opt.ci_mean:
        if n > 1:
            lo, hi = _t_ci(mean, sem, n, opt.confidence_level)
            row["CI of Mean"] = f"{_fmt(lo, sig)} to {_fmt(hi, sig)}"
        else:
            row["CI of Mean"] = "n/a"

    if opt.ci_geometric_mean:
        if all_positive and n > 1:
            log_vals = np.log(positive)
            log_mean, log_sem = float(np.mean(log_vals)), float(stats.sem(log_vals))
            lo, hi = _t_ci(log_mean, log_sem, n, opt.confidence_level)
            row["CI of Geo. Mean"] = f"{_fmt(np.exp(lo), sig)} to {_fmt(np.exp(hi), sig)}"
        else:
            row["CI of Geo. Mean"] = "n/a"

    if opt.ci_harmonic_mean:
        if all_positive and n > 1:
            recip = 1.0 / positive
            r_mean, r_sem = float(np.mean(recip)), float(stats.sem(recip))
            lo, hi = _t_ci(r_mean, r_sem, n, opt.confidence_level)
            # Reciprocal transform inverts the order of the bounds.
            hi_h = 1.0 / lo if lo > 0 else float("inf")
            lo_h = 1.0 / hi if hi > 0 else float("inf")
            row["CI of Harm. Mean"] = f"{_fmt(lo_h, sig)} to {_fmt(hi_h, sig)}"
        else:
            row["CI of Harm. Mean"] = "n/a"

    if opt.ci_median:
        ci = _median_ci(values, opt.confidence_level)
        row["CI of Median"] = f"{_fmt(ci[0], sig)} to {_fmt(ci[1], sig)}" if ci else "n/a"

    if opt.trimmed_mean:
        share = float(np.clip(opt.trim_percent, 0, 49.9)) / 100
        row[_trim_title(opt)] = _fmt(float(stats.trim_mean(values, share)), sig)

    if opt.robust_spread:
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        mad = float(np.median(np.abs(values - med)))
        row["IQR"] = _fmt(float(q3 - q1), sig)
        row["MAD"] = _fmt(mad, sig)
        # 1.4826 x MAD estimates the SD when the bulk of the values is normal, whatever a few odd ones do
        row["Robust SD"] = _fmt(1.4826 * mad, sig)

    if opt.geometric_sd:
        row["Geo. SD factor"] = _fmt(float(np.exp(np.std(np.log(positive), ddof=1))), sig) if all_positive and n > 1 else "n/a"

    if opt.ci_sd:
        if n > 1 and sd > 0:
            tail = (1 - opt.confidence_level) / 2
            lo = sd * np.sqrt((n - 1) / stats.chi2.ppf(1 - tail, n - 1))
            hi = sd * np.sqrt((n - 1) / stats.chi2.ppf(tail, n - 1))
            row["CI of SD"] = f"{_fmt(lo, sig)} to {_fmt(hi, sig)}"
        else:
            row["CI of SD"] = "n/a"

    return row


def _trim_title(opt: DescriptiveStatsOptions) -> str:
    return f"Mean, {float(np.clip(opt.trim_percent, 0, 49.9)):g}% trimmed"


def _active_columns(opt: DescriptiveStatsOptions) -> list[str]:
    columns = ["Group", "N"]
    if opt.min_max_range:
        columns += ["Minimum", "Maximum", "Range"]
    if opt.mean_sd_sem:
        columns += ["Mean", "SD", "SEM"]
    if opt.quartiles:
        columns += ["Q1 (25%)", "Median", "Q3 (75%)"]
    if opt.column_sum:
        columns.append("Sum")
    if opt.coefficient_of_variation:
        columns.append("CV%")
    if opt.skewness_kurtosis:
        columns += ["Skewness", "Kurtosis"]
    if opt.percentile:
        columns.append(f"P{np.clip(opt.percentile_value, 0, 100):g}")
    if opt.geometric_mean:
        columns.append("Geo. Mean")
    if opt.harmonic_mean:
        columns.append("Harm. Mean")
    if opt.quadratic_mean:
        columns.append("Quad. Mean")
    if opt.ci_mean:
        columns.append("CI of Mean")
    if opt.ci_geometric_mean:
        columns.append("CI of Geo. Mean")
    if opt.ci_harmonic_mean:
        columns.append("CI of Harm. Mean")
    if opt.ci_median:
        columns.append("CI of Median")
    if opt.trimmed_mean:
        columns.append(_trim_title(opt))
    if opt.robust_spread:
        columns += ["IQR", "MAD", "Robust SD"]
    if opt.geometric_sd:
        columns.append("Geo. SD factor")
    if opt.ci_sd:
        columns.append("CI of SD")
    return columns


@register("descriptive_statistics")
def run(data: SheetData, options: DescriptiveStatsOptions | None = None) -> AnalysisResult:
    opt = options or DescriptiveStatsOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    columns = _active_columns(opt)
    rows: list[list[str]] = []
    n_summarized = 0
    biggest: tuple[str, int] = ("", -1)

    for label in labels:
        if opt.subcolumn_mode == "separate":
            for rep, values in sorted(data.group_replicate_arrays(label).items()):
                row_label = f"{label} : Y{rep}"
                row_dict = _compute_row(row_label, values, opt)
                rows.append([row_dict.get(c, "-") for c in columns])
                n_summarized += 1
                if values.size > biggest[1]:
                    biggest = (row_label, values.size)
        else:
            values = data.group_values(label, mode=opt.subcolumn_mode)
            row_dict = _compute_row(label, values, opt)
            rows.append([row_dict.get(c, "-") for c in columns])
            n_summarized += 1
            if values.size > biggest[1]:
                biggest = (label, values.size)

    table = ResultTable(title="Each column, summarised", headers=columns, rows=rows)

    mode_note = {
        "average": "replicate subcolumns were averaged within each row before summarizing",
        "separate": "each replicate subcolumn was summarized on its own",
        "pooled": "every replicate value was pooled into one set per group",
    }[opt.subcolumn_mode]
    summary = (
        f"{n_summarized} group(s) summarized; largest is {biggest[0]} (n={biggest[1]})."
    )

    return AnalysisResult(
        title="Summarise each column",
        summary=summary,
        tables=[table],
        notes=[mode_note.capitalize() + "."],
    )
