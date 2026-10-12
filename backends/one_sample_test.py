"""One group against a value you expect: a one-sample t test or the Wilcoxon signed-rank test."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register



def _p(value, opt) -> str:
    """One P value, written the way the user asked for (analysis/p_values.py)."""
    from ..p_values import format_p

    return format_p(value, getattr(opt, "p_style", "number"),
                    digits=getattr(opt, "significant_digits", 4),
                    alpha=getattr(opt, "alpha", 0.05))

@dataclass
class OneSampleOptions:
    test: str = "t_test"  # "t_test" | "ratio_t_test" | "wilcoxon"
    hypothetical_value: float = 0.0
    tie_handling: str = "ignore"  # "ignore" | "pratt" - Wilcoxon values equal to the hypothetical value
    calc_discrepancy_ci: bool = False  # Wilcoxon only - Hodges-Lehmann CI
    subcolumns: str = "average"  # "average" | "separate" | "pooled"
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"  # how P values are written (analysis/p_values.py)


def _hodges_lehmann_ci(shifted: np.ndarray, alpha: float) -> tuple[float, float, float] | None:
    """Hodges-Lehmann estimate of the discrepancy (pseudo-median of the
    Walsh averages) with a distribution-free CI built from the ranks of
    the signed-rank statistic - the standard nonparametric companion to
    the Wilcoxon signed-rank test."""
    n = shifted.size
    if n < 1 or n > 200:
        return None
    walsh = np.array([(shifted[i] + shifted[j]) / 2 for i, j in combinations(range(n), 2)] + list(shifted))
    walsh.sort()
    m = walsh.size
    estimate = float(np.median(walsh))

    mean_w = n * (n + 1) / 4
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = stats.norm.ppf(1 - alpha / 2)
    rank_lo = int(np.floor(mean_w - z * sd_w))
    rank_hi = m - rank_lo + 1
    rank_lo = max(1, rank_lo)
    rank_hi = min(m, rank_hi)
    if rank_lo >= rank_hi:
        return estimate, float("nan"), float("nan")
    return estimate, float(walsh[rank_lo - 1]), float(walsh[rank_hi - 1])


@register("one_sample_test")
def run(data: SheetData, options: OneSampleOptions | None = None) -> AnalysisResult:
    opt = options or OneSampleOptions()
    groups = data.all_group_arrays(mode="pooled" if opt.subcolumns == "pooled" else "average")
    if not groups:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    label, values = next(iter(groups.items()))
    sig = opt.significant_digits
    mu0 = opt.hypothetical_value

    if opt.subcolumns == "separate":
        reps, matrix = data.group_replicate_matrix(label)
        if matrix.shape[1] < 2:
            value_sets = [(label, values)]
        else:
            arrays = data.group_replicate_arrays(label)
            value_sets = [(f"{label} (subcolumn {rep})", arr) for rep, arr in sorted(arrays.items())]
    else:
        value_sets = [(label, values)]

    for _, v in value_sets:
        if v.size < 2:
            raise AnalysisError(f"'{label}' needs at least 2 values for a one-sample test.")

    if opt.test == "ratio_t_test" and any(np.any(v <= 0) for _, v in value_sets):
        raise AnalysisError("The one sample ratio t test needs every value to be > 0 (it compares geometric means).")
    if opt.test == "ratio_t_test" and mu0 <= 0:
        raise AnalysisError("The hypothetical value for a ratio t test must be > 0 (it is a geometric mean).")

    tables = []
    summaries = []
    for set_label, v in value_sets:
        n = v.size
        mean = float(np.mean(v))
        sem = float(stats.sem(v))

        if opt.test == "ratio_t_test":
            log_v = np.log(v)
            log_mu0 = np.log(mu0)
            t_stat, t_p = stats.ttest_1samp(log_v, log_mu0)
            geo_mean = float(np.exp(np.mean(log_v)))
            log_sem = float(stats.sem(log_v))
            half = stats.t.ppf(0.5 + (1 - opt.alpha) / 2, n - 1) * log_sem
            ci = f"{np.exp(np.mean(log_v) - half):.{sig}g} to {np.exp(np.mean(log_v) + half):.{sig}g}"
            table = ResultTable(
                title=f"'{set_label}' geometric mean vs hypothetical value = {mu0:g}",
                headers=["Test", "Statistic", "P value", f"Significant (α = {opt.alpha:g})"],
                rows=[["One sample ratio t test", f"t = {t_stat:.4f}", _p(t_p, opt), "Yes" if t_p < opt.alpha else "No"]],
            )
            tables.append(table)
            summaries.append(f"'{set_label}': geometric mean = {geo_mean:.{sig}g} ({int(round((1 - opt.alpha) * 100))}% CI {ci}) vs {mu0:g}, p = {_p(t_p, opt)}.")
            continue

        if opt.test == "wilcoxon":
            shifted = v - mu0
            if opt.tie_handling == "ignore":
                nonzero = shifted[shifted != 0]
            else:
                nonzero = shifted  # Pratt's method keeps zeros in the ranking
            rows = []
            if nonzero.size == 0:
                rows.append(["Wilcoxon signed-rank", "n/a", "n/a", "n/a"])
                w_p = float("nan")
            else:
                try:
                    w_stat, w_p = stats.wilcoxon(nonzero, zero_method="pratt" if opt.tie_handling == "pratt" else "wilcox")
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                rows.append([
                    "Wilcoxon signed-rank",
                    f"W = {w_stat:.4f}" if w_stat == w_stat else "n/a",
                    _p(w_p, opt),
                    "Yes" if w_p == w_p and w_p < opt.alpha else ("No" if w_p == w_p else "n/a"),
                ])
            table = ResultTable(
                title=f"'{set_label}' median vs hypothetical value = {mu0:g}",
                headers=["Test", "Statistic", "P value", f"Significant (α = {opt.alpha:g})"],
                rows=rows,
            )
            tables.append(table)
            summary = f"'{set_label}': median = {np.median(v):.{sig}g} vs {mu0:g}, Wilcoxon p = {_p(w_p, opt)}." if w_p == w_p else f"'{set_label}': all values equal the hypothetical value."

            if opt.calc_discrepancy_ci and nonzero.size >= 1:
                hl = _hodges_lehmann_ci(nonzero, opt.alpha)
                if hl is not None:
                    est, lo, hi = hl
                    ci_str = f"{lo:.{sig}g} to {hi:.{sig}g}" if lo == lo else "n/a (sample too small for a distribution-free CI)"
                    tables.append(ResultTable(
                        title=f"Median difference from the value (Hodges-Lehmann) - '{set_label}'",
                        headers=["Median difference", f"{int(round((1 - opt.alpha) * 100))}% CI"],
                        rows=[[f"{est:.{sig}g}", ci_str]],
                    ))
                else:
                    tables.append(ResultTable(
                        title=f"Median difference from the value - '{set_label}'",
                        headers=["Note"], rows=[["Sample too large (n > 200) for the exact Walsh-average CI used here."]],
                    ))
            summaries.append(summary)
            continue

        # ordinary one-sample t test
        t_stat, t_p = stats.ttest_1samp(v, mu0)
        half_width = stats.t.ppf(0.5 + (1 - opt.alpha) / 2, n - 1) * sem
        ci = f"{mean - half_width:.{sig}g} to {mean + half_width:.{sig}g}"
        table = ResultTable(
            title=f"'{set_label}' mean vs hypothetical value = {mu0:g}",
            headers=["Test", "Statistic", "P value", f"Significant (α = {opt.alpha:g})"],
            rows=[["One sample t test", f"t = {t_stat:.4f}", _p(t_p, opt), "Yes" if t_p < opt.alpha else "No"]],
        )
        tables.append(table)
        summaries.append(f"'{set_label}': mean = {mean:.{sig}g} ({int(round((1 - opt.alpha) * 100))}% CI {ci}) vs {mu0:g}, p = {_p(t_p, opt)}.")

    notes = [f"Only the first group ('{label}') was tested."]
    if opt.test == "wilcoxon" and opt.tie_handling == "pratt":
        notes.append("Values equal to the hypothetical value were kept in the ranking using Pratt's method rather than discarded.")

    return AnalysisResult(title="One-Sample Test", summary=" ".join(summaries), tables=tables, notes=notes)
