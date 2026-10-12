"""Is it normally distributed? Pick one or more distributions
to test, one or more test methods, and an optional relative-likelihood
(Gaussian vs lognormal) comparison."""

from __future__ import annotations

from dataclasses import dataclass

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
class NormalityOptions:
    test_normal: bool = True
    test_lognormal: bool = False
    compute_relative_likelihood: bool = False
    # one test by default - several on the same question invite picking the P you like
    method_dagostino: bool = False
    method_anderson: bool = False
    method_shapiro: bool = True
    method_ks: bool = False
    create_qq_plot: bool = True
    subcolumns: str = "average"  # "average" | "separate" | "pooled"
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"  # how P values are written (analysis/p_values.py)


def _dallal_wilkinson_lilliefors_p(d_stat: float, n: int) -> float:
    """Lilliefors-corrected P value for the KS statistic when the mean and
    SD were estimated from the same data: the Dallal-Wilkinson (1986)
    formula for P < 0.1, and Stephens' modified statistic above that.
    Needs n >= 5."""
    if n < 5:
        return float("nan")
    if d_stat <= 0:
        return 1.0
    if n <= 100:
        kd, nd = d_stat, n
    else:
        kd, nd = d_stat * (n / 100) ** 0.49, 100
    p = np.exp(-7.01256 * kd**2 * (nd + 2.78019) + 2.99587 * kd * np.sqrt(nd + 2.78019)
               - 0.122119 + 0.974598 / np.sqrt(nd) + 1.67997 / nd)
    if p > 0.1:
        kk = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * d_stat
        if kk <= 0.302:
            p = 1.0
        elif kk <= 0.5:
            p = 2.76773 - 19.828315 * kk + 80.709644 * kk**2 - 138.55152 * kk**3 + 81.218052 * kk**4
        elif kk <= 0.9:
            p = -4.901232 + 40.662806 * kk - 97.490286 * kk**2 + 94.029866 * kk**3 - 32.355711 * kk**4
        elif kk <= 1.31:
            p = 6.198765 - 19.558097 * kk + 23.186922 * kk**2 - 12.234627 * kk**3 + 2.423045 * kk**4
        else:
            p = 0.0
    return float(np.clip(p, 0.0, 1.0))


def _run_methods(values: np.ndarray, opt: NormalityOptions, log_scale: bool) -> list[list[str]]:
    rows = []
    n = values.size
    spread = float(np.ptp(values)) if n else 0.0
    if n and (spread == 0 or not np.isfinite(spread) or spread < 1e-12 * max(1.0, float(np.max(np.abs(values))))):
        return [["Every test", "-", "n/a", "every value is the same: there is no shape to judge"]]
    if opt.method_dagostino:
        if n >= 8:
            stat, p = stats.normaltest(values)
            rows.append(["D'Agostino-Pearson omnibus", f"K2 = {stat:.4f}", _p(p, opt), "looks normal" if p >= opt.alpha else "doesn't look normal"])
        else:
            rows.append(["D'Agostino-Pearson omnibus", "n/a", "n/a", f"needs n >= 8 (n = {n})"])
    if opt.method_anderson:
        result = stats.anderson(values, dist="norm", method="interpolate")
        rows.append(["Anderson-Darling", f"A2 = {result.statistic:.4f}", _p(result.pvalue, opt), "looks normal" if result.pvalue >= opt.alpha else "doesn't look normal"])
    if opt.method_shapiro:
        if 3 <= n <= 5000:
            stat, p = stats.shapiro(values)
            rows.append(["Shapiro-Wilk", f"W = {stat:.4f}", _p(p, opt), "looks normal" if p >= opt.alpha else "doesn't look normal"])
        else:
            rows.append(["Shapiro-Wilk", "n/a", "n/a", f"needs 3 <= n <= 5000 (n = {n})"])
    if opt.method_ks:
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        if sd > 0:
            standardized = (values - mean) / sd
            d_stat, _ = stats.kstest(standardized, "norm")
            p = _dallal_wilkinson_lilliefors_p(d_stat, n)
            rows.append(["Kolmogorov-Smirnov (Lilliefors)", f"D = {d_stat:.4f}", _p(p, opt), "looks normal" if (p == p and p >= opt.alpha) else "doesn't look normal" if p == p else "n/a"])
        else:
            rows.append(["Kolmogorov-Smirnov (Lilliefors)", "n/a", "n/a", "all values identical"])
    return rows


def _aicc_normal_vs_lognormal(values: np.ndarray) -> tuple[float, float, str]:
    n = values.size
    ss_normal = float(np.sum((values - np.mean(values)) ** 2))
    ll_normal = -n / 2 * (np.log(2 * np.pi) + np.log(ss_normal / n) + 1)
    k = 2
    aicc_normal = 2 * k - 2 * ll_normal + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    log_v = np.log(values)
    ss_log = float(np.sum((log_v - np.mean(log_v)) ** 2))
    ll_lognormal = -n / 2 * (np.log(2 * np.pi) + np.log(ss_log / n) + 1) - float(np.sum(log_v))
    aicc_lognormal = 2 * k - 2 * ll_lognormal + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    preferred = "Gaussian" if aicc_normal < aicc_lognormal else "Lognormal"
    return aicc_normal, aicc_lognormal, preferred


@register("normality_testing")
def run(data: SheetData, options: NormalityOptions | None = None) -> AnalysisResult:
    opt = options or NormalityOptions()
    if not opt.test_normal and not opt.test_lognormal:
        raise AnalysisError("Select at least one distribution to test (Normal and/or Lognormal).")
    if not any((opt.method_dagostino, opt.method_anderson, opt.method_shapiro, opt.method_ks)):
        raise AnalysisError("Select at least one test method.")

    mode = "pooled" if opt.subcolumns == "pooled" else "average"
    groups = data.all_group_arrays(mode=mode)
    if not groups:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    tables = []
    charts = []
    notes = []
    verdict_bits = []

    for label, values in groups.items():
        if opt.subcolumns == "separate":
            arrays = data.group_replicate_arrays(label)
            value_sets = [(f"{label} (subcolumn {rep})", arr) for rep, arr in sorted(arrays.items())] or [(label, values)]
        else:
            value_sets = [(label, values)]

        for set_label, v in value_sets:
            if v.size < 3:
                notes.append(f"'{set_label}' has too few points (n = {v.size}) for normality testing; skipped.")
                if opt.create_qq_plot:
                    from ..chart_helpers import unavailable_chart

                    charts.append(unavailable_chart(f"Q–Q plot - '{set_label}'",
                                                    f"A Q–Q plot needs at least 3 values; '{set_label}' has {v.size}."))
                continue

            if opt.test_normal:
                rows = _run_methods(v, opt, log_scale=False)
                tables.append(ResultTable(title=f"Normal (Gaussian) distribution - '{set_label}'", headers=["Method", "Statistic", "P value", "Reading (α = %.3g)" % opt.alpha], rows=rows))
                verdict_bits.append(f"{set_label}: {rows[0][3] if rows else 'n/a'} ({rows[0][0] if rows else ''})")

            if opt.test_lognormal:
                if np.any(v <= 0):
                    notes.append(f"'{set_label}' has non-positive values, so it can't be tested for lognormality.")
                else:
                    rows = _run_methods(np.log(v), opt, log_scale=True)
                    tables.append(ResultTable(title=f"Lognormal distribution - '{set_label}'", headers=["Method", "Statistic", "P value", "Reading (α = %.3g)" % opt.alpha], rows=rows))

            if opt.compute_relative_likelihood:
                if np.any(v <= 0):
                    notes.append(f"'{set_label}' has non-positive values, so Gaussian-vs-lognormal likelihood can't be computed.")
                else:
                    aicc_n, aicc_ln, preferred = _aicc_normal_vs_lognormal(v)
                    tables.append(ResultTable(
                        title=f"Relative likelihood: Gaussian vs Lognormal - '{set_label}'",
                        headers=["Model", "AICc"],
                        rows=[["Gaussian", f"{aicc_n:.4g}"], ["Lognormal", f"{aicc_ln:.4g}"]],
                    ))
                    notes.append(f"'{set_label}': AICc favors the {preferred} model.")

            if opt.create_qq_plot:
                sample = np.log(v) if (opt.test_lognormal and not opt.test_normal and np.all(v > 0)) else v
                order = np.argsort(sample)
                sorted_v = sample[order]
                n = sorted_v.size
                theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
                from ..chart_helpers import qq_chart

                charts.append(qq_chart(f"Q–Q plot - '{set_label}'", theoretical, sorted_v,
                                       "Actual value (log)" if sample is not v else "Actual value"))
                tables.append(ResultTable(
                    title=f"Each value against the normal quantile it should match - '{set_label}'",
                    headers=["Theoretical quantile", "Actual value"],
                    rows=[[f"{t:.4g}", f"{s:.{opt.significant_digits}g}"] for t, s in zip(theoretical, sorted_v)][:200],
                ))

    if not tables:
        raise AnalysisError("No group had enough data (n >= 3) to run a normality test.")

    small = [label for label, v in groups.items() if 3 <= v.size < 8]
    if small:
        notes.insert(0, "Fewer than 8 values in " + ", ".join(small) + ": a normality test this small rarely "
                     "flags anything, so 'looks normal' here is weak reassurance.")
    big = [label for label, v in groups.items() if v.size > 300]
    if big:
        notes.append("Over 300 values in " + ", ".join(big) + ": tests flag even harmless departures at this "
                     "size - judge by the Q–Q plot.")
    summary = "; ".join(verdict_bits) if verdict_bits else "Normality/lognormality tests computed."
    return AnalysisResult(title="Is it normally distributed?", summary=summary, tables=tables, notes=notes, charts=charts)
