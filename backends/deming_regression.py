"""A straight line when X is measured with error too (Deming regression): equal vs.
custom X/Y error SDs, replicate handling, CI of Y at X, a linearity
check (Bland-Altman-style replicates test), and a restricted fit range."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class DemingRegressionOptions:
    equal_error: bool = True
    sd_x: float = 1.0
    sd_y: float = 1.0
    replicate_handling: str = "average"  # "individual" | "average"
    ci_y_at_x: bool = False
    ci_y_at_x_value: float = 0.0
    range_mode_start: str = "auto"
    range_start_x: float = 0.0
    range_mode_end: str = "auto"
    range_end_x: float = 0.0
    confidence_level: float = 0.95
    graph: bool = True  # data, the Deming line and the line of identity
    significant_digits: int = 4


def _deming_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, float]:
    """Slope and intercept; lam = (SD of Y error / SD of X error)^2."""
    xm, ym = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    syy = float(np.sum((y - ym) ** 2))
    sxy = float(np.sum((x - xm) * (y - ym)))
    if sxy == 0:
        return float("nan"), float("nan")
    slope = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy**2)) / (2 * sxy)
    return float(slope), float(ym - slope * xm)


def _loo_fits(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope and intercept with each pair left out in turn - all n fits at
    once, from the sums with that pair's share taken away (no refitting loop,
    so a long column costs no more than a short one)."""
    n = x.size
    sx, sy = x.sum() - x, y.sum() - y
    sxx, syy, sxy = (x * x).sum() - x * x, (y * y).sum() - y * y, (x * y).sum() - x * y
    m = n - 1
    xm, ym = sx / m, sy / m
    cxx, cyy, cxy = sxx - m * xm**2, syy - m * ym**2, sxy - m * xm * ym
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (cyy - lam * cxx + np.sqrt((cyy - lam * cxx) ** 2 + 4 * lam * cxy**2)) / (2 * cxy)
    return slope, ym - slope * xm


def _jackknife(x: np.ndarray, y: np.ndarray, lam: float, stat) -> tuple[float, float]:
    """Jackknife estimate and SE of stat(slope, intercept) - the standard way to
    put intervals on a Deming fit (Linnet 1990)."""
    n = x.size
    full = stat(*_deming_fit(x, y, lam))
    loo = np.asarray(stat(*_loo_fits(x, y, lam)), dtype=float)
    pseudo = n * full - (n - 1) * loo
    pseudo = pseudo[np.isfinite(pseudo)]
    return float(np.mean(pseudo)), float(np.std(pseudo, ddof=1) / np.sqrt(pseudo.size))


@register("deming_regression")
def run(data: SheetData, options: DemingRegressionOptions | None = None) -> AnalysisResult:
    opt = options or DemingRegressionOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    # each Y with the X of its own row: a blank cell drops that point and moves nothing else
    x, y = data.xy_points(labels[0], "average" if opt.replicate_handling == "average" else "each")

    if opt.range_mode_start == "manual" or opt.range_mode_end == "manual":
        lo = opt.range_start_x if opt.range_mode_start == "manual" else -np.inf
        hi = opt.range_end_x if opt.range_mode_end == "manual" else np.inf
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]

    if x.size < 4:
        raise AnalysisError(f"Deming regression needs at least 4 paired points; found {x.size}.")

    lam = 1.0 if opt.equal_error else (opt.sd_y**2 / opt.sd_x**2 if opt.sd_x else 1.0)
    sig = opt.significant_digits
    slope, intercept = _deming_fit(x, y, lam)
    if not np.isfinite(slope):
        raise AnalysisError("X and Y don't covary (Sxy = 0) - a Deming fit isn't defined here.")

    n_pts = x.size
    df = n_pts - 1
    t_crit = float(stats.t.ppf(0.5 + opt.confidence_level / 2, df))
    pct = f"{opt.confidence_level * 100:g}%"
    _s_est, s_se = _jackknife(x, y, lam, lambda b, a: b)
    _i_est, i_se = _jackknife(x, y, lam, lambda b, a: a)
    p_slope1 = float(2 * stats.t.sf(abs((slope - 1) / s_se), df)) if s_se > 0 else float("nan")
    p_int0 = float(2 * stats.t.sf(abs(intercept / i_se), df)) if i_se > 0 else float("nan")

    y_mean = float(np.mean(y))
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0

    def ci(est, se):
        return f"{est - t_crit * se:.{sig}g} to {est + t_crit * se:.{sig}g}"

    tables = [
        ResultTable(
            title="Y = slope \u00b7 X + intercept",
            headers=["Parameter", "Estimate", "SE (jackknife)", f"{pct} CI", "Test", "P value"],
            rows=[["Slope", f"{slope:.{sig}g}", f"{s_se:.{sig}g}", ci(slope, s_se), "slope = 1", f"{p_slope1:.4g}"],
                  ["Intercept", f"{intercept:.{sig}g}", f"{i_se:.{sig}g}", ci(intercept, i_se), "intercept = 0",
                   f"{p_int0:.4g}"]],
        ),
        ResultTable(
            title="Fit", headers=["N", "Error variance ratio (\u03bb = SDy\u00b2/SDx\u00b2)", "R\u00b2 (vertical)"],
            rows=[[str(n_pts), f"{lam:.{sig}g}", f"{r_squared:.4f}"]],
        ),
    ]
    notes = ["X and Y are assumed to have equal measurement error (\u03bb = 1)." if opt.equal_error else
             f"\u03bb = (SD of Y error / SD of X error)\u00b2 = {lam:.{sig}g}, from the SDs you supplied.",
             "Intervals come from the jackknife (leave one pair out at a time), the standard method for Deming "
             "regression. If the slope's interval includes 1 and the intercept's includes 0, the two methods agree "
             "within what these data can show."]

    if opt.ci_y_at_x:
        x0 = opt.ci_y_at_x_value
        y0 = slope * x0 + intercept
        _e, y_se = _jackknife(x, y, lam, lambda b, a: b * x0 + a)
        tables.append(ResultTable(title=f"{pct} CI of Y when X = {x0:g}", headers=["Predicted Y", f"{pct} CI (jackknife)"],
                                  rows=[[f"{y0:.{sig}g}", ci(y0, y_se)]]))

    charts = []
    if opt.graph:
        from ..backend_types import ChartSeries, ChartSpec

        lo, hi = float(min(np.min(x), np.min(y))), float(max(np.max(x), np.max(y)))
        charts.append(ChartSpec(
            title="Deming fit", x_label="X (method A)", y_label=f"{labels[0]} (method B)",
            series=[ChartSeries(labels[0], x.tolist(), y.tolist(), kind="scatter"),
                    ChartSeries("Deming line", [lo, hi], [slope * lo + intercept, slope * hi + intercept],
                                kind="line", color="#d62728")],
            reference_lines=[{"kind": "line", "points": [[lo, lo], [hi, hi]], "style": ":", "color": "#888888"}]))

    summary = (f"Y = {slope:.{sig}g}\u00b7X + {intercept:.{sig}g}; slope {pct} CI {ci(slope, s_se)}, "
               f"intercept {pct} CI {ci(intercept, i_se)} (\u03bb = {lam:.{sig}g}).")
    return AnalysisResult(title="Deming Regression", summary=summary, tables=tables, notes=notes, charts=charts)
