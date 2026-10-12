"""A curve through XY data without a model: straight segments, an interpolating
spline (cubic or Akima), a smoothing spline with a chosen number of knots, or
LOWESS with a chosen neighbourhood size."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, LSQUnivariateSpline, UnivariateSpline

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

# how local each LOWESS fit is -> the number of neighbouring points it looks at
LOWESS_PRESETS = {
    "Close to the data - 6 neighbours": 6,
    "Balanced - 10 neighbours (usual)": 10,
    "Broad trend - 20 neighbours": 20,
}
LOWESS_DEFAULT = "Balanced - 10 neighbours (usual)"


@dataclass
class SplineLowessOptions:
    method: str = "smoothing_spline"  # "linear" | "cubic" | "akima" | "smoothing_spline" | "lowess"
    n_knots: int = 4
    lowess_preset: str = LOWESS_DEFAULT
    n_segments: int = 20
    report_residuals: bool = False


def _lowess_reference(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """The point-by-point version, kept to check _lowess against."""
    n = x.size
    k = max(2, min(window, n))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    smoothed = np.empty(n)
    for i in range(n):
        dist = np.abs(xs - xs[i])
        idx = np.argsort(dist, kind="stable")[:k]
        d_max = dist[idx].max() or 1.0
        w = np.clip((1 - (dist[idx] / d_max) ** 3) ** 3, 0, None)
        X = np.column_stack([np.ones(k), xs[idx]])
        W = np.diag(w)
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ ys[idx])
            smoothed[i] = beta[0] + beta[1] * xs[i]
        except np.linalg.LinAlgError:
            smoothed[i] = np.average(ys[idx], weights=w) if w.sum() else ys[i]
    unsorted = np.empty(n)
    unsorted[order] = smoothed
    return unsorted


def _lowess(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Locally weighted straight lines (tricube weights over the ``window``
    nearest points). On sorted X the nearest points are always a run of
    neighbours, so each point's window is found with a sliding pointer and
    every local fit is solved at once - linear in the number of points
    instead of re-sorting the whole column for each one."""
    n = x.size
    k = max(2, min(window, n))
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    starts = np.empty(n, dtype=np.int64)
    lo = 0
    for i in range(n):
        while lo + k < n and xs[i] - xs[lo] > xs[lo + k] - xs[i]:
            lo += 1
        starts[i] = lo
    idx = starts[:, None] + np.arange(k)[None, :]
    xw, yw = xs[idx], ys[idx]
    dist = np.abs(xw - xs[:, None])
    d_max = dist.max(axis=1, keepdims=True)
    d_max[d_max == 0] = 1.0
    w = np.clip((1 - (dist / d_max) ** 3) ** 3, 0, None)
    sw = w.sum(axis=1)
    swx = (w * xw).sum(axis=1)
    swy = (w * yw).sum(axis=1)
    swxx = (w * xw * xw).sum(axis=1)
    swxy = (w * xw * yw).sum(axis=1)
    det = sw * swxx - swx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (sw * swxy - swx * swy) / det
        intercept = (swy - slope * swx) / sw
        line = intercept + slope * xs
        flat = swy / sw  # all window X equal: the weighted mean
    scale = np.maximum(np.abs(sw * swxx), 1e-300)
    smoothed = np.where(np.abs(det) > 1e-12 * scale, line, np.where(sw > 0, flat, ys))
    unsorted = np.empty(n)
    unsorted[order] = smoothed
    return unsorted


@register("spline_lowess_smoothing")
def run(data: SheetData, options: SplineLowessOptions | None = None) -> AnalysisResult:
    opt = options or SplineLowessOptions()
    x, y = data.paired_xy()
    if x.size < 5:
        raise AnalysisError(f"Smoothing needs at least 5 paired (X, Y) points; found {x.size}.")

    large_note = ""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    if opt.method == "linear":
        smoothed_at_x = np.interp(x, x_unique, y_unique)
        method_label = "Straight segments between the points"
    elif opt.method == "cubic":
        if x_unique.size < 4:
            raise AnalysisError("Cubic spline needs at least 4 distinct X values.")
        spline = CubicSpline(x_unique, y_unique)
        smoothed_at_x = spline(x)
        method_label = "Cubic spline (through every point)"
    elif opt.method == "akima":
        if x_unique.size < 5:
            raise AnalysisError("Akima spline needs at least 5 distinct X values.")
        spline = Akima1DInterpolator(x_unique, y_unique)
        smoothed_at_x = spline(x)
        method_label = "Akima spline (through every point)"
    elif opt.method == "lowess":
        window = LOWESS_PRESETS.get(opt.lowess_preset, 10)
        smoothed_at_x = _lowess(x, y, window)
        method_label = f"LOWESS, {window} neighbours"
    else:  # smoothing_spline
        if x_unique.size < 4:
            raise AnalysisError("Smoothing spline needs at least 4 distinct X values.")
        # a least-squares cubic spline with the chosen number of interior
        # knots, placed at quantiles of X so each stretch holds a similar
        # share of the data. (An earlier version fed a smoothing factor of
        # n·var/knots to scipy's automatic spline, which fixed R² at about
        # 1 - 1/knots whatever the data - and took minutes on long columns.)
        k = max(0, min(opt.n_knots, x_unique.size - 4))
        interior = np.unique(np.quantile(x_unique, np.linspace(0, 1, k + 2)[1:-1])) if k else np.array([])
        interior = interior[(interior > x_unique[0]) & (interior < x_unique[-1])]
        try:
            spline = LSQUnivariateSpline(x_unique, y_unique, interior)
        except ValueError:  # knots too crowded for the data: fewer knots
            spline = LSQUnivariateSpline(x_unique, y_unique, interior[::2] if interior.size > 1 else [])
        k = len(spline.get_knots()) - 2
        smoothed_at_x = spline(x)
        method_label = f"Smoothing spline ({k} knots)"

    residuals = y - smoothed_at_x
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_segments = max(2, opt.n_segments)
    curve_x = np.linspace(x_unique.min(), x_unique.max(), n_segments)
    if opt.method == "linear":
        curve_y = np.interp(curve_x, x_unique, y_unique)
    elif opt.method == "cubic":
        curve_y = CubicSpline(x_unique, y_unique)(curve_x)
    elif opt.method == "akima":
        curve_y = Akima1DInterpolator(x_unique, y_unique)(curve_x)
    elif opt.method == "lowess":
        curve_y = np.interp(curve_x, *zip(*sorted(zip(x, smoothed_at_x))))
    else:
        curve_y = spline(curve_x)  # the spline already fitted above

    table = ResultTable(
        title=f"{method_label} - curve at {n_segments} segments",
        headers=["X", "Y (curve)"],
        rows=[[f"{cx:.4g}", f"{cy:.4g}"] for cx, cy in zip(curve_x, curve_y)],
    )
    tables = [table]
    if opt.report_residuals:
        tables.append(ResultTable(
            title="Residuals (observed - curve)",
            headers=["X", "Y (observed)", "Y (curve)", "Residual"],
            rows=[[f"{xi:.4g}", f"{yi:.4g}", f"{si:.4g}", f"{yi - si:.4g}"] for xi, yi, si in zip(x, y, smoothed_at_x)],
        ))

    summary = f"{method_label} through {x.size} points; R² (curve vs observed) = {r_squared:.4f}."

    return AnalysisResult(title="Smooth without a model", summary=summary, tables=tables,
                          notes=[large_note] if large_note else [])
