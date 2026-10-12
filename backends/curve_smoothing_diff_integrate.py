"""A trace made smoother, or turned into its slope or its running total.
Taking the slope or the total, and smoothing, are independent choices (you
can smooth AND differentiate the smoothed curve, for example), not
mutually exclusive radio buttons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class CurveOpsOptions:
    operation: str = "none"  # "none" | "first_derivative" | "second_derivative" | "integrate"
    integral_start_y: float = 0.0
    smooth: bool = True
    smooth_neighbors: int = 4  # per side
    smooth_polyorder: int = 2


def compute(x: np.ndarray, y: np.ndarray, opt: CurveOpsOptions) -> tuple[np.ndarray, list[str], str]:
    """The trace after smoothing and the chosen operation: (new Y, what was done, what Y now is).
    ``x`` must be in rising order."""
    from scipy.signal import savgol_filter

    steps = []
    if opt.smooth:
        window = 2 * max(1, opt.smooth_neighbors) + 1
        window = min(window, x.size if x.size % 2 == 1 else x.size - 1)
        window = max(3, window)
        polyorder = min(opt.smooth_polyorder, window - 1)
        y = savgol_filter(y, window_length=window, polyorder=polyorder)
        steps.append(f"smoothed over {window} points, polynomial of degree {polyorder}")

    if opt.operation == "first_derivative":
        y = np.gradient(y, x)
        steps.append("slope taken")
        y_label = "dY/dX"
    elif opt.operation == "second_derivative":
        y = np.gradient(np.gradient(y, x), x)
        steps.append("slope taken twice")
        y_label = "d²Y/dX²"
    elif opt.operation == "integrate":
        y = opt.integral_start_y + np.concatenate([[0.0], np.cumsum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)])
        steps.append(f"area built up from {opt.integral_start_y:g}")
        y_label = "Running area"
    else:
        y_label = "Y"
    return y, steps, y_label


@register("curve_smoothing_diff_integrate")
def run(data: SheetData, options: CurveOpsOptions | None = None) -> AnalysisResult:
    opt = options or CurveOpsOptions()
    x, y = data.paired_xy()
    if x.size < 5:
        raise AnalysisError(f"This needs at least 5 paired (X, Y) points; found {x.size}.")

    order = np.argsort(x)
    x, y = x[order], y[order]
    y, steps, y_label = compute(x, y, opt)

    table = ResultTable(
        title=", ".join(steps).capitalize() if steps else "The trace as it is",
        headers=["X", y_label],
        rows=[[f"{xi:.4g}", f"{yi:.4g}"] for xi, yi in zip(x, y)],
    )
    summary = f"{', '.join(steps).capitalize() if steps else 'Nothing changed'}, across {x.size} points."
    if opt.operation == "integrate":
        summary += f" Total area = {y[-1] - opt.integral_start_y:.4g}."

    return AnalysisResult(title="Smooth / Differentiate / Integrate Curve", summary=summary, tables=[table])
