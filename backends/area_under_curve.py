"""The area between a trace and a level: the level is set or taken from the
trace's own ends, bumps too low or too narrow are left out, and the result gives each detected peak's area
separately as well as the grand total."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class AreaUnderCurveOptions:
    baseline_mode: str = "fixed"  # "fixed" | "mean_ends"
    baseline_y: float = 0.0
    mean_first_n: int = 1
    mean_last_n: int = 0
    min_height_mode: str = "percent"  # "percent" | "y_units"
    min_height_percent: float = 10.0
    min_height_units: float = 0.0
    min_width_enabled: bool = False
    min_width_points: int = 2
    allow_below_baseline: bool = False
    significant_digits: int = 4


def _baseline_value(y: np.ndarray, opt: AreaUnderCurveOptions) -> float:
    if opt.baseline_mode == "mean_ends":
        first = y[: max(1, opt.mean_first_n)]
        last = y[-max(1, opt.mean_last_n):] if opt.mean_last_n > 0 else np.array([])
        combined = np.concatenate([first, last]) if last.size else first
        return float(np.mean(combined))
    return opt.baseline_y


def outline(x: np.ndarray, y: np.ndarray, opt: AreaUnderCurveOptions):
    """What the settings make of one trace (``x`` rising): the baseline, the
    height a bump must reach above it to count, the positions of the bumps
    that do, the trace measured from the baseline, and that same trace as it
    is searched for bumps."""
    from scipy.signal import find_peaks

    baseline = _baseline_value(y, opt)
    signal = y - baseline
    working = np.abs(signal) if opt.allow_below_baseline else np.clip(signal, 0, None)
    span = float(np.max(working)) if working.size else 0.0
    height_thresh = span * opt.min_height_percent / 100.0 if opt.min_height_mode == "percent" else opt.min_height_units
    width_thresh = opt.min_width_points if opt.min_width_enabled else 1
    peaks, _props = find_peaks(working, height=height_thresh, width=width_thresh)
    return baseline, height_thresh, peaks, signal, working


@register("area_under_curve")
def run(data: SheetData, options: AreaUnderCurveOptions | None = None) -> AnalysisResult:
    opt = options or AreaUnderCurveOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    sig = opt.significant_digits

    tables = []
    summary_rows = []
    for label in labels:
        x, y = data.xy_points(label)      # each Y with the X of its own row, blanks dropped after pairing
        if x.size < 2:
            summary_rows.append([label, "0", "-"])
            continue

        order = np.argsort(x)
        x, y = x[order], y[order]
        baseline, _height, peaks, signal, working = outline(x, y, opt)

        total_area = float(_trapezoid(np.clip(signal, 0, None) if not opt.allow_below_baseline else signal, x))
        peak_rows = []
        for p in peaks:
            # Walk outward to the nearest valley on each side - NOT
            # scipy's topographic "prominence" base, which extends all
            # the way to a signal edge whenever a neighboring peak is
            # shorter (it never finds a point *higher* than this peak to
            # stop at), silently attributing a neighbor's area to this one.
            left = int(p)
            while left > 0 and working[left - 1] < working[left]:
                left -= 1
            right = int(p)
            while right < working.size - 1 and working[right + 1] < working[right]:
                right += 1
            peak_area = float(_trapezoid(signal[left : right + 1], x[left : right + 1]))
            peak_rows.append([f"{x[p]:.{sig}g}", f"{y[p]:.{sig}g}", f"{peak_area:.{sig}g}"])

        tables.append(ResultTable(
            title=f"'{label}' - baseline Y = {baseline:.{sig}g}, {len(peaks)} peak(s) detected",
            headers=["Peak X", "Peak Y", "Peak area"], rows=peak_rows or [["-", "-", "no peaks met the thresholds"]],
        ))
        summary_rows.append([label, str(len(peaks)), f"{total_area:.{sig}g}"])

    table_summary = ResultTable(title="Total area by group", headers=["Group", "Peaks", "Total AUC"], rows=summary_rows)
    tables.insert(0, table_summary)

    biggest = max((r for r in summary_rows if r[2] != "-"), key=lambda r: float(r[2]), default=None)
    summary = f"{len(labels)} group(s); largest total AUC is {biggest[0]} ({biggest[2]})." if biggest else "No group had enough data to integrate."

    return AnalysisResult(title="Area Under the Curve", summary=summary, tables=tables)
