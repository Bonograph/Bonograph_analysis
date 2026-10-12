"""Take a drifting baseline out from under the peaks of a trace.

X is the position; each data set is a trace. The baseline is estimated from
the trace itself - the slowly varying part that lies under the peaks - and
taken away; the flattened traces go on a new sheet. The sheet analysed is
left as it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _signal

_trapezoid = getattr(np, "trapezoid", None) or np.trapz
METHODS = {"smooth": "a smooth curve that settles under the peaks", "polynomial": "a polynomial fitted under the peaks",
           "ends": "a straight line between the two ends"}


@dataclass
class BaselineOptions:
    method: str = "smooth"        # "smooth" (asymmetric least squares) | "polynomial" | "ends"
    direction: str = "up"         # peaks point "up" or "down" from the baseline
    stiffness: float = 5.0        # "smooth": log10 of the smoothness penalty - 2 follows every bend, 9 is nearly a straight line
    degree: int = 3               # "polynomial"
    end_points: int = 5           # "ends": points averaged at each end
    create_new_sheet: bool = True
    significant_digits: int = 4


def baseline_of(x: np.ndarray, y: np.ndarray, opt: BaselineOptions) -> np.ndarray:
    sign = -1.0 if opt.direction == "down" else 1.0
    if opt.method == "polynomial":
        return sign * _signal.polynomial_baseline(x, sign * y, int(opt.degree))
    if opt.method == "ends":
        return _signal.ends_baseline(x, y, int(opt.end_points))
    # the penalty is on differences between neighbouring points, so for the same stiffness in X a
    # finely sampled trace needs a larger one: it is scaled to a trace of 1000 points
    scale = (x.size / 1000.0) ** 4
    return sign * _signal.als_baseline(sign * y, smoothness=10.0 ** float(np.clip(opt.stiffness, 0, 12)) * scale)


@register("baseline_spectrum")
def run(data: SheetData, options: BaselineOptions | None = None) -> AnalysisResult:
    opt = options or BaselineOptions()
    if opt.method not in METHODS:
        opt.method = "smooth"
    sets = _signal.traces(data, least=5)
    if not sets:
        raise AnalysisError("This needs a trace: numbers in the X column and at least 5 rows of values in a data set.")
    sig = opt.significant_digits

    def num(value) -> str:
        return f"{value:.{sig}g}"

    rows, charts, flattened = [], [], {}
    for label, x, y in sets:
        base = baseline_of(x, y, opt)
        flat = y - base
        flattened[label] = (x, flat)
        wrong_side = float(np.mean(flat < 0 if opt.direction != "down" else flat > 0))
        rows.append([label, str(x.size), f"{num(base.min())} to {num(base.max())}", num(float(_trapezoid(np.abs(base), x))),
                     num(float(_trapezoid(flat, x))), num(float(flat.max() if opt.direction != "down" else flat.min())),
                     f"{wrong_side:.0%}"])
        gx, gy, gb, gf = _signal.for_graph(x, y, base, flat)
        charts.append(ChartSpec(f"{label}: the trace and the baseline taken from it", x_label="X", y_label="Y", group="baseline", option=label,
                                series=[ChartSeries("Trace", gx, gy, kind="line", color="#7d8590"),
                                        ChartSeries("Baseline", gx, gb, kind="line", color="#d1342f"),
                                        ChartSeries("Flattened", gx, gf, kind="line", color="#1f5aa6")],
                                reference_lines=[{"kind": "h", "value": 0.0}]))
    side = "below" if opt.direction != "down" else "above"
    tables = [ResultTable("What was taken away", ["Data set", "Points", "Baseline runs from", "Area under the baseline",
                                                  "Area left in the flattened trace", "Largest value left", f"Share of points left {side} zero"],
                          rows)]
    how = {"smooth": f"a smooth curve (asymmetric least squares, stiffness {opt.stiffness:g})",
           "polynomial": f"a polynomial of degree {int(opt.degree)}, fitted again and again to the lower of the trace and the last fit",
           "ends": f"a straight line between the means of the first and the last {int(opt.end_points)} points"}[opt.method]
    summary = f"The baseline of {len(sets)} trace{'s' if len(sets) != 1 else ''} was estimated as {how} and taken away."
    notes = ["Look at the graph before using the result: the baseline should run under the peaks and through the stretches where "
             "there is no peak. If it climbs into a broad peak, make it stiffer; if it cuts corners where the true baseline bends, "
             "make it less stiff."]
    if opt.method == "smooth":
        notes.append("Asymmetric least squares (Eilers and Boelens 2005): a smooth curve is fitted with a small weight (0.01) on points "
                     "above it and a large one on points below, so it settles to the bottom of the trace. Stiffness is the base-10 "
                     "logarithm of the smoothness penalty, for a trace of 1000 points; each step of 1 makes the curve about 1.8 "
                     "times stiffer in X.")
    elif opt.method == "polynomial":
        notes.append("Iterated polynomial (Lieber and Mahadevan-Jansen 2003): after each fit the trace is replaced by the lower of "
                     "itself and the fit, which cuts the peaks away step by step. A high degree can swing wildly at the ends.")
    else:
        notes.append("The line between the ends assumes the trace starts and ends on its baseline, with no peak in the first or "
                     "last points, and that the drift between them is straight.")
    notes.append(f"A flattened trace sits a little {side} zero where there is only noise: about half of such points should. "
                 f"A share far above 50% over the whole trace means the baseline was set too {'high' if side == 'below' else 'low'}.")
    result = AnalysisResult(title="Flatten the baseline", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.create_new_sheet:
        # one X column for all: where the data sets share their X values each keeps its own row
        grid = np.unique(np.concatenate([x for x, _f in flattened.values()]))
        columns = {}
        for label, (x, flat) in flattened.items():
            column = np.full(grid.size, np.nan)
            column[np.searchsorted(grid, x)] = flat
            columns[label] = column.tolist()
        result.new_sheet = NewSheetData("Baseline removed", "X", grid.tolist(), columns)
        notes.append("The flattened traces are on a new sheet, 'Baseline removed'. The sheet analysed is unchanged.")
    return result
