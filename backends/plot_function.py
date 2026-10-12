"""Draw an equation: pick an equation from the library, set its
parameter values (one set per curve, or a family of curves), and
tabulate Y = f(X) over a range - this doesn't fit data, it evaluates a
known formula, so it produces a new sheet directly rather than reading
one."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register
from .function_library import EQUATIONS


@dataclass
class PlotFunctionOptions:
    equation_key: str = "straight_line"
    plot_family: bool = False
    family_count: int = 2
    x_min: float = 0.0
    x_max: float = 100.0
    n_points: int = 100
    significant_digits: int = 4
    parameter_sets: list[dict[str, float]] = field(default_factory=list)  # one dict per curve
    column_titles: list[str] = field(default_factory=list)  # aligned with parameter_sets
    # what to report
    plot_function: bool = True
    plot_first_derivative: bool = False
    plot_second_derivative: bool = False
    plot_integral: bool = False
    integral_x0: float = 0.0
    integral_y0: float = 0.0
    column_label_mode: str = "manual"  # "manual" | "parameter_value"
    column_label_parameter: str = ""


@register("plot_function")
def run(data: SheetData, options: PlotFunctionOptions | None = None) -> AnalysisResult:
    opt = options or PlotFunctionOptions()
    spec = EQUATIONS.get(opt.equation_key)
    if spec is None:
        raise AnalysisError(f"'{opt.equation_key}' isn't implemented in this build's function library yet.")
    if opt.x_max <= opt.x_min:
        raise AnalysisError("The X range's maximum must be greater than its minimum.")
    if opt.n_points < 2:
        raise AnalysisError("Need at least 2 points to plot.")

    if not any((opt.plot_function, opt.plot_first_derivative, opt.plot_second_derivative, opt.plot_integral)):
        raise AnalysisError("Nothing to draw yet: choose the curve itself, its slope, its curvature or its running area.")

    n_curves = opt.family_count if opt.plot_family else 1
    x = np.linspace(opt.x_min, opt.x_max, opt.n_points)
    sig = opt.significant_digits

    columns: dict[str, list[float]] = {}
    notes = [f"Equation: {spec.equation}"]
    for i in range(n_curves):
        overrides = opt.parameter_sets[i] if i < len(opt.parameter_sets) else {}
        values = [overrides.get(name, default) for name, default in spec.params]
        try:
            with np.errstate(all="ignore"):
                y = spec.fn(x, *values)
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Couldn't evaluate '{spec.label}': {exc}") from None
        y = np.asarray(y, dtype=float)
        if y.shape != x.shape:
            y = np.broadcast_to(y, x.shape)

        if opt.column_label_mode == "parameter_value" and opt.column_label_parameter:
            pval = overrides.get(opt.column_label_parameter)
            base_title = f"{opt.column_label_parameter} = {pval:g}" if pval is not None else spec.label
        else:
            base_title = opt.column_titles[i] if i < len(opt.column_titles) and opt.column_titles[i] else (spec.label if n_curves == 1 else f"{spec.label} (curve {i+1})")

        if opt.plot_function:
            columns[base_title] = y.tolist()
        if opt.plot_first_derivative or opt.plot_second_derivative:
            dy = np.gradient(y, x)
            if opt.plot_first_derivative:
                columns[f"{base_title} (1st derivative)"] = dy.tolist()
            if opt.plot_second_derivative:
                d2y = np.gradient(dy, x)
                columns[f"{base_title} (2nd derivative)"] = d2y.tolist()
        if opt.plot_integral:
            cumulative = np.concatenate([[0.0], np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(x))])
            # Shift so the integral curve passes through (integral_x0, integral_y0), the constant of integration.
            idx0 = int(np.argmin(np.abs(x - opt.integral_x0)))
            cumulative = cumulative - cumulative[idx0] + opt.integral_y0
            columns[f"{base_title} (integral)"] = cumulative.tolist()

    preview_rows = [[f"{xv:.{sig}g}"] + [f"{columns[c][i]:.{sig}g}" if columns[c][i] == columns[c][i] else "-" for c in columns] for i, xv in enumerate(x[:200])]
    table = ResultTable(title=f"{spec.label}: {spec.equation}", headers=[spec.x_label] + list(columns.keys()), rows=preview_rows)

    new_sheet = NewSheetData(suggested_name=spec.label, x_title=spec.x_label, x_values=[float(v) for v in x], columns=columns)
    summary = f"{spec.label} tabulated at {opt.n_points} point(s) from {spec.x_label} = {opt.x_min:g} to {opt.x_max:g}" + (f" ({n_curves} curves)." if n_curves > 1 else ".")

    return AnalysisResult(title="Draw an equation", summary=summary, tables=[table], notes=notes, new_sheet=new_sheet)
