"""Doses onto a log scale and back: special handling for X=0, a unit
change (multiply/divide by a constant), and an optional log transform.
Y values are left unchanged; results go to a new sheet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register

LOG_BASES = {"log10 (common)": 10.0, "ln (natural)": np.e, "log2": 2.0}


@dataclass
class TransformXOptions:
    replace_zero: bool = False
    zero_replacement: float = 1e-10
    change_units: bool = False
    units_mode: str = "multiply"  # "multiply" | "divide"
    units_constant: float = 10.0
    log_transform: bool = True  # the point of the analysis: doses onto a log scale
    log_base: str = "log10 (common)"
    create_new_sheet: bool = True
    zero_blank: bool = False  # a dose of 0 that is not replaced is left blank instead of stopping the run


@register("transform_x_concentration")
def run(data: SheetData, options: TransformXOptions | None = None) -> AnalysisResult:
    opt = options or TransformXOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    if not (opt.replace_zero or opt.change_units or opt.log_transform):
        raise AnalysisError("Nothing to do yet - change the units, take a log, or give a dose of 0 a stand-in.")

    n_rows = data.n_rows
    x = np.pad(data.x[:n_rows], (0, max(0, n_rows - data.x[:n_rows].size)), constant_values=np.nan).astype(float)
    steps = []

    if opt.replace_zero:
        x = np.where(x == 0, opt.zero_replacement, x)
        steps.append(f"X=0 replaced with {opt.zero_replacement:g}")
    elif opt.zero_blank and opt.log_transform and np.any(x == 0):
        steps.append(f"{int(np.sum(x == 0))} dose(s) of 0 left blank")
        x = np.where(x == 0, np.nan, x)

    if opt.change_units:
        if opt.units_constant == 0:
            raise AnalysisError("The unit-change constant can't be zero.")
        x = x * opt.units_constant if opt.units_mode == "multiply" else x / opt.units_constant
        steps.append(f"units {'x' if opt.units_mode == 'multiply' else '/'} {opt.units_constant:g}")

    if opt.log_transform:
        base = LOG_BASES.get(opt.log_base, 10.0)
        if np.any(x[~np.isnan(x)] <= 0):
            raise AnalysisError("A logarithm needs every dose to be above 0. Give a dose of 0 a stand-in, "
                                "or have it left blank; a negative dose has to be corrected on the sheet.")
        x = np.log(x) / np.log(base)
        steps.append(opt.log_base)

    new_columns: dict[str, list[float]] = {}
    for label in labels:
        reps, matrix = data.group_replicate_matrix(label)
        if not matrix.size:
            new_columns[label] = [float("nan")] * n_rows
            continue
        for j, rep in sorted(enumerate(reps), key=lambda pair: pair[1]):   # every repeat column comes along
            raw = np.pad(matrix[:, j], (0, max(0, n_rows - matrix.shape[0])), constant_values=np.nan)[:n_rows]
            new_columns[label if len(reps) == 1 else f"{label} : Y{rep}"] = [float(v) for v in raw]

    old_x = data.x[:n_rows]
    table = ResultTable(
        title=f"X transformed: {'; '.join(steps)}",
        headers=["Row", "Old X", "New X"],
        rows=[
            [str(i + 1), f"{ox:.4g}" if not np.isnan(ox) else "-", f"{nx:.4g}" if not np.isnan(nx) else "-"]
            for i, (ox, nx) in enumerate(zip(old_x, x))
        ],
    )
    summary = f"{'; '.join(steps)} across {n_rows} row(s); Y values unchanged."

    result_kwargs = dict(title="Doses to a log scale", summary=summary, tables=[table])
    if opt.create_new_sheet:
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Concentration Scale", x_title="X", x_values=[float(v) for v in x], columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
