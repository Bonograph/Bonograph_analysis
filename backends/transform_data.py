"""Change the values on a sheet and write the result to a new one, leaving the
source data untouched.

X and Y each have a recipe (see ``analysis/recipes.py``): a chain of steps
such as "log, then times 100", applied to every value in turn. A recipe with
no step leaves that axis as it is. A step may be an equation that uses the
row's own X and Y, so "X becomes y, Y becomes x" turns the axes round.

Settings saved by earlier versions named one formula and one constant per
axis; they still run (``y_steps`` and ``x_steps`` are then None).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register
from .. import recipes
from ..safe_eval import evaluate

# Preset "standard function" formulas - K is filled in from k_value.
STANDARD_FORMULAS: dict[str, str] = {
    "Y = K*Y": "K*y",
    "Y = Y + K": "y + K",
    "Y = Y / K": "y / K",
    "Y = Y^K": "y ** K",
    "Y = log10(Y)": "log10(y)",
    "Y = ln(Y)": "log(y)",
    "Y = sqrt(Y)": "sqrt(y)",
    "Y = 1/Y": "1 / y",
    "Y = 10^Y": "10 ** y",
    "Y = e^Y": "exp(y)",
}
X_STANDARD_FORMULAS: dict[str, str] = {
    "X = K*X": "K*x",
    "X = X + K": "x + K",
    "X = log10(X)": "log10(x)",
    "X = ln(X)": "log(x)",
}


@dataclass
class TransformDataOptions:
    interchange_xy: bool = False
    transform_x: bool = False
    x_formula: str = "X = K*X"
    x_k: float = 1.0
    transform_y: bool = True
    y_formula: str = "Y = log10(Y)"
    function_source: str = "standard"  # "standard" | "custom"
    custom_formula: str = "y"
    y_k: float = 1.0
    replicate_handling: str = "individual"  # "individual" | "average"
    create_new_sheet: bool = True
    # the recipes; None means "settings from an earlier version: use the fields above"
    y_steps: list | None = None
    x_steps: list | None = None


def _run_recipes(data: SheetData, opt: TransformDataOptions) -> AnalysisResult:
    labels = data.group_labels()
    y_steps, x_steps = list(opt.y_steps or []), list(opt.x_steps or [])
    if not y_steps and not x_steps:
        raise AnalysisError("Nothing to do yet - add a step to the Y values or to the X values.")
    for axis, steps in (("Y", y_steps), ("X", x_steps)):
        wrong = recipes.check(steps)
        if wrong:
            raise AnalysisError(f"{axis} values: {wrong}")

    n_rows = data.n_rows
    x_old = np.pad(np.asarray(data.x[:n_rows], dtype=float), (0, max(0, n_rows - len(data.x[:n_rows]))),
                   constant_values=np.nan)

    def column(values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]

    # an X step may use y: that is the first data set's value on the same row
    first = column(data.group_rows(labels[0], "average"))
    x_new = recipes.apply_many(x_steps, x_old, x=x_old, y=first).tolist()

    series: list[tuple[str, np.ndarray]] = []
    for label in labels:
        reps, matrix = data.group_replicate_matrix(label)
        if opt.replicate_handling == "average" or matrix.ndim < 2 or matrix.shape[1] <= 1:
            series.append((label, column(data.group_rows(label, "average"))))
        else:   # every repeat column, each on its own rows
            for j, rep in sorted(enumerate(reps), key=lambda pair: pair[1]):
                series.append((f"{label} : Y{rep}", column(matrix[:, j])))

    new_columns: dict[str, list[float]] = {}
    rows, lost = [], 0
    for name, values in series:
        changed = recipes.apply_many(y_steps, values, x=x_old, y=values).tolist()
        had = int(np.sum(~np.isnan(values)))
        kept = int(np.sum(~np.isnan(np.asarray(changed, dtype=float))))
        lost += had - kept
        new_columns[name] = changed
        rows.append([name, str(had), str(kept), str(had - kept)])
    x_had = int(np.sum(~np.isnan(x_old)))
    x_lost = x_had - int(np.sum(~np.isnan(np.asarray(x_new, dtype=float))))
    if y_steps and all(row[2] == "0" for row in rows):
        raise AnalysisError(f"The Y recipe ({recipes.describe(y_steps)}) has no answer for any value on this sheet.")
    if x_steps and x_had and x_lost == x_had:
        raise AnalysisError(f"The X recipe ({recipes.describe(x_steps)}) has no answer for any X value on this sheet.")

    table = ResultTable(title="What was changed", headers=["Data set", "Values", "Changed", "Left blank"], rows=rows)
    summary = f"Y: {recipes.describe(y_steps)}.  X: {recipes.describe(x_steps)}.  {len(series)} data set(s)."
    notes = []
    if lost or x_lost:
        notes.append(f"{lost + x_lost} value(s) have no answer under this recipe (the log of zero or less, a division "
                     "by zero, the root of a negative number) and are left blank.")
    kwargs = dict(title="Transform values", summary=summary, tables=[table], notes=notes)
    if opt.create_new_sheet:
        kwargs["new_sheet"] = NewSheetData(suggested_name="Transformed Data", x_title="X", x_values=x_new,
                                           columns=new_columns)
    return AnalysisResult(**kwargs)


@register("transform_data")
def run(data: SheetData, options: TransformDataOptions | None = None) -> AnalysisResult:
    opt = options or TransformDataOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    if opt.y_steps is not None or opt.x_steps is not None:
        return _run_recipes(data, opt)

    n_rows = data.n_rows
    x_values = [float(v) if not np.isnan(v) else float("nan") for v in data.x[:n_rows]]
    while len(x_values) < n_rows:
        x_values.append(float("nan"))

    if opt.interchange_xy:
        # Swapping X/Y only makes sense against a single series - use the
        # first group: interchange first, then transform.
        first_label = labels[0]
        _reps, matrix = data.group_replicate_matrix(first_label)
        y_raw = matrix[:, 0] if matrix.size else np.full(n_rows, np.nan)
        y_raw = np.pad(y_raw, (0, max(0, n_rows - y_raw.size)), constant_values=np.nan)[:n_rows]
        new_x_values = [float(v) for v in y_raw]
        x_values, new_x_values = new_x_values, x_values
    else:
        new_x_values = None

    if opt.transform_x:
        x_expr = X_STANDARD_FORMULAS.get(opt.x_formula, "x")
        x_values = [
            evaluate(x_expr, x=v, K=opt.x_k) if not np.isnan(v) else v for v in x_values
        ]

    if opt.function_source == "custom":
        y_expr = opt.custom_formula
    else:
        y_expr = STANDARD_FORMULAS.get(opt.y_formula, "y")

    new_columns: dict[str, list[float]] = {}
    preview_rows = []
    for label in labels:
        if opt.replicate_handling == "average":
            values = data.group_rows(label, "average")   # row by row: a blank stays a blank on its own row
            values = np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]
            xs_for_row = x_values
        else:
            _reps, matrix = data.group_replicate_matrix(label)
            values = matrix[:, 0] if matrix.size else np.full(n_rows, np.nan)
            values = np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]
            xs_for_row = x_values

        transformed = []
        n_ok = 0
        for row_idx, y_val in enumerate(values):
            if row_idx >= len(xs_for_row) or np.isnan(y_val):
                transformed.append(float("nan"))
                continue
            x_val = xs_for_row[row_idx]
            try:
                if opt.transform_y:
                    transformed.append(float(evaluate(y_expr, y=float(y_val), x=x_val, K=opt.y_k)))
                else:
                    transformed.append(float(y_val))
                n_ok += 1
            except AnalysisError:
                transformed.append(float("nan"))
        new_columns[label] = transformed
        preview_rows.append([label, str(n_ok)])

    if not opt.transform_y and not opt.transform_x and not opt.interchange_xy:
        raise AnalysisError("Nothing to do - tick 'Change X to' or 'Change Y to', or swap X and Y.")
    if opt.transform_y and all(n == "0" for _l, n in preview_rows):
        raise AnalysisError(f"'{y_expr}' didn't produce a valid number for any value - check the formula/K.")

    table = ResultTable(title="Values transformed per group", headers=["Group", "N transformed"], rows=preview_rows)
    pieces = []
    if opt.interchange_xy:
        pieces.append("X/Y interchanged")
    if opt.transform_x:
        pieces.append(f"X: {opt.x_formula}")
    if opt.transform_y:
        pieces.append(f"Y: {y_expr}" if opt.function_source == "custom" else f"Y: {opt.y_formula}")
    summary = f"{', '.join(pieces) or 'No transform applied'} across {len(labels)} group(s)."

    result_kwargs = dict(title="Transform Data", summary=summary, tables=[table])
    if opt.create_new_sheet:
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Transformed Data", x_title="X", x_values=x_values, columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
