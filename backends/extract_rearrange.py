"""Rebuild the table in another shape. Picks an independent (X) variable, a response (Y)
variable, and optional grouping/subject/row-title variables from the
sheet's full variable list (X plus every group), then produces a
genuinely reshaped new sheet - not just a preview."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class ExtractRearrangeOptions:
    # the kind of table to build
    table_type: str = "xy"  # "xy" | "column" | "grouped" | "contingency"
    include_variable_prefix: bool = False
    only_complete_rows: bool = False
    # which variable goes where
    x_variable: str = ""
    y_variable: str = ""
    grouping_variable: str = ""  # "" = none
    subject_variable: str = ""  # "" = none
    row_title_variable: str = ""  # "" = none


def _variable_values(name: str, all_labels: list[str], all_matrix: np.ndarray, data: SheetData) -> np.ndarray:
    if name in all_labels:
        return all_matrix[:, all_labels.index(name)]
    if name == "X":
        return data.x
    return np.array([])


@register("extract_rearrange")
def run(data: SheetData, options: ExtractRearrangeOptions | None = None) -> AnalysisResult:
    opt = options or ExtractRearrangeOptions()
    all_labels, all_matrix = data.full_variable_matrix()
    if not opt.x_variable or not opt.y_variable:
        raise AnalysisError("Choose the variable that holds the values, and the one to set them against.")

    x = _variable_values(opt.x_variable, all_labels, all_matrix, data)
    y = _variable_values(opt.y_variable, all_labels, all_matrix, data)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    grouping = _variable_values(opt.grouping_variable, all_labels, all_matrix, data)[:n] if opt.grouping_variable else None
    subject = _variable_values(opt.subject_variable, all_labels, all_matrix, data)[:n] if opt.subject_variable else None
    row_var = _variable_values(opt.row_title_variable, all_labels, all_matrix, data)[:n] if opt.row_title_variable else None

    mask = ~np.isnan(x) & ~np.isnan(y)
    if opt.only_complete_rows:
        if grouping is not None:
            mask &= ~np.isnan(grouping)
        if subject is not None:
            mask &= ~np.isnan(subject)
    if row_var is not None:
        mask &= ~np.isnan(row_var)  # a value with no row to go in can't be placed
    x, y = x[mask], y[mask]
    row_var = row_var[mask] if row_var is not None else None
    grouping = grouping[mask] if grouping is not None else None
    subject = subject[mask] if subject is not None else None

    if x.size == 0:
        raise AnalysisError("No complete rows remain for the chosen variables.")

    prefix = f"{opt.grouping_variable}: " if opt.include_variable_prefix and opt.grouping_variable else ""

    if opt.table_type == "contingency":
        if grouping is None or subject is None:
            raise AnalysisError("A contingency table needs both a grouping variable and a Subject/replicate variable (used as the two categorical axes).")
        row_levels = sorted(set(grouping.tolist()))
        col_levels = sorted(set(subject.tolist()))
        rows = []
        for rl in row_levels:
            row = [f"{rl:g}"]
            for cl in col_levels:
                row.append(str(int(np.sum((grouping == rl) & (subject == cl)))))
            rows.append(row)
        table = ResultTable(title="Cross-tab (counts)", headers=[opt.grouping_variable] + [f"{opt.subject_variable} = {cl:g}" for cl in col_levels], rows=rows)
        summary = f"{len(row_levels)}x{len(col_levels)} contingency table of '{opt.grouping_variable}' vs '{opt.subject_variable}'."
        return AnalysisResult(title="Rebuild the table", summary=summary, tables=[table])

    # XY / Column / Grouped: pivot Y (and its paired X, for XY) into one column per grouping level,
    # further split into subcolumns per subject/replicate level if chosen.
    if grouping is not None:
        group_levels = sorted(set(grouping.tolist()))
    else:
        group_levels = [None]
    if subject is not None:
        subject_levels = sorted(set(subject.tolist()))
    else:
        subject_levels = [None]

    x_values: list[float] = []
    columns: dict[str, list[float]] = {}
    max_rows = 0
    pivot_notes = []
    if row_var is not None and opt.table_type in ("column", "grouped"):
        # a real pivot: one row per level of the row variable, so each value
        # lands in its own row/column/subcolumn cell
        row_levels = sorted(set(row_var.tolist()))
        averaged = 0
        for glevel in group_levels:
            g_mask = np.ones(x.size, dtype=bool) if glevel is None else grouping == glevel
            base_title = f"{prefix}{glevel:g}" if glevel is not None else "Column A"
            for slevel in subject_levels:
                s_mask = g_mask if slevel is None else g_mask & (subject == slevel)
                col = []
                for rl in row_levels:
                    cell = y[s_mask & (row_var == rl)]
                    averaged += int(cell.size > 1)
                    col.append(float(np.mean(cell)) if cell.size else float("nan"))
                columns[base_title if slevel is None else f"{base_title} (subcolumn {slevel:g})"] = col
        x_values = [float(v) for v in row_levels]
        max_rows = len(row_levels)
        if averaged:
            pivot_notes.append(f"{averaged} cell(s) received more than one value and hold their mean - choose a "
                               "Subject/replicate variable to keep them apart.")
    for gi, glevel in enumerate(group_levels if not columns else []):
        g_mask = np.ones(x.size, dtype=bool) if glevel is None else grouping == glevel
        base_title = f"{prefix}{glevel:g}" if glevel is not None else "Column A"
        for slevel in subject_levels:
            s_mask = g_mask if slevel is None else g_mask & (subject == slevel)
            col_title = base_title if slevel is None else f"{base_title} (subcolumn {slevel:g})"
            col_values = y[s_mask].tolist()
            columns[col_title] = col_values
            max_rows = max(max_rows, len(col_values))
        if opt.table_type == "xy" and gi == 0:
            x_values = x[g_mask].tolist()

    if (opt.table_type != "xy" and not pivot_notes and row_var is None) or not x_values:
        x_values = list(range(1, max_rows + 1))

    new_sheet = NewSheetData(
        suggested_name=f"Extracted ({opt.y_variable} by {opt.grouping_variable or 'none'})",
        x_title=opt.x_variable if opt.table_type == "xy" else (opt.row_title_variable or "Row"),
        x_values=[float(v) for v in x_values][:max_rows] if len(x_values) >= max_rows else [float(v) for v in x_values] + [float("nan")] * (max_rows - len(x_values)),
        columns={k: v for k, v in columns.items()},
    )

    type_names = {"xy": "Bivariate", "column": "Column", "grouped": "Grouped"}
    summary = f"Extracted '{opt.y_variable}'" + (f" grouped by '{opt.grouping_variable}'" if opt.grouping_variable else "") + f" into a {type_names.get(opt.table_type, opt.table_type)}-format table ({len(columns)} column(s), up to {max_rows} rows)."
    notes = list(pivot_notes)
    if not opt.grouping_variable:
        notes.append("No grouping variable was chosen, so all Y values went into a single data set (Column A).")

    preview_rows = [[f"{v:.4g}" for v in [x_values[i] if i < len(x_values) else float('nan')]] + [f"{columns[c][i]:.4g}" if i < len(columns[c]) else "" for c in columns] for i in range(min(max_rows, 50))]
    preview_table = ResultTable(title="Preview of the extracted table", headers=[new_sheet.x_title] + list(columns.keys()), rows=preview_rows)

    return AnalysisResult(title="Rebuild the table", summary=summary, tables=[preview_table], notes=notes, new_sheet=new_sheet)
