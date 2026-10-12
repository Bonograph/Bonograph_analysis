"""Turn the table on its side: transpose into a grouped-data table (each original column
becomes a row; X values come from row position), the natural transpose
shape for this app's group-column sheet layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class TransposeOptions:
    row_titles: str = "column_titles"  # "column_titles" | "letters" | "none"
    dataset_titles: str = "row_numbers"  # "x_values" | "row_numbers"
    create_new_sheet: bool = True


@register("transpose_xy")
def run(data: SheetData, options: TransposeOptions | None = None) -> AnalysisResult:
    opt = options or TransposeOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    n_rows = data.n_rows
    x = np.pad(data.x[:n_rows], (0, max(0, n_rows - data.x[:n_rows].size)), constant_values=np.nan)

    matrices = {}
    for label in labels:
        values = data.group_rows(label, "average")       # row by row: a blank stays a blank on its own row
        matrices[label] = np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]

    # Original columns (groups) become the new rows; original rows become
    # the new columns (data sets).
    if opt.row_titles == "letters":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        new_x_labels = [letters[i % 26] for i in range(len(labels))]
    elif opt.row_titles == "none":
        new_x_labels = [str(i + 1) for i in range(len(labels))]
    else:
        new_x_labels = list(labels)

    if opt.dataset_titles == "x_values":
        new_col_titles = [f"{v:.4g}" if not np.isnan(v) else f"Row {i + 1}" for i, v in enumerate(x)]
    elif opt.dataset_titles == "row_numbers":
        new_col_titles = [f"Row {i + 1}" for i in range(n_rows)]
    else:
        new_col_titles = [f"Row {i + 1}" for i in range(n_rows)]  # our sheet has no separate "row titles" to reuse

    new_columns: dict[str, list[float]] = {}
    for row_idx, col_title in enumerate(new_col_titles):
        new_columns[col_title] = [float(matrices[label][row_idx]) for label in labels]

    preview_headers = ["New row (old group)"] + new_col_titles[:8]
    preview_rows = []
    for i, label in enumerate(new_x_labels):
        row = [label]
        for col_title in new_col_titles[:8]:
            row.append(f"{new_columns[col_title][i]:.4g}")
        preview_rows.append(row)
    table = ResultTable(title="Transposed table preview (first 8 new data sets)", headers=preview_headers, rows=preview_rows)

    summary = f"Transposed {len(labels)} original group(s) into rows, {n_rows} original row(s) into new data sets."

    result_kwargs = dict(title="Turn the table on its side", summary=summary, tables=[table])
    if opt.create_new_sheet:
        new_x_values = [float(i) for i in range(len(new_x_labels))]
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Transposed Data", x_title="Group", x_values=new_x_values, columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
