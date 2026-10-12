"""Summarise each row: a chosen central-tendency statistic (mean, median,
geometric mean) plus a companion statistic (SD, SEM, CI, N, min/max),
computed per data-set column for each row."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import NewSheetData, AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

CENTRAL_STATS = {
    "means": lambda v: float(np.mean(v)),
    "medians": lambda v: float(np.median(v)),
    "geometric means": lambda v: float(stats.gmean(v)) if np.all(v > 0) else float("nan"),
}


@dataclass
class RowStatisticsOptions:
    scope: str = "per_column"  # "per_column" | "whole_row"
    central_stat: str = "means"  # key into CENTRAL_STATS
    companion_stat: str = "sd_n"  # "sd_n" | "sem_n" | "ci_n" | "min_max"
    confidence_level: float = 0.95
    create_new_sheet: bool = False


def _companion(values: np.ndarray, opt: RowStatisticsOptions) -> tuple[str, str]:
    n = values.size
    if n == 0:
        return "-", "-"
    if opt.companion_stat == "sem_n":
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return f"{sem:.4g}", str(n)
    if opt.companion_stat == "ci_n":
        if n > 1:
            sem = float(stats.sem(values))
            half = stats.t.ppf(0.5 + opt.confidence_level / 2, n - 1) * sem
            mean = float(np.mean(values))
            return f"{mean - half:.4g} to {mean + half:.4g}", str(n)
        return "n/a", str(n)
    if opt.companion_stat == "min_max":
        return f"{float(np.min(values)):.4g} to {float(np.max(values)):.4g}", str(n)
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return f"{sd:.4g}", str(n)


_COMPANION_LABELS = {"sd_n": "SD", "sem_n": "SEM", "ci_n": "CI", "min_max": "Range"}


@register("row_wise_statistics")
def run(data: SheetData, options: RowStatisticsOptions | None = None) -> AnalysisResult:
    opt = options or RowStatisticsOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    central_fn = CENTRAL_STATS.get(opt.central_stat, CENTRAL_STATS["means"])
    companion_label = _COMPANION_LABELS.get(opt.companion_stat, "SD")

    n_rows = data.n_rows

    if opt.scope == "whole_row":
        _labels, matrix = data.first_replicate_matrix()
        if matrix.size == 0:
            raise AnalysisError("Need at least 1 group column with data.")
        rows = []
        sheet_x, sheet_centre = [], []
        for r in range(matrix.shape[0]):
            row_vals = matrix[r][~np.isnan(matrix[r])]
            if row_vals.size == 0:
                continue
            sheet_x.append(float(data.x[r]) if r < data.x.size and not np.isnan(data.x[r]) else float(r + 1))
            sheet_centre.append(float(central_fn(row_vals)))
            x_label = f"{data.x[r]:.4g}" if r < data.x.size and not np.isnan(data.x[r]) else str(r + 1)
            companion, n = _companion(row_vals, opt)
            rows.append([x_label, f"{central_fn(row_vals):.4g}", companion, n])
        table = ResultTable(
            title=f"{opt.central_stat.capitalize()} across all groups, per row",
            headers=["Row (X)", opt.central_stat.capitalize(), companion_label, "N"],
            rows=rows,
        )
        summary = f"{len(rows)} row(s) summarized across {len(labels)} groups (whole-row scope)."
        new_sheet = None
        if opt.create_new_sheet:
            new_sheet = NewSheetData(suggested_name=f"Row {opt.central_stat}", x_title="X",
                                     x_values=sheet_x, columns={opt.central_stat.capitalize(): sheet_centre})
        return AnalysisResult(title="Summarise each row", summary=summary, tables=[table], new_sheet=new_sheet)

    # per_column (the default): one summary per data set
    headers = ["Row (X)"]
    for label in labels:
        headers += [f"{label}: {opt.central_stat.capitalize()}", f"{label}: {companion_label}", f"{label}: N"]

    rows = []
    sheet_x: list[float] = []
    sheet_cols: dict[str, list[float]] = {label: [] for label in labels}
    for r in range(n_rows):
        row_cells = []
        row_numbers = {}
        x_val = data.x[r] if r < data.x.size else float("nan")
        row_label = f"{x_val:.4g}" if not np.isnan(x_val) else str(r + 1)
        any_data = False
        for label in labels:
            _reps, matrix = data.group_replicate_matrix(label)
            if r < matrix.shape[0]:
                row_vals = matrix[r][~np.isnan(matrix[r])]
            else:
                row_vals = np.array([])
            if row_vals.size:
                any_data = True
                companion, n = _companion(row_vals, opt)
                row_cells += [f"{central_fn(row_vals):.4g}", companion, n]
                row_numbers[label] = float(central_fn(row_vals))
            else:
                row_cells += ["-", "-", "0"]
        if any_data:
            rows.append([row_label] + row_cells)
            sheet_x.append(float(x_val) if not np.isnan(x_val) else float(r + 1))
            for label in labels:
                sheet_cols[label].append(row_numbers.get(label, float("nan")))

    table = ResultTable(title=f"{opt.central_stat.capitalize()}, {companion_label} and N per data set column", headers=headers, rows=rows)
    summary = f"{len(rows)} row(s) x {len(labels)} data set(s); showing {opt.central_stat} with {companion_label}."

    new_sheet = None
    if opt.create_new_sheet:  # one column per data set, holding its row summary - ready to graph
        new_sheet = NewSheetData(suggested_name=f"Row {opt.central_stat}", x_title="X", x_values=sheet_x,
                                 columns={f"{label} ({opt.central_stat})": vals for label, vals in sheet_cols.items()})
    return AnalysisResult(title="Summarise each row", summary=summary, tables=[table], new_sheet=new_sheet)
