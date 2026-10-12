"""Take a baseline away from each value, or combine the two another way. The
baseline is another data set (the first one, or each set's right-hand
neighbour) or rows of the same data set (its first, its last, or the mean of
both ends)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register

# each value (v) against its baseline (b); the label is what the settings list shows
CALCULATIONS = {
    "Subtract it:  v − b": lambda v, b: v - b,
    "Divide by it:  v ÷ b": lambda v, b: v / b,
    "As a % of it:  100 × v ÷ b": lambda v, b: 100 * v / b,
    "Change from it, as a %:  100 × (v − b) ÷ b": lambda v, b: 100 * (v - b) / b,
    "Change from it, as a fraction:  (v − b) ÷ b": lambda v, b: (v - b) / b,
    "Add it:  v + b": lambda v, b: v + b,
    "Multiply by it:  v × b": lambda v, b: v * b,
}


@dataclass
class BaselineOptions:
    baseline_source: str = "columns"  # "columns" | "rows"
    # "columns" mode: which data sets serve as everyone else's baseline
    column_mode: str = "every_other"  # "every_other" | "first_column"
    assume_linear_with_x: bool = False
    # "rows" mode: which rows, within each column, form its own baseline
    row_mode: str = "first"  # "first" | "last" | "mean_ends"
    mean_first_n: int = 3
    mean_last_n: int = 3
    remove_baseline_from_results: bool = False
    calculation: str = next(iter(CALCULATIONS))
    subcolumn_mode: str = "ignore"  # "repeated_measures" | "replicates" | "ignore"
    create_new_sheet: bool = True


def _row_baseline(values: np.ndarray, opt: BaselineOptions) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    if opt.row_mode == "last":
        return float(valid[-1])
    if opt.row_mode == "mean_ends":
        first_n = valid[: max(1, opt.mean_first_n)]
        last_n = valid[-max(1, opt.mean_last_n):]
        return float(np.mean(np.concatenate([first_n, last_n])))
    return float(valid[0])


@register("baseline_correction")
def run(data: SheetData, options: BaselineOptions | None = None) -> AnalysisResult:
    opt = options or BaselineOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    calc = CALCULATIONS.get(opt.calculation)
    if calc is None:
        raise AnalysisError(f"Unknown calculation '{opt.calculation}'.")

    n_rows = data.n_rows
    x_values = [float(v) for v in np.pad(data.x[:n_rows], (0, max(0, n_rows - data.x[:n_rows].size)), constant_values=np.nan)]

    if opt.baseline_source == "columns":
        if opt.column_mode == "every_other" and len(labels) < 2:
            raise AnalysisError("Pairs need at least 2 data sets: each one followed by its baseline.")
        if opt.column_mode == "first_column" and len(labels) < 2:
            raise AnalysisError("Using the first column as baseline needs at least 2 groups.")

    new_columns: dict[str, list[float]] = {}
    baseline_rows = []

    def _group_row_series(label: str) -> np.ndarray:
        values = data.group_rows(label, "average")       # row by row: a blank stays a blank on its own row
        return np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]

    if opt.baseline_source == "columns":
        if opt.column_mode == "every_other":
            pairs = [(labels[i], labels[i + 1]) for i in range(0, len(labels) - 1, 2)]
        else:  # first_column
            pairs = [(label, labels[0]) for label in labels[1:]]

        for value_label, baseline_label in pairs:
            value_series = _group_row_series(value_label)
            baseline_series = _group_row_series(baseline_label)
            if opt.assume_linear_with_x:
                mask = ~(np.isnan(x_values) | np.isnan(baseline_series))
                if np.sum(mask) >= 2:
                    x_arr = np.asarray(x_values)
                    slope, intercept = np.polyfit(x_arr[mask], baseline_series[mask], 1)
                    baseline_series = slope * x_arr + intercept
            with np.errstate(divide="ignore", invalid="ignore"):
                result = calc(value_series, baseline_series)
            new_columns[f"{value_label} - {baseline_label}"] = [float(v) for v in result]
            baseline_rows.append([value_label, baseline_label, "linear fit" if opt.assume_linear_with_x else "as measured"])
        if not opt.remove_baseline_from_results:
            for _v, b in pairs:
                if b not in new_columns:
                    new_columns[b] = [float(v) for v in _group_row_series(b)]
    elif opt.subcolumn_mode == "ignore":
        for label in labels:
            values = _group_row_series(label)
            baseline_val = _row_baseline(values, opt)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = calc(values, baseline_val)
            new_columns[label] = [float(v) for v in result]
            baseline_rows.append([label, f"{baseline_val:.4g}", opt.row_mode])
    else:
        # "repeated_measures" / "replicates": each replicate subcolumn
        # gets its own baseline from its own rows, kept as its own column.
        for label in labels:
            for rep, raw_values in sorted(data.group_replicate_arrays(label).items()):
                _reps, matrix = data.group_replicate_matrix(label)
                col_idx = _reps.index(rep) if rep in _reps else 0
                padded = np.pad(matrix[:, col_idx], (0, max(0, n_rows - matrix.shape[0])), constant_values=np.nan)[:n_rows]
                baseline_val = _row_baseline(padded, opt)
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = calc(padded, baseline_val)
                col_label = f"{label} : Y{rep}"
                new_columns[col_label] = [float(v) for v in result]
                baseline_rows.append([col_label, f"{baseline_val:.4g}", opt.row_mode])

    table = ResultTable(
        title="Baseline used",
        headers=["Group", "Baseline", "Detail"] if opt.baseline_source == "rows" else ["Value column", "Baseline column", "Detail"],
        rows=baseline_rows,
    )
    summary = f"Applied '{opt.calculation.split(':')[0].lower()}' using {'other columns' if opt.baseline_source == 'columns' else 'row-based'} baselines across {len(new_columns)} series."

    result_kwargs = dict(title="Baseline Correction", summary=summary, tables=[table])
    if opt.create_new_sheet:
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Baseline Corrected", x_title="X", x_values=x_values, columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
