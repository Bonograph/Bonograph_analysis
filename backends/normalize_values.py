"""Put columns on a common scale:
how replicate subcolumns are combined, how 0% and 100% are defined, and
whether results are shown as fractions or percentages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class NormalizeOptions:
    subcolumn_mode: str = "average"  # "average" | "separate"
    zero_definition: str = "smallest_mean"  # "smallest_mean" | "first_mean" | "fixed"
    zero_fixed_value: float = 0.0
    hundred_definition: str = "largest_mean"  # "largest_mean" | "last_mean" | "fixed" | "sum" | "mean"
    hundred_fixed_value: float = 100.0
    present_as: str = "percentages"  # "fractions" | "percentages"
    create_new_sheet: bool = True


def _zero_ref(values: np.ndarray, opt: NormalizeOptions) -> float:
    valid = values[~np.isnan(values)]
    if opt.zero_definition == "fixed":
        return opt.zero_fixed_value
    if opt.zero_definition == "first_mean":
        first, last = valid[0], valid[-1]
        return float(min(first, last))
    return float(np.min(valid))


def _hundred_ref(values: np.ndarray, opt: NormalizeOptions) -> float:
    valid = values[~np.isnan(values)]
    if opt.hundred_definition == "fixed":
        return opt.hundred_fixed_value
    if opt.hundred_definition == "last_mean":
        first, last = valid[0], valid[-1]
        return float(max(first, last))
    if opt.hundred_definition == "sum":
        return float(np.sum(valid))
    if opt.hundred_definition == "mean":
        return float(np.mean(valid))
    return float(np.max(valid))


@register("normalize_values")
def run(data: SheetData, options: NormalizeOptions | None = None) -> AnalysisResult:
    opt = options or NormalizeOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    n_rows = data.n_rows
    x_values = [float(v) for v in np.pad(data.x[:n_rows], (0, max(0, n_rows - data.x[:n_rows].size)), constant_values=np.nan)]
    scale = 1.0 if opt.present_as == "fractions" else 100.0

    new_columns: dict[str, list[float]] = {}
    rows = []

    def _process(label: str, values: np.ndarray) -> None:
        if not np.any(~np.isnan(values)):      # nothing in it: carried over blank
            new_columns[label] = [float("nan")] * len(values)
            rows.append([label, "-", "-"])
            return
        zero_ref = _zero_ref(values, opt)
        hundred_ref = _hundred_ref(values, opt)
        span = hundred_ref - zero_ref
        if span == 0:
            raise AnalysisError(f"'{label}': the 0% and 100% references are equal - normalization is undefined.")
        normalized = (values - zero_ref) / span * scale
        new_columns[label] = [float(v) for v in normalized]
        rows.append([label, f"{zero_ref:.4g}", f"{hundred_ref:.4g}"])

    if opt.subcolumn_mode == "separate":
        for label in labels:
            reps, matrix = data.group_replicate_matrix(label)
            for j, rep in sorted(enumerate(reps), key=lambda pair: pair[1]):
                values = matrix[:, j]                 # the subcolumn as it sits on the sheet: a blank stays on its own row
                padded = np.pad(values, (0, max(0, n_rows - values.size)), constant_values=np.nan)[:n_rows]
                _process(f"{label} : Y{rep}", padded)
    else:
        for label in labels:
            _reps, matrix = data.group_replicate_matrix(label)
            with np.errstate(invalid="ignore"):
                row_mean = np.nanmean(matrix, axis=1) if matrix.size else np.full(n_rows, np.nan)
            row_mean = np.pad(row_mean, (0, max(0, n_rows - row_mean.size)), constant_values=np.nan)[:n_rows]
            _process(label, row_mean)

    if all(row[1] == "-" for row in rows):
        raise AnalysisError("There are no values to rescale on this sheet.")
    unit = "%" if opt.present_as == "percentages" else ""
    table = ResultTable(title="Normalization references", headers=["Group", "0% reference", "100% reference"], rows=rows)
    summary = f"Normalized {len(new_columns)} series to {opt.present_as} (0{unit} to 100{unit})."

    result_kwargs = dict(title="Put on a common scale", summary=summary, tables=[table])
    if opt.create_new_sheet:
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Normalized Data", x_title="X", x_values=x_values, columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
