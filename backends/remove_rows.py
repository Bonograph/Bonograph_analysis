"""Drop rows you do not want:
either exclude rows outside an X range, or sort by X and average every
K rows into one output row (optionally only above a threshold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class RemoveRowsOptions:
    mode: str = "exclude_range"  # "exclude_range" | "average_k"
    x_min: float = 0.0
    x_max: float = 1.0
    k_rows: int = 2
    only_above_threshold: bool = False
    threshold: float = 0.0
    create_new_sheet: bool = True


@register("remove_rows")
def run(data: SheetData, options: RemoveRowsOptions | None = None) -> AnalysisResult:
    opt = options or RemoveRowsOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    n_rows = data.n_rows
    x = np.pad(data.x[:n_rows], (0, max(0, n_rows - data.x[:n_rows].size)), constant_values=np.nan)
    matrices = {}
    for label in labels:
        _reps, matrix = data.group_replicate_matrix(label)
        raw = matrix[:, 0] if matrix.size else np.full(n_rows, np.nan)
        matrices[label] = np.pad(raw, (0, max(0, n_rows - raw.size)), constant_values=np.nan)[:n_rows]

    if opt.mode == "average_k":
        if opt.k_rows < 2:
            raise AnalysisError("K must be at least 2 to average rows together.")
        order = np.argsort(x)
        x_sorted = x[order]
        sorted_cols = {label: values[order] for label, values in matrices.items()}

        if opt.only_above_threshold:
            keep_intact = x_sorted <= opt.threshold
            to_average_mask = ~keep_intact
        else:
            keep_intact = np.zeros(n_rows, dtype=bool)
            to_average_mask = np.ones(n_rows, dtype=bool)

        new_x: list[float] = list(x_sorted[keep_intact])
        new_columns: dict[str, list[float]] = {label: list(values[keep_intact]) for label, values in sorted_cols.items()}

        idxs = np.where(to_average_mask)[0]
        for start in range(0, len(idxs), opt.k_rows):
            chunk = idxs[start : start + opt.k_rows]
            if chunk.size == 0:
                continue
            new_x.append(float(np.nanmean(x_sorted[chunk])))
            for label, values in sorted_cols.items():
                new_columns[label].append(float(np.nanmean(values[chunk])))

        reason = f"averaged every {opt.k_rows} rows"
        if opt.only_above_threshold:
            reason += f" where X > {opt.threshold:g} (rows at/below kept intact)"
        n_kept = len(new_x)
    else:
        if opt.x_max <= opt.x_min:
            raise AnalysisError("X max must be greater than X min.")
        keep = (x >= opt.x_min) & (x <= opt.x_max)
        n_kept = int(np.sum(keep))
        if n_kept == 0:
            raise AnalysisError("This filter would remove every row - nothing left to write to a new sheet.")
        new_x = [float(v) for v in x[keep]]
        new_columns = {label: [float(v) for v in values[keep]] for label, values in matrices.items()}
        reason = f"kept rows with {opt.x_min:g} ≤ X ≤ {opt.x_max:g}"

    table = ResultTable(
        title="Row filter result",
        headers=["Rows before", "Rows after"],
        rows=[[str(n_rows), str(n_kept)]],
    )
    summary = f"{reason}: {n_kept} row(s) remain (from {n_rows})."

    result_kwargs = dict(title="Drop rows you don't want", summary=summary, tables=[table])
    if opt.create_new_sheet:
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Filtered Data", x_title="X", x_values=new_x, columns=new_columns
        )
    return AnalysisResult(**result_kwargs)
