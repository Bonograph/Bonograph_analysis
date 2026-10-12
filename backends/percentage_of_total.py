"""Each value as its share of the total: divide each value by its column/row/grand total (or all
three), present as fractions or percentages, with an optional Wilson/
Brown-style binomial confidence interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class PercentageOfTotalOptions:
    divide_by: str = "column"  # "column" | "row" | "grand" | "all"
    present_as: str = "percentages"  # "fractions" | "percentages"
    calculate_ci: bool = False
    ci_level: float = 0.95
    ci_method: str = "wilson"  # "wilson" | "clopper_pearson"


def _wilson_ci(successes: float, total: float, conf: float) -> tuple[float, float] | None:
    if total <= 0 or successes < 0 or successes > total:
        return None
    z = stats.norm.ppf(0.5 + conf / 2)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _clopper_pearson_ci(successes: float, total: float, conf: float) -> tuple[float, float] | None:
    if total <= 0 or successes < 0 or successes > total:
        return None
    alpha = 1 - conf
    lo = stats.beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes) if successes < total else 1.0
    return float(lo), float(hi)


def _fractions_by_column(matrix: np.ndarray) -> np.ndarray:
    totals = np.nansum(matrix, axis=0)
    totals[totals == 0] = np.nan
    return matrix / totals


def _fractions_by_row(matrix: np.ndarray) -> np.ndarray:
    totals = np.nansum(matrix, axis=1, keepdims=True)
    totals[totals == 0] = np.nan
    return matrix / totals


@register("percentage_of_total")
def run(data: SheetData, options: PercentageOfTotalOptions | None = None) -> AnalysisResult:
    opt = options or PercentageOfTotalOptions()
    labels, matrix = data.first_replicate_matrix()
    if matrix.size == 0:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    if np.any(matrix[~np.isnan(matrix)] < 0):
        raise AnalysisError("A share of the total needs values that are zero or more (counts or amounts).")

    scale = 100.0 if opt.present_as == "percentages" else 1.0
    grand_total = float(np.nansum(matrix))
    if grand_total == 0:
        raise AnalysisError("The grand total is zero, so fractions/percentages are undefined.")

    modes = ["column", "row", "grand"] if opt.divide_by == "all" else [opt.divide_by]
    tables = []
    new_columns: dict[str, list[float]] = {}

    for mode in modes:
        if mode == "column":
            fractions = _fractions_by_column(matrix) * scale
            title = f"Divided by column total ({opt.present_as})"
        elif mode == "row":
            fractions = _fractions_by_row(matrix) * scale
            title = f"Divided by row total ({opt.present_as})"
        else:
            fractions = matrix / grand_total * scale
            title = f"Divided by grand total ({opt.present_as})"

        rows = [
            [f"{x:.4g}" if not np.isnan(x) else "-"] + [f"{v:.4g}" if not np.isnan(v) else "-" for v in fractions[r]]
            for r, x in enumerate(data.x[: matrix.shape[0]])
        ]
        tables.append(ResultTable(title=title, headers=["X"] + labels, rows=rows))
        if len(modes) == 1:
            for j, label in enumerate(labels):
                new_columns[label] = [float(v) for v in fractions[:, j]]

    ci_note = None
    if opt.calculate_ci:
        # CI is always computed on the raw counts (binomial), independent
        # of whether results are displayed as fractions or percentages.
        ci_rows = []
        col_totals = np.nansum(matrix, axis=0)
        for j, label in enumerate(labels):
            for r, x in enumerate(data.x[: matrix.shape[0]]):
                v = matrix[r, j]
                if np.isnan(v):
                    continue
                ci = _clopper_pearson_ci(v, col_totals[j], opt.ci_level) if opt.ci_method == "clopper_pearson" else _wilson_ci(v, col_totals[j], opt.ci_level)
                if ci:
                    lo, hi = ci
                    ci_rows.append([label, f"{x:.4g}", f"{lo * 100:.2f}% to {hi * 100:.2f}%"])
        pct = int(round(opt.ci_level * 100))
        method_name = "Clopper-Pearson (exact)" if opt.ci_method == "clopper_pearson" else "Wilson score"
        tables.append(ResultTable(title=f"{pct}% {method_name} confidence intervals (column basis)", headers=["Group", "X", "CI"], rows=ci_rows))
        ci_note = "The intervals treat every value as a count of whole items, not a value already scaled or averaged."

    summary = f"Divided {matrix.shape[0]}x{matrix.shape[1]} table by {opt.divide_by} total(s), shown as {opt.present_as}."
    notes = [ci_note] if ci_note else []

    result_kwargs = dict(title="Turn counts into percentages", summary=summary, tables=tables, notes=notes)
    if new_columns and opt.divide_by != "all":
        result_kwargs["new_sheet"] = NewSheetData(
            suggested_name="Percent of total" if opt.present_as == "percentages" else "Share of total",
            x_title="X",
            x_values=[float(v) for v in data.x[: matrix.shape[0]]],
            columns=new_columns,
        )
    return AnalysisResult(**result_kwargs)
