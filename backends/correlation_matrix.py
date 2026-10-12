"""Every pair of columns at once: pairwise Pearson correlation between X and every
group's first replicate column."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@register("correlation_matrix")
def run(data: SheetData) -> AnalysisResult:
    labels, matrix = data.full_variable_matrix()
    # leave out variables with no values at all (an unused X column, say)
    has_values = ~np.all(np.isnan(matrix), axis=0) if matrix.size else np.zeros(len(labels), dtype=bool)
    labels = [lbl for lbl, keep in zip(labels, has_values) if keep]
    matrix = matrix[:, has_values] if matrix.size else matrix
    complete = matrix[~np.any(np.isnan(matrix), axis=1)] if matrix.size else matrix
    if len(labels) < 2:
        raise AnalysisError("Need at least 2 variables (X plus a group column) to correlate.")
    if complete.shape[0] < 3:
        raise AnalysisError(
            f"Need at least 3 complete rows across all {len(labels)} "
            f"variables; found {complete.shape[0]}."
        )

    n = len(labels)
    r_matrix = np.eye(n)
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, p = stats.pearsonr(complete[:, i], complete[:, j])
            r_matrix[i, j] = r_matrix[j, i] = r
            p_matrix[i, j] = p_matrix[j, i] = p

    rows = [
        [labels[i]] + [f"{r_matrix[i, j]:.3f}" for j in range(n)]
        for i in range(n)
    ]
    table = ResultTable(title="Pearson r", headers=["Variable"] + labels, rows=rows)

    p_rows = [
        [labels[i]] + ["-" if i == j else f"{p_matrix[i, j]:.4g}" for j in range(n)]
        for i in range(n)
    ]
    p_table = ResultTable(title="P values", headers=["Variable"] + labels, rows=p_rows)

    strongest_i, strongest_j = np.unravel_index(
        np.argmax(np.abs(r_matrix - np.eye(n))), r_matrix.shape
    )
    summary = (
        f"{n} variables, {complete.shape[0]} complete rows; strongest correlation is "
        f"{labels[strongest_i]} vs {labels[strongest_j]} (r = {r_matrix[strongest_i, strongest_j]:.3f})."
    )

    return AnalysisResult(title="Every pair at once", summary=summary, tables=[table, p_table])
