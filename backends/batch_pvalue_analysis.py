"""Many P values at once: which of them still stand once allowance is made
for how many were tested, by an FDR approach or a statistical-significance
(family-wise) approach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class BatchPValueOptions:
    approach: str = "fdr"  # "fdr" | "significance"
    fdr_method: str = "two_stage_bky"  # "two_stage_bky" | "benjamini_hochberg" | "benjamini_yekutieli"
    desired_fdr_percent: float = 1.0
    sig_method: str = "holm_sidak"  # "holm_sidak" | "bonferroni" | "none"
    alpha: float = 0.05
    graph_ranked: bool = False


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)[::-1]
    m = len(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


def _benjamini_yekutieli(p: np.ndarray) -> np.ndarray:
    m = len(p)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m * c_m / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


def _two_stage_bky(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Krieger-Yekutieli two-stage adaptive FDR: estimate the
    number of true null hypotheses from a first-pass BH pass at level
    q/(1+q), then rerun BH at level q against that estimated m0."""
    m = len(p)
    bh_first = _benjamini_hochberg(p)
    r1 = int(np.sum(bh_first <= q / (1 + q)))
    if r1 == 0 or r1 == m:
        return _benjamini_hochberg(p)
    m0_hat = m - r1
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m0_hat / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


def _holm_sidak(p: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        remaining = m - rank
        step_adj = 1 - (1 - p[idx]) ** remaining
        running_max = max(running_max, step_adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _bonferroni(p: np.ndarray) -> np.ndarray:
    return np.clip(p * p.size, 0, 1)


@register("batch_pvalue_analysis")
def run(data: SheetData, options: BatchPValueOptions | None = None) -> AnalysisResult:
    opt = options or BatchPValueOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    p_values = data.group_values(labels[0])
    if p_values.size < 2:
        raise AnalysisError("Need at least 2 p-values to correct for multiple testing.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise AnalysisError(f"'{labels[0]}' should contain p-values between 0 and 1 - check the sheet's values.")

    if opt.approach == "fdr":
        q = opt.desired_fdr_percent / 100.0
        if opt.fdr_method == "benjamini_hochberg":
            adjusted, method_name = _benjamini_hochberg(p_values), "Benjamini-Hochberg FDR"
        elif opt.fdr_method == "benjamini_yekutieli":
            adjusted, method_name = _benjamini_yekutieli(p_values), "Benjamini-Yekutieli FDR"
        else:
            adjusted, method_name = _two_stage_bky(p_values, q), "Two-stage step-up (Benjamini-Krieger-Yekutieli)"
        threshold = q
        decision_col = "Discovery (FDR)"
    else:
        if opt.sig_method == "bonferroni":
            adjusted, method_name = _bonferroni(p_values), "Bonferroni"
        elif opt.sig_method == "none":
            adjusted, method_name = p_values.copy(), "Uncorrected"
        else:
            adjusted, method_name = _holm_sidak(p_values, opt.alpha), "Holm-Sidak"
        threshold = opt.alpha
        decision_col = "Significant"

    order = np.argsort(p_values)
    rows = [
        [str(rank + 1), f"{p_values[i]:.4g}", f"{adjusted[i]:.4g}", "Yes" if adjusted[i] < threshold else "No"]
        for rank, i in enumerate(order)
    ]
    table = ResultTable(
        title=f"{p_values.size} p-values from '{labels[0]}', ranked smallest first ({method_name})",
        headers=["Rank", "P value", "Adjusted P value", decision_col],
        rows=rows,
    )
    n_flagged = int(np.sum(adjusted < threshold))
    if opt.approach == "fdr":
        summary = f"{n_flagged} of {p_values.size} p-values are discoveries at a {opt.desired_fdr_percent:g}% false discovery rate ({method_name})."
    else:
        summary = f"{n_flagged} of {p_values.size} p-values remain significant at alpha = {opt.alpha:g} ({method_name})."

    new_sheet = None
    if opt.graph_ranked:
        ranked_p = [float(p_values[i]) for i in order]
        new_sheet = NewSheetData(
            suggested_name=f"Ranked P values ({labels[0]})",
            x_title="Rank",
            x_values=[float(r + 1) for r in range(len(order))],
            columns={"P value": ranked_p, "Adjusted P value": [float(adjusted[i]) for i in order]},
        )

    charts = []
    if opt.graph_ranked:
        from ..chart_helpers import xy_chart

        ranks = np.arange(1, len(order) + 1)
        charts.append(xy_chart(
            "Ranked P values", ranks,
            {"P value": [float(p_values[i]) for i in order], "Adjusted P value": [float(adjusted[i]) for i in order]},
            kind="scatter", x_label="Rank", y_label="P value",
            reference_lines=[{"kind": "h", "value": float(threshold)}],
        ))
    return AnalysisResult(title="Many P values at once", summary=summary, tables=[table], new_sheet=new_sheet, charts=charts)
