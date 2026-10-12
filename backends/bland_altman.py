"""Do two ways of measuring agree? Bias and limits of agreement after Bland
and Altman: which column is method A and which is method B, how to calculate the paired comparison, and whether to create a new
graph of the results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class BlandAltmanOptions:
    method_a_index: int = 0
    method_b_index: int = 1
    calculate: str = "diff_a_minus_b"  # diff_a_minus_b | ratio_a_over_b | pct_diff_a_minus_b | diff_b_minus_a | ratio_b_over_a | pct_diff_b_minus_a
    significant_digits: int = 4
    create_new_graph: bool = True


def _compute(a: np.ndarray, b: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    if mode == "diff_a_minus_b":
        return a - b, "A − B"
    if mode == "ratio_a_over_b":
        return a / b, "A / B"
    if mode == "pct_diff_a_minus_b":
        return 100 * (a - b) / ((a + b) / 2), "A − B, % of the pair's mean"
    if mode == "diff_b_minus_a":
        return b - a, "B − A"
    if mode == "ratio_b_over_a":
        return b / a, "B / A"
    if mode == "pct_diff_b_minus_a":
        return 100 * (b - a) / ((a + b) / 2), "B − A, % of the pair's mean"
    raise ValueError(f"Unknown calculate mode: {mode!r}")


@register("bland_altman")
def run(data: SheetData, options: BlandAltmanOptions | None = None) -> AnalysisResult:
    opt = options or BlandAltmanOptions()
    labels = data.group_labels()
    if len(labels) < 2:
        raise AnalysisError(f"Bland-Altman needs two paired measurement methods (two groups); this sheet has {len(labels)}.")

    ai = max(0, min(opt.method_a_index, len(labels) - 1))
    bi = max(0, min(opt.method_b_index, len(labels) - 1))
    if ai == bi:
        raise AnalysisError("Data set with method A and data set with method B must be two different data sets.")

    a_label, b_label = labels[ai], labels[bi]
    # the two methods on the same row are one subject: a blank in either drops that subject only
    rows = data.row_matrix([a_label, b_label])
    both = ~np.any(np.isnan(rows), axis=1)
    unpaired = int(np.sum(~np.all(np.isnan(rows), axis=1) & ~both))
    a, b = rows[both, 0], rows[both, 1]
    n = a.size
    if n < 3:
        raise AnalysisError("Need at least 3 rows with a measurement by both methods.")

    if opt.calculate in ("ratio_a_over_b",) and np.any(b == 0):
        raise AnalysisError(f"Can't compute A/B - '{b_label}' contains zero values.")
    if opt.calculate in ("ratio_b_over_a",) and np.any(a == 0):
        raise AnalysisError(f"Can't compute B/A - '{a_label}' contains zero values.")

    values, value_label = _compute(a, b, opt.calculate)
    means = (a + b) / 2
    bias = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    lower_loa = bias - 1.96 * sd
    upper_loa = bias + 1.96 * sd
    sig = opt.significant_digits

    table = ResultTable(
        title=f"Agreement between '{a_label}' (A) and '{b_label}' (B) (n = {n})",
        headers=["Statistic", "Value"],
        rows=[
            [f"Bias (mean {value_label})", f"{bias:.{sig}g}"],
            [f"SD of {value_label}", f"{sd:.{sig}g}"],
            ["Lower 95% limit of agreement", f"{lower_loa:.{sig}g}"],
            ["Upper 95% limit of agreement", f"{upper_loa:.{sig}g}"],
            ["Average of the two methods (range)", f"{means.min():.{sig}g} to {means.max():.{sig}g}"],
        ],
    )
    summary = f"Bias ({value_label}) = {bias:.{sig}g}, 95% limits of agreement = [{lower_loa:.{sig}g}, {upper_loa:.{sig}g}]."
    notes = [f"'{a_label}' and '{b_label}' were matched row-by-row as paired measurements of the same subjects by two different methods."]
    if unpaired:
        notes.insert(0, f"{unpaired} row{'s' if unpaired != 1 else ''} with a measurement by only one of the two methods "
                        f"{'were' if unpaired != 1 else 'was'} left out.")

    new_sheet = None
    if opt.create_new_graph:
        order = np.argsort(means)
        new_sheet = NewSheetData(
            suggested_name=f"Bland-Altman: {a_label} vs {b_label}",
            x_title="Average of the two methods",
            x_values=[float(v) for v in means[order]],
            columns={value_label: [float(v) for v in values[order]]},
        )

    from ..chart_helpers import bland_altman_chart

    chart = bland_altman_chart(f"Bland-Altman: {a_label} vs {b_label}", means, values, bias, lower_loa, upper_loa, value_label)
    return AnalysisResult(title="Do two methods agree?", summary=summary, tables=[table], notes=notes, new_sheet=new_sheet, charts=[chart])
