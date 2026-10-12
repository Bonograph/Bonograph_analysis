"""Many pairs at once: every pair of groups (or every group against a
control) tested directly, without an overall ANOVA first, and adjusted for
the number of comparisons.

This is for when the pairwise comparisons *are* the question. When the
question is "does any group differ?", use 'Three groups or more', which
runs the overall test and then the same follow-ups.

One kind of test runs - Welch t tests, or Mann-Whitney tests when nothing
is assumed about shape - never both side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..backend_types import AnalysisError, AnalysisResult
from ..data_source import SheetData
from ..registry import register
from . import _common


@dataclass
class MultipleComparisonsOptions:
    distribution: str = "gaussian"  # "gaussian" (Welch t tests) | "nonparametric" (Mann-Whitney)
    scope: str = "all_pairs"  # "all_pairs" | "vs_control"
    control_index: int = 0
    correction: str = "holm"  # "holm" | "fdr" | "none"
    confidence_level: float = 0.95


def _holm_bonferroni(p_values: list[float]) -> list[float]:  # kept for older callers
    return _common.holm(p_values).tolist()


@register("multiple_comparisons")
def run(data: SheetData, options: MultipleComparisonsOptions | None = None) -> AnalysisResult:
    opt = options or MultipleComparisonsOptions()
    groups = {k: v for k, v in data.all_group_arrays().items() if v.size >= 2}
    if len(groups) < 3:
        raise AnalysisError(
            "Comparing many pairs needs at least 3 groups with 2+ values each; "
            f"this sheet has {len(groups)}. Use 'Two groups' for just two.")
    labels = list(groups)
    arrays = [groups[k] for k in labels]
    ranks = opt.distribution == "nonparametric"
    method = opt.correction if opt.correction in ("holm", "fdr", "none") else "holm"
    table, described = _common.pairwise(arrays, labels, opt.scope, opt.control_index, method,
                                        opt.confidence_level, ranks=ranks)
    n_sig = sum(1 for row in table.rows if row[-1] == "Yes")
    name = {"holm": "Holm", "fdr": "Benjamini-Hochberg", "none": "no"}[method]
    summary = (f"{len(table.rows)} comparisons across {len(groups)} groups; {n_sig} significant "
               f"after {name} adjustment." if method != "none"
               else f"{len(table.rows)} comparisons across {len(groups)} groups; {n_sig} with P < 0.05 (unadjusted).")
    notes = [described]
    small = _common.small_n_note({k: v.size for k, v in groups.items()})
    if small:
        notes.insert(0, small)
    if method == "none":
        notes.append("Without an adjustment, the chance of at least one false positive grows with every pair.")
    return AnalysisResult(title="Many pairs at once", summary=summary, tables=[table], notes=notes)
