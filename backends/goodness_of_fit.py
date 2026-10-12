"""Do observed counts match what was expected? Pick a data set, type in
the expected count (or percentage) for each row by hand, and choose
between a chi-square test or (for exactly two rows) an exact binomial
test."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register



def _p(value, opt) -> str:
    """One P value, written the way the user asked for (analysis/p_values.py)."""
    from ..p_values import format_p

    return format_p(value, getattr(opt, "p_style", "number"),
                    digits=getattr(opt, "significant_digits", 4),
                    alpha=getattr(opt, "alpha", 0.05))

@dataclass
class GoodnessOfFitOptions:
    dataset_index: int = 0
    expected_as_percent: bool = False
    two_row_method: str = "binomial"  # "binomial" (exact, recommended) | "chi_square" - only used with exactly 2 rows
    expected_values: list[float] = field(default_factory=list)
    confidence_level: float = 0.95
    significant_digits: int = 4
    p_style: str = "number"  # how P values are written (analysis/p_values.py)


def _wilson_ci(successes: float, total: float, conf: float) -> tuple[float, float] | None:
    if total <= 0 or successes < 0 or successes > total:
        return None
    z = stats.norm.ppf(0.5 + conf / 2)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


@register("goodness_of_fit")
def run(data: SheetData, options: GoodnessOfFitOptions | None = None) -> AnalysisResult:
    opt = options or GoodnessOfFitOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    idx = max(0, min(opt.dataset_index, len(labels) - 1))
    label = labels[idx]
    on_rows = data.group_rows(label, "average")
    sheet_rows = np.flatnonzero(~np.isnan(on_rows))          # a row with no count is not a category
    observed = on_rows[sheet_rows]
    if observed.size < 2:
        raise AnalysisError("Need at least 2 categories (rows) of observed counts.")
    if np.any(observed < 0):
        raise AnalysisError("Observed counts can't be negative - check the sheet's values.")

    if len(opt.expected_values) != observed.size:
        raise AnalysisError(
            f"Enter an expected value for every row - '{label}' has {observed.size} rows "
            f"but {len(opt.expected_values)} expected value(s) were entered."
        )
    expected_raw = np.array(opt.expected_values, dtype=float)
    if np.any(expected_raw < 0):
        raise AnalysisError("Expected values can't be negative.")
    if np.sum(expected_raw) == 0:
        raise AnalysisError("The expected values can't all be zero.")

    sig = opt.significant_digits
    if opt.expected_as_percent:
        total_pct = float(np.sum(expected_raw))
        if abs(total_pct - 100.0) > 0.5:
            raise AnalysisError(f"The expected percentages add up to {total_pct:g}%, not 100%. Correct them, or "
                                "choose 'Relative amounts' to have them scaled automatically.")
        expected = expected_raw / total_pct * observed.sum()
        expected_display = [f"{v:.4g}%" for v in expected_raw]
    else:
        expected = expected_raw * (observed.sum() / expected_raw.sum())
        expected_display = [f"{v:.{sig}g}" for v in expected_raw]

    table = ResultTable(
        title="Observed vs. expected",
        headers=["Row", "Observed", "Expected (entered)", "Expected (scaled to total)"],
        rows=[[str(int(r) + 1), f"{o:.{sig}g}", ed, f"{e:.{sig}g}"] for r, o, ed, e in zip(sheet_rows, observed, expected_display, expected)],
    )

    pct = int(round(opt.confidence_level * 100))
    if observed.size == 2 and opt.two_row_method == "binomial":
        n = int(round(observed.sum()))
        successes = int(round(observed[0]))
        p0 = expected[0] / expected.sum()
        result = stats.binomtest(successes, n, p0)
        p_value = result.pvalue
        ci = _wilson_ci(successes, n, opt.confidence_level)
        ci_text = f"{ci[0]*100:.{sig}g}% to {ci[1]*100:.{sig}g}%" if ci else "n/a"
        test_table = ResultTable(
            title="Binomial test",
            headers=["Observed proportion", f"{pct}% CI (Wilson/Brown)", "Hypothesized proportion", "P value"],
            rows=[[f"{successes/n*100:.{sig}g}%", ci_text, f"{p0*100:.{sig}g}%", _p(p_value, opt)]],
        )
        test_name = "Binomial test"
        dof = None
    else:
        chi2, p_value = stats.chisquare(observed, f_exp=expected)
        dof = observed.size - 1
        cohen_w = float(np.sqrt(chi2 / observed.sum())) if observed.sum() > 0 else float("nan")
        size = "small" if cohen_w < 0.3 else "medium" if cohen_w < 0.5 else "large"
        test_table = ResultTable(title="Chi-square test for goodness of fit",
                                 headers=["Chi-square", "df", "P value", "Cohen's w (effect size)"],
                                 rows=[[f"{chi2:.4f}", str(dof), _p(p_value, opt), f"{cohen_w:.3f} ({size})"]])
        test_name = "Chi-square test"

    summary = f"'{label}': {test_name} p = {_p(p_value, opt)} ({'reject' if p_value < 0.05 else 'fail to reject'} the fit at α = 0.05)."
    notes = ["Expected values were scaled so their total matches the observed total."]
    if dof is not None and np.any(expected < 5):
        notes.insert(0, f"{int(np.sum(expected < 5))} row(s) expect fewer than 5, so the chi-square P value is only "
                        "approximate. Combine sparse rows, or collect more data.")

    return AnalysisResult(title="Counts against expectations", summary=summary, tables=[table, test_table], notes=notes)
