"""Made-up counts in a 2x2 table: a cross-sectional study design where
subjects are independently assigned a risk-factor exposure, then an
outcome probability that depends on that exposure - for testing
chi-square/Fisher's exact analyses or teaching. A generator, not an
analysis of the sheet's existing data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class SimulateContingencyOptions:
    sample_size: int = 100
    row1_title: str = "Exposed"
    row2_title: str = "Not exposed"
    p_exposed: float = 0.5
    col_a_title: str = "Bad outcome"
    col_b_title: str = "Good outcome"
    p_outcome_row1: float = 0.3
    p_outcome_row2: float = 0.1
    seed_mode: str = "auto"  # "auto" | "fixed"
    seed: int = 0


@register("simulate_contingency_table")
def run(data: SheetData, options: SimulateContingencyOptions | None = None) -> AnalysisResult:
    opt = options or SimulateContingencyOptions()
    if opt.sample_size < 2:
        raise AnalysisError("Sample size needs to be at least 2.")
    for name, p in (("row exposure", opt.p_exposed), ("row 1 outcome", opt.p_outcome_row1), ("row 2 outcome", opt.p_outcome_row2)):
        if not (0 <= p <= 1):
            raise AnalysisError(f"The {name} probability must be between 0% and 100%.")

    rng = np.random.default_rng(None if opt.seed_mode == "auto" else opt.seed)
    n_row1 = int(rng.binomial(opt.sample_size, opt.p_exposed))
    n_row2 = opt.sample_size - n_row1
    col_a_row1 = int(rng.binomial(n_row1, opt.p_outcome_row1)) if n_row1 else 0
    col_a_row2 = int(rng.binomial(n_row2, opt.p_outcome_row2)) if n_row2 else 0
    col_b_row1, col_b_row2 = n_row1 - col_a_row1, n_row2 - col_a_row2

    table_data = np.array([[col_a_row1, col_b_row1], [col_a_row2, col_b_row2]])
    chi2, chi_p, _dof, _exp = stats.chi2_contingency(table_data)
    _odds, fisher_p = stats.fisher_exact(table_data)

    table = ResultTable(
        title=f"Simulated {opt.sample_size}-subject cross-sectional table",
        headers=["", opt.col_a_title, opt.col_b_title, "Total"],
        rows=[
            [opt.row1_title, str(col_a_row1), str(col_b_row1), str(n_row1)],
            [opt.row2_title, str(col_a_row2), str(col_b_row2), str(n_row2)],
        ],
    )
    test_table = ResultTable(title="Test on this simulated draw", headers=["Test", "P value"], rows=[["Chi-square", f"{chi_p:.4g}"], ["Fisher's exact", f"{fisher_p:.4g}"]])

    true_rr = opt.p_outcome_row1 / opt.p_outcome_row2 if opt.p_outcome_row2 else float("inf")
    summary = f"{opt.sample_size} subjects ({n_row1} {opt.row1_title}, {n_row2} {opt.row2_title}); true relative risk = {true_rr:.3g}; chi-square p = {chi_p:.4g} on this draw."

    return AnalysisResult(
        title="Simulated counts",
        summary=summary,
        tables=[table, test_table],
        notes=["Generated data - not derived from anything on the sheet. Re-run with a different seed to see how much the test result varies by chance."],
    )
