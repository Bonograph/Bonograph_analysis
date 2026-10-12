"""Run an experiment many times over (a Monte Carlo study): repeatedly simulates a two-group comparison
under given sample size, effect size, and noise, to estimate statistical
power empirically. A generator/study-design tool, not an analysis of the
sheet's existing data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class MonteCarloOptions:
    n_per_group: int = 10
    effect_size: float = 1.0  # difference in means, in units of SD
    sd: float = 1.0
    n_simulations: int = 2000
    alpha: float = 0.05
    seed: int = 0


@register("monte_carlo_simulation")
def run(data: SheetData, options: MonteCarloOptions | None = None) -> AnalysisResult:
    opt = options or MonteCarloOptions()
    if opt.n_per_group < 2:
        raise AnalysisError("Need at least 2 subjects per group.")
    if opt.n_simulations < 100:
        raise AnalysisError("Use at least 100 simulations for a stable power estimate.")

    rng = np.random.default_rng(opt.seed)
    p_values = np.empty(opt.n_simulations)
    for i in range(opt.n_simulations):
        a = rng.normal(0, opt.sd, opt.n_per_group)
        b = rng.normal(opt.effect_size, opt.sd, opt.n_per_group)
        _t, p_values[i] = stats.ttest_ind(a, b)

    power = float(np.mean(p_values < opt.alpha))
    se_power = float(np.sqrt(power * (1 - power) / opt.n_simulations))

    table = ResultTable(
        title="Simulated power",
        headers=["N per group", "Effect size (Δ/SD)", "α", "Simulations", "Estimated power"],
        rows=[[
            str(opt.n_per_group),
            f"{opt.effect_size / opt.sd:.3g}",
            f"{opt.alpha:g}",
            str(opt.n_simulations),
            f"{power * 100:.1f}% ± {se_power * 100:.1f}%",
        ]],
    )
    summary = (
        f"With n = {opt.n_per_group}/group and effect size {opt.effect_size / opt.sd:.2f} SD, "
        f"the estimated power to detect it at α = {opt.alpha:g} is {power * 100:.1f}%."
    )

    return AnalysisResult(
        title="Run it many times over",
        summary=summary,
        tables=[table],
        notes=[
            "Power is the fraction of simulated two-sample t-tests (Welch's) "
            "that came back significant when the true effect and noise were "
            "exactly as specified.",
        ],
    )
