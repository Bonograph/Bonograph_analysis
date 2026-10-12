"""Made-up groups with a known answer.
Generates synthetic groups (one population mean per data set, entered
individually or drawn from a Gaussian distribution of means) with
configurable random scatter and outliers - for testing group-comparison
analyses or teaching, not derived from the sheet's existing data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class SimulateColumnOptions:
    # the groups
    n_datasets: int = 1
    mean_mode: str = "individual"  # "individual" | "random_gaussian"
    population_mean: float = 0.0
    population_sd: float = 1.0
    dataset_rows: list[int] = field(default_factory=lambda: [5])
    dataset_means: list[float] = field(default_factory=lambda: [0.0])
    dataset_titles: list[str] = field(default_factory=lambda: [""])
    # the scatter
    error_type: str = "none"
    error_sd: float = 1.0
    error_sd2: float = 1.0
    error_relative_pct: float = 10.0
    double_gaussian_pct: float = 50.0
    t_df: int = 2
    binomial_value_type: str = "fractions"
    binomial_n: int = 1
    add_outliers: bool = False
    outlier_count: int = 0
    outlier_lower: float = 0.0
    outlier_upper: float = 0.0
    seed_mode: str = "auto"
    seed: int = 0
    significant_digits: int = 4


def _add_error(y: np.ndarray, opt: SimulateColumnOptions, rng: np.random.Generator) -> np.ndarray:
    if opt.error_type == "gaussian_abs":
        return y + rng.normal(0, opt.error_sd, y.shape)
    if opt.error_type == "gaussian_relative":
        return y + rng.normal(0, np.abs(y) * opt.error_relative_pct / 100.0, y.shape)
    if opt.error_type == "double_gaussian":
        use_sd1 = rng.uniform(0, 100, y.shape) < opt.double_gaussian_pct
        return y + np.where(use_sd1, rng.normal(0, opt.error_sd, y.shape), rng.normal(0, opt.error_sd2, y.shape))
    if opt.error_type == "poisson":
        return rng.poisson(np.clip(y, 0, None)).astype(float)
    if opt.error_type == "t_dist":
        return y + stats.t.rvs(opt.t_df, size=y.shape, random_state=rng) * opt.error_sd
    if opt.error_type == "cauchy":
        return y + rng.standard_cauchy(y.shape) * opt.error_sd
    if opt.error_type == "binomial":
        n_trials = max(1, opt.binomial_n)
        if opt.binomial_value_type == "counts":
            p = np.clip(y / n_trials, 0, 1)
        elif opt.binomial_value_type == "percentages":
            p = np.clip(y / 100.0, 0, 1)
        else:
            p = np.clip(y, 0, 1)
        counts = rng.binomial(n_trials, p)
        if opt.binomial_value_type == "counts":
            return counts.astype(float)
        if opt.binomial_value_type == "percentages":
            return 100.0 * counts / n_trials
        return counts / n_trials
    return y


def _apply_outliers(y: np.ndarray, opt: SimulateColumnOptions, rng: np.random.Generator) -> np.ndarray:
    if not opt.add_outliers or opt.outlier_count <= 0:
        return y
    y = y.copy()
    count = min(opt.outlier_count, y.size)
    idx = rng.choice(y.size, size=count, replace=False)
    y[idx] = y[idx] + rng.uniform(opt.outlier_lower, opt.outlier_upper, count)
    return y


@register("simulate_column_data")
def run(data: SheetData, options: SimulateColumnOptions | None = None) -> AnalysisResult:
    opt = options or SimulateColumnOptions()
    n = max(1, opt.n_datasets)
    if len(opt.dataset_rows) < n or len(opt.dataset_means) < n:
        raise AnalysisError("Give every group its number of values and its mean.")

    rng = np.random.default_rng(None if opt.seed_mode == "auto" else opt.seed)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if opt.mean_mode == "random_gaussian":
        dataset_means = rng.normal(opt.population_mean, opt.population_sd, n).tolist()
    else:
        dataset_means = opt.dataset_means[:n]

    columns: dict[str, list[float]] = {}
    for i in range(n):
        rows = max(1, opt.dataset_rows[i] if i < len(opt.dataset_rows) else 5)
        mean = dataset_means[i]
        title = opt.dataset_titles[i] if i < len(opt.dataset_titles) and opt.dataset_titles[i] else f"Set {letters[i] if i < len(letters) else i+1}"
        y = np.full(rows, mean)
        y = _add_error(y, opt, rng)
        y = _apply_outliers(y, opt, rng)
        columns[title] = y.tolist()

    max_rows = max(len(v) for v in columns.values())
    sig = opt.significant_digits
    rows_preview = [[f"{columns[c][i]:.{sig}g}" if i < len(columns[c]) else "" for c in columns] for i in range(min(max_rows, 200))]
    table = ResultTable(title=f"{n} simulated data set(s)", headers=list(columns.keys()), rows=rows_preview)

    new_sheet = NewSheetData(suggested_name="Simulated groups", x_title="Row", x_values=[float(i + 1) for i in range(max_rows)], columns=columns)

    error_note = {"none": "no random scatter", "gaussian_abs": f"Gaussian, SD = {opt.error_sd:g}", "gaussian_relative": f"Gaussian, {opt.error_relative_pct:g}% of |Y|"}.get(opt.error_type, opt.error_type)
    summary = f"{n} data set(s) simulated ({error_note})" + (f", {opt.outlier_count} outlier(s) injected." if opt.add_outliers and opt.outlier_count > 0 else ".")
    notes = ["Generated data - not derived from anything on the sheet."]
    if opt.mean_mode == "random_gaussian":
        notes.append(f"Each data set's own population mean was drawn from a Gaussian distribution (mean = {opt.population_mean:g}, SD = {opt.population_sd:g}) rather than entered individually.")

    return AnalysisResult(title="Simulated groups", summary=summary, tables=[table], notes=notes, new_sheet=new_sheet)
