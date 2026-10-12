"""A made-up curve with scatter. Generates a synthetic X/Y dataset from the same equation
library "Draw an equation" uses, with configurable random error and
optional outliers added - for testing curve-fitting analyses or
teaching, not derived from the sheet's existing data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register
from .function_library import EQUATIONS


@dataclass
class SimulateXYOptions:
    # the X values
    use_sheet_x: bool = False
    x_start: float = 0.0
    x_step_mode: str = "plus"  # "plus" | "times"
    x_step_value: float = 1.0
    x_count_mode: str = "count"  # "count" | "stop_at"
    x_count: int = 20
    x_stop_at: float = 20.0
    # the curve
    use_sheet_y: bool = False
    equation_key: str = "straight_line"
    # parameter values and names
    parameter_sets: list[dict[str, float]] = field(default_factory=list)
    column_titles: list[str] = field(default_factory=list)
    n_datasets: int = 1
    replicates_per_dataset: int = 1
    # the scatter
    error_type: str = "none"  # "none"|"gaussian_abs"|"gaussian_relative"|"double_gaussian"|"poisson"|"t_dist"|"cauchy"|"binomial"
    error_sd: float = 1.0
    error_sd2: float = 1.0
    error_relative_pct: float = 10.0
    double_gaussian_pct: float = 50.0
    t_df: int = 2
    binomial_value_type: str = "fractions"  # "fractions" | "percentages" | "counts"
    binomial_n: int = 1
    add_outliers: bool = False
    outlier_count: int = 0
    outlier_lower: float = 0.0
    outlier_upper: float = 0.0
    seed_mode: str = "auto"  # "auto" | "fixed"
    seed: int = 0
    significant_digits: int = 4


def _generate_x(opt: SimulateXYOptions, data: SheetData) -> np.ndarray:
    if opt.use_sheet_x:
        x = data.x[~np.isnan(data.x)]
        if x.size < 2:
            raise AnalysisError("The current sheet's X column has fewer than 2 values to reuse.")
        return x
    xs = [opt.x_start]
    if opt.x_count_mode == "count":
        n = max(2, opt.x_count)
        for _ in range(n - 1):
            xs.append(xs[-1] + opt.x_step_value if opt.x_step_mode == "plus" else xs[-1] * opt.x_step_value)
    else:
        while True:
            nxt = xs[-1] + opt.x_step_value if opt.x_step_mode == "plus" else xs[-1] * opt.x_step_value
            if (opt.x_step_value >= 0 and nxt > opt.x_stop_at) or (opt.x_step_value < 0 and nxt < opt.x_stop_at):
                break
            xs.append(nxt)
            if len(xs) > 100000:
                raise AnalysisError("Generating X values didn't converge - check the step and stop-at values.")
    return np.array(xs)


def _add_error(y: np.ndarray, opt: SimulateXYOptions, rng: np.random.Generator) -> np.ndarray:
    if opt.error_type == "gaussian_abs":
        return y + rng.normal(0, opt.error_sd, y.shape)
    if opt.error_type == "gaussian_relative":
        return y + rng.normal(0, np.abs(y) * opt.error_relative_pct / 100.0, y.shape)
    if opt.error_type == "double_gaussian":
        use_sd1 = rng.uniform(0, 100, y.shape) < opt.double_gaussian_pct
        noise = np.where(use_sd1, rng.normal(0, opt.error_sd, y.shape), rng.normal(0, opt.error_sd2, y.shape))
        return y + noise
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


def _apply_outliers(y: np.ndarray, opt: SimulateXYOptions, rng: np.random.Generator) -> np.ndarray:
    if not opt.add_outliers or opt.outlier_count <= 0:
        return y
    y = y.copy()
    n = y.size
    count = min(opt.outlier_count, n)
    idx = rng.choice(n, size=count, replace=False)
    y[idx] = y[idx] + rng.uniform(opt.outlier_lower, opt.outlier_upper, count)
    return y


@register("simulate_xy_data")
def run(data: SheetData, options: SimulateXYOptions | None = None) -> AnalysisResult:
    opt = options or SimulateXYOptions()
    x = _generate_x(opt, data)
    n = x.size
    sig = opt.significant_digits
    rng = np.random.default_rng(None if opt.seed_mode == "auto" else opt.seed)

    if opt.use_sheet_y:
        labels = data.group_labels()
        if not labels:
            raise AnalysisError("The current sheet has no Y column to reuse.")
        if opt.use_sheet_x:
            # the sheet's own pairs: a blank X or Y drops that point, and every other Y keeps its X
            x, base_y = data.xy_points(labels[0])
            if x.size < 2:
                raise AnalysisError("The current sheet has fewer than 2 rows with both an X and a Y to reuse.")
            n = x.size
        else:
            base_y = data.group_values(labels[0])
            n = min(n, base_y.size)
            x, base_y = x[:n], base_y[:n]
        base_curves = [base_y]
        titles = opt.column_titles[:1] or [labels[0]]
        equation_note = f"Y values reused from '{labels[0]}' on the current sheet."
    else:
        spec = EQUATIONS.get(opt.equation_key)
        if spec is None:
            raise AnalysisError(f"'{opt.equation_key}' isn't implemented in this build's function library yet.")
        n_curves = max(1, opt.n_datasets)
        base_curves = []
        titles = []
        for i in range(n_curves):
            overrides = opt.parameter_sets[i] if i < len(opt.parameter_sets) else {}
            values = [overrides.get(name, default) for name, default in spec.params]
            try:
                with np.errstate(all="ignore"):
                    y = np.asarray(spec.fn(x, *values), dtype=float)
            except Exception as exc:  # noqa: BLE001
                raise AnalysisError(f"Couldn't evaluate '{spec.label}': {exc}") from None
            if y.shape != x.shape:
                y = np.broadcast_to(y, x.shape)
            base_curves.append(y)
            titles.append(opt.column_titles[i] if i < len(opt.column_titles) and opt.column_titles[i] else (spec.label if n_curves == 1 else f"{spec.label} (dataset {i+1})"))
        equation_note = f"Equation: {spec.equation}"

    n_reps = max(1, opt.replicates_per_dataset)
    columns: dict[str, list[float]] = {}
    for title, y in zip(titles, base_curves):
        for rep in range(n_reps):
            noisy = _add_error(y, opt, rng)
            noisy = _apply_outliers(noisy, opt, rng)
            col_title = title if n_reps == 1 else f"{title} (rep {rep+1})"
            columns[col_title] = noisy.tolist()

    new_sheet = NewSheetData(suggested_name="Simulated XY data", x_title="X", x_values=[float(v) for v in x], columns=columns)
    preview_rows = [[f"{xv:.{sig}g}"] + [f"{columns[c][i]:.{sig}g}" for c in columns] for i, xv in enumerate(x[:200])]
    table = ResultTable(title="Simulated data", headers=["X"] + list(columns.keys()), rows=preview_rows)

    error_names = {
        "none": "no random error added", "gaussian_abs": f"Gaussian noise, SD = {opt.error_sd:g}",
        "gaussian_relative": f"Gaussian noise, {opt.error_relative_pct:g}% of |Y|",
        "double_gaussian": f"double Gaussian ({opt.double_gaussian_pct:g}% with SD={opt.error_sd:g}, rest with SD={opt.error_sd2:g})",
        "poisson": "Poisson noise (Y treated as expected count)",
        "t_dist": f"t-distributed noise (df={opt.t_df}, scale={opt.error_sd:g})",
        "cauchy": f"Cauchy/Lorentzian noise (scale={opt.error_sd:g})",
        "binomial": f"binomial noise (N={opt.binomial_n}, values as {opt.binomial_value_type})",
    }
    error_note = error_names.get(opt.error_type, opt.error_type)
    summary = f"{n} simulated point(s), {len(columns)} column(s) ({error_note})" + (f", {opt.outlier_count} outlier(s) injected." if opt.add_outliers and opt.outlier_count > 0 else ".")

    return AnalysisResult(title="Simulated XY data", summary=summary, tables=[table], notes=[equation_note], new_sheet=new_sheet)
