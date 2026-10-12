"""A straight line through the points, by least squares.

Beside the slope and intercept (each with its standard error, interval and
P value) the report can answer:

* is a straight line the right shape - a runs test, and, when some X values
  are repeated, the lack-of-fit F test that weighs the scatter about the
  line against the scatter among repeats at the same X;
* which points pull the line - leverage, studentised residuals and Cook's
  distance (Cook 1977), with a test of the most outlying point;
* would a line that ignores odd points agree - the Theil-Sen line (Theil
  1950; Sen 1968): the median of the slopes between every pair of points;
* what the line predicts - Y at an X, and X at a Y with Fieller's limits,
  which are exact for inverse prediction where a symmetric interval is not.

The line may be forced through a point; every interval then follows the
one-parameter model, not the two-parameter formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register

MOST_FOR_PAIRS = 3000   # the robust line looks at every pair of points: beyond this it uses an even selection
BLUE, RED, GREEN = "#4a90d9", "#d62728", "#2ca02c"


@dataclass
class LinearRegressionOptions:
    replicate_handling: str = "average"  # "individual" | "average"
    force_point: bool = False
    force_x: float = 0.0
    force_y: float = 0.0
    show_confidence_band: bool = False
    confidence_band_level: float = 0.95
    band_type: str = "confidence"  # "confidence" (where the true line lies) | "prediction" (where new points fall)
    confidence_level: float = 0.95
    graph: bool = True  # the data with the best-fit line
    residual_table: bool = False
    runs_test: bool = False
    ci_y_at_x: bool = False
    ci_y_at_x_value: float = 0.0
    ci_x_at_y: bool = False
    ci_x_at_y_value: float = 0.0
    range_mode_start: str = "auto"  # "auto" | "manual"
    range_start_x: float = 0.0
    range_mode_end: str = "auto"
    range_end_x: float = 0.0
    interpolate_unknowns: bool = False
    significant_digits: int = 4
    # is a line the right shape, when X values are repeated
    lack_of_fit: bool = False
    # which points pull the line
    influence: bool = False
    influence_plot: bool = False
    # a line that ignores odd points
    robust_line: bool = False


def _runs_test(residuals: np.ndarray) -> tuple[int, float] | None:
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    if signs.size < 4:
        return None
    n1, n2 = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n = n1 + n2
    mean_runs = 2 * n1 * n2 / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1)) if n > 1 else 0
    if var_runs <= 0:
        return runs, float("nan")
    z = (runs - mean_runs) / np.sqrt(var_runs)
    p = float(2 * stats.norm.sf(abs(z)))
    return runs, p


class Line:
    """A least-squares line, free or forced through a point, with what its intervals need."""

    def __init__(self, x: np.ndarray, y: np.ndarray, through: tuple[float, float] | None = None):
        self.x, self.y, self.through, self.n = x, y, through, x.size
        if through is None:
            self.X = np.column_stack([np.ones(self.n), x])
            target = y
        else:
            self.X = (x - through[0])[:, None]
            target = y - through[1]
        self.p = self.X.shape[1]
        gram = self.X.T @ self.X
        if np.linalg.matrix_rank(gram) < self.p:
            raise AnalysisError("Every X value is the same, so no line can be fitted." if through is None else
                                "Every X value equals the X of the point the line is forced through.")
        self.inverse = np.linalg.inv(gram)
        coefficients = self.inverse @ self.X.T @ target
        if through is None:
            self.intercept, self.slope = float(coefficients[0]), float(coefficients[1])
        else:
            self.slope = float(coefficients[0])
            self.intercept = float(through[1] - self.slope * through[0])
        self.fitted = self.slope * x + self.intercept
        self.residuals = y - self.fitted
        self.df = self.n - self.p
        self.ss_residual = float(np.sum(self.residuals ** 2))
        self.mse = self.ss_residual / self.df if self.df > 0 else float("nan")
        self.cov = self.mse * self.inverse
        total = float(np.sum((y - np.mean(y)) ** 2)) if through is None else float(np.sum(target ** 2))
        self.r_squared = 1 - self.ss_residual / total if total > 0 else 0.0
        self.leverage = np.clip(np.einsum("ij,jk,ik->i", self.X, self.inverse, self.X), 0, 1)

    def row(self, at):
        """The design row(s) for X = ``at``."""
        at = np.atleast_1d(np.asarray(at, dtype=float))
        return np.column_stack([np.ones(at.size), at]) if self.through is None else (at - self.through[0])[:, None]

    def predict(self, at):
        return self.slope * np.asarray(at, dtype=float) + self.intercept

    def se_mean(self, at) -> np.ndarray:
        """Standard error of the line's height at X = ``at``."""
        rows = self.row(at)
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", rows, self.cov, rows), 0, None))

    def slope_se(self) -> float:
        return float(np.sqrt(self.cov[-1, -1]))

    def intercept_se(self) -> float:
        return float(np.sqrt(self.cov[0, 0])) if self.through is None else 0.0

    def x_at(self, target: float, level: float) -> tuple[float, float, float, str]:
        """The X at which the line reaches ``target``, with Fieller's limits: (X, lower, upper, remark)."""
        if self.slope == 0:
            return float("nan"), float("nan"), float("nan"), "the slope is zero"
        t2 = float(stats.t.ppf(0.5 + level / 2, self.df)) ** 2 if self.df > 0 else float("nan")
        if self.through is None:
            a, b, shift = self.intercept, self.slope, 0.0
            vaa, vab, vbb = self.cov[0, 0], self.cov[0, 1], self.cov[1, 1]
        else:   # in u = X - (the forced X): the height there is known exactly
            a, b, shift = self.through[1], self.slope, self.through[0]
            vaa, vab, vbb = 0.0, 0.0, self.cov[0, 0]
        estimate = (target - a) / b + shift
        qa = b * b - t2 * vbb
        qb = 2 * (b * (a - target) - t2 * vab)
        qc = (a - target) ** 2 - t2 * vaa
        disc = qb * qb - 4 * qa * qc
        if not np.isfinite(t2) or qa <= 0 or disc < 0:
            return estimate, float("nan"), float("nan"), "no limits: the slope is not clearly different from zero"
        root = np.sqrt(disc)
        return estimate, float((-qb - root) / (2 * qa)) + shift, float((-qb + root) / (2 * qa)) + shift, ""

    def influence(self):
        """Leverage, internally and externally studentised residuals, and Cook's distance of each point."""
        h = np.clip(self.leverage, 0, 1 - 1e-12)
        s = np.sqrt(self.mse)
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = self.residuals / (s * np.sqrt(1 - h))
            outer = inner * np.sqrt(np.clip((self.df - 1) / np.clip(self.df - inner ** 2, 1e-12, None), 0, None))
            cook = inner ** 2 * h / (self.p * (1 - h))
        return h, inner, outer, cook


def lack_of_fit(x: np.ndarray, y: np.ndarray, through: tuple[float, float] | None):
    """The scatter about the line weighed against the scatter among points
    that share an X: (F, df lack of fit, df pure error, P), or a string
    saying why the test cannot be run."""
    levels, where = np.unique(x, return_inverse=True)
    line = Line(x, y, through)
    df_pure, df_lack = x.size - levels.size, levels.size - line.p
    if df_pure < 1:
        return "no X value is repeated, so there is no scatter among repeats to compare the line with"
    if df_lack < 1:
        return "there are too few distinct X values to tell a line from any other shape"
    means = np.array([np.mean(y[where == k]) for k in range(levels.size)])
    pure = float(np.sum((y - means[where]) ** 2))
    lack = max(line.ss_residual - pure, 0.0)
    if pure <= 0:
        return "the repeats at each X are identical, so their scatter cannot be measured"
    f = (lack / df_lack) / (pure / df_pure)
    return f, df_lack, df_pure, float(stats.f.sf(f, df_lack, df_pure))


def theil_sen(x: np.ndarray, y: np.ndarray, level: float):
    """The Theil-Sen line: (slope, intercept, lower and upper limit of the slope, points used)."""
    if x.size > MOST_FOR_PAIRS:
        keep = np.linspace(0, x.size - 1, MOST_FOR_PAIRS).astype(int)
        order = np.argsort(x, kind="stable")
        x, y = x[order][keep], y[order][keep]
    slope, intercept, low, high = stats.theilslopes(y, x, alpha=level)
    return float(slope), float(intercept), float(low), float(high), int(x.size)


@register("linear_regression")
def run(data: SheetData, options: LinearRegressionOptions | None = None) -> AnalysisResult:
    opt = options or LinearRegressionOptions()

    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    mode = "average" if opt.replicate_handling == "average" else "pooled"
    # each Y with the X of its own row: a blank cell drops that point and moves nothing else
    x, y = data.xy_points(labels[0], "average" if opt.replicate_handling == "average" else "each")

    lo, hi = -np.inf, np.inf
    if opt.range_mode_start == "manual" or opt.range_mode_end == "manual":
        lo = opt.range_start_x if opt.range_mode_start == "manual" else -np.inf
        hi = opt.range_end_x if opt.range_mode_end == "manual" else np.inf
        range_mask = (x >= lo) & (x <= hi)
        x, y = x[range_mask], y[range_mask]

    if x.size < 3:
        raise AnalysisError(f"A straight-line fit needs at least 3 points with both an X and a Y; found {x.size}.")

    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{level * 100:g}%"
    through = (opt.force_x, opt.force_y) if opt.force_point else None
    line = Line(x, y, through)
    slope, intercept, df = line.slope, line.intercept, line.df
    t_crit = float(stats.t.ppf(0.5 + level / 2, df)) if df > 0 else float("nan")
    slope_se, intercept_se = line.slope_se(), line.intercept_se()
    p_value = float(2 * stats.t.sf(abs(slope / slope_se), df)) if df > 0 and slope_se > 0 else float("nan")
    notes = []

    def span(mid: float, se: float) -> str:
        return f"{mid - t_crit * se:.{sig}g} to {mid + t_crit * se:.{sig}g}" if np.isfinite(t_crit) and np.isfinite(se) else "n/a"

    intercept_p = (float(2 * stats.t.sf(abs(intercept / intercept_se), df)) if not opt.force_point and df > 0 and intercept_se > 0
                   else float("nan"))
    tables = [ResultTable(
        title="Y = slope · X" + ("" if not opt.force_point else f" (forced through X={opt.force_x:g}, Y={opt.force_y:g})") + " + intercept",
        headers=["Parameter", "Estimate", "Std. error", f"{pct} CI", "P value (against 0)"],
        rows=[["Slope", f"{slope:.{sig}g}", f"{slope_se:.{sig}g}", span(slope, slope_se), f"{p_value:.4g}"],
              ["Intercept", f"{intercept:.{sig}g}"] + (["fixed", "fixed", "-"] if opt.force_point else
                                                       [f"{intercept_se:.{sig}g}", span(intercept, intercept_se), f"{intercept_p:.4g}"])])]
    tables.append(ResultTable(
        title="How close the points lie to the line",
        headers=["R²", "Residual SD", "Points", "P value (slope ≠ 0)"],
        rows=[[f"{line.r_squared:.4f}", f"{np.sqrt(line.mse):.{sig}g}" if df > 0 else "n/a", str(x.size), f"{p_value:.4g}"]]))

    # ---- is a straight line the right shape?
    if opt.runs_test:
        rt = _runs_test(line.residuals[np.argsort(x, kind="stable")])
        if rt:
            runs, p = rt
            tables.append(ResultTable(
                title="Do the points wander either side of the line in runs?",
                headers=["Runs", "P value", "Reading"],
                rows=[[str(runs), f"{p:.4g}", "Too few runs for a line: the truth is probably curved" if p < 0.05 else
                       "No sign of a curve"]]))
        else:
            notes.append("The runs test needs at least 4 points off the line.")
    if opt.lack_of_fit:
        # every repeat counts as a point here, however the line itself was fitted
        each_x, each_y = data.xy_points(labels[0], "each")
        inside = (each_x >= lo) & (each_x <= hi)
        outcome = lack_of_fit(each_x[inside], each_y[inside], through)
        if isinstance(outcome, str):
            notes.append(f"The lack-of-fit test was not run: {outcome}.")
        else:
            f, df_lack, df_pure, p = outcome
            tables.append(ResultTable(
                title="Is the scatter about the line more than the scatter among repeats?",
                headers=["F (lack of fit)", "df", "P value", "Reading"],
                rows=[[f"{f:.4g}", f"{df_lack}, {df_pure}", f"{p:.4g}",
                       "The points miss the line by more than repeats differ: a line is the wrong shape" if p < 0.05 else
                       "A line fits as well as the repeats allow"]]))
            if opt.replicate_handling == "average" and each_x.size != x.size:
                notes.append("The lack-of-fit test uses every repeat as its own point, since it needs the scatter among repeats; "
                             "the line itself was fitted to the means.")

    # ---- which points pull the line?
    if (opt.influence or opt.influence_plot) and df > 1:
        leverage, inner, outer, cook = line.influence()
        if opt.influence:
            order = np.argsort(-cook)[:10]
            flagged = int(np.sum(cook > 4 / x.size))
            tables.append(ResultTable(
                title=f"The points that pull the line hardest ({flagged} with Cook's distance above 4/n = {4 / x.size:.3g})",
                headers=["X", "Y", "Residual", "Studentised residual", "Leverage", "Cook's distance"],
                rows=[[f"{x[i]:.{sig}g}", f"{y[i]:.{sig}g}", f"{line.residuals[i]:.{sig}g}", f"{outer[i]:.{sig}g}",
                       f"{leverage[i]:.{sig}g}", f"{cook[i]:.{sig}g}"] for i in order]))
            worst = int(np.nanargmax(np.abs(outer)))
            adjusted = min(1.0, x.size * 2 * float(stats.t.sf(abs(outer[worst]), df - 1)))
            tables.append(ResultTable(
                title="Is the point furthest from the line an outlier?",
                headers=["X", "Y", "Studentised residual", "P value, allowing for the choice of the worst point", "Reading"],
                rows=[[f"{x[worst]:.{sig}g}", f"{y[worst]:.{sig}g}", f"{outer[worst]:.{sig}g}", f"{adjusted:.4g}",
                       "Further off than chance would put any point" if adjusted < 0.05 else "Within what chance allows"]]))
        if opt.influence_plot:
            charts_influence = ChartSpec(
                title="How hard each point pulls the line",
                series=[ChartSeries("Points", x.tolist(), cook.tolist(), kind="scatter", color=BLUE)],
                x_label="X", y_label="Cook's distance", reference_lines=[{"kind": "h", "value": 4 / x.size, "style": ":"}])
        else:
            charts_influence = None
    else:
        charts_influence = None
        if opt.influence or opt.influence_plot:
            notes.append("Influence needs at least two more points than the line has parameters.")

    # ---- would a line that ignores odd points agree?
    robust = None
    if opt.robust_line:
        if opt.force_point:
            notes.append("The robust line is not forced through a point, so it was not worked out for this fit.")
        else:
            r_slope, r_intercept, r_low, r_high, used = theil_sen(x, y, level)
            robust = (r_slope, r_intercept)
            agrees = slope - t_crit * slope_se <= r_slope <= slope + t_crit * slope_se
            tables.append(ResultTable(
                title="A line that ignores odd points (Theil-Sen)",
                headers=["Slope", f"{pct} CI", "Intercept", "Against the least-squares line"],
                rows=[[f"{r_slope:.{sig}g}", f"{r_low:.{sig}g} to {r_high:.{sig}g}", f"{r_intercept:.{sig}g}",
                       "Its slope lies inside the least-squares interval: the two agree" if agrees else
                       "Its slope lies outside the least-squares interval: a few points are steering the fit"]]))
            if used < x.size:
                notes.append(f"The robust line compares every pair of points; with {x.size} points it used an even selection "
                             f"of {used}.")

    # ---- what does the line predict?
    if opt.ci_y_at_x:
        x0 = opt.ci_y_at_x_value
        y0 = float(line.predict(x0))
        se_line = float(line.se_mean(x0)[0]) if df > 0 else float("nan")
        se_new = float(np.sqrt(se_line ** 2 + line.mse)) if df > 0 else float("nan")
        tables.append(ResultTable(
            title=f"Y when X = {x0:g}, with its {pct} CI",
            headers=["Predicted Y", f"{pct} CI of the line there", f"{pct} interval for one new point"],
            rows=[[f"{y0:.{sig}g}", span(y0, se_line), span(y0, se_new)]]))
        if not float(np.min(x)) <= x0 <= float(np.max(x)):
            notes.append(f"X = {x0:g} lies outside the X values fitted: that prediction is an extrapolation.")

    if opt.ci_x_at_y:
        y0 = opt.ci_x_at_y_value
        x0, x_lo, x_hi, remark = line.x_at(y0, level)
        if remark == "the slope is zero":
            notes.append("X cannot be read off a flat line: the slope is zero.")
        else:
            tables.append(ResultTable(
                title=f"X when Y = {y0:g}, with its {pct} CI",
                headers=["Predicted X", f"{pct} CI (Fieller)"],
                rows=[[f"{x0:.{sig}g}", remark or f"{min(x_lo, x_hi):.{sig}g} to {max(x_lo, x_hi):.{sig}g}"]]))

    if opt.residual_table:
        tables.append(ResultTable(
            title="Residuals", headers=["X", "Y", "Fitted", "Residual"],
            rows=[[f"{xi:.4g}", f"{yi:.4g}", f"{fi:.4g}", f"{ri:.4g}"] for xi, yi, fi, ri in zip(x, y, line.fitted, line.residuals)]))

    if opt.interpolate_unknowns and len(labels) > 1:
        unknown_y = data.group_values(labels[1], mode=mode)
        if slope == 0:
            notes.append("X cannot be read off a flat line: the slope is zero.")
        else:
            rows = [[f"{yv:.4g}", f"{(yv - intercept) / slope:.{sig}g}"] for yv in unknown_y]
            tables.append(ResultTable(title=f"X read off the line for each Y in '{labels[1]}'", headers=["Y", "X"], rows=rows))

    summary = f"Y = {slope:.{sig}g}·X + {intercept:.{sig}g}  (R² = {line.r_squared:.4f}, p = {p_value:.4g})"

    # ---- graphs: the fit, with its band; the residuals; the pull of each point
    charts = []
    if opt.graph or opt.show_confidence_band:
        grid_x = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        line_y = line.predict(grid_x)
        series = []
        if opt.show_confidence_band and df > 0 and np.isfinite(line.mse):
            se_band = line.se_mean(grid_x)
            if opt.band_type == "prediction":   # a new point adds its own scatter
                se_band = np.sqrt(se_band ** 2 + line.mse)
            band_t = stats.t.ppf(0.5 + opt.confidence_band_level / 2, df)
            kind_word = "prediction" if opt.band_type == "prediction" else "confidence"
            series.append(ChartSeries(f"{opt.confidence_band_level * 100:g}% {kind_word} band",
                                      grid_x.tolist(), (line_y - band_t * se_band).tolist(), kind="band",
                                      y_upper=(line_y + band_t * se_band).tolist(), color=BLUE))
        series.append(ChartSeries(labels[0], x.tolist(), y.tolist(), kind="scatter", color=BLUE))
        series.append(ChartSeries("Best-fit line", grid_x.tolist(), line_y.tolist(), kind="line", color=RED))
        if robust is not None:
            series.append(ChartSeries("Robust line (Theil-Sen)", grid_x.tolist(), (robust[0] * grid_x + robust[1]).tolist(),
                                      kind="line", color=GREEN))
        charts.append(ChartSpec(title="Best-fit line", series=series, x_label="X", y_label=labels[0]))
    if opt.residual_table:
        charts.append(ChartSpec(title="Residuals against fitted values",
                                series=[ChartSeries("Residuals", line.fitted.tolist(), line.residuals.tolist(), kind="scatter")],
                                x_label="Fitted Y", y_label="Residual",
                                reference_lines=[{"kind": "h", "value": 0.0}]))
    if charts_influence is not None:
        charts.append(charts_influence)

    return AnalysisResult(title="Straight-line fit", summary=summary, tables=tables, notes=notes, charts=charts)
