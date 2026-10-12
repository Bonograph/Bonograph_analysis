"""Curve fitting: least-squares fit of a chosen model to every data set on
the sheet, with a confidence interval for each parameter and a graph of the
fitted curve through the points.

Each data set is fitted on its own, using every replicate value (a
triplicate at each substrate concentration is three points, not one), so a
sheet with one data set per inhibitor concentration gives one Vmax and Km per
concentration side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register


def _linear(x, a, b):
    return a * x + b


def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def _cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def _exp_growth(x, a, b):
    return a * np.exp(b * x)


def _exp_plateau(x, a, b, c):
    return c + a * np.exp(-b * x)


def _sigmoid_4pl(x, bottom, top, ec50, hill):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)


def _sigmoid_5pl(x, bottom, top, ec50, hill, asymmetry):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bottom + (top - bottom) / (1 + (ec50 / x) ** hill) ** asymmetry


def _hyperbola(x, top, half):
    """Y = top·X / (half + X) - Michaelis-Menten, and one-site binding."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return top * x / (half + x)


def _allosteric(x, vmax, k_half, h):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xh = np.power(np.clip(x, 0, None), h)
        return vmax * xh / (np.power(k_half, h) + xh)


def _substrate_inhibition(x, vmax, km, ki):
    with np.errstate(divide="ignore", invalid="ignore"):
        return vmax * x / (km + x * (1 + x / ki))


def _gaussian(x, amplitude, mean, sd):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sd**2))


@dataclass
class _Model:
    key: str
    label: str
    category: str
    equation: str
    param_names: tuple[str, ...]
    fn: Callable
    p0: Callable[[np.ndarray, np.ndarray], tuple]
    # parameters that only make sense positive (a Km, a Vmax) are held there,
    # so noisy data can't return a negative Km that fits but means nothing
    lower: tuple[float, ...] | None = None
    x_label: str = "X"
    y_label: str = "Y"
    log_x: bool = False  # draw on a log axis when the X values span decades


def _p0_linear(x, y):
    return (1.0, float(np.mean(y)))


def _p0_quadratic(x, y):
    return (0.0, 1.0, float(np.mean(y)))


def _p0_cubic(x, y):
    return (0.0, 0.0, 1.0, float(np.mean(y)))


def _p0_exp_growth(x, y):
    sign = 1.0 if np.mean(y) >= 0 else -1.0
    return (sign * max(abs(float(y[0])), 1e-3), 0.1)


def _p0_exp_plateau(x, y):
    return (float(y[0] - y[-1]) or 1.0, 0.1, float(y[-1]))


def _dose_slope_sign(x, y) -> float:
    """+1 when Y rises with dose (stimulation), -1 when it falls (inhibition)."""
    pos = x > 0
    if pos.sum() >= 3 and np.ptp(np.log(x[pos])) > 0:
        return 1.0 if np.corrcoef(np.log(x[pos]), y[pos])[0, 1] >= 0 else -1.0
    return 1.0


def _p0_sigmoid(x, y):
    # standard form: bottom at low dose, top at high dose; Hill > 0 rises, < 0 falls
    return (float(np.min(y)), float(np.max(y)), float(np.median(x)) or 1.0, _dose_slope_sign(x, y))


def _p0_sigmoid5(x, y):
    return (float(np.min(y)), float(np.max(y)), float(np.median(x)) or 1.0, _dose_slope_sign(x, y), 1.0)


def _p0_hyperbola(x, y):
    # Half the top is reached at the K: start from the X whose Y is nearest it.
    top = float(np.max(y)) * 1.2 or 1.0
    positive = x[x > 0]
    half = float(x[np.argmin(np.abs(y - top / 2))]) if positive.size else 1.0
    return (top, half if half > 0 else (float(np.median(positive)) if positive.size else 1.0))


def _p0_allosteric(x, y):
    top, half = _p0_hyperbola(x, y)
    return (top, half, 1.0)


def _p0_substrate_inhibition(x, y):
    top, half = _p0_hyperbola(x, y)
    return (top * 1.5, half, float(np.max(x)) * 2 or 10.0)


def _p0_gaussian(x, y):
    return (float(np.max(y) - np.min(y)), float(x[np.argmax(y)]), float(np.std(x)) or 1.0)


MODELS: dict[str, _Model] = {
    m.key: m
    for m in [
        _Model("michaelis_menten", "Michaelis-Menten", "Enzyme kinetics",
               "Y = Vmax·X / (Km + X)", ("Vmax", "Km"), _hyperbola, _p0_hyperbola,
               lower=(0.0, 0.0), x_label="[Substrate]", y_label="Velocity"),
        _Model("allosteric_sigmoidal", "Cooperative enzyme (Hill)", "Enzyme kinetics",
               "Y = Vmax·X^h / (K′^h + X^h)", ("Vmax", "K′ (half-saturation)", "h (Hill)"), _allosteric,
               _p0_allosteric, lower=(0.0, 0.0, 0.0), x_label="[Substrate]", y_label="Velocity"),
        _Model("substrate_inhibition", "Slowed by excess substrate", "Enzyme kinetics",
               "Y = Vmax·X / (Km + X·(1 + X/Ki))", ("Vmax", "Km", "Ki"), _substrate_inhibition,
               _p0_substrate_inhibition, lower=(0.0, 0.0, 0.0), x_label="[Substrate]", y_label="Velocity"),
        _Model("hyperbola", "Saturation binding, one site (hyperbola)", "Binding",
               "Y = Bmax·X / (Kd + X)", ("Bmax", "Kd"), _hyperbola, _p0_hyperbola,
               lower=(0.0, 0.0), x_label="[Ligand]", y_label="Specific binding"),
        _Model("sigmoid_4pl", "Sigmoid, four parameters (4PL)", "Dose-response",
               "Y = bottom + (top − bottom) / (1 + (EC50/X)^Hill)",
               ("bottom", "top", "EC50", "Hill slope"), _sigmoid_4pl, _p0_sigmoid, log_x=True,
               x_label="Dose", y_label="Response"),
        _Model("sigmoid_5pl", "Lopsided sigmoid, five parameters (5PL)", "Dose-response",
               "Y = bottom + (top − bottom) / (1 + (EC50/X)^Hill)^S",
               ("bottom", "top", "EC50", "Hill slope", "S (asymmetry)"), _sigmoid_5pl, _p0_sigmoid5,
               log_x=True, x_label="Dose", y_label="Response"),
        _Model("linear", "Straight line", "Lines", "Y = a·X + b", ("a", "b"), _linear, _p0_linear),
        _Model("quadratic", "Quadratic", "Polynomials", "Y = a·X² + b·X + c", ("a", "b", "c"), _quadratic, _p0_quadratic),
        _Model("cubic", "Cubic", "Polynomials", "Y = a·X³ + b·X² + c·X + d", ("a", "b", "c", "d"), _cubic, _p0_cubic),
        _Model("exp_growth", "Exponential growth", "Exponential", "Y = a·e^(b·X)", ("a", "b"), _exp_growth, _p0_exp_growth),
        _Model("exp_plateau", "Decay to a plateau", "Exponential", "Y = c + a·e^(−b·X)",
               ("a", "b", "c"), _exp_plateau, _p0_exp_plateau),
        _Model("gaussian", "Bell curve (Gaussian peak)", "Peaks", "Y = amplitude·exp(-(X-mean)²/(2·SD²))",
               ("Amplitude", "Mean", "SD"), _gaussian, _p0_gaussian),
    ]
}

CATEGORIES: dict[str, list[str]] = {}
for _m in MODELS.values():
    CATEGORIES.setdefault(_m.category, []).append(_m.key)


@dataclass
class NonlinearRegressionOptions:
    model: str = "quadratic"
    relative_weighting: bool = False
    range_mode_start: str = "auto"
    range_start_x: float = 0.0
    range_mode_end: str = "auto"
    range_end_x: float = 0.0
    confidence_level: float = 0.95
    significant_digits: int = 4
    # which data sets to fit; empty = every one on the sheet
    data_sets: list[str] = field(default_factory=list)


@dataclass
class _Fit:
    label: str
    x: np.ndarray
    y: np.ndarray
    params: np.ndarray
    errors: np.ndarray
    r_squared: float
    sy_x: float
    df: int


def _data_sets(data: SheetData, opt: NonlinearRegressionOptions) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, X, Y) for each data set, every replicate value as its own
    point, blanks dropped and the fit range applied."""
    lo = opt.range_start_x if opt.range_mode_start == "manual" else -np.inf
    hi = opt.range_end_x if opt.range_mode_end == "manual" else np.inf
    x_all = np.asarray(data.x, dtype=float)
    out = []
    for label in data.group_labels():
        if opt.data_sets and label not in opt.data_sets:
            continue
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.size == 0:
            continue
        n = min(len(x_all), matrix.shape[0])
        xs = np.repeat(x_all[:n, None], matrix.shape[1], axis=1).ravel()
        ys = matrix[:n].ravel()
        keep = np.isfinite(xs) & np.isfinite(ys) & (xs >= lo) & (xs <= hi)
        out.append((label, xs[keep], ys[keep]))
    return out


def _fit_one(model: _Model, label: str, x: np.ndarray, y: np.ndarray, opt) -> _Fit:
    k = len(model.param_names)
    if x.size < k + 1:
        raise AnalysisError(f"'{label}' has {x.size} paired (X, Y) points; fitting "
                            f"{model.label} needs at least {k + 1}.")
    if np.unique(x).size < k:
        raise AnalysisError(f"'{label}' has only {np.unique(x).size} distinct X values; "
                            f"{model.label} has {k} parameters to pin down.")
    sigma = np.where(y != 0, np.abs(y), 1.0) if opt.relative_weighting else None
    p0 = np.asarray(model.p0(x, y), dtype=float)
    bounds = (-np.inf, np.inf)
    if model.lower is not None:
        lower = np.asarray(model.lower, dtype=float)
        p0 = np.maximum(p0, lower + 1e-9)
        bounds = (lower, np.full(k, np.inf))
    try:
        params, cov = curve_fit(model.fn, x, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        raise AnalysisError(f"{model.label} didn't converge on '{label}': {exc}") from None
    fitted = model.fn(x, *params)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    df = max(1, x.size - k)
    return _Fit(label, x, y, params, np.sqrt(np.clip(np.diag(cov), 0, None)),
                1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), float(np.sqrt(ss_res / df)), df)


def _colours(n: int) -> list[str]:
    from graphs.engine.style import PALETTES

    base = PALETTES["classic"]
    return [base[i % len(base)] for i in range(n)]


# Saturation models whose first two parameters are the plateau and the X at
# half of it, drawn with the textbook guides: the plateau, the half-maximum
# drop lines and, for a true hyperbola, the initial slope.
# key -> (plateau symbol, subscript, half-X symbol, subscript, X symbol, draw slope?)
_SATURATION = {
    "michaelis_menten": ("V", "max", "K", "m", "[S]", True),
    "hyperbola": ("B", "max", "K", "d", "[L]", True),
    "allosteric_sigmoidal": ("V", "max", "K", "", "[S]", False),
}

_CURVE_RED = "#d62728"
_SLOPE_BLUE = "#3b5bdb"
_GUIDE_GREY = "#555555"


def _tex(symbol: str, sub: str) -> str:
    return rf"\mathit{{{symbol}}}_{{\mathrm{{{sub}}}}}" if sub else rf"\mathit{{{symbol}}}'"


def _points(fit: _Fit, colour: str, labelled: bool) -> list[ChartSeries]:
    xs, inverse, counts = np.unique(fit.x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=fit.y)
    means = sums / counts
    sq = np.bincount(inverse, weights=(fit.y - means[inverse]) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sds = np.where(counts > 1, np.sqrt(sq / np.maximum(counts - 1, 1)), 0.0)
    return [ChartSeries(fit.label if labelled else "", xs.tolist(), means.tolist(), kind="scatter",
                        color=colour, y_err=sds.tolist() if (sds > 0).any() else None)]


def _curve_chart(model: _Model, fits: list[_Fit]) -> ChartSpec:
    """The points and the fitted curve through them, one colour per data set.

    Where an X value was measured more than once the point shown is the mean
    with an SD bar - the fit itself still used every replicate.
    """
    all_x = np.concatenate([f.x for f in fits])
    positive = all_x[all_x > 0]
    log_x = bool(model.log_x and positive.size == all_x.size and positive.size
                 and positive.max() / positive.min() >= 100)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    if model.key in _SATURATION and x_min >= 0 and x_max > 0:
        return _saturation_chart(model, fits, x_max)
    if log_x:
        grid = np.logspace(np.log10(x_min), np.log10(x_max), 240)
    else:
        start = 0.0 if model.lower is not None and x_min >= 0 else x_min  # saturation curves start at the origin
        grid = np.linspace(start, x_max, 240)

    series = []
    for fit, colour in zip(fits, _colours(len(fits))):
        series += _points(fit, colour, len(fits) > 1)
        curve = model.fn(grid, *fit.params)
        ok = np.isfinite(curve)
        series.append(ChartSeries("", grid[ok].tolist(), curve[ok].tolist(), kind="line", color=colour))

    return ChartSpec(title=f"{model.label} fit", series=series, x_label=model.x_label,
                     y_label=model.y_label, x_log=log_x)


def _saturation_chart(model: _Model, fits: list[_Fit], x_max: float) -> ChartSpec:
    """The Michaelis-Menten picture: the curve rising from the origin to its
    plateau, a dashed line at the plateau (Vmax), dotted drop lines from
    half of it down to the half-saturation constant (Km) and, for a true
    hyperbola, the initial-rate line V = Vmax·[S]/Km. Past the last measured
    X the curve is extrapolated, so that stretch is dashed."""
    top_sym, top_sub, half_sym, half_sub, x_sym, slope = _SATURATION[model.key]
    top_tex, half_tex = _tex(top_sym, top_sub), _tex(half_sym, half_sub)
    single = len(fits) == 1
    colours = [_CURVE_RED] if single else _colours(len(fits))

    tops = [float(f.params[0]) for f in fits]
    halves = [float(f.params[1]) for f in fits]
    far = max(halves)
    x_end = max(x_max, min(5 * far, 4 * x_max)) if far > 0 else x_max
    y_top = 1.18 * max(max(tops), max(float(f.y.max()) for f in fits))

    series, refs, notes = [], [], []
    for fit, colour, top, half in zip(fits, colours, tops, halves):
        points = _points(fit, "#333333" if single else colour, not single)
        if not single:
            points[0].label = rf"{fit.label}  (${top_tex}$ {top:.3g}, ${half_tex}$ {half:.3g})"
        series += points
        measured = np.linspace(0.0, x_max, 240)
        series.append(ChartSeries("", measured.tolist(), model.fn(measured, *fit.params).tolist(),
                                  kind="line", color=colour))
        if x_end > x_max * 1.001:
            beyond = np.linspace(x_max, x_end, 120)
            refs.append({"kind": "path", "points": np.column_stack([beyond, model.fn(beyond, *fit.params)]).tolist(),
                         "color": colour, "style": [0, [4, 3]], "width": 1.6, "zorder": 2})

        refs.append({"kind": "path", "points": [[0.0, top], [x_end * 1.03, top]], "color": colour,
                     "style": [0, [6, 4]], "width": 1.0})
        if 0 < half <= x_end:
            refs.append({"kind": "path", "points": [[0.0, top / 2], [half, top / 2], [half, 0.0]],
                         "color": _GUIDE_GREY if single else colour, "style": [0, [2, 2.5]], "width": 1.0})

        if not single:
            continue
        notes.append({"x": x_end * 1.03, "y": top, "text": rf"${top_tex} = {top:.3g}$", "offset": [-2, 6],
                      "ha": "right", "size": 11, "color": "#222222"})
        if 0 < half <= x_end:
            notes.append({"x": half, "y": top / 2, "text": rf"$\frac{{1}}{{2}}\,{top_tex}$", "offset": [6, -12],
                          "size": 10, "color": "#222222"})
            notes.append({"x": half, "y": 0.0, "text": rf"${half_tex} = {half:.3g}$", "offset": [5, 6],
                          "size": 11, "color": "#222222"})
        if slope and half > 0:
            # the tangent at the origin, drawn until it leaves the plot
            x_stop = min(x_end, half * y_top / top)
            refs.append({"kind": "path", "points": [[0.0, 0.0], [x_stop, top / half * x_stop]],
                         "color": _SLOPE_BLUE, "style": "-", "width": 1.1, "zorder": 2})
            x_lab = x_stop * 0.93
            notes.append({"x": x_lab, "y": top / half * x_lab,
                          "text": rf"$\leftarrow\ \mathit{{{top_sym}}}_0 = \frac{{{top_tex}\,{x_sym}}}{{{half_tex}}}$",
                          "offset": [6, 0], "size": 11, "color": _SLOPE_BLUE, "va": "center"})

    enzyme = model.key == "michaelis_menten"
    title = "Michaelis–Menten curve" if enzyme else f"{model.label} fit"
    y_label = r"Initial velocity, $\mathit{V}_0$" if enzyme else model.y_label
    x_label = r"Substrate, $[\mathit{S}]$" if enzyme else model.x_label
    return ChartSpec(title=title, series=series, x_label=x_label, y_label=y_label,
                     reference_lines=refs, annotations=notes, x_limits=(0.0, x_end * 1.03),
                     y_limits=(0.0, y_top))


def _residual_chart(model: _Model, fits: list[_Fit]) -> ChartSpec:
    series = []
    for fit, colour in zip(fits, _colours(len(fits))):
        residuals = fit.y - model.fn(fit.x, *fit.params)
        series.append(ChartSeries(fit.label if len(fits) > 1 else "", fit.x.tolist(), residuals.tolist(),
                                  kind="scatter", color=colour))
    all_x = np.concatenate([f.x for f in fits])
    log_x = bool(model.log_x and (all_x > 0).all() and all_x.max() / all_x.min() >= 100)
    return ChartSpec(title="Residuals", series=series, x_label=model.x_label, y_label="Residual (Y − fit)",
                     reference_lines=[{"kind": "h", "value": 0.0}], x_log=log_x)


@register("nonlinear_regression")
def run(data: SheetData, options: NonlinearRegressionOptions | None = None) -> AnalysisResult:
    opt = options or NonlinearRegressionOptions()
    model = MODELS.get(opt.model)
    if model is None:
        raise AnalysisError(f"There is no model called '{opt.model}'. Choose one from the list.")

    sets = _data_sets(data, opt)
    if not sets or not any(x.size for _label, x, _y in sets):
        raise AnalysisError("Curve fitting needs numbers in the X column, paired with at least one "
                            "column of Y values.")

    fits, skipped = [], []
    for label, x, y in sets:
        try:
            fits.append(_fit_one(model, label, x, y, opt))
        except AnalysisError as exc:
            skipped.append(str(exc))
    if not fits:
        raise AnalysisError(skipped[0] if len(skipped) == 1 else "No data set could be fitted. " + " ".join(skipped))

    sig = opt.significant_digits
    pct = int(round(opt.confidence_level * 100))
    tables = []
    for fit in fits:
        t_crit = stats.t.ppf(0.5 + opt.confidence_level / 2, fit.df)
        tables.append(ResultTable(
            title=f"{fit.label}: {model.equation}" if len(fits) > 1 else f"{model.label}: {model.equation}",
            headers=["Parameter", "Best fit", "Std. error", f"{pct}% CI"],
            rows=[[name, f"{val:.{sig}g}", f"{err:.{sig}g}",
                   f"{val - t_crit * err:.{sig}g} to {val + t_crit * err:.{sig}g}"]
                  for name, val, err in zip(model.param_names, fit.params, fit.errors)],
        ))
    if len(fits) > 1:
        # the parameters side by side - usually the point of fitting several sets
        tables.insert(0, ResultTable(
            title="Fitted parameters, for each data set",
            headers=["Parameter"] + [f.label for f in fits],
            rows=[[name] + [f"{f.params[i]:.{sig}g} ± {f.errors[i]:.{sig}g}" for f in fits]
                  for i, name in enumerate(model.param_names)],
        ))
    tables.append(ResultTable(
        title="Goodness of fit",
        headers=["Data set", "R²", "Residual SD", "Points", "Degrees of freedom", "Weighting"],
        rows=[[f.label, f"{f.r_squared:.4f}", f"{f.sy_x:.{sig}g}", str(f.x.size), str(f.df),
               "1/Y²" if opt.relative_weighting else "None"] for f in fits],
    ))

    first = fits[0]
    params = ", ".join(f"{n} = {v:.{sig}g}" for n, v in zip(model.param_names, first.params))
    summary = (f"{model.label}: {params}; R² = {first.r_squared:.4f}" if len(fits) == 1
               else f"{model.label} fitted to {len(fits)} data sets. {first.label}: {params}.")

    notes = list(skipped)
    if model.lower is not None:
        notes.append("Parameters were kept positive while fitting; a value sitting at 0 means the data "
                     "could not pin it down.")
    replicated = [f.label for f in fits if np.unique(f.x).size < f.x.size]
    if replicated:
        notes.append("Every replicate value was fitted as its own point; the graph shows their mean ± SD.")

    return AnalysisResult(
        title="Curve fit",
        summary=summary,
        tables=tables,
        notes=notes,
        charts=[_curve_chart(model, fits), _residual_chart(model, fits)],
    )
