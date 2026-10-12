"""Find the peaks in a trace and fit them all at once.

X is the position (wavelength, time, angle); each data set is a trace. The
peaks are found from their prominence, then a sum of peak shapes - Gaussian,
Lorentzian or Voigt - is fitted to the whole trace together with a baseline,
so that overlapping peaks share the signal between them properly.

For each peak: where it is, how high, how wide (full width at half height),
its area and its share of the total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, special, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _signal

SHAPES = {"gaussian": "Gaussian", "lorentzian": "Lorentzian", "voigt": "Voigt"}
BASELINES = {"none": -1, "constant": 0, "linear": 1, "curved": 2}       # degree of the polynomial fitted under the peaks
GAUSS_AREA = float(np.sqrt(np.pi / (4 * np.log(2))))                   # area of a Gaussian of height 1 and width 1
SIGMA_TO_FWHM = float(2 * np.sqrt(2 * np.log(2)))
FIT_POINTS = 20_000        # a trace longer than this is averaged down for the fit and the graphs


@dataclass
class PeakOptions:
    shape: str = "gaussian"           # "gaussian" | "lorentzian" | "voigt"
    direction: str = "up"             # "up" | "down" (dips)
    baseline: str = "linear"          # "none" | "constant" | "linear" | "curved"
    min_height_percent: float = 5.0   # a peak must stand this far out of its surroundings, in % of the trace's range
    max_peaks: int = 8
    fit: bool = True                  # False: only find and measure the peaks
    confidence_level: float = 0.95
    significant_digits: int = 4


# ── shapes: height 1 at the centre ───────────────────────────────────────────
def gaussian(x, centre, width):
    return np.exp(-4 * np.log(2) * ((x - centre) / width) ** 2)


def lorentzian(x, centre, width):
    return 1.0 / (1.0 + 4 * ((x - centre) / width) ** 2)


def voigt(x, centre, sigma, gamma):
    """A Gaussian of SD sigma convolved with a Lorentzian of half-width gamma, scaled to height 1."""
    return special.voigt_profile(x - centre, sigma, gamma) / special.voigt_profile(0.0, sigma, gamma)


def voigt_width(sigma: float, gamma: float) -> float:
    """Full width at half height of a Voigt profile (Olivero and Longbothum 1977, within 0.02%)."""
    lorentz, gauss = 2 * gamma, SIGMA_TO_FWHM * sigma
    return float(0.5346 * lorentz + np.sqrt(0.2166 * lorentz**2 + gauss**2))


def find_peaks(x: np.ndarray, y: np.ndarray, min_percent: float, most: int) -> list[dict]:
    """Peaks of y by prominence - how far each stands out of its
    surroundings - with a first estimate of height and width. The trace is
    lightly smoothed for the search only, so noise is not taken for peaks."""
    n = y.size
    window = int(np.clip(n // 40, 5, 31)) | 1
    smooth = signal.savgol_filter(y, window, 2) if n >= window + 2 else y
    span = float(np.ptp(smooth)) or 1.0
    found, props = signal.find_peaks(smooth, prominence=max(min_percent, 0.01) / 100 * span)
    if not found.size:
        return []
    keep = np.argsort(-props["prominences"])[:most]
    found, prominences = found[keep], props["prominences"][keep]
    widths, _h, left, right = signal.peak_widths(smooth, found, rel_height=0.5)
    positions = np.arange(n)
    out = []
    for i, prominence, lo, hi in zip(found, prominences, left, right):
        width = float(np.interp(hi, positions, x) - np.interp(lo, positions, x))
        out.append({"index": int(i), "centre": float(x[i]), "height": float(prominence), "top": float(y[i]),
                    "width": width if width > 0 else float(np.ptp(x)) / n * 3})
    return sorted(out, key=lambda p: p["centre"])


def _model(shape: str, n_peaks: int, degree: int, x: np.ndarray, x_mid: float, x_half: float):
    """(function(params) -> fitted trace, parameters per peak). The baseline's
    polynomial is in (x - middle) / half-range, so its terms stay of one size."""
    per = 4 if shape == "voigt" else 3
    u = (x - x_mid) / x_half

    def parts(params):
        peaks = []
        for k in range(n_peaks):
            p = params[k * per:(k + 1) * per]
            if shape == "gaussian":
                peaks.append(p[0] * gaussian(x, p[1], p[2]))
            elif shape == "lorentzian":
                peaks.append(p[0] * lorentzian(x, p[1], p[2]))
            else:
                peaks.append(p[0] * voigt(x, p[1], p[2], p[3]))
        base = np.zeros(x.size)
        for d in range(degree + 1):
            base = base + params[n_peaks * per + d] * u**d
        return peaks, base

    return parts, per


def _measures(shape: str, p: np.ndarray) -> np.ndarray:
    """(height, centre, full width at half height, area) of one peak from its fitted parameters."""
    if shape == "gaussian":
        return np.array([p[0], p[1], p[2], p[0] * p[2] * GAUSS_AREA])
    if shape == "lorentzian":
        return np.array([p[0], p[1], p[2], p[0] * p[2] * np.pi / 2])
    return np.array([p[0], p[1], voigt_width(p[2], p[3]), p[0] / special.voigt_profile(0.0, p[2], p[3])])


def fit_peaks(x: np.ndarray, y: np.ndarray, start: list[dict], shape: str, degree: int) -> dict:
    """Least-squares fit of len(start) peaks of ``shape`` plus a polynomial baseline."""
    n_peaks = len(start)
    x_mid, x_half = float((x[0] + x[-1]) / 2), float(np.ptp(x) / 2) or 1.0
    parts, per = _model(shape, n_peaks, degree, x, x_mid, x_half)
    span, y_span = float(np.ptp(x)), float(np.ptp(y)) or 1.0
    step = span / max(x.size - 1, 1)
    p0, lower, upper = [], [], []
    for peak in start:
        width = float(np.clip(peak["width"], 2 * step, span))
        if shape == "voigt":
            p0 += [peak["height"], peak["centre"], width / SIGMA_TO_FWHM * 0.7, width / 2 * 0.4]
            lower += [0.0, x[0], step / 20, step / 2000]
            upper += [10 * y_span, x[-1], span, span]
        else:
            p0 += [peak["height"], peak["centre"], width]
            lower += [0.0, x[0], step / 4]
            upper += [10 * y_span, x[-1], 2 * span]
    floor = float(np.percentile(y, 10))
    for d in range(degree + 1):
        p0.append(floor if d == 0 else 0.0)
        lower.append(-np.inf)
        upper.append(np.inf)
    p0, lower, upper = np.array(p0, dtype=float), np.array(lower), np.array(upper)
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    def residuals(params):
        peaks, base = parts(params)
        return base + np.sum(peaks, axis=0) - y

    u = (x - x_mid) / x_half

    def slopes(params):
        """The derivative of every fitted value with respect to every parameter, worked out exactly."""
        out = np.empty((x.size, params.size))
        for k in range(n_peaks):
            height, centre, width = params[k * 3:(k + 1) * 3]
            d = (x - centre) / width
            if shape == "gaussian":
                g = np.exp(-4 * np.log(2) * d**2)
                out[:, k * 3] = g
                out[:, k * 3 + 1] = height * g * 8 * np.log(2) * d / width
                out[:, k * 3 + 2] = height * g * 8 * np.log(2) * d**2 / width
            else:
                g = 1.0 / (1.0 + 4 * d**2)
                out[:, k * 3] = g
                out[:, k * 3 + 1] = height * g**2 * 8 * d / width
                out[:, k * 3 + 2] = height * g**2 * 8 * d**2 / width
        for d_ in range(degree + 1):
            out[:, n_peaks * 3 + d_] = u**d_
        return out

    scale = np.where(np.isfinite(upper - lower), np.maximum(np.abs(p0), 1e-6 * y_span), np.maximum(np.abs(p0), y_span))
    sol = optimize.least_squares(residuals, p0, jac="2-point" if shape == "voigt" else slopes, bounds=(lower, upper), x_scale=scale,
                                 max_nfev=(60 if shape == "voigt" else 200) * p0.size if x.size <= 5000 else 150)
    dof = max(1, x.size - p0.size)
    rss = float((sol.fun**2).sum())
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * rss / dof
    except np.linalg.LinAlgError:
        cov = np.full((p0.size, p0.size), np.nan)
    peaks, base = parts(sol.x)
    total_ss = float(((y - y.mean()) ** 2).sum())
    rows = []
    for k in range(n_peaks):
        sl = slice(k * per, (k + 1) * per)
        p = sol.x[sl]
        value = _measures(shape, p)
        # standard errors of height, centre, width and area by the delta method
        grad = np.zeros((4, per))
        for j in range(per):
            h = max(abs(p[j]) * 1e-6, 1e-12)
            up, down = p.copy(), p.copy()
            up[j] += h
            down[j] -= h
            grad[:, j] = (_measures(shape, up) - _measures(shape, down)) / (2 * h)
        with np.errstate(invalid="ignore"):
            errors = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", grad, cov[sl, sl], grad), 0, None))
        rows.append({"height": value[0], "centre": value[1], "width": value[2], "area": value[3], "se": errors,
                     "at_limit": bool(np.any(np.isclose(p, lower[sl], rtol=1e-6, atol=0) | np.isclose(p, upper[sl], rtol=1e-6, atol=0))),
                     "gauss_share": (SIGMA_TO_FWHM * p[2]) / (SIGMA_TO_FWHM * p[2] + 2 * p[3]) if shape == "voigt" else None})
    return {"peaks": rows, "components": peaks, "baseline": base, "fitted": base + np.sum(peaks, axis=0), "rss": rss, "dof": dof,
            "r2": 1 - rss / total_ss if total_ss > 0 else float("nan"), "ok": bool(sol.success), "sd": float(np.sqrt(rss / dof))}


@register("peak_fitting")
def run(data: SheetData, options: PeakOptions | None = None) -> AnalysisResult:
    opt = options or PeakOptions()
    sets = _signal.traces(data, least=8)
    if not sets:
        raise AnalysisError("Peaks need a trace: numbers in the X column and at least 8 rows of values in a data set.")
    shape = opt.shape if opt.shape in SHAPES else "gaussian"
    degree = BASELINES.get(opt.baseline, 1)
    sign = -1.0 if opt.direction == "down" else 1.0
    word = "dip" if sign < 0 else "peak"
    sig = opt.significant_digits
    t_level = opt.confidence_level
    pct = f"{round(t_level * 100)}%"

    def num(value) -> str:
        return "" if value is None or not np.isfinite(value) else f"{value:.{sig}g}"

    from graphs.engine.style import PALETTES

    palette = PALETTES["classic"]
    tables, charts, notes, lines = [], [], [], []
    many = len(sets) > 1
    for label, x, y_raw in sets[:12]:
        if np.unique(x).size < x.size:
            notes.append(f"{label}: some X values occur more than once; the trace should have one value at each X.")
        y = sign * y_raw
        found = find_peaks(x, y, opt.min_height_percent, max(1, int(opt.max_peaks)))
        title = f"{label}: " if many else ""
        if not found:
            tables.append(ResultTable(f"{title}no {word} found", ["", ""],
                                      [["Why", f"Nothing stands {opt.min_height_percent:g}% of the trace's range out of its surroundings. "
                                               "Lower that figure, or check the direction."]]))
            lines.append(f"{label}: no {word}")
            charts.append(ChartSpec(f"{label}: the trace", series=[ChartSeries("", *_signal.for_graph(x, y_raw), kind="line", color="#555555")],
                                    x_label="X", y_label="Y", group="peaks", option=label))
            continue

        if x.size > FIT_POINTS:
            # a very long trace: neighbouring points are averaged for the fit and the graphs (the search above used them all)
            group = int(np.ceil(x.size / FIT_POINTS))
            whole = (x.size // group) * group
            x, y, y_raw = (v[:whole].reshape(-1, group).mean(axis=1) for v in (x, y, y_raw))
            notes.append(f"{label}: the trace has more than {FIT_POINTS:,} points, so neighbours were averaged in groups of {group} "
                         "for the fit and the graphs. A peak only a few points wide is blurred by this.")
        thin = slice(None, None, max(1, int(np.ceil(x.size / 4000))))      # every k-th point is enough for a graph
        fit = None
        if opt.fit:
            if x.size <= len(found) * (4 if shape == "voigt" else 3) + degree + 2:
                notes.append(f"{label}: too few points to fit {len(found)} {word}s; they are measured from the trace instead.")
            else:
                try:
                    fit = fit_peaks(x, y, found, shape, degree)
                except Exception as exc:  # noqa: BLE001 - a fit that fails still leaves the peaks that were found
                    notes.append(f"{label}: the fit did not work ({exc}); the {word}s are measured from the trace instead.")
        if fit is None:
            rows = [[str(k + 1), num(p["centre"]), num(sign * p["height"]), num(p["width"]), num(sign * p["top"])] for k, p in enumerate(found)]
            tables.append(ResultTable(f"{title}{word}s found (measured from the trace, not fitted)",
                                      [word.capitalize(), "Position", "Height above its surroundings", "Width at half height", "Y at the top"], rows))
            lines.append(f"{label}: {len(found)} {word}{'s' if len(found) != 1 else ''}")
            charts.append(ChartSpec(f"{label}: the trace and its {word}s",
                                    series=[ChartSeries("Trace", x[thin].tolist(), y_raw[thin].tolist(), kind="line", color="#555555"),
                                            ChartSeries(f"{word.capitalize()}s", [p["centre"] for p in found], [sign * p["top"] for p in found],
                                                        kind="scatter", color="#d1342f")],
                                    x_label="X", y_label="Y", group="peaks", option=label))
            continue

        t_crit = stats.t.ppf(0.5 + t_level / 2, fit["dof"])
        peaks = sorted(fit["peaks"], key=lambda p: p["centre"])
        total_area = sum(p["area"] for p in peaks) or 1.0
        rows = []
        for k, p in enumerate(peaks):
            se = p["se"]
            gap = ""
            if k + 1 < len(peaks):          # how well this peak is parted from the next (chromatographic resolution)
                gap = num(1.18 * (peaks[k + 1]["centre"] - p["centre"]) / (p["width"] + peaks[k + 1]["width"]))
            rows.append([str(k + 1), num(p["centre"]), f"{num(p['centre'] - t_crit * se[1])} to {num(p['centre'] + t_crit * se[1])}",
                         num(sign * p["height"]), num(se[0]), num(p["width"]), num(se[2]), num(sign * p["area"]), num(se[3]),
                         f"{100 * p['area'] / total_area:.1f}", gap])
        headers = [word.capitalize(), "Position", f"{pct} CI of position", "Height", "SE", "Width at half height", "SE", "Area", "SE",
                   "Share of area (%)", "Resolution from the next"]
        tables.append(ResultTable(f"{title}{len(peaks)} {SHAPES[shape]} {word}{'s' if len(peaks) != 1 else ''}, fitted together", headers, rows))
        tables.append(ResultTable(f"{title}how well the fit follows the trace", ["R squared", "Residual SD", "Points", "Degrees of freedom", "Baseline"],
                                  [[f"{fit['r2']:.5f}", num(fit["sd"]), str(x.size), str(fit["dof"]),
                                    {"none": "none (zero)", "constant": "a constant", "linear": "a sloping line", "curved": "a curve (quadratic)"}[opt.baseline
                                     if opt.baseline in BASELINES else "linear"]]]))
        if shape == "voigt":
            tables.append(ResultTable(f"{title}make-up of each Voigt {word}", [word.capitalize(), "Gaussian share of the width (%)", "Reading"],
                                      [[str(k + 1), f"{100 * p['gauss_share']:.0f}",
                                        "mostly Gaussian" if p["gauss_share"] > 0.7 else "mostly Lorentzian" if p["gauss_share"] < 0.3 else "mixed"]
                                       for k, p in enumerate(peaks)]))
        stuck = [str(k + 1) for k, p in enumerate(peaks) if p["at_limit"]]
        if stuck:
            notes.append(f"{label}: {word} {', '.join(stuck)} ended at a limit of the fit (zero height, the edge of the trace, or the "
                         "narrowest or widest width allowed). Its figures are not to be trusted; it may not be a real peak.")
        if not fit["ok"]:
            notes.append(f"{label}: the fit stopped before it settled; treat its figures as rough.")
        biggest = max(peaks, key=lambda p: p["area"])
        lines.append(f"{label}: {len(peaks)} {word}{'s' if len(peaks) != 1 else ''}, the largest at {num(biggest['centre'])} "
                     f"({100 * biggest['area'] / total_area:.0f}% of the area), R² = {fit['r2']:.4f}")

        order = np.argsort([p["centre"] for p in fit["peaks"]])
        series = [ChartSeries("Trace", x[thin].tolist(), y_raw[thin].tolist(), kind="scatter" if x.size <= 150 else "line", color="#7d8590"),
                  ChartSeries("Fit", x[thin].tolist(), (sign * fit["fitted"])[thin].tolist(), kind="line", color="#111111")]
        for rank, k in enumerate(order):
            series.append(ChartSeries(f"{word.capitalize()} {rank + 1}", x[thin].tolist(), (sign * (fit["components"][k] + fit["baseline"]))[thin].tolist(),
                                      kind="line", color=palette[rank % len(palette)]))
        if degree >= 0:
            series.append(ChartSeries("Baseline", x[thin].tolist(), (sign * fit["baseline"])[thin].tolist(), kind="line", color="#b8bec7"))
        charts.append(ChartSpec(f"{label}: the trace, the fit and each {word}", series=series, x_label="X", y_label="Y", group="peaks", option=label))
        charts.append(ChartSpec(f"{label}: what the fit leaves over", x_label="X", y_label="Trace minus fit", group="peak_residuals", option=label,
                                series=[ChartSeries("", x[thin].tolist(), (sign * (y - fit["fitted"]))[thin].tolist(), kind="scatter" if x.size <= 150 else "line",
                                                    color="#1f5aa6")],
                                reference_lines=[{"kind": "h", "value": 0.0}]))

    if len(sets) > 12:
        notes.append(f"Only the first 12 of {len(sets)} data sets were analysed.")
    summary = ". ".join(lines) + "."
    notes += [f"A {word} is found when it stands at least {opt.min_height_percent:g}% of the trace's range out of its surroundings "
              f"(its prominence); at most {int(opt.max_peaks)} are kept, the most prominent first.",
              "All the peaks and the baseline are fitted in one go, so two that overlap are each given their own share. Area is the "
              "area under the fitted shape alone, without the baseline; width is the full width at half height."
              if opt.fit else "The peaks were found and measured but not fitted: widths and heights are read off the trace.",
              {"gaussian": "Gaussian: the shape given by many small random broadenings - typical of chromatography and of instrument-limited lines.",
               "lorentzian": "Lorentzian: sharper at the top with long tails - the natural shape of a resonance or a lifetime-broadened line.",
               "voigt": "Voigt: a Gaussian and a Lorentzian combined, for lines broadened in both ways. It has one parameter more per "
                        "peak, so it needs clean, well-sampled peaks to be pinned down."}[shape],
              "Resolution is 1.18 times the distance between two neighbours over the sum of their widths: above about 1.5 they are "
              "parted to the baseline, under 1 they overlap heavily and their separate areas rest on the shape assumed.",
              "A residual plot with waves in it means the shape or the baseline is wrong, or a peak has been missed."]
    if opt.fit:
        notes.append("Method: nonlinear least squares (trust-region reflective) on all peaks at once; standard errors from the fit's "
                     "covariance, carried to width and area by the delta method.")
    return AnalysisResult(title="Peaks in a trace", summary=summary, tables=tables, notes=notes, charts=charts)
