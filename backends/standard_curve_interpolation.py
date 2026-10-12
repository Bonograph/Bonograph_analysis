"""Read unknowns off a standard curve: a model list (line through
polynomials and sigmoidal dose-response curves), outlier handling,
optional 1/Y^2 weighting, and a confidence interval on each interpolated
value.

Fits a standard curve to the sheet's first group (X = known standard
concentrations, Y = measured signal), then back-calculates the X
(concentration) for every unknown Y value in the second group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats
from scipy.optimize import brentq, curve_fit

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


def _line(x, a, b):
    return a * x + b


def _semilog_line(x, a, b):
    return a * np.log10(x) + b


def _sigmoid_4pl(x, bottom, top, ec50, hill):
    # divide: a standard at X=0 with a negative trial Hill slope gives 0**-h
    # while the optimizer searches - inf is the correct limit, not an error.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bottom + (top - bottom) / (1 + (x / ec50) ** hill)


def _sigmoid_4pl_logx(logx, bottom, top, log_ec50, hill):
    with np.errstate(over="ignore", invalid="ignore"):
        return bottom + (top - bottom) / (1 + 10 ** ((logx - log_ec50) * hill))


def _sigmoid_5pl(x, bottom, top, ec50, hill, asymmetry):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bottom + (top - bottom) / (1 + (x / ec50) ** hill) ** asymmetry


def _hyperbola(x, vmax, km):
    return vmax * x / (km + x)


def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def _cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


@dataclass
class _Model:
    key: str
    label: str
    n_params: int
    fn: Callable
    p0: Callable[[np.ndarray, np.ndarray], tuple]
    x_domain: str = "x"  # "x" | "logx" - whether fn takes X directly or log10(X)


MODELS: dict[str, _Model] = {
    m.key: m
    for m in [
        _Model("line", "Straight line", 2, _line, lambda x, y: (1.0, float(np.mean(y)))),
        _Model("semilog", "Straight line against log X", 2, _semilog_line, lambda x, y: (1.0, float(np.mean(y))), x_domain="logx"),
        _Model(
            "4pl_x", "Sigmoid, four parameters", 4, _sigmoid_4pl,
            lambda x, y: (float(np.min(y)), float(np.max(y)), float(np.median(x)) or 1.0, 1.0),
        ),
        _Model(
            "4pl_logx", "Sigmoid, four parameters, on log X", 4, _sigmoid_4pl_logx,
            lambda x, y: (float(np.min(y)), float(np.max(y)), float(np.median(x)), 1.0), x_domain="logx",
        ),
        _Model(
            "5pl_x", "Lopsided sigmoid, five parameters", 5, _sigmoid_5pl,
            lambda x, y: (float(np.min(y)), float(np.max(y)), float(np.median(x)) or 1.0, 1.0, 1.0),
        ),
        _Model("hyperbola", "Saturation (hyperbola)", 2, _hyperbola, lambda x, y: (float(np.max(y)), float(np.median(x)) or 1.0)),
        _Model("quadratic", "Quadratic", 3, _quadratic, lambda x, y: (0.0, 1.0, float(np.mean(y)))),
        _Model("cubic", "Cubic", 4, _cubic, lambda x, y: (0.0, 0.0, 1.0, float(np.mean(y)))),
    ]
}


@dataclass
class StandardCurveOptions:
    model: str = "4pl_x"
    outlier_handling: str = "none"  # "none" | "robust" | "eliminate" | "report"
    outlier_q: float = 1.0
    relative_weighting: bool = False
    report_ci: bool = True
    confidence_level: float = 0.95


def _grubbs_outliers(resid: np.ndarray, q_percent: float) -> np.ndarray:
    """Flag the standards that sit far off the curve: points whose residual is an outlier
    by Grubbs' test at alpha = Q/100, run iteratively."""
    flagged = np.zeros(resid.size, dtype=bool)
    remaining = np.arange(resid.size)
    alpha = max(0.0001, q_percent / 100.0)
    while remaining.size >= 3:
        vals = resid[remaining]
        mean, sd = np.mean(vals), np.std(vals, ddof=1)
        if sd == 0:
            break
        dev = np.abs(vals - mean)
        idx = int(np.argmax(dev))
        n = remaining.size
        g = dev[idx] / sd
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        if g > g_crit:
            flagged[remaining[idx]] = True
            remaining = np.delete(remaining, idx)
        else:
            break
    return flagged


@register("standard_curve_interpolation")
def run(data: SheetData, options: StandardCurveOptions | None = None) -> AnalysisResult:
    opt = options or StandardCurveOptions()
    model = MODELS.get(opt.model, MODELS["4pl_x"])
    labels = data.group_labels()
    if len(labels) < 1:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    x_std, y_std = data.paired_xy()
    min_points = model.n_params + 1
    if x_std.size < min_points:
        raise AnalysisError(f"'{model.label}' needs at least {min_points} paired (X, Y) points; found {x_std.size}.")
    if model.x_domain == "logx" and np.any(x_std <= 0):
        raise AnalysisError(f"'{model.label}' needs every standard's X to be > 0 (it fits on log10(X)).")

    fit_x = np.log10(x_std) if model.x_domain == "logx" else x_std
    fit_y = y_std

    notes = []
    if opt.outlier_handling in ("eliminate", "report"):
        try:
            prelim, _ = curve_fit(model.fn, fit_x, fit_y, p0=model.p0(fit_x, fit_y), maxfev=10000)
        except RuntimeError as exc:
            raise AnalysisError(
                f"'{model.label}' didn't converge on this standard curve (while screening for outliers): {exc}"
            ) from None
        resid = fit_y - model.fn(fit_x, *prelim)
        flagged = _grubbs_outliers(resid, opt.outlier_q)
        if flagged.any():
            which = ", ".join(f"X = {v:.4g}" for v in x_std[flagged])
            action = "and left out of the fit" if opt.outlier_handling == "eliminate" else "but kept in the fit"
            notes.append(f"{int(flagged.sum())} standard(s) flagged as outliers ({which}), {action} "
                         f"(Grubbs' test on the residuals, α = {opt.outlier_q:g}%).")
            if opt.outlier_handling == "eliminate":
                keep = ~flagged
                fit_x, fit_y, x_std = fit_x[keep], fit_y[keep], x_std[keep]
        else:
            notes.append(f"No standard was flagged as an outlier (Grubbs' test on the residuals, α = {opt.outlier_q:g}%).")

    sigma = None
    if opt.relative_weighting:
        sigma = np.where(fit_y != 0, np.abs(fit_y), 1.0)

    fit_kwargs = dict(p0=model.p0(fit_x, fit_y), sigma=sigma, absolute_sigma=False, maxfev=20000)
    if opt.outlier_handling == "robust":
        # the robust loss needs the scale of a typical residual (MAD of a first fit), not
        # the spread of Y itself - at that size it would behave just like least squares
        try:
            first, _ = curve_fit(model.fn, fit_x, fit_y, p0=model.p0(fit_x, fit_y), maxfev=10000)
            r0 = fit_y - model.fn(fit_x, *first)
            scale = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
        except RuntimeError:
            scale = 0.0
        fit_kwargs.update(method="trf", loss="soft_l1", f_scale=scale if scale > 0 else (float(np.std(fit_y)) or 1.0))
        notes.append("Robust fit (soft-L1 loss): points far from the curve count for less, so a single bad "
                     "standard pulls it less.")
    try:
        popt, pcov = curve_fit(model.fn, fit_x, fit_y, **fit_kwargs)
    except RuntimeError as exc:
        raise AnalysisError(f"'{model.label}' didn't converge on this standard curve: {exc}") from None

    fitted = model.fn(fit_x, *popt)
    ss_res = float(np.sum((fit_y - fitted) ** 2))
    ss_tot = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
    resid_sd = np.sqrt(ss_res / max(1, fit_x.size - model.n_params))

    domain_lo, domain_hi = float(np.min(fit_x)), float(np.max(fit_x))

    def forward(v: float) -> float:
        return float(model.fn(v, *popt))

    def inverse(y_val: float) -> float:
        lo_y, hi_y = forward(domain_lo), forward(domain_hi)
        span_lo, span_hi = min(lo_y, hi_y), max(lo_y, hi_y)
        if not (span_lo <= y_val <= span_hi):
            raise AnalysisError(f"Y = {y_val:.4g} is outside the standard curve's range ({span_lo:.4g} to {span_hi:.4g}).")
        fit_val = brentq(lambda v: forward(v) - y_val, domain_lo, domain_hi)
        return 10**fit_val if model.x_domain == "logx" else fit_val

    if len(labels) < 2:
        raise AnalysisError("Add a second group column with the unknown samples' measured Y values to interpolate.")
    unknown_y = data.group_values(labels[1])
    if unknown_y.size == 0:
        raise AnalysisError(f"'{labels[1]}' has no values to interpolate.")

    pct = int(round(opt.confidence_level * 100))
    headers = ["Unknown Y (signal)", "Interpolated X (concentration)"]
    if opt.report_ci:
        headers.append(f"{pct}% CI (approx.)")

    rows = []
    for y_val in unknown_y:
        try:
            x_val = inverse(float(y_val))
            row = [f"{y_val:.4g}", f"{x_val:.4g}"]
            if opt.report_ci:
                # Approximate CI via local linearization: dx = dy / slope.
                eps = (domain_hi - domain_lo) * 1e-4 or 1e-6
                slope = (forward(min(domain_hi, (np.log10(x_val) if model.x_domain == "logx" else x_val) + eps))
                         - forward(max(domain_lo, (np.log10(x_val) if model.x_domain == "logx" else x_val) - eps))) / (2 * eps)
                if slope:
                    z = stats.norm.ppf(0.5 + opt.confidence_level / 2)
                    half_fit = z * resid_sd / abs(slope)
                    fit_domain_val = np.log10(x_val) if model.x_domain == "logx" else x_val
                    lo = fit_domain_val - half_fit
                    hi = fit_domain_val + half_fit
                    lo_x = 10**lo if model.x_domain == "logx" else lo
                    hi_x = 10**hi if model.x_domain == "logx" else hi
                    row.append(f"{min(lo_x, hi_x):.4g} to {max(lo_x, hi_x):.4g}")
                else:
                    row.append("n/a")
            rows.append(row)
        except AnalysisError as exc:
            row = [f"{y_val:.4g}", f"n/a ({exc})"]
            if opt.report_ci:
                row.append("-")
            rows.append(row)

    table = ResultTable(title=f"Standard curve: {model.label} (R² = {r_squared:.4f})", headers=headers, rows=rows)
    summary = f"{unknown_y.size} unknown(s) interpolated from a {model.label.lower()} fit ({x_std.size} standards, R² = {r_squared:.4f})."

    return AnalysisResult(title="Read off a standard curve", summary=summary, tables=[table], notes=notes)
