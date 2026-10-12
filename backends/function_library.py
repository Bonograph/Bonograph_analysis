"""The equation library behind "Draw an equation" and "Simulated XY
data": closed-form formulas grouped into this app's own families (lines,
polynomials, exponentials, growth, dose-response, binding, enzyme
kinetics, peaks, waves, radiation survival). Nothing here is fitted -
the analyses tabulate Y at given parameter values over an X range.

Equations that need an iterative or implicit solution (tight binding,
segmental regression and the like) are deliberately left out rather
than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EquationSpec:
    key: str
    label: str
    category: tuple[str, ...]
    equation: str
    params: tuple[tuple[str, float], ...]  # (name, default)
    fn: Callable[..., np.ndarray]
    x_label: str = "X"


def _safe_pow(base: np.ndarray, exp: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.power(np.clip(base, 1e-300, None), exp)


# ---------------- Lines ----------------

def _straight_line(x, slope, intercept):
    return slope * x + intercept


def _line_through_point(x, slope, x0, y0):
    return y0 + slope * (x - x0)


def _line_through_origin(x, slope):
    return slope * x


def _horizontal_line(x, y0):
    return np.full_like(x, y0)


def _semilog_line_x_log(x, slope, intercept):
    with np.errstate(invalid="ignore"):
        return slope * np.log10(x) + intercept


def _semilog_line_y_log(x, slope, intercept):
    return 10 ** (slope * x + intercept)


def _log_log_line(x, slope, intercept):
    with np.errstate(invalid="ignore"):
        return 10 ** (slope * np.log10(x) + intercept)


# ---------------- Polynomial ----------------

def _poly(coeffs_desc_order):
    def fn(x, *coeffs):
        return np.polyval(list(reversed(coeffs)), x)
    return fn


def _polynomial_fn(x, *coeffs):
    # coeffs given in increasing-power order (a0, a1, a2, ...)
    result = np.zeros_like(x, dtype=float)
    for power, c in enumerate(coeffs):
        result = result + c * x**power
    return result


# ---------------- Exponential ----------------

def _one_phase_decay(x, y0, plateau, k):
    return plateau + (y0 - plateau) * np.exp(-k * x)


def _two_phase_decay(x, y0, plateau, pct_fast, k_fast, k_slow):
    frac = pct_fast / 100.0
    span = y0 - plateau
    return plateau + span * (frac * np.exp(-k_fast * x) + (1 - frac) * np.exp(-k_slow * x))


def _three_phase_decay(x, y0, plateau, pct1, k1, pct2, k2, k3):
    f1, f2 = pct1 / 100.0, pct2 / 100.0
    f3 = max(0.0, 1 - f1 - f2)
    span = y0 - plateau
    return plateau + span * (f1 * np.exp(-k1 * x) + f2 * np.exp(-k2 * x) + f3 * np.exp(-k3 * x))


def _plateau_then_decay(x, y0, plateau, k, x0):
    return np.where(x < x0, y0, plateau + (y0 - plateau) * np.exp(-k * (x - x0)))


def _one_phase_association(x, y0, plateau, k):
    return y0 + (plateau - y0) * (1 - np.exp(-k * x))


def _two_phase_association(x, y0, plateau, pct_fast, k_fast, k_slow):
    frac = pct_fast / 100.0
    span = plateau - y0
    return y0 + span * (frac * (1 - np.exp(-k_fast * x)) + (1 - frac) * (1 - np.exp(-k_slow * x)))


def _plateau_then_association(x, y0, plateau, k, x0):
    return np.where(x < x0, y0, y0 + (plateau - y0) * (1 - np.exp(-k * (x - x0))))


def _exponential_growth(x, y0, k):
    return y0 * np.exp(k * x)


# ---------------- Gaussian / Lorentzian ----------------

def _gaussian_fn(x, amplitude, mean, sd):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sd**2))


def _sum_two_gaussians(x, amp1, mean1, sd1, amp2, mean2, sd2):
    return _gaussian_fn(x, amp1, mean1, sd1) + _gaussian_fn(x, amp2, mean2, sd2)


def _lognormal(x, amplitude, mean_log, sd_log):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, amplitude * np.exp(-((np.log(x) - mean_log) ** 2) / (2 * sd_log**2)), 0.0)


def _cumulative_gaussian_fraction(x, mean, sd):
    from scipy.stats import norm
    return norm.cdf(x, loc=mean, scale=sd)


def _cumulative_gaussian_percent(x, mean, sd):
    return 100.0 * _cumulative_gaussian_fraction(x, mean, sd)


def _cumulative_gaussian_counts(x, n, mean, sd):
    return n * _cumulative_gaussian_fraction(x, mean, sd)


def _lorentzian(x, amplitude, center, width):
    return amplitude / (1 + ((x - center) / (width / 2)) ** 2)


def _sum_two_lorentzians(x, amp1, c1, w1, amp2, c2, w2):
    return _lorentzian(x, amp1, c1, w1) + _lorentzian(x, amp2, c2, w2)


# ---------------- Sine waves ----------------

def _sine_wave(x, amplitude, freq, phase):
    return amplitude * np.sin(2 * np.pi * freq * x + phase)


def _sine_wave_baseline(x, amplitude, freq, phase, baseline):
    return baseline + _sine_wave(x, amplitude, freq, phase)


def _damped_sine_wave(x, amplitude, freq, phase, decay):
    return amplitude * np.exp(-decay * x) * np.sin(2 * np.pi * freq * x + phase)


def _sinc(x, amplitude, freq):
    with np.errstate(invalid="ignore"):
        arg = 2 * np.pi * freq * x
        return amplitude * np.where(arg == 0, 1.0, np.sin(arg) / arg)


# ---------------- Growth curves ----------------

def _exponential_malthusian(x, y0, k):
    return y0 * np.exp(k * x)


def _exponential_plateau_growth(x, y0, plateau, k):
    return plateau - (plateau - y0) * np.exp(-k * x)


def _gompertz(x, y0, plateau, k, x0):
    with np.errstate(over="ignore"):
        return y0 + (plateau - y0) * np.exp(-np.exp(-k * (x - x0)))


def _logistic_growth(x, y0, plateau, k, x0):
    with np.errstate(over="ignore"):
        return y0 + (plateau - y0) / (1 + np.exp(-k * (x - x0)))


# ---------------- Dose-response ----------------

def _dr_4pl(x, bottom, top, logec50, hill):
    with np.errstate(over="ignore", invalid="ignore"):
        return bottom + (top - bottom) / (1 + 10 ** ((logec50 - x) * hill))


def _dr_4pl_conc(x, bottom, top, ec50, hill):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return bottom + (top - bottom) / (1 + _safe_pow(ec50 / np.clip(x, 1e-300, None), hill))


def _dr_3pl(x, top, logec50, hill):
    return _dr_4pl(x, 0.0, top, logec50, hill)


def _dr_normalized(x, logec50, hill):
    with np.errstate(over="ignore"):
        return 100.0 / (1 + 10 ** ((logec50 - x) * hill))


def _dr_inhibition_4pl(x, bottom, top, logic50, hill):
    with np.errstate(over="ignore"):
        return bottom + (top - bottom) / (1 + 10 ** ((x - logic50) * hill))


def _asymmetric_5pl(x, bottom, top, logec50, hill, s):
    with np.errstate(over="ignore"):
        return bottom + (top - bottom) / (1 + 10 ** ((logec50 - x) * hill)) ** s


def _biphasic(x, bottom, span1, logec50_1, hill1, span2, logec50_2, hill2):
    with np.errstate(over="ignore"):
        return bottom + span1 / (1 + 10 ** ((logec50_1 - x) * hill1)) + span2 / (1 + 10 ** ((logec50_2 - x) * hill2))


def _bell_shaped(x, bottom, top, logec50_rise, hill_rise, logec50_fall, hill_fall):
    with np.errstate(over="ignore"):
        rise = 1 / (1 + 10 ** ((logec50_rise - x) * hill_rise))
        fall = 1 / (1 + 10 ** ((x - logec50_fall) * hill_fall))
        return bottom + (top - bottom) * rise * fall


# ---------------- Binding ----------------

def _one_site_specific(x, bmax, kd):
    return bmax * x / (kd + x)


def _one_site_total(x, bmax, kd, ns_slope):
    return bmax * x / (kd + x) + ns_slope * x


def _two_site_specific(x, bmax1, kd1, bmax2, kd2):
    return bmax1 * x / (kd1 + x) + bmax2 * x / (kd2 + x)


def _two_site_total(x, bmax1, kd1, bmax2, kd2, ns_slope):
    return _two_site_specific(x, bmax1, kd1, bmax2, kd2) + ns_slope * x


def _specific_binding_hill(x, bmax, kd, hill):
    return bmax * _safe_pow(x, hill) / (_safe_pow(kd, hill) + _safe_pow(x, hill))


def _one_site_competitive_logic50(x, top, bottom, logic50):
    return _dr_inhibition_4pl(x, bottom, top, logic50, 1.0)


def _two_site_competitive_logic50(x, top, bottom, fraction1, logic50_1, logic50_2):
    with np.errstate(over="ignore"):
        site1 = 1 / (1 + 10 ** (x - logic50_1))
        site2 = 1 / (1 + 10 ** (x - logic50_2))
        return bottom + (top - bottom) * (fraction1 * site1 + (1 - fraction1) * site2)


# ---------------- Enzyme kinetics ----------------

def _michaelis_menten(x, vmax, km):
    return vmax * x / (km + x)


def _substrate_inhibition(x, vmax, km, ki):
    return vmax * x / (km + x * (1 + x / ki))


def _competitive_inhibition_enzyme(x, vmax, km, ki, i):
    return vmax * x / (km * (1 + i / ki) + x)


def _noncompetitive_inhibition_enzyme(x, vmax, km, ki, i):
    return (vmax / (1 + i / ki)) * x / (km + x)


def _uncompetitive_inhibition_enzyme(x, vmax, km, ki, i):
    return (vmax / (1 + i / ki)) * x / (km / (1 + i / ki) + x)


def _allosteric_sigmoidal(x, vmax, k_prime, h):
    return vmax * _safe_pow(x, h) / (_safe_pow(k_prime, h) + _safe_pow(x, h))


# ---------------- Linear-quadratic (cell survival) ----------------

def _linear_quadratic_survival_fraction(x, alpha, beta):
    with np.errstate(over="ignore"):
        return np.exp(-(alpha * x + beta * x**2))


def _linear_quadratic_survival_percent(x, alpha, beta):
    return 100.0 * _linear_quadratic_survival_fraction(x, alpha, beta)


EQUATIONS: dict[str, EquationSpec] = {}


def _add(key, label, category, equation, params, fn, x_label="X"):
    EQUATIONS[key] = EquationSpec(key, label, category, equation, tuple(params), fn, x_label)


# Lines
_add("straight_line", "Straight line", ("Lines",), "Y = slope*X + intercept", [("slope", 1.0), ("intercept", 0.0)], _straight_line)
_add("line_through_point", "Straight line through a chosen point", ("Lines",), "Y = Y0 + slope*(X - X0)", [("slope", 1.0), ("X0", 0.0), ("Y0", 0.0)], _line_through_point)
_add("line_through_origin", "Straight line through the origin", ("Lines",), "Y = slope*X", [("slope", 1.0)], _line_through_origin)
_add("horizontal_line", "Flat line (a constant)", ("Lines",), "Y = Y0", [("Y0", 1.0)], _horizontal_line)
_add("semilog_line_x_log", "Line against log X", ("Lines", "log X"), "Y = slope*log10(X) + intercept", [("slope", 1.0), ("intercept", 0.0)], _semilog_line_x_log)
_add("semilog_line_y_log", "Line on log Y (exponential)", ("Lines", "log Y"), "Y = 10^(slope*X + intercept)", [("slope", 1.0), ("intercept", 0.0)], _semilog_line_y_log)
_add("log_log_line", "Power law (a line on log-log axes)", ("Lines", "log-log"), "Y = 10^(slope*log10(X) + intercept)", [("slope", 1.0), ("intercept", 0.0)], _log_log_line)

# Polynomial
for _order, _name in ((1, "Polynomial, degree 1"), (2, "Quadratic (degree 2)"), (3, "Cubic (degree 3)"), (4, "Polynomial, degree 4"), (5, "Polynomial, degree 5"), (6, "Polynomial, degree 6")):
    _params = [(f"a{i}", 1.0 if i == 1 else 0.0) for i in range(_order + 1)]
    _add(f"poly_{_order}", _name, ("Polynomials", "around 0") if _order > 1 else ("Retired",), "Y = " + " + ".join(f"a{i}*X^{i}" for i in range(_order + 1)), _params, _polynomial_fn)


def _centered_polynomial_fn(x, x0, *coeffs):
    return _polynomial_fn(x - x0, *coeffs)


for _order, _name in ((1, "Polynomial around X0, degree 1"), (2, "Quadratic around X0"), (3, "Cubic around X0"), (4, "Polynomial around X0, degree 4"), (5, "Polynomial around X0, degree 5"), (6, "Polynomial around X0, degree 6")):
    _params = [("X0", 0.0)] + [(f"a{i}", 1.0 if i == 1 else 0.0) for i in range(_order + 1)]
    _add(f"poly_centered_{_order}", _name, ("Polynomials", "around X0") if _order > 1 else ("Retired",), "Y = " + " + ".join(f"a{i}*(X-X0)^{i}" for i in range(_order + 1)), _params, _centered_polynomial_fn)

# Exponential
_add("one_phase_decay", "Decay to a plateau, one rate", ("Exponential", "decay"), "Y = Plateau + (Y0-Plateau)*exp(-k*X)", [("Y0", 100.0), ("Plateau", 0.0), ("k", 0.5)], _one_phase_decay)
_add("plateau_then_decay", "Delay, then decay", ("Exponential", "decay"), "Y0 until X0, then one-phase decay", [("Y0", 100.0), ("Plateau", 0.0), ("k", 0.5), ("X0", 5.0)], _plateau_then_decay)
_add("two_phase_decay", "Decay to a plateau, two rates", ("Exponential", "decay"), "Weighted sum of a fast and slow decay", [("Y0", 100.0), ("Plateau", 0.0), ("Pct1", 50.0), ("k1", 1.0), ("k2", 0.1)], _two_phase_decay)
_add("three_phase_decay", "Decay to a plateau, three rates", ("Exponential", "decay"), "Weighted sum of three decays", [("Y0", 100.0), ("Plateau", 0.0), ("Pct1", 33.0), ("k1", 2.0), ("Pct2", 33.0), ("k2", 0.5), ("k3", 0.05)], _three_phase_decay)
_add("one_phase_association", "Rise to a plateau, one rate", ("Exponential", "rise"), "Y = Y0 + (Plateau-Y0)*(1-exp(-k*X))", [("Y0", 0.0), ("Plateau", 100.0), ("k", 0.5)], _one_phase_association)
_add("plateau_then_association", "Delay, then rise", ("Exponential", "rise"), "Y0 until X0, then one-phase association", [("Y0", 0.0), ("Plateau", 100.0), ("k", 0.5), ("X0", 5.0)], _plateau_then_association)
_add("two_phase_association", "Rise to a plateau, two rates", ("Exponential", "rise"), "Weighted sum of a fast and slow association", [("Y0", 0.0), ("Plateau", 100.0), ("Pct1", 50.0), ("k1", 1.0), ("k2", 0.1)], _two_phase_association)
_add("exponential_growth_eq", "Unlimited exponential growth", ("Growth",), "Y = Y0*exp(k*X)", [("Y0", 1.0), ("k", 0.5)], _exponential_growth)

# Gaussian
_add("gaussian", "Bell curve (Gaussian peak)", ("Peaks & distributions", "peak"), "Y = Amplitude*exp(-(X-Mean)^2/(2*SD^2))", [("Amplitude", 1.0), ("Mean", 0.0), ("SD", 1.0)], _gaussian_fn)
_add("sum_two_gaussians", "Two overlapping bell curves", ("Peaks & distributions", "peak"), "Sum of two Gaussian peaks", [("Amp1", 1.0), ("Mean1", 0.0), ("SD1", 1.0), ("Amp2", 1.0), ("Mean2", 5.0), ("SD2", 1.0)], _sum_two_gaussians)
_add("lognormal", "Skewed peak (lognormal)", ("Peaks & distributions", "peak"), "Y = Amplitude*exp(-(ln(X)-Mean)^2/(2*SD^2))", [("Amplitude", 1.0), ("Mean", 0.0), ("SD", 1.0)], _lognormal)
_add("cumulative_gaussian_pct", "Normal S-curve, as a percent", ("Peaks & distributions", "cumulative"), "Y = 100*Phi((X-Mean)/SD)", [("Mean", 0.0), ("SD", 1.0)], _cumulative_gaussian_percent)
_add("cumulative_gaussian_frac", "Normal S-curve, as a fraction", ("Peaks & distributions", "cumulative"), "Y = Phi((X-Mean)/SD)", [("Mean", 0.0), ("SD", 1.0)], _cumulative_gaussian_fraction)
_add("cumulative_gaussian_counts", "Normal S-curve (cumulative Gaussian)", ("Peaks & distributions", "cumulative"), "Y = N*Phi((X-Mean)/SD)", [("N", 100.0), ("Mean", 0.0), ("SD", 1.0)], _cumulative_gaussian_counts)
_add("lorentzian", "Sharp peak (Lorentzian)", ("Peaks & distributions", "peak"), "Y = Amplitude/(1+((X-Center)/(Width/2))^2)", [("Amplitude", 1.0), ("Center", 0.0), ("Width", 1.0)], _lorentzian)
_add("sum_two_lorentzians", "Two sharp peaks", ("Peaks & distributions", "peak"), "Sum of two Lorentzian peaks", [("Amp1", 1.0), ("Center1", 0.0), ("Width1", 1.0), ("Amp2", 1.0), ("Center2", 5.0), ("Width2", 1.0)], _sum_two_lorentzians)

# Sine waves
_add("sine_wave", "Sine wave", ("Waves",), "Y = Amplitude*sin(2*pi*Freq*X + Phase)", [("Amplitude", 1.0), ("Freq", 1.0), ("Phase", 0.0)], _sine_wave)
_add("damped_sine_wave", "Dying-away sine wave", ("Waves",), "Y = Amplitude*exp(-Decay*X)*sin(2*pi*Freq*X + Phase)", [("Amplitude", 1.0), ("Freq", 1.0), ("Phase", 0.0), ("Decay", 0.1)], _damped_sine_wave)
_add("sinc", "sin(x)/x wave", ("Waves",), "Y = Amplitude*sin(2*pi*Freq*X)/(2*pi*Freq*X)", [("Amplitude", 1.0), ("Freq", 1.0)], _sinc)
_add("sine_wave_baseline", "Sine wave around a baseline", ("Waves",), "Y = Baseline + Amplitude*sin(2*pi*Freq*X + Phase)", [("Amplitude", 1.0), ("Freq", 1.0), ("Phase", 0.0), ("Baseline", 0.0)], _sine_wave_baseline)

# Growth curves
_add("exp_malthusian", "Unlimited exponential growth", ("Growth",), "Y = Y0*exp(k*X)", [("Y0", 1.0), ("k", 0.2)], _exponential_malthusian)
_add("exp_plateau_growth", "Growth slowing to a ceiling", ("Growth",), "Y = Plateau - (Plateau-Y0)*exp(-k*X)", [("Y0", 1.0), ("Plateau", 100.0), ("k", 0.2)], _exponential_plateau_growth)
_add("gompertz_growth", "Gompertz S-curve", ("Growth",), "Y = Y0 + (Plateau-Y0)*exp(-exp(-k*(X-X0)))", [("Y0", 1.0), ("Plateau", 100.0), ("k", 0.3), ("X0", 5.0)], _gompertz)
_add("logistic_growth", "Logistic S-curve", ("Growth",), "Y = Y0 + (Plateau-Y0)/(1+exp(-k*(X-X0)))", [("Y0", 1.0), ("Plateau", 100.0), ("k", 0.5), ("X0", 5.0)], _logistic_growth)


def _beta_growth_then_decay(x, y0, ymax, x0, n):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.clip(x, 1e-300, None) / x0
        return y0 + (ymax - y0) * _safe_pow(ratio, n) * np.exp(n * (1 - ratio))


_add("beta_growth_decay", "Rise and fall (beta function)", ("Growth",), "Y = Y0 + (Ymax-Y0)*(X/X0)^n*exp(n*(1-X/X0))", [("Y0", 1.0), ("Ymax", 100.0), ("X0", 5.0), ("n", 2.0)], _beta_growth_then_decay)

# Dose-response: rising
_add("dr_stim_3pl_log", "Rising sigmoid, slope fixed at 1", ("Dose-response", "log dose"), "Y = Bottom + (Top-Bottom)/(1+10^(logEC50-X))", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0)], lambda x, bottom, top, logec50: _dr_4pl(x, bottom, top, logec50, 1.0), x_label="log(dose)")
_add("dr_stim_4pl_log", "Rising sigmoid, any slope", ("Dose-response", "log dose"), "Y = Bottom + (Top-Bottom)/(1+10^((logEC50-X)*Hill))", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0), ("Hill", 1.0)], _dr_4pl, x_label="log(dose)")
_add("dr_stim_norm_log", "Rising sigmoid from 0 to 100%, slope 1", ("Dose-response", "log dose"), "Y = 100/(1+10^(logEC50-X))", [("logEC50", 0.0)], lambda x, logec50: _dr_normalized(x, logec50, 1.0), x_label="log(dose)")
_add("dr_stim_norm_var_log", "Rising sigmoid from 0 to 100%, any slope", ("Dose-response", "log dose"), "Y = 100/(1+10^((logEC50-X)*Hill))", [("logEC50", 0.0), ("Hill", 1.0)], _dr_normalized, x_label="log(dose)")
_add("dr_stim_4pl_conc", "Rising sigmoid, any slope", ("Dose-response", "dose"), "Y = Bottom + (Top-Bottom)/(1+(EC50/X)^Hill)", [("Bottom", 0.0), ("Top", 100.0), ("EC50", 1.0), ("Hill", 1.0)], _dr_4pl_conc, x_label="dose")

# Dose-response: falling
_add("dr_inhib_3pl_log", "Falling sigmoid, slope fixed at 1", ("Dose-response", "log dose"), "Y = Bottom + (Top-Bottom)/(1+10^((X-logIC50)))", [("Bottom", 0.0), ("Top", 100.0), ("logIC50", 0.0)], lambda x, bottom, top, logic50: _dr_inhibition_4pl(x, bottom, top, logic50, 1.0), x_label="log(dose)")
_add("dr_inhib_4pl_log", "Falling sigmoid, any slope", ("Dose-response", "log dose"), "Y = Bottom + (Top-Bottom)/(1+10^((X-logIC50)*Hill))", [("Bottom", 0.0), ("Top", 100.0), ("logIC50", 0.0), ("Hill", 1.0)], _dr_inhibition_4pl, x_label="log(dose)")
_add("dr_inhib_norm_log", "Falling sigmoid from 100 to 0%, slope 1", ("Dose-response", "log dose"), "Y = 100/(1+10^(X-logIC50))", [("logIC50", 0.0)], lambda x, logic50: _dr_inhibition_4pl(x, 0.0, 100.0, logic50, 1.0), x_label="log(dose)")

# Dose-response: other shapes
_add("dr_asymmetric_5pl_conc", "Lopsided sigmoid (5 parameters)", ("Dose-response", "dose"), "Richards/5PL sigmoid", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0), ("Hill", 1.0), ("S", 1.0)], lambda x, bottom, top, logec50, hill, s: _asymmetric_5pl(np.log10(np.clip(x, 1e-300, None)), bottom, top, logec50, hill, s), x_label="dose")
_add("dr_asymmetric_5pl_log", "Lopsided sigmoid (5 parameters)", ("Dose-response", "log dose"), "Richards/5PL sigmoid", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0), ("Hill", 1.0), ("S", 1.0)], _asymmetric_5pl, x_label="log(dose)")
_add("dr_biphasic_conc", "Two-step sigmoid", ("Dose-response", "dose"), "Sum of two dose-response curves", [("Bottom", 0.0), ("Height1", 50.0), ("logEC50_1", -1.0), ("Hill1", 1.0), ("Height2", 50.0), ("logEC50_2", 1.0), ("Hill2", 1.0)], lambda x, bottom, span1, e1, h1, span2, e2, h2: _biphasic(np.log10(np.clip(x, 1e-300, None)), bottom, span1, e1, h1, span2, e2, h2), x_label="dose")
_add("dr_bell_conc", "Up, then down (bell-shaped response)", ("Dose-response", "dose"), "Rises then falls with concentration", [("Bottom", 0.0), ("Top", 100.0), ("logEC50_rise", -1.0), ("Hill_rise", 1.0), ("logEC50_fall", 1.0), ("Hill_fall", 1.0)], lambda x, *p: _bell_shaped(np.log10(np.clip(x, 1e-300, None)), *p), x_label="dose")

# Binding: saturation
_add("one_site_total", "Saturation, one site, plus nonspecific", ("Binding", "total"), "Y = Bmax*X/(Kd+X) + Nonspecific*X", [("Bmax", 100.0), ("Kd", 1.0), ("Nonspecific", 0.1)], _one_site_total)
_add("one_site_specific", "Saturation, one site (hyperbola)", ("Binding", "specific"), "Y = Bmax*X/(Kd+X)", [("Bmax", 100.0), ("Kd", 1.0)], _one_site_specific)
_add("two_site_specific", "Saturation, two sites", ("Binding", "specific"), "Sum of two one-site binding curves", [("Bmax1", 50.0), ("Kd1", 0.5), ("Bmax2", 50.0), ("Kd2", 5.0)], _two_site_specific)
_add("two_site_total", "Saturation, two sites, plus nonspecific", ("Binding", "total"), "Two-site specific binding plus a linear nonspecific term", [("Bmax1", 50.0), ("Kd1", 0.5), ("Bmax2", 50.0), ("Kd2", 5.0), ("Nonspecific", 0.1)], _two_site_total)
_add("specific_binding_hill", "Saturation with cooperativity (Hill)", ("Binding", "specific"), "Y = Bmax*X^h/(Kd^h+X^h)", [("Bmax", 100.0), ("Kd", 1.0), ("Hill", 1.0)], _specific_binding_hill)

# Binding: competition
_add("one_site_fit_logic50", "Displaced by a competitor, one site", ("Binding", "competition"), "Y = Bottom + (Top-Bottom)/(1+10^(X-logIC50))", [("Top", 100.0), ("Bottom", 0.0), ("logIC50", 0.0)], _one_site_competitive_logic50, x_label="log(competitor)")
_add("two_site_fit_logic50", "Displaced by a competitor, two sites", ("Binding", "competition"), "Weighted sum of two competitive-binding curves", [("Top", 100.0), ("Bottom", 0.0), ("Fraction1", 0.5), ("logIC50_1", -1.0), ("logIC50_2", 1.0)], _two_site_competitive_logic50, x_label="log(competitor)")

# Binding: over time
_add("kinetics_association", "Binding over time", ("Binding", "kinetics"), "Y = Ymax*(1-exp(-k*X))", [("Ymax", 100.0), ("k", 0.3)], lambda x, ymax, k: _one_phase_association(x, 0.0, ymax, k), x_label="time")
_add("kinetics_dissociation", "Unbinding over time", ("Binding", "kinetics"), "Y = Y0*exp(-k*X)", [("Y0", 100.0), ("k", 0.3)], lambda x, y0, k: _one_phase_decay(x, y0, 0.0, k), x_label="time")

# Enzyme kinetics
_add("michaelis_menten", "Michaelis–Menten", ("Enzyme kinetics", "velocity"), "Y = Vmax*X/(Km+X)", [("Vmax", 100.0), ("Km", 1.0)], _michaelis_menten, x_label="[substrate]")
_add("allosteric_sigmoidal", "Cooperative enzyme (Hill)", ("Enzyme kinetics", "velocity"), "Y = Vmax*X^h/(K'^h+X^h)", [("Vmax", 100.0), ("Kprime", 1.0), ("h", 1.5)], _allosteric_sigmoidal, x_label="[substrate]")
_add("substrate_inhibition", "Slowed by excess substrate", ("Enzyme kinetics", "inhibition"), "Y = Vmax*X/(Km+X*(1+X/Ki))", [("Vmax", 100.0), ("Km", 1.0), ("Ki", 10.0)], _substrate_inhibition, x_label="[substrate]")
_add("competitive_inhibition_enz", "Competitive inhibitor", ("Enzyme kinetics", "inhibition"), "Y = Vmax*X/(Km*(1+I/Ki)+X)", [("Vmax", 100.0), ("Km", 1.0), ("Ki", 1.0), ("I", 1.0)], _competitive_inhibition_enzyme, x_label="[substrate]")
_add("noncompetitive_inhibition_enz", "Noncompetitive inhibitor", ("Enzyme kinetics", "inhibition"), "Y = (Vmax/(1+I/Ki))*X/(Km+X)", [("Vmax", 100.0), ("Km", 1.0), ("Ki", 1.0), ("I", 1.0)], _noncompetitive_inhibition_enzyme, x_label="[substrate]")
_add("uncompetitive_inhibition_enz", "Uncompetitive inhibitor", ("Enzyme kinetics", "inhibition"), "Y = (Vmax/(1+I/Ki))*X/(Km/(1+I/Ki)+X)", [("Vmax", 100.0), ("Km", 1.0), ("Ki", 1.0), ("I", 1.0)], _uncompetitive_inhibition_enzyme, x_label="[substrate]")

# Radiation survival (linear-quadratic model)
def _lq_death_fraction(x, alpha, beta):
    return 1.0 - _linear_quadratic_survival_fraction(x, alpha, beta)


def _lq_death_percent(x, alpha, beta):
    return 100.0 * _lq_death_fraction(x, alpha, beta)


def _lq_alive_count(x, n0, alpha, beta):
    return n0 * _linear_quadratic_survival_fraction(x, alpha, beta)


def _lq_dead_count(x, n0, alpha, beta):
    return n0 * _lq_death_fraction(x, alpha, beta)


_add("lq_cell_death_fraction", "Surviving fraction", ("Radiation survival", "survival"), "Y = exp(-(alpha*X + beta*X^2))", [("alpha", 0.2), ("beta", 0.02)], _linear_quadratic_survival_fraction, x_label="dose")
_add("lq_survival_percent", "Surviving cells, percent", ("Radiation survival", "survival"), "Y = 100*exp(-(alpha*X + beta*X^2))", [("alpha", 0.2), ("beta", 0.02)], _linear_quadratic_survival_percent, x_label="dose")
_add("lq_survival_alive_count", "Surviving cells", ("Radiation survival", "survival"), "Y = N0*exp(-(alpha*X + beta*X^2))", [("N0", 1000.0), ("alpha", 0.2), ("beta", 0.02)], _lq_alive_count, x_label="dose")
_add("lq_cell_death_fraction_dead", "Killed fraction", ("Radiation survival", "killed"), "Y = 1 - exp(-(alpha*X + beta*X^2))", [("alpha", 0.2), ("beta", 0.02)], _lq_death_fraction, x_label="dose")
_add("lq_survival_percent_dead", "Killed cells, percent", ("Radiation survival", "killed"), "Y = 100*(1 - exp(-(alpha*X + beta*X^2)))", [("alpha", 0.2), ("beta", 0.02)], _lq_death_percent, x_label="dose")
_add("lq_survival_dead_count", "Killed cells", ("Radiation survival", "killed"), "Y = N0*(1 - exp(-(alpha*X + beta*X^2)))", [("N0", 1000.0), ("alpha", 0.2), ("beta", 0.02)], _lq_dead_count, x_label="dose")

# Keys kept only so settings saved by earlier versions still open (listed as Retired)
_add("classic_one_site_binding", "Saturation, one site (hyperbola)", ("Retired",), "Y = Bmax*X/(Kd+X)", [("Bmax", 100.0), ("Kd", 1.0)], _one_site_specific)
_add("classic_two_site_binding", "Saturation, two sites", ("Retired",), "Sum of two one-site binding curves", [("Bmax1", 50.0), ("Kd1", 0.5), ("Bmax2", 50.0), ("Kd2", 5.0)], _two_site_specific)
_add("classic_sigmoidal_dr", "Rising sigmoid, slope fixed at 1", ("Retired",), "Y = Bottom + (Top-Bottom)/(1+10^(logEC50-X))", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0)], lambda x, bottom, top, logec50: _dr_4pl(x, bottom, top, logec50, 1.0))
_add("classic_sigmoidal_dr_var", "Rising sigmoid, any slope", ("Retired",), "Y = Bottom + (Top-Bottom)/(1+10^((logEC50-X)*Hill))", [("Bottom", 0.0), ("Top", 100.0), ("logEC50", 0.0), ("Hill", 1.0)], _dr_4pl)
_add("classic_boltzmann", "Boltzmann sigmoid", ("Dose-response", "voltage"), "Y = Bottom + (Top-Bottom)/(1+exp((V50-X)/Slope))", [("Bottom", 0.0), ("Top", 100.0), ("V50", 0.0), ("Slope", 1.0)], lambda x, bottom, top, v50, slope: bottom + (top - bottom) / (1 + np.exp((v50 - x) / slope)))
_add("classic_one_phase_decay", "Decay to a plateau, one rate", ("Retired",), "Y = Plateau + (Y0-Plateau)*exp(-k*X)", [("Y0", 100.0), ("Plateau", 0.0), ("k", 0.5)], _one_phase_decay)
_add("classic_two_phase_decay", "Decay to a plateau, two rates", ("Retired",), "Weighted sum of a fast and slow decay", [("Y0", 100.0), ("Plateau", 0.0), ("Pct1", 50.0), ("k1", 1.0), ("k2", 0.1)], _two_phase_decay)
_add("classic_one_phase_assoc", "Rise to a plateau, one rate", ("Retired",), "Y = Y0 + (Plateau-Y0)*(1-exp(-k*X))", [("Y0", 0.0), ("Plateau", 100.0), ("k", 0.5)], _one_phase_association)
_add("classic_two_phase_assoc", "Rise to a plateau, two rates", ("Retired",), "Weighted sum of a fast and slow association", [("Y0", 0.0), ("Plateau", 100.0), ("Pct1", 50.0), ("k1", 1.0), ("k2", 0.1)], _two_phase_association)
_add("classic_exp_growth", "Unlimited exponential growth", ("Retired",), "Y = Y0*exp(k*X)", [("Y0", 1.0), ("k", 0.5)], _exponential_growth)
_add("classic_power_series", "Two power terms, A·X^B + C·X^D", ("Polynomials", "power"), "Y = A*X^B + C*X^D", [("A", 1.0), ("B", 1.0), ("C", 0.0), ("D", 2.0)], lambda x, a, b, c, d: a * _safe_pow(x, b) + c * _safe_pow(x, d))
for _order in (1, 2, 3, 4):
    _params = [(f"a{i}", 1.0 if i == 1 else 0.0) for i in range(_order + 1)]
    _add(f"classic_poly_{_order}", f"Polynomial, degree {_order}", ("Retired",), "Y = " + " + ".join(f"a{i}*X^{i}" for i in range(_order + 1)), _params, _polynomial_fn)
_add("classic_sine_wave", "Sine wave", ("Retired",), "Y = Amplitude*sin(2*pi*Freq*X + Phase)", [("Amplitude", 1.0), ("Freq", 1.0), ("Phase", 0.0)], _sine_wave)
_add("classic_gaussian", "Bell curve (Gaussian peak)", ("Retired",), "Y = Amplitude*exp(-(X-Mean)^2/(2*SD^2))", [("Amplitude", 1.0), ("Mean", 0.0), ("SD", 1.0)], _gaussian_fn)


def category_tree() -> dict:
    """Nested {category_path_segment: {...}} with a leaf list under each
    full category path, for building the Function list tree widget."""
    tree: dict = {}
    for spec in EQUATIONS.values():
        node = tree
        for segment in spec.category:
            node = node.setdefault(segment, {})
        node.setdefault("__leaves__", []).append(spec.key)
    return tree


# Variants that differ from a listed model only in scale (a fraction or a percent of the same
# curve) are no longer listed: the listed model has the scale as a parameter. Their keys stay
# valid for settings saved earlier.
for _key in ("cumulative_gaussian_pct", "cumulative_gaussian_frac", "lq_cell_death_fraction", "lq_survival_percent",
             "lq_cell_death_fraction_dead", "lq_survival_percent_dead"):
    EQUATIONS[_key].category = ("Retired",)


# ── how the equations are presented ─────────────────────────────────────────
# The registry above keeps every key it has ever had (saved settings and the
# examples refer to them). What follows decides how they are shown: families
# a scientist would look under, clear names, the real formula for every one,
# what each parameter means, and an X range where the curve's shape shows.

from dataclasses import replace as _replace  # noqa: E402

# (family, glyph, one line on what's in it) - in the order they are listed
FAMILIES: list[tuple[str, str, str]] = [
    ("Lines", "╱", "Straight lines on linear or log axes"),
    ("Polynomials", "∿", "Quadratic, cubic and higher, and power series"),
    ("Exponential", "↘", "Falling or rising to a plateau, at one, two or three rates"),
    ("Growth", "↗", "Population and tumour growth, with and without a plateau"),
    ("Dose-response", "∫", "Sigmoids rising or falling, lopsided, two-step or bell-shaped"),
    ("Binding", "◔", "Saturation, competition and kinetics of receptor binding"),
    ("Enzyme kinetics", "⊗", "Michaelis–Menten, allosteric and inhibition"),
    ("Peaks & distributions", "∩", "Gaussian, lognormal, Lorentzian and cumulative curves"),
    ("Waves", "∼", "Sine waves: plain, damped, with a baseline"),
    ("Radiation survival", "⊙", "Linear-quadratic cell survival and death"),
]
FAMILY_NAMES = [f[0] for f in FAMILIES]

# Retired keys stay valid (settings saved by earlier versions name them), but
# are listed once, as the equation each duplicates.
ALIASES: dict[str, str] = {
    "classic_one_site_binding": "one_site_specific",
    "classic_two_site_binding": "two_site_specific",
    "classic_sigmoidal_dr": "dr_stim_3pl_log",
    "classic_sigmoidal_dr_var": "dr_stim_4pl_log",
    "classic_one_phase_decay": "one_phase_decay",
    "classic_two_phase_decay": "two_phase_decay",
    "classic_one_phase_assoc": "one_phase_association",
    "classic_two_phase_assoc": "two_phase_association",
    "classic_exp_growth": "exp_malthusian",
    "classic_poly_1": "straight_line",
    "classic_poly_2": "poly_2",
    "classic_poly_3": "poly_3",
    "classic_poly_4": "poly_4",
    "classic_sine_wave": "sine_wave",
    "classic_gaussian": "gaussian",
    "exponential_growth_eq": "exp_malthusian",
    "poly_1": "straight_line",
    "poly_centered_1": "line_through_point",
}

# every formula written out (several only had a description)
_FORMULAS = {
    "plateau_then_decay": "Y = Y0 if X < X0 else Plateau + (Y0-Plateau)*exp(-k*(X-X0))",
    "plateau_then_association": "Y = Y0 if X < X0 else Y0 + (Plateau-Y0)*(1-exp(-k*(X-X0)))",
    "two_phase_decay": "Y = Plateau + (Y0-Plateau)*(P*exp(-k1*X) + (1-P)*exp(-k2*X)), P = Pct1/100",
    "three_phase_decay": "Y = Plateau + (Y0-Plateau)*(P1*exp(-k1*X) + P2*exp(-k2*X) + (1-P1-P2)*exp(-k3*X)), P1 = Pct1/100, P2 = Pct2/100",
    "specific_binding_hill": "Y = Bmax*X^Hill/(Kd^Hill + X^Hill)",
    "two_phase_association": "Y = Y0 + (Plateau-Y0)*(P*(1-exp(-k1*X)) + (1-P)*(1-exp(-k2*X))), P = Pct1/100",
    "sum_two_gaussians": "Y = Amp1*exp(-(X-Mean1)^2/(2*SD1^2)) + Amp2*exp(-(X-Mean2)^2/(2*SD2^2))",
    "sum_two_lorentzians": "Y = Amp1/(1+((X-Center1)/(Width1/2))^2) + Amp2/(1+((X-Center2)/(Width2/2))^2)",
    "dr_asymmetric_5pl_conc": "Y = Bottom + (Top-Bottom)/(1+10^((logEC50-log(X))*Hill))^S",
    "dr_asymmetric_5pl_log": "Y = Bottom + (Top-Bottom)/(1+10^((logEC50-X)*Hill))^S",
    "dr_biphasic_conc": "Y = Bottom + Height1/(1+10^((logEC50_1-log(X))*Hill1)) + Height2/(1+10^((logEC50_2-log(X))*Hill2))",
    "dr_bell_conc": "Y = Bottom + (Top-Bottom)*rise*fall; rise = 1/(1+10^((logEC50_rise-log(X))*Hill_rise)), fall = 1/(1+10^((log(X)-logEC50_fall)*Hill_fall))",
    "two_site_specific": "Y = Bmax1*X/(Kd1+X) + Bmax2*X/(Kd2+X)",
    "two_site_total": "Y = Bmax1*X/(Kd1+X) + Bmax2*X/(Kd2+X) + Nonspecific*X",
    "two_site_fit_logic50": "Y = Bottom + (Top-Bottom)*(F/(1+10^(X-logIC50_1)) + (1-F)/(1+10^(X-logIC50_2))), F = Fraction1",
    "classic_two_site_binding": "Y = Bmax1*X/(Kd1+X) + Bmax2*X/(Kd2+X)",
    "classic_two_phase_decay": "Y = Plateau + (Y0-Plateau)*(P*exp(-k1*X) + (1-P)*exp(-k2*X)), P = Pct1/100",
    "classic_two_phase_assoc": "Y = Y0 + (Plateau-Y0)*(P*(1-exp(-k1*X)) + (1-P)*(1-exp(-k2*X))), P = Pct1/100",
    "dr_stim_norm_log": "Y = 100/(1+10^(logEC50-X))",
}

# polynomials start with their highest term switched on, so each order shows
# its own shape (with only a1 = 1 every order drew the same straight line)
for _key, _spec in list(EQUATIONS.items()):
    if _key.startswith(("poly_", "poly_centered_", "classic_poly_")) and not _key.endswith("_1"):
        _order = len([n for n, _d in _spec.params if n.startswith("a")]) - 1
        EQUATIONS[_key] = _replace(_spec, params=tuple((n, 0.5 if n == f"a{_order}" else d) for n, d in _spec.params))

for _key, _formula in _FORMULAS.items():
    EQUATIONS[_key] = _replace(EQUATIONS[_key], equation=_formula)

# what each parameter means (numbered copies - Bmax1, Kd2, logEC50_1 - share one entry)
PARAM_HELP = {
    "slope": "change in Y per unit of X", "intercept": "Y where X is 0", "X0": "X where it starts or is centred",
    "Y0": "Y at the start", "Plateau": "Y it levels off at",
    "Pct": "share of the change that goes at this rate, %", "k": "rate constant (half-life = ln 2 / k)",
    "Amplitude": "peak height", "Amp": "peak height", "Mean": "centre of the peak", "SD": "spread (standard deviation)",
    "Center": "centre of the peak", "Width": "full width at half maximum", "Freq": "cycles per unit of X",
    "Phase": "shift, in radians", "Baseline": "level it oscillates around", "Decay": "damping rate",
    "Top": "plateau at high X", "Bottom": "plateau at low X", "logEC50": "log of the X giving a half response",
    "EC50": "X giving a half response", "Hill": "steepness (1 = standard)",
    "Hill_rise": "steepness of the rise", "Hill_fall": "steepness of the fall", "logEC50_rise": "log X half-way up",
    "logEC50_fall": "log X half-way down", "Height": "rise of this phase", "logIC50": "log of the X giving half inhibition",
    "S": "asymmetry (1 = symmetric)", "Bmax": "maximum specific binding", "Kd": "concentration giving half-maximal binding",
    "Nonspecific": "nonspecific binding per unit of X", "Vmax": "maximum velocity", "Km": "[substrate] giving half of Vmax",
    "Ki": "inhibition constant", "I": "inhibitor concentration", "Kprime": "[substrate] giving half of Vmax",
    "h": "Hill coefficient (cooperativity)", "alpha": "linear, single-hit term (per unit dose)",
    "beta": "quadratic, two-hit term (per unit dose squared)", "N0": "cells at dose 0 (1 gives a fraction, 100 a percent)", "N": "the top of the curve (1 gives a fraction, 100 a percent)",
    "Ymax": "highest value", "Fraction1": "fraction of sites of the first kind", "V50": "X half-way up",
    "Slope": "width of the transition", "A": "coefficient", "B": "power", "C": "coefficient", "D": "power", "n": "shape",
}


def param_help(name: str) -> str:
    if name in PARAM_HELP:
        return PARAM_HELP[name]
    import re as _re

    base = _re.sub(r"_?\d+$", "", name)
    if base in PARAM_HELP:
        return PARAM_HELP[base]
    if _re.fullmatch(r"a\d+", name):
        return "the constant term" if name == "a0" else f"coefficient of X^{name[1:]}"
    return ""


# an X range where the default curve shows its shape
_X_RANGES = {
    "Lines": (0.0, 10.0), "Polynomials": (-3.0, 3.0), "Exponential": (0.0, 10.0), "Growth": (0.0, 20.0),
    "Binding": (0.0, 10.0), "Enzyme kinetics": (0.0, 10.0), "Peaks & distributions": (-4.0, 4.0),
    "Waves": (0.0, 3.0), "Radiation survival": (0.0, 10.0), "Dose-response": (-3.0, 3.0),
}
_KEY_X_RANGES = {
    "semilog_line_x_log": (0.1, 100.0), "log_log_line": (0.1, 100.0), "semilog_line_y_log": (0.0, 2.0),
    "plateau_then_decay": (0.0, 15.0), "plateau_then_association": (0.0, 15.0), "two_phase_decay": (0.0, 30.0),
    "two_phase_association": (0.0, 30.0), "three_phase_decay": (0.0, 40.0), "exp_malthusian": (0.0, 10.0),
    "exponential_growth_eq": (0.0, 5.0), "gompertz_growth": (0.0, 15.0), "logistic_growth": (0.0, 15.0),
    "beta_growth_decay": (0.0, 15.0), "exp_plateau_growth": (0.0, 25.0),
    "sum_two_gaussians": (-3.0, 8.0), "sum_two_lorentzians": (-3.0, 8.0), "lognormal": (0.01, 5.0),
    "lorentzian": (-5.0, 5.0), "damped_sine_wave": (0.0, 10.0), "sinc": (-3.0, 3.0),
    "dr_stim_4pl_conc": (0.0, 10.0), "dr_asymmetric_5pl_conc": (0.0, 10.0), "dr_biphasic_conc": (0.0, 30.0),
    "dr_bell_conc": (0.0, 30.0), "classic_boltzmann": (-5.0, 5.0), "classic_power_series": (0.0, 5.0),
    "kinetics_association": (0.0, 20.0), "kinetics_dissociation": (0.0, 20.0), "one_site_fit_logic50": (-3.0, 3.0),
    "two_site_fit_logic50": (-4.0, 4.0), "substrate_inhibition": (0.0, 40.0),
}


def resolve(key: str) -> str:
    """The equation a key is listed as (a retired duplicate -> the original)."""
    return ALIASES.get(key, key)


def family_of(key: str) -> tuple[str, str]:
    """(family, tag) for an equation - its category, as registered."""
    category = EQUATIONS[key].category
    return category[0], (category[1] if len(category) > 1 else "")


def listed_equations(family: str | None = None) -> list[str]:
    """The keys shown in the picker, in library order: no retired duplicates."""
    keys = [k for k, spec in EQUATIONS.items() if k not in ALIASES and spec.category[0] != "Retired"]
    return [k for k in keys if family is None or family_of(k)[0] == family]


def suggested_x_range(key: str) -> tuple[float, float]:
    key = resolve(key)
    return _KEY_X_RANGES.get(key) or _X_RANGES.get(family_of(key)[0], (0.0, 10.0))


def search_text(key: str) -> str:
    """Everything a search should match for an equation: its name, formula,
    family, tag, parameter names, and the old names of any duplicate retired
    in its favour."""
    spec = EQUATIONS[key]
    family, tag = family_of(key)
    old_names = [EQUATIONS[k].label for k, target in ALIASES.items() if target == key]
    return " ".join([spec.label, spec.equation, family, tag, spec.x_label,
                     " ".join(name for name, _d in spec.params), *old_names]).lower()


def search(query: str) -> list[str]:
    """Listed equations matching every word of ``query``, best first: name
    matches before formula / parameter matches."""
    words = [w for w in query.lower().replace("-", " ").split() if w]
    if not words:
        return listed_equations()
    scored = []
    for order, key in enumerate(listed_equations()):
        text = search_text(key).replace("-", " ").replace("\u2013", " ")
        if all(w in text for w in words):
            name = EQUATIONS[key].label.lower().replace("-", " ").replace("\u2013", " ")
            scored.append((0 if all(w in name for w in words) else 1, order, key))
    return [k for _s, _o, k in sorted(scored)]
