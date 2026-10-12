"""Which of two models fits the data better?

Both models are fitted to each data set. A model with more parameters always
fits closer; the question is whether it fits closer by more than its extra
parameters would buy by chance. Two answers are given: Akaike's criterion
(AICc), which works for any two models, and the extra sum-of-squares F test,
which is valid when the simpler model is a special case of the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import nonlinear_regression as nlr
from .global_fit import aicc, extra_ss_f, weighted_rss

# (simpler, fuller): the simpler is the fuller with a parameter fixed
NESTED = {("linear", "quadratic"), ("linear", "cubic"), ("quadratic", "cubic"), ("sigmoid_4pl", "sigmoid_5pl"),
          ("michaelis_menten", "allosteric_sigmoidal"), ("michaelis_menten", "substrate_inhibition"),
          ("hyperbola", "allosteric_sigmoidal"), ("hyperbola", "substrate_inhibition")}
SAME = {frozenset(("michaelis_menten", "hyperbola"))}       # one equation under two names


@dataclass
class CompareModelsOptions:
    model_a: str = "michaelis_menten"
    model_b: str = "allosteric_sigmoidal"
    relative_weighting: bool = False
    range_mode_start: str = "auto"
    range_start_x: float = 0.0
    range_mode_end: str = "auto"
    range_end_x: float = 0.0
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"
    data_sets: list[str] = field(default_factory=list)


@register("compare_models")
def run(data: SheetData, options: CompareModelsOptions | None = None) -> AnalysisResult:
    opt = options or CompareModelsOptions()
    first, second = nlr.MODELS.get(opt.model_a), nlr.MODELS.get(opt.model_b)
    if first is None or second is None:
        raise AnalysisError("Choose two models from the list.")
    if first.key == second.key:
        raise AnalysisError("The two models are the same. Choose two different ones.")
    if frozenset((first.key, second.key)) in SAME:
        raise AnalysisError(f"'{first.label}' and '{second.label}' are one equation under two names; there is nothing to compare.")
    # the one with fewer parameters is the simpler
    simple, full = (first, second) if len(first.param_names) <= len(second.param_names) else (second, first)
    nested = (simple.key, full.key) in NESTED
    same_size = len(simple.param_names) == len(full.param_names)
    sets = [(label, x, y) for label, x, y in nlr._data_sets(data, opt) if x.size]
    if not sets:
        raise AnalysisError("Curve fitting needs numbers in the X column, paired with at least one column of Y values.")
    sig = opt.significant_digits

    def num(value) -> str:
        return "" if not np.isfinite(value) else f"{value:.{sig}g}"

    def p_text(p) -> str:
        return "" if not np.isfinite(p) else format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    tables, charts, verdicts, skipped = [], [], [], []
    overview = []
    many = len(sets) > 1
    for label, x, y in sets[:12]:
        try:
            fit_s, fit_f = nlr._fit_one(simple, label, x, y, opt), nlr._fit_one(full, label, x, y, opt)
        except AnalysisError as exc:
            skipped.append(str(exc))
            continue
        n = x.size
        ks, kf = len(simple.param_names), len(full.param_names)
        rss_s = weighted_rss(simple, x, y, fit_s.params, opt.relative_weighting)
        rss_f = weighted_rss(full, x, y, fit_f.params, opt.relative_weighting)
        a_s, a_f = aicc(rss_s, n, ks), aicc(rss_f, n, kf)
        delta = a_s - a_f                                   # above 0: the fuller model is the better by AICc
        prob_full = 1.0 / (1.0 + np.exp(-0.5 * delta)) if np.isfinite(delta) else float("nan")
        title = f"{label}: " if many else ""
        tables.append(ResultTable(f"{title}the two fits side by side",
                                  ["Model", "Parameters", "Sum of squares", "df", "R²", "Typical residual", "AICc", "Chance it is the better"],
                                  [[simple.label, str(ks), num(rss_s), str(n - ks), f"{fit_s.r_squared:.4f}", num(fit_s.sy_x), num(a_s),
                                    "" if not np.isfinite(prob_full) else f"{1 - prob_full:.1%}"],
                                   [full.label, str(kf), num(rss_f), str(n - kf), f"{fit_f.r_squared:.4f}", num(fit_f.sy_x), num(a_f),
                                    "" if not np.isfinite(prob_full) else f"{prob_full:.1%}"]]))
        f_value, p_value = (extra_ss_f(rss_s, n - ks, rss_f, n - kf) if nested else (float("nan"), float("nan")))
        if nested:
            better_full = np.isfinite(p_value) and p_value < opt.alpha
            reading = (f"the extra parameter{'s' if kf - ks > 1 else ''} of {full.label} earn their place" if better_full else
                       f"no clear gain from {full.label}: keep {simple.label}")
            tables.append(ResultTable(f"{title}is the fuller model worth its extra parameters?", ["Test", "F", "df", "P value", "Reading"],
                                      [["Extra sum of squares", "" if not np.isfinite(f_value) else f"{f_value:.4f}", f"{kf - ks}, {n - kf}",
                                        p_text(p_value), reading]]))
            choice = full if better_full else simple
        else:
            choice = full if (np.isfinite(prob_full) and prob_full > 0.5) else simple
        if not np.isfinite(prob_full):
            notes_few = f"{label}: too few points for AICc with {kf} parameters ({n} points)."
            skipped.append(notes_few)
        verdicts.append((label, choice, prob_full, p_value))
        ratio = max(prob_full, 1 - prob_full) / max(min(prob_full, 1 - prob_full), 1e-12) if np.isfinite(prob_full) else float("nan")
        overview.append([label, choice.label, "" if not np.isfinite(prob_full) else f"{max(prob_full, 1 - prob_full):.1%}",
                         "" if not np.isfinite(ratio) else (f"{ratio:.3g} to 1" if ratio < 1e6 else "more than a million to 1"),
                         p_text(p_value) if nested else "not a special case: no F test"])
        tables.append(ResultTable(f"{title}fitted parameters", ["Model", "Parameter", "Best fit", "Std. error"],
                                  [[m.label if i == 0 else "", name, num(v), num(e)]
                                   for m, f in ((simple, fit_s), (full, fit_f)) for i, (name, v, e) in enumerate(zip(m.param_names, f.params, f.errors))]))

        positive = x[x > 0]
        log_x = bool((simple.log_x or full.log_x) and positive.size == x.size and positive.size and positive.max() / positive.min() >= 100)
        grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 240) if log_x else np.linspace(float(x.min()), float(x.max()), 240)
        series = nlr._points(nlr._Fit(label, x, y, fit_s.params, fit_s.errors, 0, 0, 1), "#555555", False)
        for m, f, colour in ((simple, fit_s, "#1f5aa6"), (full, fit_f, "#d62728")):
            with np.errstate(all="ignore"):
                curve = m.fn(grid, *f.params)
            ok = np.isfinite(curve)
            series.append(ChartSeries(m.label, grid[ok].tolist(), curve[ok].tolist(), kind="line", color=colour))
        charts.append(ChartSpec(f"{label}: the two fitted curves", series=series, x_label="X", y_label="Y", x_log=log_x, group="compare", option=label))
        charts.append(ChartSpec(f"{label}: what each fit leaves over", x_label="X", y_label="Residual (Y − fit)", x_log=log_x,
                                group="compare_residuals", option=label, reference_lines=[{"kind": "h", "value": 0.0}],
                                series=[ChartSeries(simple.label, x.tolist(), (y - simple.fn(x, *fit_s.params)).tolist(), kind="scatter", color="#1f5aa6"),
                                        ChartSeries(full.label, x.tolist(), (y - full.fn(x, *fit_f.params)).tolist(), kind="scatter", color="#d62728")]))
    if not verdicts:
        raise AnalysisError(skipped[0] if skipped else "Neither model could be fitted.")
    tables.insert(0, ResultTable("Which model to keep", ["Data set", "Keep", "Chance it is the better (AICc)", "Odds", "P value (F test)"], overview))

    label, choice, prob, p_value = verdicts[0]
    summary = f"{label}: keep {choice.label}" if many else f"Keep {choice.label}"
    if np.isfinite(prob):
        summary += f" - by AICc it is {max(prob, 1 - prob):.0%} likely to be the better of the two"
    if nested and np.isfinite(p_value):
        summary += f"; extra sum-of-squares F test P = {p_value:.4g}"
    summary += "."
    if many:
        kept = {c.label for _l, c, _p, _q in verdicts}
        summary += (f" All {len(verdicts)} data sets agree." if len(kept) == 1 else " The data sets do not all agree: see the first table.")

    notes = list(skipped)
    if nested:
        notes.append(f"{simple.label} is {full.label} with {len(full.param_names) - len(simple.param_names)} parameter"
                     f"{'s' if len(full.param_names) - len(simple.param_names) > 1 else ''} fixed, so the F test applies. Where the test and "
                     "AICc disagree, the F test is the more cautious about adding parameters.")
    elif same_size:
        notes.append("The two models have the same number of parameters, so the one that fits closer is simply the better; AICc "
                     "turns the gap into a chance.")
    else:
        notes.append(f"{simple.label} is not a special case of {full.label}, so the F test does not apply; AICc is the comparison to use.")
    notes += ["'Chance it is the better' is the Akaike weight: of these two models only. Both may be wrong; look at the residuals. "
              "A model that fits should leave residuals scattered evenly about zero, with no run of the same sign.",
              "Choose between models on what they mean as well: a fuller model that wins narrowly, with a parameter far from any "
              "sensible value or a very wide interval, has not earned its place.",
              "Compare models on the same data, with the same weighting. Both tests assume normal scatter about the curve.",
              "Method: extra sum-of-squares F test after Bates and Watts, Nonlinear Regression Analysis and Its Applications "
              "(1988); AICc after Hurvich and Tsai, Biometrika (1989), and Burnham and Anderson, Model Selection and "
              "Multimodel Inference (2002)."]
    if len(sets) > 12:
        notes.append(f"Only the first 12 of {len(sets)} data sets were compared.")
    return AnalysisResult(title="Which model fits better?", summary=summary, tables=tables, notes=notes, charts=charts)
