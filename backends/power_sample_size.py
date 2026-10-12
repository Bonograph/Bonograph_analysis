"""Power and sample size, before the data exist: how many subjects a study
needs, what power a given number buys, or the smallest effect it can find.

Six designs. The t tests and the ANOVA use the exact noncentral t and F
distributions; two proportions use the usual normal approximation (pooled
variance under the null), and a correlation uses Fisher's z transformation.
Nothing is read from the sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register

DESIGNS = {"two_means": "Two separate groups, means compared", "paired_means": "The same subjects measured twice",
           "one_mean": "One group against a fixed value", "anova": "Three groups or more, means compared",
           "two_proportions": "Two groups, proportions compared", "correlation": "A correlation between two measurements"}


@dataclass
class PowerOptions:
    design: str = "two_means"
    solve_for: str = "n"            # "n" | "power" | "effect"
    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True
    difference: float = 5.0         # means: the difference worth finding (anova: largest difference between two group means)
    sd: float = 10.0                # means: SD within a group (paired: SD of the differences)
    n: int = 20                     # per group (one group, pairs: in all)
    ratio: float = 1.0              # second group is this many times the first
    groups: int = 3                 # anova
    p1: float = 0.50                # proportions
    p2: float = 0.30
    r: float = 0.30                 # correlation
    dropout_percent: float = 0.0    # subjects expected to be lost: the number to enrol is raised to allow for them


# ── power of each design ─────────────────────────────────────────────────────
def _t_power(ncp: float, df: float, alpha: float, two_sided: bool) -> float:
    if df < 1:
        return 0.0
    if two_sided:
        crit = stats.t.ppf(1 - alpha / 2, df)
        return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(crit, df, abs(ncp)))


def _power(opt: PowerOptions, n: float, effect: float) -> float:
    """Power with ``n`` per group (or in all, for one group, pairs and a
    correlation) to find ``effect``: a standardised difference d for the
    means, Cohen's f for the ANOVA, p2 for proportions, r for a correlation."""
    a, two = opt.alpha, opt.two_sided
    if opt.design == "two_means":
        n2 = n * opt.ratio
        return _t_power(effect * np.sqrt(n * n2 / (n + n2)), n + n2 - 2, a, two)
    if opt.design in ("paired_means", "one_mean"):
        return _t_power(effect * np.sqrt(n), n - 1, a, two)
    if opt.design == "anova":
        k = opt.groups
        total = n * k
        if total - k < 1:
            return 0.0
        crit = stats.f.ppf(1 - a, k - 1, total - k)
        return float(stats.ncf.sf(crit, k - 1, total - k, effect**2 * total))
    if opt.design == "two_proportions":
        p1, p2, n2 = opt.p1, effect, n * opt.ratio
        pooled = (n * p1 + n2 * p2) / (n + n2)
        se0 = np.sqrt(pooled * (1 - pooled) * (1 / n + 1 / n2))
        se1 = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n2)
        if se1 == 0:
            return 1.0 if p1 != p2 else a
        z = stats.norm.ppf(1 - a / 2) if two else stats.norm.ppf(1 - a)
        gap = abs(p1 - p2)
        power = stats.norm.cdf((gap - z * se0) / se1)
        if two:
            power += stats.norm.cdf((-gap - z * se0) / se1)
        return float(power)
    # correlation, by Fisher's z
    if n <= 3:
        return 0.0
    shift = abs(np.arctanh(effect)) * np.sqrt(n - 3)
    z = stats.norm.ppf(1 - a / 2) if two else stats.norm.ppf(1 - a)
    power = stats.norm.sf(z - shift)
    if two:
        power += stats.norm.cdf(-z - shift)
    return float(power)


def _least_n(opt: PowerOptions) -> int:
    return {"two_means": 2, "paired_means": 2, "one_mean": 2, "anova": 2, "two_proportions": 2, "correlation": 4}[opt.design]


def _n_for(opt: PowerOptions, effect: float, target: float) -> int | None:
    """The smallest whole number per group that reaches ``target`` power."""
    lo = _least_n(opt)
    if _power(opt, lo, effect) >= target:
        return lo
    hi = lo
    while _power(opt, hi, effect) < target:
        hi *= 2
        if hi > 10_000_000:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _power(opt, mid, effect) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _effect_from(opt: PowerOptions) -> float:
    """The effect the user described, in the unit ``_power`` works in."""
    if opt.design in ("two_means", "paired_means", "one_mean"):
        return abs(opt.difference) / opt.sd
    if opt.design == "anova":
        # the least favourable pattern for a given largest difference: two means at the ends,
        # the rest half-way - Cohen's f = (difference / SD) x sqrt(1 / (2k))
        return abs(opt.difference) / opt.sd * np.sqrt(1.0 / (2 * opt.groups))
    if opt.design == "two_proportions":
        return opt.p2
    return opt.r


def _effect_words(opt: PowerOptions, effect: float) -> str:
    if opt.design in ("two_means", "paired_means", "one_mean"):
        return f"a difference of {effect * opt.sd:.4g} (SD {opt.sd:g}; standardised d = {effect:.3g})"
    if opt.design == "anova":
        gap = effect * opt.sd / np.sqrt(1.0 / (2 * opt.groups))
        return f"group means up to {gap:.4g} apart (SD {opt.sd:g}; Cohen's f = {effect:.3g})"
    if opt.design == "two_proportions":
        return f"proportions of {opt.p1:.3g} and {effect:.3g}"
    return f"a correlation of {effect:.3g}"


def _detectable(opt: PowerOptions) -> float | None:
    """The smallest effect that ``n`` finds with the wanted power."""
    def gap(effect):
        return _power(opt, opt.n, effect) - opt.power

    try:
        if opt.design == "two_proportions":
            direction = 1 if opt.p2 >= opt.p1 else -1
            far = 1 - 1e-9 if direction > 0 else 1e-9
            if gap(far) < 0:
                return None
            return float(optimize.brentq(gap, opt.p1 + direction * 1e-9, far))
        if opt.design == "correlation":
            return float(optimize.brentq(gap, 1e-9, 1 - 1e-9)) if gap(1 - 1e-9) >= 0 else None
        top = 0.01
        while not gap(top) >= 0:         # widen until the power is reached (very large effects overflow the distribution)
            top *= 2
            if top > 60:
                return None
        return float(optimize.brentq(gap, 1e-9, top))
    except ValueError:
        return None


def _people(opt: PowerOptions, n: int) -> tuple[str, int]:
    """How ``n`` reads for this design, and the total it makes."""
    if opt.design in ("two_means", "two_proportions"):
        n2 = int(np.ceil(n * opt.ratio))
        if n2 == n:
            return f"{n} per group ({2 * n} in all)", 2 * n
        return f"{n} and {n2} ({n + n2} in all)", n + n2
    if opt.design == "anova":
        return f"{n} per group ({n * opt.groups} in all)", n * opt.groups
    if opt.design == "paired_means":
        return f"{n} pairs", n
    return f"{n} subjects", n


def _check(opt: PowerOptions) -> None:
    if opt.design not in DESIGNS:
        raise AnalysisError("Choose the kind of study.")
    if not 0 < opt.alpha < 0.5:
        raise AnalysisError("The chance of a false positive (alpha) must be above 0 and below 0.5.")
    if opt.solve_for != "power" and not opt.alpha < opt.power < 1:
        raise AnalysisError("The power wanted must lie between alpha and 1 (0.8 and 0.9 are usual).")
    if opt.solve_for != "n" and opt.n < _least_n(opt):
        raise AnalysisError(f"This design needs at least {_least_n(opt)} subjects per group.")
    if opt.design in ("two_means", "two_proportions") and not 0.05 <= opt.ratio <= 20:
        raise AnalysisError("The size of the second group must be between 0.05 and 20 times the first.")
    if opt.design in ("two_means", "paired_means", "one_mean", "anova"):
        if opt.sd <= 0:
            raise AnalysisError("The standard deviation must be above 0.")
        if opt.solve_for != "effect" and opt.difference == 0:
            raise AnalysisError("The difference to find must not be 0: no study can find a difference of nothing.")
    if opt.design == "anova" and opt.groups < 3:
        raise AnalysisError("Three groups or more: for two groups choose 'Two separate groups'.")
    if opt.design == "two_proportions":
        if not (0 < opt.p1 < 1 and 0 <= opt.p2 <= 1):
            raise AnalysisError("Proportions must lie between 0 and 1 (30% is 0.30).")
        if opt.solve_for != "effect" and opt.p1 == opt.p2:
            raise AnalysisError("The two proportions are the same: there is no difference to find.")
    if opt.design == "correlation" and opt.solve_for != "effect" and not 0 < abs(opt.r) < 1:
        raise AnalysisError("The correlation to find must lie between 0 and 1 (or 0 and -1).")
    if not 0 <= opt.dropout_percent < 95:
        raise AnalysisError("Expected losses must be between 0 and 95%.")


@register("power_sample_size")
def run(data: SheetData | None = None, options: PowerOptions | None = None) -> AnalysisResult:
    opt = options or PowerOptions()
    _check(opt)
    # an F test over several groups has no direction to choose
    sides = "" if opt.design == "anova" else (", two-sided" if opt.two_sided else ", one-sided")
    effect = _effect_from(opt)
    notes: list[str] = []

    if opt.solve_for == "n":
        n = _n_for(opt, effect, opt.power)
        if n is None:
            raise AnalysisError("More than ten million subjects per group would be needed: the effect is too small to find.")
        reached = _power(opt, n, effect)
        who, total = _people(opt, n)
        summary = f"{who} give {reached:.1%} power to find {_effect_words(opt, effect)} at α = {opt.alpha:g}{sides}."
        answer = [["Number needed", who], ["Power reached", f"{reached:.1%} (wanted: {opt.power:.0%})"]]
    elif opt.solve_for == "power":
        n = int(opt.n)
        reached = _power(opt, n, effect)
        who, total = _people(opt, n)
        summary = f"{who}: {reached:.1%} power to find {_effect_words(opt, effect)} at α = {opt.alpha:g}{sides}."
        answer = [["Subjects", who], ["Power", f"{reached:.1%}"]]
        if reached < 0.5:
            notes.append("Power under 50%: the study is more likely to miss this effect than to find it.")
    else:
        n = int(opt.n)
        found = _detectable(opt)
        who, total = _people(opt, n)
        if found is None:
            raise AnalysisError(f"With {who}, no effect however large reaches {opt.power:.0%} power.")
        effect = found
        reached = _power(opt, n, effect)
        summary = (f"{who} can find {_effect_words(opt, effect)} with {opt.power:.0%} power at α = {opt.alpha:g}{sides}. "
                   "Smaller effects are more likely to be missed.")
        answer = [["Subjects", who], ["Smallest effect found with that power", _effect_words(opt, effect)]]

    if opt.dropout_percent > 0:
        enrol = int(np.ceil(n / (1 - opt.dropout_percent / 100)))
        answer.append([f"To enrol, allowing {opt.dropout_percent:g}% losses", _people(opt, enrol)[0]])
        summary += f" Enrol {_people(opt, enrol)[0]} to allow for {opt.dropout_percent:g}% losses."
    setting = [["Kind of study", DESIGNS[opt.design]], ["Chance of a false positive (α)", f"{opt.alpha:g}{sides}"]]
    if opt.design in ("two_means", "two_proportions") and opt.ratio != 1:
        setting.append(["Second group", f"{opt.ratio:g} times the first"])
    tables = [ResultTable("The answer", ["", ""], answer + setting)]

    # power at other numbers of subjects
    least = _least_n(opt)
    grid = sorted({max(least, int(round(n * f))) for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)})
    tables.append(ResultTable("Power with other numbers of subjects", ["Number", "In all", "Power"],
                              [[_people(opt, g)[0].split(" (")[0], str(_people(opt, g)[1]), f"{_power(opt, g, effect):.1%}"] for g in grid]))

    # the number needed for other effects, at the wanted power
    target = opt.power if opt.solve_for != "power" else max(0.8, min(0.99, round(reached, 2)))
    if opt.design == "two_proportions":
        others = [opt.p1 + (effect - opt.p1) * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
        others = [p for p in others if 0 <= p <= 1 and p != opt.p1]
    elif opt.design == "correlation":
        others = [e for e in (effect * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)) if 0 < abs(e) < 1]
    else:
        others = [effect * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
    rows = []
    for e in others:
        need = _n_for(opt, e, target)
        rows.append([_effect_words(opt, e), _people(opt, need)[0] if need else "more than ten million"])
    tables.append(ResultTable(f"Numbers needed for other effects, for {target:.0%} power", ["Effect to find", "Number needed"], rows))

    # graph: power against the number of subjects, for the effect and for effects a quarter smaller and larger
    top = max(int(n * 2.2), least + 8)
    xs = np.unique(np.linspace(least, top, 60).astype(int))
    series = []
    for e, name, colour in zip(others[1:4] if len(others) >= 4 else [effect],
                               ("A quarter smaller", "The effect entered" if opt.solve_for != "effect" else "The effect found",
                                "A quarter larger") if len(others) >= 4 else ("The effect",),
                               ("#9aa9c4", "#1f5aa6", "#9aa9c4") if len(others) >= 4 else ("#1f5aa6",)):
        series.append(ChartSeries(name, xs.tolist(), [_power(opt, int(x), e) * 100 for x in xs], kind="line", color=colour))
    unit = {"paired_means": "Pairs", "one_mean": "Subjects", "correlation": "Subjects"}.get(opt.design, "Subjects per group")
    charts = [ChartSpec("Power against the number of subjects", series=series, x_label=unit, y_label="Power (%)",
                        y_limits=(0, 102),
                        reference_lines=[{"kind": "h", "value": target * 100, "color": "#d1342f", "style": "--", "width": 0.9},
                                         {"kind": "v", "value": n, "color": "#555555", "style": ":", "width": 1.0}])]

    method = {"two_means": "the noncentral t distribution (Student's t test, equal SDs)",
              "paired_means": "the noncentral t distribution (paired t test; the SD is that of the differences between the two measurements)",
              "one_mean": "the noncentral t distribution (one-sample t test)",
              "anova": "the noncentral F distribution (one-way ANOVA)",
              "two_proportions": "the normal approximation to the difference of two proportions (no continuity correction)",
              "correlation": "Fisher's z transformation of the correlation coefficient"}[opt.design]
    notes += [f"Worked out from {method}.",
              "The answer is only as good as the effect and the SD put in. Take the SD from earlier data or a pilot, and "
              "choose the effect as the smallest one that would matter - not the one hoped for."]
    if opt.design == "anova":
        notes.append("For several groups the effect is given as the largest difference between two group means. The other "
                     "means are placed half-way, the pattern that needs the most subjects - any other pattern with the same "
                     "largest difference has more power than shown.")
    if opt.design == "paired_means":
        notes.append("The SD of the differences is smaller than the SD of the measurements when the two are correlated: "
                     "SD of differences = SD × √(2 (1 - r)). With r = 0.5 the two are equal.")
    if opt.design == "two_proportions" and min(n * opt.p1, n * (1 - opt.p1), n * effect, n * (1 - effect)) < 5:
        notes.append("Fewer than 5 subjects are expected in one of the cells: the normal approximation is rough here, and an "
                     "exact test will have somewhat less power.")
    if opt.design == "correlation":
        notes.append("Fisher's approximation is within one or two subjects of the exact answer.")
    if not opt.two_sided and opt.design != "anova":
        notes.append("One-sided: an effect in the other direction, however large, would have to be called 'no effect'.")
    notes.append("References: Cohen, Statistical Power Analysis for the Behavioral Sciences (1988); "
                 "Chow, Shao, Wang and Lokhnygina, Sample Size Calculations in Clinical Research (2017).")
    return AnalysisResult(title="How many subjects?", summary=summary, tables=tables, notes=notes, charts=charts)
