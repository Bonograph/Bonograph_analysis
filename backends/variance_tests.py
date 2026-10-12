"""Is the spread the same in every group?

The check behind Student's t test and the ordinary ANOVA, as an analysis of
its own: Brown-Forsythe and Levene's tests (on distances from each group's
median or mean), Bartlett's test, the Fligner-Killeen rank test and, for two
groups, the F test of two variances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register


@dataclass
class VarianceOptions:
    brown_forsythe: bool = True    # distances from the median: the sturdy default
    levene: bool = False           # distances from the mean: the original form
    bartlett: bool = False         # most power when the values are normal, fooled when they are not
    fligner: bool = False          # ranks: for clearly non-normal values
    f_test: bool = True            # two groups only
    subcolumns: str = "average"    # "average" | "pooled"
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


@register("variance_tests")
def run(data: SheetData, options: VarianceOptions | None = None) -> AnalysisResult:
    opt = options or VarianceOptions()
    mode = "pooled" if opt.subcolumns == "pooled" else "average"
    groups = {label: v[np.isfinite(v)] for label, v in data.all_group_arrays(mode=mode).items()}
    small = [label for label, v in groups.items() if v.size < 3]
    groups = {label: v for label, v in groups.items() if v.size >= 3}
    if len(groups) < 2:
        raise AnalysisError("Comparing spreads needs two groups or more, each with at least 3 values.")
    names, values = list(groups), list(groups.values())
    k = len(names)
    total = sum(v.size for v in values)
    sig = opt.significant_digits

    def num(value) -> str:
        return f"{value:.{sig}g}"

    def p_text(p) -> str:
        return format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    sds = np.array([v.std(ddof=1) for v in values])
    if np.all(sds == 0):
        raise AnalysisError("Every group has no spread at all - all its values are the same.")

    rows, verdicts = [], []

    def add(name: str, based_on: str, statistic: str, df: str, p: float) -> None:
        verdicts.append((name, p))
        rows.append([name, based_on, statistic, df, p_text(p), "spreads differ" if p < opt.alpha else "no clear difference"])

    if opt.brown_forsythe:
        w, p = stats.levene(*values, center="median")
        add("Brown-Forsythe", "distances from each group's median", f"F = {w:.4f}", f"{k - 1}, {total - k}", float(p))
    if opt.levene:
        w, p = stats.levene(*values, center="mean")
        add("Levene", "distances from each group's mean", f"F = {w:.4f}", f"{k - 1}, {total - k}", float(p))
    if opt.bartlett and np.all(sds > 0):
        chi2, p = stats.bartlett(*values)
        add("Bartlett", "the variances themselves; trusts a normal shape", f"Chi-square = {chi2:.4f}", str(k - 1), float(p))
    if opt.fligner:
        chi2, p = stats.fligner(*values)
        add("Fligner-Killeen", "ranks of the distances from the median", f"Chi-square = {chi2:.4f}", str(k - 1), float(p))
    if opt.f_test and k == 2 and np.all(sds > 0):
        big, little = (0, 1) if sds[0] >= sds[1] else (1, 0)
        f_value = float(sds[big] ** 2 / sds[little] ** 2)
        df1, df2 = values[big].size - 1, values[little].size - 1
        p = float(min(1.0, 2 * stats.f.sf(f_value, df1, df2)))
        add("F test of two variances", f"variance of {names[big]} over that of {names[little]}; trusts a normal shape",
            f"F = {f_value:.4f}", f"{df1}, {df2}", p)
    if not rows:
        raise AnalysisError("Choose at least one test.")
    tables = [ResultTable("Are the spreads the same?", ["Test", "Works on", "Statistic", "df", "P value", "Reading"], rows)]

    largest, smallest = int(np.argmax(sds)), int(np.argmin(sds))
    ratio = float(sds[largest] / sds[smallest]) if sds[smallest] > 0 else float("inf")
    medians = [float(np.median(v)) for v in values]
    tables.append(ResultTable("Spread of each group",
                              ["Group", "N", "Mean", "SD", "Variance", "Interquartile range", "Typical distance from the median"],
                              [[names[i], str(v.size), num(v.mean()), num(sds[i]), num(sds[i] ** 2),
                                num(float(np.subtract(*np.percentile(v, [75, 25])))), num(float(np.mean(np.abs(v - medians[i]))))]
                               for i, v in enumerate(values)]))
    sizes = np.array([v.size for v in values])
    tables.append(ResultTable("How far apart the spreads are", ["Measure", "Value"],
                              [["Largest SD", f"{num(sds[largest])} ({names[largest]})"], ["Smallest SD", f"{num(sds[smallest])} ({names[smallest]})"],
                               ["Largest SD over smallest", "no spread in one group" if not np.isfinite(ratio) else f"{ratio:.3g}"],
                               ["Largest variance over smallest", "" if not np.isfinite(ratio) else f"{ratio ** 2:.3g}"],
                               ["Largest group over smallest", f"{sizes.max() / sizes.min():.3g}"]]))

    lead_name, lead_p = verdicts[0]
    differ = lead_p < opt.alpha
    summary = (f"{lead_name}: P = {lead_p:.4g} - {'the spreads differ' if differ else 'no clear difference between the spreads'}. "
               f"The largest SD ({names[largest]}) is {ratio:.2g} times the smallest ({names[smallest]}).")
    disagree = {p < opt.alpha for _n, p in verdicts}
    notes = []
    if len(disagree) > 1:
        notes.append("The tests disagree. Bartlett's test and the F test read any departure from a normal shape as unequal "
                     "spread; when they alone say 'differ', look at the shape of the values before believing them.")
    if differ or (np.isfinite(ratio) and ratio > 2):
        which = "Welch's t test" if k == 2 else "Welch's ANOVA with Games-Howell comparisons"
        notes.append(f"To compare the means of these groups, use {which}: it does not need equal spreads, and loses almost "
                     "nothing when they are equal.")
        if np.all(np.concatenate(values) > 0):
            notes.append("If the SD rises with the mean, the values may be lognormal: comparing their logarithms often makes "
                         "the spreads equal.")
    else:
        notes.append("No clear difference is not proof of equal spreads, least of all in small groups, where these tests "
                     "miss real differences. Welch's tests are a safe choice either way.")
    if sizes.max() / sizes.min() > 1.5:
        notes.append("The groups differ in size, which is when unequal spreads distort an ordinary t test or ANOVA most.")
    if small:
        notes.append("Left out, with fewer than 3 values: " + ", ".join(small) + ".")
    notes.append("Brown-Forsythe and Levene's tests are one-way ANOVAs on each value's distance from its group's centre "
                 "(Levene 1960; Brown and Forsythe 1974). Bartlett 1937; Fligner and Killeen 1976 (in Conover's form).")

    # ── graphs: the values themselves, and each value's distance from its group's median ──
    boxes = []
    for name, v in groups.items():
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        far = (v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)
        near = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)) & ~far
        boxes.append({"label": name, "q1": q1, "median": med, "q3": q3, "low": float(inside.min()), "high": float(inside.max()),
                      "outside": v[near].tolist(), "far": v[far].tolist(), "flagged": []})
    from graphs.engine.style import PALETTES

    palette = PALETTES["classic"]
    rng = np.random.default_rng(0)
    series = []
    for i, v in enumerate(values):
        jitter = (rng.random(v.size) - 0.5) * 0.36
        series.append(ChartSeries(names[i], (i + jitter).tolist(), np.abs(v - medians[i]).tolist(), kind="scatter", color=palette[i % len(palette)]))
    lines = [{"kind": "path", "points": [[i - 0.3, float(np.mean(np.abs(v - medians[i])))], [i + 0.3, float(np.mean(np.abs(v - medians[i])))]],
              "color": "#222222", "style": "-", "width": 1.8} for i, v in enumerate(values)]
    charts = [ChartSpec("Each value's distance from its group's median (bars: the mean distance)", series=series, x_label="Group",
                        y_label="Distance from the median", reference_lines=lines, x_categories=names, x_limits=(-0.6, k - 0.4)),
              ChartSpec("The values, group by group", kind="boxes", y_label="Value", extra={"boxes": boxes})]
    return AnalysisResult(title="Is the spread the same?", summary=summary, tables=tables, notes=notes, charts=charts)
