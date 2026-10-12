"""Compare groups after allowing for a covariate (analysis of covariance).

The sheet's X column holds the covariate - a baseline, an age, a dose - and
each data set holds one group's outcome at that X. The groups are compared
at the same value of the covariate: the fitted lines are given one common
slope, and the gap between them is the difference between the groups.

Whether one slope is fair is tested first (the group-by-covariate term).
Sums of squares are for each term given the other (Type III).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _common


@dataclass
class AncovaOptions:
    test_slopes: bool = True
    compare: str = "all_pairs"     # "none" | "all_pairs" | "vs_control"
    control_index: int = 0
    correction: str = "holm"       # "holm" | "fdr" | "none"
    show_unadjusted: bool = True   # the plain one-way ANOVA beside it, to show what the covariate changed
    at_value: str = ""             # compare the groups at this covariate value; empty = its overall mean
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def _rss(design: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of a least-squares fit."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(((y - design @ beta) ** 2).sum())


def read_groups(data: SheetData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{group: (covariate, outcome)} from an X-and-Y sheet, rows with both filled."""
    x = np.asarray(data.x, dtype=float)
    out = {}
    for label in data.group_labels():
        _reps, matrix = data.group_replicate_matrix(label)
        if not matrix.size:
            continue
        with np.errstate(all="ignore"):
            y = np.nanmean(matrix, axis=1) if matrix.shape[1] > 1 else matrix[:, 0]
        n = min(x.size, y.size)
        keep = np.isfinite(x[:n]) & np.isfinite(y[:n])
        if keep.sum() >= 2:
            out[label] = (x[:n][keep], y[:n][keep])
    return out


@register("ancova")
def run(data: SheetData, options: AncovaOptions | None = None) -> AnalysisResult:
    opt = options or AncovaOptions()
    groups = read_groups(data)
    if len(groups) < 2:
        raise AnalysisError("This needs two groups or more, each with a covariate in the X column and an outcome in its own data set.")
    names = list(groups)
    k = len(names)
    xs = np.concatenate([groups[g][0] for g in names])
    ys = np.concatenate([groups[g][1] for g in names])
    which = np.concatenate([np.full(groups[g][0].size, i) for i, g in enumerate(names)])
    n = ys.size
    if n - k - 1 < 1:
        raise AnalysisError("Too few rows: there must be more rows than groups plus one.")
    if np.ptp(xs) == 0:
        raise AnalysisError("The covariate in the X column is the same in every row, so there is nothing to allow for.")
    sig = opt.significant_digits

    def num(value) -> str:
        return f"{value:.{sig}g}"

    def p_text(p) -> str:
        return format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    dummies = np.stack([(which == i).astype(float) for i in range(k)], axis=1)        # one column per group
    rss_parallel = _rss(np.column_stack([dummies, xs]), ys)                           # own intercepts, one slope
    rss_separate = _rss(np.column_stack([dummies, dummies * xs[:, None]]), ys)        # own intercepts, own slopes
    rss_groups = _rss(dummies, ys)                                                    # no covariate
    rss_line = _rss(np.column_stack([np.ones(n), xs]), ys)                            # no groups
    df_error = n - k - 1
    mse = rss_parallel / df_error
    if mse <= 0:
        raise AnalysisError("The outcome is fitted without any error left over; there is nothing to test against.")

    # the common slope and the group lines
    means_x = np.array([groups[g][0].mean() for g in names])
    means_y = np.array([groups[g][1].mean() for g in names])
    sizes = np.array([groups[g][0].size for g in names])
    sxx_within = float(sum(((groups[g][0] - groups[g][0].mean()) ** 2).sum() for g in names))
    if sxx_within == 0:
        raise AnalysisError("Within every group the covariate takes a single value, so its slope cannot be found.")
    slope = float(sum(((groups[g][0] - groups[g][0].mean()) * (groups[g][1] - groups[g][1].mean())).sum() for g in names) / sxx_within)
    try:
        at = float(opt.at_value) if str(opt.at_value).strip() else float(xs.mean())
    except ValueError:
        at = float(xs.mean())
    adjusted = means_y - slope * (means_x - at)
    se_adjusted = np.sqrt(mse * (1 / sizes + (means_x - at) ** 2 / sxx_within))
    t_crit = stats.t.ppf(0.5 + opt.confidence_level / 2, df_error)
    pct = f"{round(opt.confidence_level * 100)}%"

    tables: list[ResultTable] = []
    notes: list[str] = []

    # ── is one slope fair? ───────────────────────────────────────────────────
    slopes_differ = False
    if opt.test_slopes and n - 2 * k >= 1 and rss_separate > 0:
        f_slopes = ((rss_parallel - rss_separate) / (k - 1)) / (rss_separate / (n - 2 * k))
        p_slopes = float(stats.f.sf(f_slopes, k - 1, n - 2 * k))
        slopes_differ = p_slopes < opt.alpha
        own = []
        for g in names:
            gx, gy = groups[g]
            sxx = ((gx - gx.mean()) ** 2).sum()
            own.append(float(((gx - gx.mean()) * (gy - gy.mean())).sum() / sxx) if sxx > 0 else float("nan"))
        tables.append(ResultTable("Do the groups share one slope?", ["Test", "F", "df", "P value", "Reading"],
                                  [["Group × covariate", f"{f_slopes:.4f}", f"{k - 1}, {n - 2 * k}", p_text(p_slopes),
                                    "the slopes differ: read the comparison of the groups with care" if slopes_differ else
                                    "no sign that the slopes differ: one slope is fair"]]))
        tables.append(ResultTable("Slope of the outcome on the covariate", ["Group", "N", "Own slope", "Common slope"],
                                  [[g, str(int(sizes[i])), num(own[i]) if np.isfinite(own[i]) else "", num(slope)] for i, g in enumerate(names)]))
        if slopes_differ:
            notes.append("The slopes differ between the groups, so the gap between the groups depends on where along the covariate "
                         "it is measured. The adjusted means below are the gap at one value only; consider reporting the separate lines.")

    # ── the analysis of covariance ───────────────────────────────────────────
    ss_cov, ss_group = rss_groups - rss_parallel, rss_line - rss_parallel
    f_cov = ss_cov / mse
    f_group = (ss_group / (k - 1)) / mse
    p_cov, p_group = float(stats.f.sf(f_cov, 1, df_error)), float(stats.f.sf(f_group, k - 1, df_error))
    eta_group = ss_group / (ss_group + rss_parallel)
    eta_cov = ss_cov / (ss_cov + rss_parallel)
    tables.insert(0, ResultTable("Groups compared at the same covariate",
                                 ["Source", "Sum of squares", "df", "Mean square", "F", "P value", "Partial eta squared"],
                                 [["Groups (covariate allowed for)", num(ss_group), str(k - 1), num(ss_group / (k - 1)), f"{f_group:.4f}", p_text(p_group), f"{eta_group:.3f}"],
                                  ["Covariate", num(ss_cov), "1", num(ss_cov), f"{f_cov:.4f}", p_text(p_cov), f"{eta_cov:.3f}"],
                                  ["Residual", num(rss_parallel), str(df_error), num(mse), "", "", ""]]))

    tables.insert(1, ResultTable(f"Group means, as measured and at covariate = {num(at)}",
                                 ["Group", "N", "Mean of covariate", "Mean as measured", "Adjusted mean", "SE", f"{pct} CI"],
                                 [[g, str(int(sizes[i])), num(means_x[i]), num(means_y[i]), num(adjusted[i]), num(se_adjusted[i]),
                                   f"{num(adjusted[i] - t_crit * se_adjusted[i])} to {num(adjusted[i] + t_crit * se_adjusted[i])}"]
                                  for i, g in enumerate(names)]))

    # ── pairs of adjusted means ──────────────────────────────────────────────
    if opt.compare != "none" and k >= 2:
        if opt.compare == "vs_control" and k > 2:
            c = max(0, min(opt.control_index, k - 1))
            pairs = [(j, c) for j in range(k) if j != c]
        else:
            pairs = list(combinations(range(k), 2))
        rows, raw = [], []
        for i, j in pairs:
            diff = adjusted[i] - adjusted[j]
            se = np.sqrt(mse * (1 / sizes[i] + 1 / sizes[j] + (means_x[i] - means_x[j]) ** 2 / sxx_within))
            t = diff / se
            p = float(2 * stats.t.sf(abs(t), df_error))
            raw.append(p)
            rows.append([f"{names[i]} − {names[j]}", num(diff), num(se), f"{num(diff - t_crit * se)} to {num(diff + t_crit * se)}", f"{t:.4f}", p_text(p)])
        headers = ["Comparison", "Difference of adjusted means", "SE", f"{pct} CI", "t", "P value"]
        if len(pairs) > 1 and opt.correction != "none":
            adj = _common.adjust(raw, opt.correction)
            for row, a in zip(rows, adj):
                row += [p_text(float(a)), "Yes" if a < opt.alpha else "No"]
            name = {"fdr": "Benjamini-Hochberg"}.get(opt.correction, "Holm")
            headers += [f"P value ({name})", f"Significant (α = {opt.alpha:g})"]
        tables.insert(2, ResultTable("Pairs of adjusted means", headers, rows))

    if opt.show_unadjusted:
        grand = ys.mean()
        ss_between = float((sizes * (means_y - grand) ** 2).sum())
        f_plain = (ss_between / (k - 1)) / (rss_groups / (n - k))
        p_plain = float(stats.f.sf(f_plain, k - 1, n - k))
        tables.append(ResultTable("What the covariate changed", ["Analysis", "F for groups", "df", "P value", "Residual SD"],
                                  [["Covariate allowed for", f"{f_group:.4f}", f"{k - 1}, {df_error}", p_text(p_group), num(np.sqrt(mse))],
                                   ["Covariate ignored (one-way ANOVA)", f"{f_plain:.4f}", f"{k - 1}, {n - k}", p_text(p_plain),
                                    num(np.sqrt(rss_groups / (n - k)))]]))

    best = int(np.argmax(adjusted)), int(np.argmin(adjusted))
    summary = (f"With the covariate allowed for, the groups {'differ' if p_group < opt.alpha else 'do not differ clearly'}: "
               f"F({k - 1}, {df_error}) = {f_group:.3f}, P = {p_group:.4g}. Adjusted means run from {num(adjusted[best[1]])} "
               f"({names[best[1]]}) to {num(adjusted[best[0]])} ({names[best[0]]}). "
               f"The covariate itself {'matters' if p_cov < opt.alpha else 'shows no clear link with the outcome'} (P = {p_cov:.4g}).")
    if slopes_differ:
        summary += " The groups' slopes differ, which this comparison assumes they do not."

    # ── graphs ───────────────────────────────────────────────────────────────
    from graphs.engine.style import PALETTES

    palette = PALETTES["classic"]
    series, lines = [], []
    for i, g in enumerate(names):
        gx, gy = groups[g]
        colour = palette[i % len(palette)]
        series.append(ChartSeries(g, gx.tolist(), gy.tolist(), kind="scatter", color=colour))
        intercept = means_y[i] - slope * means_x[i]
        lo, hi = float(gx.min()), float(gx.max())
        lines.append({"kind": "path", "points": [[lo, intercept + slope * lo], [hi, intercept + slope * hi]], "color": colour,
                      "style": "-", "width": 1.6})
    lines.append({"kind": "v", "value": at, "color": "#888888", "style": ":", "width": 1.0})
    charts = [ChartSpec("Outcome against the covariate, one slope for all groups", series=series, x_label="Covariate (X)",
                        y_label="Outcome", reference_lines=lines),
              ChartSpec(f"Adjusted means with {pct} confidence intervals", x_label="Group", y_label="Adjusted mean",
                        series=[ChartSeries("", [float(i) for i in range(k)], adjusted.tolist(), kind="scatter", color="#1f5aa6")],
                        reference_lines=[{"kind": "path", "points": [[i, adjusted[i] - t_crit * se_adjusted[i]], [i, adjusted[i] + t_crit * se_adjusted[i]]],
                                          "color": "#1f5aa6", "style": "-", "width": 1.6} for i in range(k)],
                        x_categories=names, x_limits=(-0.6, k - 0.4))]

    notes = [f"An adjusted mean is the group's mean outcome moved along the common slope to covariate = {num(at)}"
             + ("" if str(opt.at_value).strip() else ", the mean of the covariate over all rows") + ". The dotted line on the graph marks it.",
             "It assumes a straight-line link between covariate and outcome, the same slope in every group, and scatter about "
             "the lines that is normal and equally wide in each group.",
             "The covariate must not itself be changed by the treatment: use a baseline measured before it, not something "
             "measured after."] + notes
    if opt.compare != "none" and k > 2:
        notes.append("Pairs are compared with t tests on the adjusted means, using the residual of the whole model.")
    notes.append("Sums of squares are Type III: each term is tested given the other. Reference: Huitema, The Analysis of Covariance "
                 "and Alternatives (2011).")
    return AnalysisResult(title="Groups, allowing for a covariate", summary=summary, tables=tables, notes=notes, charts=charts)
