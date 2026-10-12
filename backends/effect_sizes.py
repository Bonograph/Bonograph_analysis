"""How large is the difference? Standardised effect sizes with confidence
intervals, to report beside a P value.

Two groups: Cohen's d, Hedges' g, Glass's delta, and two measures that make
no assumption about shape - the probability that a value from one group
exceeds a value from the other, and the rank-biserial correlation.
Three groups or more: eta squared, omega squared, epsilon squared and
Cohen's f for the whole design, then d for each pair.

Intervals for d and for eta squared come from the noncentral t and F
distributions (Steiger and Fouladi 1997; Cumming and Finch 2001), not from a
normal approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize, special, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class EffectSizeOptions:
    paired: bool = False             # each row is one subject measured under every condition
    compare: str = "all_pairs"       # three groups or more: "all_pairs" | "vs_control"
    control_index: int = 0
    show_rank_based: bool = True     # the measures that assume no shape
    show_reading: bool = True        # small / medium / large, by Cohen's rough guide
    confidence_level: float = 0.95
    significant_digits: int = 4


def hedges_correction(df: float) -> float:
    """The factor that removes the small-sample bias of d (Hedges 1981), exact."""
    if df < 2:
        return 1.0
    return float(np.exp(special.gammaln(df / 2) - np.log(np.sqrt(df / 2)) - special.gammaln((df - 1) / 2)))


def noncentral_t_limits(t: float, df: float, level: float) -> tuple[float, float]:
    """Confidence limits of the noncentrality parameter of an observed t."""
    if t < 0:      # the distribution is a mirror image for a negative t, and the positive side is the one worked out reliably
        low, high = noncentral_t_limits(-t, df, level)
        return -high, -low
    tail = (1 - level) / 2

    def limit(p: float) -> float:
        def f(ncp):
            return stats.nct.cdf(t, df, ncp) - p

        lo, hi = t - 10, t + 10
        for _ in range(60):
            if f(lo) > 0:
                break
            lo -= 10
        for _ in range(60):
            if f(hi) < 0:
                break
            hi += 10
        try:
            return float(optimize.brentq(f, lo, hi, xtol=1e-10))
        except ValueError:
            return float("nan")

    return limit(1 - tail), limit(tail)


def noncentral_f_limits(f_value: float, df1: float, df2: float, level: float) -> tuple[float, float]:
    """Confidence limits of the noncentrality parameter of an observed F (0 where the limit would be negative)."""
    tail = (1 - level) / 2

    def limit(p: float) -> float:
        def g(nc):
            return stats.ncf.cdf(f_value, df1, df2, nc) - p

        if stats.f.cdf(f_value, df1, df2) <= p:
            return 0.0
        hi = max(10.0, f_value * df1 * 4)
        for _ in range(60):
            if g(hi) < 0:
                break
            hi *= 2
        try:
            return float(optimize.brentq(g, 0.0, hi, xtol=1e-10))
        except ValueError:
            return float("nan")

    return limit(1 - tail), limit(tail)


def _reading(value: float, cuts: tuple[float, float, float]) -> str:
    v = abs(value)
    if not np.isfinite(v):
        return ""
    return "very small" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def _two_groups(a: np.ndarray, b: np.ndarray, level: float) -> dict:
    """a against b, separate subjects: every measure, with limits where they exist."""
    n1, n2 = a.size, b.size
    df = n1 + n2 - 2
    diff = float(a.mean() - b.mean())
    pooled = float(np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / df))
    out = {"n": f"{n1}, {n2}", "difference": diff}
    if pooled > 0:
        d = diff / pooled
        scale = np.sqrt(n1 * n2 / (n1 + n2))
        lo, hi = noncentral_t_limits(d * scale, df, level)
        j = hedges_correction(df)
        out.update(d=d, d_ci=(lo / scale, hi / scale), g=d * j, g_ci=(lo / scale * j, hi / scale * j))
        sd_b = float(b.std(ddof=1))
        if sd_b > 0:
            out["glass"] = diff / sd_b
    # P(a value of A is above a value of B), ties counted half: the Mann-Whitney U over n1 n2,
    # from the ranks of all the values together (comparing every pair would not do for long columns)
    ranks = stats.rankdata(np.concatenate([a, b]))
    prob = float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))
    out.update(superiority=prob, rank_biserial=2 * prob - 1)
    return out


def _paired(a: np.ndarray, b: np.ndarray, level: float) -> dict:
    """a against b, the same subjects: d from the differences and from the average SD."""
    diffs = a - b
    n = diffs.size
    df = n - 1
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    out = {"n": str(n), "difference": mean}
    if sd > 0:
        dz = mean / sd
        lo, hi = noncentral_t_limits(dz * np.sqrt(n), df, level)
        j = hedges_correction(df)
        out.update(d=dz, d_ci=(lo / np.sqrt(n), hi / np.sqrt(n)), g=dz * j, g_ci=(lo / np.sqrt(n) * j, hi / np.sqrt(n) * j))
    average_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    if average_sd > 0:
        out["d_av"] = mean / average_sd
    # matched pairs: share of signed ranks that favour A, as a correlation (Kerby 2014)
    nonzero = diffs[diffs != 0]
    if nonzero.size:
        ranks = stats.rankdata(np.abs(nonzero))
        out["rank_biserial"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        out["superiority"] = float((np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / n)
    return out


@register("effect_sizes")
def run(data: SheetData, options: EffectSizeOptions | None = None) -> AnalysisResult:
    opt = options or EffectSizeOptions()
    level = opt.confidence_level
    pct = f"{round(level * 100)}%"
    sig = opt.significant_digits

    def num(value) -> str:
        return "" if value is None or not np.isfinite(value) else f"{value:.{sig}g}"

    def span(pair) -> str:
        return "" if pair is None or not all(np.isfinite(pair)) else f"{num(pair[0])} to {num(pair[1])}"

    labels = data.group_labels()
    if opt.paired:
        names, matrix = data.first_replicate_matrix()
        if len(names) < 2:
            raise AnalysisError("Effect sizes compare groups: this sheet needs at least two data sets.")
        complete = ~np.any(np.isnan(matrix), axis=1)
        if complete.sum() < 3:
            raise AnalysisError("Paired effect sizes need at least 3 rows with a value in every data set.")
        groups = {name: matrix[complete, i] for i, name in enumerate(names)}
    else:
        groups = {label: v[np.isfinite(v)] for label, v in data.all_group_arrays(mode="average").items()}
        groups = {label: v for label, v in groups.items() if v.size >= 2}
        if len(groups) < 2:
            raise AnalysisError("Effect sizes compare groups: this sheet needs at least two data sets with 2 values or more.")
    names = list(groups)
    k = len(names)
    tables: list[ResultTable] = []
    notes: list[str] = []
    summary_parts: list[str] = []

    # ── the whole design, for three groups or more ───────────────────────────
    if k >= 3 and not opt.paired:
        values = list(groups.values())
        total_n = sum(v.size for v in values)
        grand = np.concatenate(values).mean()
        ss_between = float(sum(v.size * (v.mean() - grand) ** 2 for v in values))
        ss_within = float(sum(((v - v.mean()) ** 2).sum() for v in values))
        ss_total = ss_between + ss_within
        df1, df2 = k - 1, total_n - k
        if ss_total > 0 and ss_within > 0 and df2 > 0:
            ms_within = ss_within / df2
            f_value = (ss_between / df1) / ms_within
            eta = ss_between / ss_total
            lo, hi = noncentral_f_limits(f_value, df1, df2, level)
            eta_ci = (lo / (lo + total_n), hi / (hi + total_n))
            omega = max(0.0, (ss_between - df1 * ms_within) / (ss_total + ms_within))
            epsilon = max(0.0, (ss_between - df1 * ms_within) / ss_total)
            cohen_f = float(np.sqrt(eta / (1 - eta))) if eta < 1 else float("inf")
            rows = [["Eta squared", num(eta), span(eta_ci), "Share of all the variation that lies between the groups; runs high in small samples",
                     _reading(eta, (0.01, 0.06, 0.14))],
                    ["Omega squared", num(omega), "", "The same share, corrected for that bias - the one to report", _reading(omega, (0.01, 0.06, 0.14))],
                    ["Epsilon squared", num(epsilon), "", "Another corrected form, a little less cautious", _reading(epsilon, (0.01, 0.06, 0.14))],
                    ["Cohen's f", num(cohen_f), span((float(np.sqrt(lo / total_n)), float(np.sqrt(hi / total_n)))),
                     "Spread of the group means in SDs; what a power calculation asks for", _reading(cohen_f, (0.10, 0.25, 0.40))]]
            headers = ["Measure", "Value", f"{pct} CI", "What it says", "Reading"]
            if not opt.show_reading:
                headers, rows = headers[:-1], [r[:-1] for r in rows]
            tables.append(ResultTable(f"All {k} groups together (F = {f_value:.4g}, df {df1}, {df2})", headers, rows))
            summary_parts.append(f"Across the {k} groups, omega squared = {omega:.3g} (eta squared {eta:.3g}, {pct} CI {span(eta_ci)})")
            notes.append("Eta squared's interval comes from the noncentral F distribution. A lower limit of 0 means that no "
                         "difference between the groups cannot be ruled out.")

    # ── pairs ────────────────────────────────────────────────────────────────
    if k == 2:
        pairs = [(0, 1)]
    elif opt.compare == "vs_control":
        c = max(0, min(opt.control_index, k - 1))
        pairs = [(j, c) for j in range(k) if j != c]
    else:
        pairs = list(combinations(range(k), 2))
    one = _paired if opt.paired else _two_groups
    found = []
    for i, j in pairs:
        res = one(groups[names[i]], groups[names[j]], level)
        res["label"] = f"{names[i]} − {names[j]}"
        found.append(res)

    rows = []
    for res in found:
        row = [res["label"], res["n"], num(res["difference"]), num(res.get("d")), span(res.get("d_ci")), num(res.get("g")), span(res.get("g_ci"))]
        if opt.show_reading:
            row.append(_reading(res.get("g", float("nan")), (0.2, 0.5, 0.8)))
        rows.append(row)
    d_name = "d (from the differences)" if opt.paired else "Cohen's d"
    headers = ["Comparison", "N", "Difference of means", d_name, f"{pct} CI", "Hedges' g", f"{pct} CI"] + (["Reading"] if opt.show_reading else [])
    tables.insert(0, ResultTable("Standardised difference between means", headers, rows))

    extra_rows = []
    for res in found:
        if opt.paired:
            extra_rows.append([res["label"], num(res.get("d_av")), "The difference in units of the two conditions' average SD - comparable "
                               "with d from studies of separate groups"])
        elif "glass" in res:
            second = res["label"].split(" − ")[1]
            extra_rows.append([res["label"], num(res["glass"]), f"The difference in SDs of {second} alone - for when a treatment changes the spread"])
    if extra_rows:
        tables.append(ResultTable("d on the average SD" if opt.paired else "Glass's delta", ["Comparison", "Value", "What it says"], extra_rows))

    if opt.show_rank_based:
        rank_rows = []
        for res in found:
            if "superiority" not in res:
                continue
            first, second = res["label"].split(" − ")
            says = (f"In {res['superiority']:.0%} of subjects {first} is the higher" if opt.paired else
                    f"A value from {first} is above a value from {second} {res['superiority']:.0%} of the time")
            rank_rows.append([res["label"], f"{res['superiority']:.3f}", num(res["rank_biserial"]), says])
        if rank_rows:
            tables.append(ResultTable("Without assuming a shape", ["Comparison", "Probability of being higher", "Rank-biserial correlation", "In words"],
                                      rank_rows))
            notes.append("The probability of being higher (0.5 = no difference) and the rank-biserial correlation (0 = no difference, "
                         "±1 = no overlap) use only the order of the values, so skewed data and outliers do not distort them. "
                         "They go with a Mann-Whitney or Wilcoxon test.")

    best = max(found, key=lambda r: abs(r.get("g", 0.0)) if np.isfinite(r.get("g", 0.0)) else 0.0)
    if "g" in best:
        summary_parts.insert(0, f"{best['label']}: Hedges' g = {best['g']:.3g} ({pct} CI {span(best['g_ci'])})"
                             + (f", {_reading(best['g'], (0.2, 0.5, 0.8))} by Cohen's rough guide" if opt.show_reading else ""))
    summary = "; ".join(summary_parts) + "." if summary_parts else "The groups have no spread: effect sizes cannot be worked out."

    # ── graph: each g with its interval ──────────────────────────────────────
    charts = []
    drawn = [r for r in found if "g" in r and all(np.isfinite(r["g_ci"]))]
    if drawn:
        xs = list(range(len(drawn)))
        lines = [{"kind": "path", "points": [[x, r["g_ci"][0]], [x, r["g_ci"][1]]], "color": "#1f5aa6", "style": "-", "width": 1.6}
                 for x, r in zip(xs, drawn)]
        lines.append({"kind": "h", "value": 0.0, "color": "#888888", "style": "--", "width": 0.9})
        charts.append(ChartSpec(f"Hedges' g with its {pct} confidence interval", x_label="Comparison", y_label="Hedges' g (SDs)",
                                series=[ChartSeries("", [float(x) for x in xs], [r["g"] for r in drawn], kind="scatter", color="#1f5aa6")],
                                reference_lines=lines, x_categories=[r["label"] for r in drawn],
                                x_limits=(-0.6, len(drawn) - 0.4)))

    notes = [("d is the mean difference divided by the SD of the differences; it grows with the correlation between the two "
              "measurements, so it is not comparable with d from separate groups - use d on the average SD for that."
              if opt.paired else
              "Cohen's d is the difference between two means in units of their pooled SD. Hedges' g is d with the small-sample "
              "bias taken off, and is the one to report; with 20 values or more per group the two barely differ."),
             f"Intervals come from the noncentral t distribution. An interval that includes 0 goes with a P value above {1 - level:.2g}.",
             ] + notes
    if opt.show_reading:
        notes.append("'Small', 'medium' and 'large' follow Cohen's rough guide (g of 0.2, 0.5, 0.8; eta squared of 0.01, 0.06, 0.14). "
                     "He offered it for when nothing better is known: what counts as large depends on the field.")
    if k > 2:
        notes.append("Each pair uses the pooled SD of its own two groups. No adjustment is made for the number of pairs: these are "
                     "estimates of size, not tests.")
    notes.append("References: Cohen 1988; Hedges 1981; Cumming and Finch 2001; Lakens 2013 (Frontiers in Psychology 4:863).")
    return AnalysisResult(title="How large is the difference?", summary=summary, tables=tables, notes=notes, charts=charts)
