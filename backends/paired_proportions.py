"""Yes/no outcomes on the same subjects: has the proportion changed?

Each row is a subject, each data set a condition, time point or method, and
every value a code for yes or no. Two data sets: McNemar's test, which looks
only at the subjects who changed. Three or more: Cochran's Q, then McNemar's
test pair by pair.
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
class PairedProportionsOptions:
    yes_code: float = 1.0
    no_code: float = 0.0
    exact: bool = True             # exact binomial P for a pair (the chi-square form is shown beside it)
    compare: str = "all_pairs"     # three data sets or more: "none" | "all_pairs" | "vs_control"
    control_index: int = 0
    correction: str = "holm"       # "holm" | "fdr" | "none"
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def mcnemar(first: np.ndarray, second: np.ndarray, level: float = 0.95) -> dict:
    """McNemar's test on two yes/no (True/False) arrays for the same subjects."""
    n = first.size
    both = int(np.sum(first & second))
    only_first = int(np.sum(first & ~second))        # yes then no
    only_second = int(np.sum(~first & second))       # no then yes
    neither = n - both - only_first - only_second
    changed = only_first + only_second
    out = {"n": n, "both": both, "only_first": only_first, "only_second": only_second, "neither": neither, "changed": changed,
           "p1": (both + only_first) / n, "p2": (both + only_second) / n}
    if changed == 0:
        out.update(p_exact=1.0, p_chi2=1.0, chi2=0.0, p_mid=1.0)
    else:
        low = min(only_first, only_second)
        exact = float(min(1.0, 2 * stats.binom.cdf(low, changed, 0.5)))
        out["p_exact"] = exact
        out["p_mid"] = float(min(1.0, exact - stats.binom.pmf(low, changed, 0.5)))      # mid-P: half the weight of the observed table
        chi2 = (abs(only_first - only_second) - 1) ** 2 / changed
        out["chi2"] = float(max(chi2, 0.0)) if abs(only_first - only_second) >= 1 else 0.0
        out["p_chi2"] = float(stats.chi2.sf(out["chi2"], 1))
    z = stats.norm.ppf(0.5 + level / 2)
    diff = (only_second - only_first) / n            # second minus first
    se = np.sqrt(max(changed - (only_second - only_first) ** 2 / n, 0.0)) / n
    out.update(diff=diff, diff_ci=(max(-1.0, diff - z * se), min(1.0, diff + z * se)))
    if only_first > 0 and only_second > 0:
        tail = (1 - level) / 2
        lo = stats.beta.ppf(tail, only_second, only_first + 1)                # exact limits of the share of changers who became yes
        hi = stats.beta.ppf(1 - tail, only_second + 1, only_first)
        out.update(odds=only_second / only_first, odds_ci=(lo / (1 - lo), hi / (1 - hi)))
    return out


def cochran_q(matrix: np.ndarray) -> tuple[float, int, float]:
    """Cochran's Q on a subjects x conditions array of 0/1: (Q, df, P)."""
    k = matrix.shape[1]
    cols, rows = matrix.sum(axis=0), matrix.sum(axis=1)
    total = cols.sum()
    bottom = k * total - (rows**2).sum()
    if bottom == 0:
        return 0.0, k - 1, 1.0
    q = float((k - 1) * (k * (cols**2).sum() - total**2) / bottom)
    return q, k - 1, float(stats.chi2.sf(q, k - 1))


@register("paired_proportions")
def run(data: SheetData, options: PairedProportionsOptions | None = None) -> AnalysisResult:
    opt = options or PairedProportionsOptions()
    names, matrix = data.first_replicate_matrix()
    if len(names) < 2:
        raise AnalysisError("This needs two data sets or more of yes/no codes, with one subject on each row.")
    known = np.isin(matrix, (opt.yes_code, opt.no_code))
    filled = np.isfinite(matrix)
    odd = int((filled & ~known).sum())
    keep = known.all(axis=1)
    dropped = int(filled.any(axis=1).sum() - keep.sum())
    yes = matrix[keep] == opt.yes_code
    n, k = yes.shape
    if n < 2:
        raise AnalysisError(f"Fewer than 2 rows hold a yes ({opt.yes_code:g}) or no ({opt.no_code:g}) code in every data set.")
    sig = opt.significant_digits
    level = opt.confidence_level
    pct = f"{round(level * 100)}%"

    def p_text(p) -> str:
        return format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    def share(v) -> str:
        return f"{v:.1%}"

    def points(pair) -> str:
        return f"{pair[0] * 100:+.1f} to {pair[1] * 100:+.1f}"

    props = yes.mean(axis=0)
    z = stats.norm.ppf(0.5 + level / 2)
    prop_rows = []
    for j in range(k):
        centre = (props[j] + z * z / (2 * n)) / (1 + z * z / n)                       # Wilson's interval
        half = z * np.sqrt(props[j] * (1 - props[j]) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        prop_rows.append([names[j], str(int(yes[:, j].sum())), str(n), share(props[j]), f"{(centre - half) * 100:.1f} to {(centre + half) * 100:.1f}"])
    tables = [ResultTable("Share of yes in each data set", ["Data set", "Yes", "Subjects", "Share", f"{pct} CI (%)"], prop_rows)]
    notes: list[str] = []

    def pair_table(i: int, j: int, res: dict) -> ResultTable:
        return ResultTable(f"Subjects by what happened: {names[i]} then {names[j]}", ["", f"{names[j]}: yes", f"{names[j]}: no", "Total"],
                           [[f"{names[i]}: yes", str(res["both"]), str(res["only_first"]), str(res["both"] + res["only_first"])],
                            [f"{names[i]}: no", str(res["only_second"]), str(res["neither"]), str(res["only_second"] + res["neither"])],
                            ["Total", str(res["both"] + res["only_second"]), str(res["only_first"] + res["neither"]), str(n)]])

    if k == 2:
        res = mcnemar(yes[:, 0], yes[:, 1], level)
        p_lead = res["p_exact"] if opt.exact else res["p_chi2"]
        tests = [["McNemar, exact (binomial)", "", f"{res['changed']} changed", p_text(res["p_exact"])],
                 ["McNemar, mid-P", "", "", p_text(res["p_mid"])],
                 ["McNemar, chi-square with continuity correction", f"{res['chi2']:.4f}", "1", p_text(res["p_chi2"])]]
        tables.insert(0, ResultTable("Has the share of yes changed?", ["Test", "Chi-square", "df", "P value"], tests))
        size = [[f"Change in the share of yes ({names[1]} − {names[0]}), percentage points", f"{res['diff'] * 100:+.1f}", points(res["diff_ci"])]]
        if "odds" in res:
            size.append([f"Odds of changing to yes rather than to no ({res['only_second']} : {res['only_first']})", f"{res['odds']:.{sig}g}",
                         f"{res['odds_ci'][0]:.{sig}g} to {res['odds_ci'][1]:.{sig}g}"])
        tables.insert(1, ResultTable("Size of the change", ["Measure", "Estimate", f"{pct} CI"], size))
        tables.append(pair_table(0, 1, res))
        summary = (f"The share of yes goes from {share(res['p1'])} ({names[0]}) to {share(res['p2'])} ({names[1]}): "
                   f"{'a clear change' if p_lead < opt.alpha else 'no clear change'}, McNemar P = {p_lead:.4g}. "
                   f"{res['changed']} of {n} subjects changed: {res['only_second']} to yes, {res['only_first']} to no.")
        if res["changed"] < 10:
            notes.append(f"Only {res['changed']} subjects changed. The test rests on them alone, so it has little power here whatever the "
                         "number of subjects.")
        notes.append("McNemar's test uses only the subjects who changed: those who said yes both times or no both times say nothing "
                     "about a change. The exact form is right at any size; the mid-P form is a little less cautious and is often "
                     "recommended (Fagerland, Lydersen and Laake 2013); the chi-square form is the classical one, for comparison.")
    else:
        q, df, p_q = cochran_q(yes.astype(float))
        tables.insert(0, ResultTable("Does the share of yes differ between the data sets?", ["Test", "Q", "df", "P value"],
                                     [["Cochran's Q", f"{q:.4f}", str(df), p_text(p_q)]]))
        summary = (f"The share of yes runs from {share(props.min())} ({names[int(props.argmin())]}) to {share(props.max())} "
                   f"({names[int(props.argmax())]}): {'the data sets differ' if p_q < opt.alpha else 'no clear difference'}, "
                   f"Cochran's Q({df}) = {q:.3f}, P = {p_q:.4g}. {n} subjects.")
        if opt.compare != "none":
            if opt.compare == "vs_control":
                c = max(0, min(opt.control_index, k - 1))
                pairs = [(c, j) for j in range(k) if j != c]
            else:
                pairs = list(combinations(range(k), 2))
            rows, raw = [], []
            for i, j in pairs:
                res = mcnemar(yes[:, i], yes[:, j], level)
                p = res["p_exact"] if opt.exact else res["p_chi2"]
                raw.append(p)
                rows.append([f"{names[j]} − {names[i]}", f"{res['diff'] * 100:+.1f}", points(res["diff_ci"]),
                             f"{res['only_second']} to yes, {res['only_first']} to no", p_text(p)])
            headers = ["Comparison", "Change (points)", f"{pct} CI", "Subjects who changed", "P value (McNemar)"]
            if opt.correction != "none" and len(pairs) > 1:
                for row, a in zip(rows, _common.adjust(raw, opt.correction)):
                    row += [p_text(float(a)), "Yes" if a < opt.alpha else "No"]
                headers += [f"P value ({'Benjamini-Hochberg' if opt.correction == 'fdr' else 'Holm'})", f"Significant (α = {opt.alpha:g})"]
            tables.insert(1, ResultTable("Pairs of data sets", headers, rows))
        pattern, counts = np.unique(yes, axis=0, return_counts=True)
        order = np.argsort(-counts)[:8]
        tables.append(ResultTable("Most common patterns across the data sets", ["Pattern (" + ", ".join(names) + ")", "Subjects"],
                                  [[" ".join("yes" if v else "no" for v in pattern[i]), str(int(counts[i]))] for i in order]))
        notes.append("Cochran's Q extends McNemar's test to several data sets; subjects who gave the same answer every time do not "
                     "affect it. It is a large-sample test: with fewer than about 20 subjects who changed, read it with care.")

    notes += ["The data sets must be the same subjects, row by row. For separate groups of subjects, compare counts with a "
              "chi-square or Fisher's exact test instead.",
              "The interval for a change is the usual large-sample one for paired proportions; intervals for a single share are Wilson's."]
    if dropped:
        notes.insert(0, f"{dropped} row{'s' if dropped != 1 else ''} without a yes or no code in every data set "
                        f"{'were' if dropped != 1 else 'was'} left out" + (f" ({odd} cell{'s' if odd != 1 else ''} held another value)." if odd else "."))
    notes.append("References: McNemar 1947; Cochran 1950.")
    charts = [ChartSpec("Share of yes in each data set", x_label="Data set", y_label="Yes (%)", y_limits=(0, 100),
                        series=[ChartSeries("", [float(j) for j in range(k)], (props * 100).tolist(), kind="bar", color="#1f5aa6")],
                        x_categories=names)]
    return AnalysisResult(title="Yes/no, same subjects", summary=summary, tables=tables, notes=notes, charts=charts)
