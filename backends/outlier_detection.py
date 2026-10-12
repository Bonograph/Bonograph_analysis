"""Any value out of line? Four published rules, each for its own case:

* Rosner's generalized ESD test - for several outliers at once in a group,
  at an error rate you set;
* Grubbs' test - for one outlier, and its repeated form;
* the box-plot rule (Tukey's fences) - a set number of interquartile ranges
  beyond the quartiles;
* robust z-scores - distance from the median in units of the median
  absolute deviation.

Values are only flagged. The sheet is never changed; a clean copy can be
made on a new sheet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class OutlierOptions:
    method: str = "esd"  # "esd" (generalized ESD) | "grubbs" | "iterative_grubbs" | "fences" | "robust"
    q_percent: float = 1.0  # the error rate allowed, in percent, for the ESD and Grubbs tests
    subcolumns: str = "average"  # "average" | "separate" | "pooled"
    fence_factor: float = 1.5     # "fences": how many interquartile ranges beyond the quartiles (far out = twice that)
    robust_cut: float = 3.5       # "robust": the modified z-score beyond which a value is flagged
    show_details: bool = True     # the working of the rule: statistic, critical value, limits
    show_ends: bool = True        # the five lowest and five highest values of each group
    show_effect: bool = True      # mean, SD and median with and without the flagged values
    show_order_graph: bool = True
    show_box_graph: bool = True
    make_clean_copy: bool = False  # a new sheet with the flagged values left blank (the sheet itself is never changed)


def _grubbs_once(values: np.ndarray, alpha: float) -> tuple[int, float, float] | None:
    n = values.size
    if n < 3:
        return None
    mean, sd = np.mean(values), np.std(values, ddof=1)
    if sd == 0:
        return None
    abs_dev = np.abs(values - mean)
    idx = int(np.argmax(abs_dev))
    g_stat = abs_dev[idx] / sd
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return (idx, float(g_stat), float(g_crit)) if g_stat > g_crit else None


def _iterative_grubbs(values: np.ndarray, alpha: float) -> list[int]:
    """Repeatedly remove the single most extreme point while Grubbs'
    test still flags it - this method is known not to be
    recommended, since removing one point changes the mean/SD used to
    test the next, but it's exactly what "Iterative Grubbs'" means."""
    remaining = list(range(values.size))
    outlier_indices: list[int] = []
    working = values.copy()
    while working.size >= 3:
        hit = _grubbs_once(working, alpha)
        if hit is None:
            break
        idx, _g, _crit = hit
        outlier_indices.append(remaining.pop(idx))
        working = np.delete(working, idx)
    return outlier_indices


def _rosner_esd(values: np.ndarray, q_percent: float) -> list[int]:
    """Rosner's Generalized ESD test for up to n/2 outliers at
    significance level q_percent/100: the test made for finding several
    outliers in one group without the first hiding the next.

    The value furthest from the mean is always the smallest or the largest
    one still in, so the data are sorted once and trimmed from either end,
    with the mean and SD kept as running sums and every critical value
    computed in one call - the same steps as removing points one at a
    time, in a fraction of the time on long columns."""
    q = q_percent / 100.0
    values = np.asarray(values, dtype=float)
    n = values.size
    max_outliers = max(1, n // 2 - 1)
    order = np.argsort(values, kind="stable")
    sorted_v = values[order]
    steps = np.arange(1, max_outliers + 1)
    dfs = n - steps - 1
    with np.errstate(all="ignore"):
        t_crit = np.where(dfs > 0, stats.t.ppf(1 - q / (2 * (n - steps + 1)), np.maximum(dfs, 1)), np.inf)
        denom = np.sqrt((dfs + t_crit**2) * (n - steps + 1))
        lambda_crit = np.where((dfs > 0) & (denom > 0), (n - steps) * t_crit / denom, np.inf)

    # running sums, centred on the median for numerical stability
    shift = float(np.median(sorted_v))
    centred = sorted_v - shift
    total, total_sq = float(centred.sum()), float((centred**2).sum())
    lo, hi, remaining = 0, n - 1, n
    removed: list[tuple[int, float]] = []
    for i in range(max_outliers):
        if remaining < 3:
            break
        mean_c = total / remaining
        var = (total_sq - remaining * mean_c**2) / (remaining - 1)
        if var <= 0:
            break
        sd = np.sqrt(var)
        dev_lo, dev_hi = abs(centred[lo] - mean_c), abs(centred[hi] - mean_c)
        take_hi = dev_hi >= dev_lo
        pos = hi if take_hi else lo
        r_stat = (dev_hi if take_hi else dev_lo) / sd
        removed.append((int(order[pos]), float(r_stat) if r_stat >= lambda_crit[i] else -1.0))
        total -= centred[pos]
        total_sq -= centred[pos] ** 2
        remaining -= 1
        if take_hi:
            hi -= 1
        else:
            lo += 1

    # Rosner's rule: the number of outliers is the largest step whose test passed.
    last_pass = -1
    for k, (_idx, verdict) in enumerate(removed):
        if verdict >= 0:
            last_pass = k
    return [idx for idx, _ in removed[: last_pass + 1]]


def _grubbs_values(values: np.ndarray, alpha: float) -> dict | None:
    """The numbers of one Grubbs' test: the value furthest from the mean, G,
    its critical value, and an approximate two-sided P value
    (n times the tail of the matching t statistic, at most 1)."""
    n = values.size
    if n < 3:
        return None
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    if sd == 0:
        return None
    idx = int(np.argmax(np.abs(values - mean)))
    g = abs(values[idx] - mean) / sd
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    inside = (n - 1) ** 2 - n * g**2
    if inside <= 0:
        p = 0.0
    else:
        t = np.sqrt(n * (n - 2) * g**2 / inside)
        p = float(min(1.0, 2 * n * stats.t.sf(t, n - 2)))
    return {"index": idx, "n": n, "mean": mean, "sd": sd, "value": float(values[idx]), "g": float(g), "critical": float(g_crit),
            "p": p, "flagged": bool(g > g_crit)}


def _esd_steps(values: np.ndarray, q_percent: float) -> list[dict]:
    """Rosner's test step by step, as a reader would check it: at each step
    the value furthest from the mean of what is left, its statistic R and the
    critical value. Slow and plain on purpose - it is run only to show the
    working, on the first steps."""
    q = q_percent / 100.0
    left = list(range(values.size))
    n = values.size
    steps = []
    for i in range(1, max(1, n // 2 - 1) + 1):
        v = values[left]
        if v.size < 3:
            break
        sd = np.std(v, ddof=1)
        if sd == 0:
            break
        mean = np.mean(v)
        k = int(np.argmax(np.abs(v - mean)))
        r = abs(v[k] - mean) / sd
        df = n - i - 1
        if df <= 0:
            break
        t_crit = stats.t.ppf(1 - q / (2 * (n - i + 1)), df)
        critical = (n - i) * t_crit / np.sqrt((df + t_crit**2) * (n - i + 1))
        steps.append({"step": i, "index": left[k], "value": float(v[k]), "r": float(r), "critical": float(critical),
                      "passed": bool(r > critical)})
        left.pop(k)
        if len(steps) >= 12:
            break
    return steps


def _fences(values: np.ndarray, factor: float) -> dict:
    """The box-plot rule (Tukey 1977): limits ``factor`` interquartile ranges
    beyond the quartiles; twice as far is 'far out'. Quartiles are found as
    the graphs find them (linear interpolation between the ordered values)."""
    q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    return {"q1": q1, "median": median, "q3": q3, "iqr": iqr, "low": q1 - factor * iqr, "high": q3 + factor * iqr,
            "far_low": q1 - 2 * factor * iqr, "far_high": q3 + 2 * factor * iqr}


def _robust_scores(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Modified z-scores (Iglewicz and Hoaglin 1993): 0.6745 (x - median) / MAD,
    which one wild value cannot stretch as it stretches the SD. Returns
    (scores, median, MAD); all zeros when half the values are identical."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return np.zeros(values.size), median, 0.0
    return 0.6745 * (values - median) / mad, median, mad


def _num(value: float) -> str:
    return f"{value:.4g}"


@register("outlier_detection")
def run(data: SheetData, options: OutlierOptions | None = None) -> AnalysisResult:
    opt = options or OutlierOptions()
    mode = "pooled" if opt.subcolumns == "pooled" else "average"
    groups = data.all_group_arrays(mode=mode)
    if not groups:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    alpha = opt.q_percent / 100.0
    factor = max(0.1, float(opt.fence_factor))
    cut = max(0.5, float(opt.robust_cut))
    method_names = {"esd": "Generalized ESD (Rosner)", "grubbs": "Grubbs'", "iterative_grubbs": "Iterative Grubbs'",
                    "fences": "the box-plot rule", "robust": "robust z-scores"}
    method = opt.method if opt.method in method_names else "esd"
    method_name = method_names[method]
    setting = {"fences": f"{factor:g} × IQR", "robust": f"beyond {cut:g}"}.get(method, f"α = {opt.q_percent:g}%")

    def flag(v: np.ndarray) -> list[int]:
        if method == "grubbs":
            hit = _grubbs_once(v, alpha)
            return [hit[0]] if hit else []
        if method == "iterative_grubbs":
            return list(_iterative_grubbs(v, alpha))
        if method == "fences":
            f = _fences(v, factor)
            return [int(i) for i in np.flatnonzero((v < f["low"]) | (v > f["high"]))]
        if method == "robust":
            scores, _median, _mad = _robust_scores(v)
            return [int(i) for i in np.flatnonzero(np.abs(scores) > cut)]
        return list(_rosner_esd(v, opt.q_percent))

    # every set of values that is judged on its own, with the sheet row of each value where that is known
    sets: list[dict] = []
    for label, values in groups.items():
        _reps, matrix = data.group_replicate_matrix(label)
        rows_with_data = (np.where(~np.all(np.isnan(matrix), axis=1))[0] + 1) if matrix.size else np.array([])
        if opt.subcolumns == "separate":
            arrays = data.group_replicate_arrays(label)
            value_sets = [(f"{label} (subcolumn {rep})", arr) for rep, arr in sorted(arrays.items())] or [(label, values)]
        else:
            value_sets = [(label, values)]
        for set_label, v in value_sets:
            known = opt.subcolumns != "pooled" and set_label == label and len(rows_with_data) == v.size
            sets.append({"label": set_label, "group": label, "v": np.asarray(v, dtype=float),
                         "rows": [int(r) for r in rows_with_data] if known else None})

    rows: list[list[str]] = []
    total_outliers = 0
    few = []
    for item in sets:
        v = item["v"]
        item["flagged"] = []
        if v.size < 3:
            rows.append([item["label"], str(v.size), "too few values", ""])
            continue
        if v.size < 7:
            few.append(item["label"])
        idxs = flag(v)
        item["flagged"] = idxs
        total_outliers += len(idxs)
        text = ", ".join(_num(v[i]) for i in idxs) if idxs else "none"
        where = ", ".join(str(item["rows"][i]) for i in idxs) if idxs and item["rows"] else ""
        rows.append([item["label"], str(v.size), text, where])

    tables = [ResultTable(title=f"Values flagged by {method_name} ({setting})",
                          headers=["Group", "N", "Flagged value(s)", "Row"], rows=rows)]
    judged = [item for item in sets if item["v"].size >= 3]

    def place(item: dict, i: int) -> str:
        return str(item["rows"][i]) if item["rows"] else ""

    # ── each flagged value, and what the other rules say of it ───────────────
    flagged_rows = []
    for item in judged:
        v = item["v"]
        if not item["flagged"]:
            continue
        mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        scores, _median, mad = _robust_scores(v)
        fence = _fences(v, 1.5)
        others = {"Generalized ESD": set(_rosner_esd(v, opt.q_percent)) if method != "esd" else None,
                  "Grubbs'": ({h[0]} if (h := _grubbs_once(v, alpha)) else set()) if method not in ("grubbs", "iterative_grubbs") else None,
                  "box-plot rule": set(np.flatnonzero((v < fence["low"]) | (v > fence["high"])).tolist()) if method != "fences" else None,
                  "robust z-score": set(np.flatnonzero(np.abs(scores) > 3.5).tolist()) if method != "robust" and mad > 0 else None}
        for i in item["flagged"]:
            agree = [name for name, found in others.items() if found is not None and i in found]
            flagged_rows.append([item["label"], place(item, i), _num(v[i]), f"{(v[i] - mean) / sd:+.2f}" if sd > 0 else "",
                                 f"{scores[i]:+.2f}" if mad > 0 else "", ", ".join(agree) if agree else "no other rule"])
    if flagged_rows:
        tables.append(ResultTable("Each flagged value, and the other rules",
                                  ["Group", "Row", "Value", "SDs from the mean", "Robust z-score", "Also flagged by"], flagged_rows))

    # ── the working of the rule ──────────────────────────────────────────────
    if opt.show_details and judged:
        if method == "grubbs":
            detail = []
            for item in judged:
                g = _grubbs_values(item["v"], alpha)
                if g is None:
                    detail.append([item["label"], str(item["v"].size), _num(float(np.mean(item["v"]))), "0", "", "", "", "", "all values are the same"])
                    continue
                detail.append([item["label"], str(g["n"]), _num(g["mean"]), _num(g["sd"]), _num(g["value"]), f"{g['g']:.4f}",
                               f"{g['critical']:.4f}", "<0.0001" if g["p"] < 0.0001 else f"{g['p']:.4f}",
                               "out of line" if g["flagged"] else "in line"])
            tables.append(ResultTable("Grubbs' test, in numbers",
                                      ["Group", "N", "Mean", "SD", "Furthest value", "G", "Critical G", "P (approximate)", "Reading"], detail))
        elif method == "iterative_grubbs":
            detail = []
            for item in judged:
                working, step = item["v"].copy(), 0
                while working.size >= 3 and step < 12:
                    g = _grubbs_values(working, alpha)
                    if g is None:
                        break
                    step += 1
                    detail.append([item["label"], str(step), str(g["n"]), _num(g["value"]), f"{g['g']:.4f}", f"{g['critical']:.4f}",
                                   "out of line: taken out, test repeated" if g["flagged"] else "in line: stop"])
                    if not g["flagged"]:
                        break
                    working = np.delete(working, g["index"])
            tables.append(ResultTable("Grubbs' test, round by round",
                                      ["Group", "Round", "Values left", "Furthest value", "G", "Critical G", "Reading"], detail))
        elif method == "esd":
            detail = []
            for item in judged:
                counted = len(item["flagged"])
                for st in _esd_steps(item["v"], opt.q_percent)[:max(3, counted + 2)]:
                    detail.append([item["label"], str(st["step"]), _num(st["value"]), f"{st['r']:.4f}", f"{st['critical']:.4f}",
                                   "above" if st["passed"] else "below", "yes" if st["step"] <= counted else "no"])
            tables.append(ResultTable("Generalized ESD, step by step",
                                      ["Group", "Step", "Furthest value left", "R", "Critical value", "R against critical",
                                       "Counted as out of line"], detail))
        elif method == "fences":
            detail = []
            for item in judged:
                v, f = item["v"], _fences(item["v"], factor)
                far = int(np.sum((v < f["far_low"]) | (v > f["far_high"])))
                detail.append([item["label"], str(v.size), _num(f["q1"]), _num(f["median"]), _num(f["q3"]), _num(f["iqr"]),
                               f"{_num(f['low'])} to {_num(f['high'])}", f"{_num(f['far_low'])} to {_num(f['far_high'])}",
                               str(len(item["flagged"]) - far), str(far)])
            tables.append(ResultTable("The box-plot rule, in numbers",
                                      ["Group", "N", "Lower quartile", "Median", "Upper quartile", "IQR",
                                       f"In line between ({factor:g} × IQR)", f"Far out beyond ({2 * factor:g} × IQR)",
                                       "Outside", "Far outside"], detail))
        else:
            detail = []
            for item in judged:
                _scores, median, mad = _robust_scores(item["v"])
                if mad == 0:
                    detail.append([item["label"], str(item["v"].size), _num(median), "0", "half the values are the same: no score"])
                else:
                    half = cut * mad / 0.6745
                    detail.append([item["label"], str(item["v"].size), _num(median), _num(mad),
                                   f"{_num(median - half)} to {_num(median + half)}"])
            tables.append(ResultTable("Robust z-scores, in numbers",
                                      ["Group", "N", "Median", "Median absolute deviation", f"In line between (score within {cut:g})"],
                                      detail))

    # ── the ends of each group ───────────────────────────────────────────────
    if opt.show_ends and judged:
        end_rows = []
        for item in judged:
            v = item["v"]
            order = np.argsort(v, kind="stable")
            take = min(5, v.size // 2)

            def listed(indices) -> str:
                return ", ".join(_num(v[i]) + (f" (row {item['rows'][i]})" if item["rows"] else "")
                                 + ("*" if i in item["flagged"] else "") for i in indices)

            end_rows.append([item["label"], listed(order[:take]), listed(order[::-1][:take])])
        tables.append(ResultTable("Lowest and highest values (* = flagged)", ["Group", "Lowest, from the bottom up", "Highest, from the top down"],
                                  end_rows))

    # ── what leaving them out would change ───────────────────────────────────
    if opt.show_effect and any(item["flagged"] for item in judged):
        effect = []
        for item in judged:
            if not item["flagged"]:
                continue
            v = item["v"]
            kept = np.delete(v, item["flagged"])
            sd_kept = _num(float(np.std(kept, ddof=1))) if kept.size > 1 else ""
            effect.append([item["label"], str(len(item["flagged"])), f"{_num(float(np.mean(v)))} → {_num(float(np.mean(kept)))}",
                           f"{_num(float(np.std(v, ddof=1)))} → {sd_kept}",
                           f"{_num(float(np.median(v)))} → {_num(float(np.median(kept)))}"])
        tables.append(ResultTable("With every value → without the flagged ones", ["Group", "Flagged", "Mean", "SD", "Median"], effect))

    summary = (f"{method_name[0].upper() + method_name[1:]} flagged {total_outliers} value(s) across {len(groups)} group(s) "
               f"({setting}). Nothing was removed from the sheet.") if method in ("fences", "robust") else (
        f"{method_name} flagged {total_outliers} value(s) across {len(groups)} group(s) "
        f"at α = {opt.q_percent:g}%. Nothing was removed from the sheet.")
    notes = ["Nothing was removed. Take a value out only for a reason beyond its being extreme - a recording "
             "error or a failed sample - and report that you did.",
             "These tests assume the rest of the values come from a normal distribution; in skewed data they "
             "flag the long tail. Look at the values on a log scale first if the data are skewed."]
    if method in ("fences", "robust"):
        notes[1] = ("This rule is a screen, not a test: it has no P value, and in a large sample from a normal distribution "
                    + ("the box-plot rule flags about 7 values in 1,000 by chance alone." if method == "fences" else
                       "a cut of 3.5 flags about 5 values in 10,000 by chance alone.")
                    + " In skewed data it flags the long tail.")
    if few:
        notes.insert(0, "Fewer than 7 values in " + ", ".join(few) + ": outlier tests have almost no power "
                     "this small.")
    if method == "iterative_grubbs":
        notes.append("Iterative Grubbs' is not generally recommended - removing one point changes the mean/SD used to test the next.")
    if method == "esd":
        notes.append("Generalized ESD finds any number of outliers up to a third of the values, while keeping "
                     "the chance of a false alarm at α.")
    if method == "fences":
        notes.append(f"Box-plot rule (Tukey 1977): a value is outside when it lies more than {factor:g} interquartile ranges "
                     f"beyond a quartile, and far outside beyond {2 * factor:g}. Quartiles are interpolated between the ordered values.")
    if method == "robust":
        notes.append("Robust z-score (Iglewicz and Hoaglin 1993): 0.6745 × (value - median) / median absolute deviation. "
                     "The median and that deviation are barely moved by the outliers themselves, so one wild value cannot hide another.")
    if method == "grubbs" and opt.show_details:
        notes.append("Grubbs' P value is approximate: the number of values times the two-sided tail of the matching t statistic.")
    if flagged_rows:
        notes.append("'Also flagged by' runs the other rules on the same values (ESD and Grubbs' at the same α, the box-plot "
                     "rule at 1.5 × IQR, robust z-scores beyond 3.5). Rules that agree make a stronger case; a value only one rule "
                     "flags deserves a second look before anything is done about it.")

    # ── graphs ───────────────────────────────────────────────────────────────
    charts: list[ChartSpec] = []
    if opt.show_order_graph:
        for item in judged:
            v = item["v"]
            x = np.array(item["rows"], dtype=float) if item["rows"] else np.arange(1, v.size + 1, dtype=float)
            is_out = np.zeros(v.size, dtype=bool)
            is_out[item["flagged"]] = True
            series = [ChartSeries("In line", x[~is_out].tolist(), v[~is_out].tolist(), kind="scatter", color="#4a78b5")]
            if is_out.any():
                series.append(ChartSeries("Flagged", x[is_out].tolist(), v[is_out].tolist(), kind="scatter", color="#d1342f"))
            if method == "fences":
                f = _fences(v, factor)
                centre, low, high, what = f["median"], f["low"], f["high"], "median and box-plot limits"
            elif method == "robust":
                _sc, median, mad = _robust_scores(v)
                half = cut * mad / 0.6745
                centre, low, high, what = median, median - half, median + half, "median and robust limits"
            else:
                g = _grubbs_values(v, alpha)
                mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
                half = (g["critical"] if g else 3.0) * sd
                centre, low, high, what = mean, mean - half, mean + half, "mean and Grubbs' limits"
            lines = [{"kind": "h", "value": centre, "color": "#555555", "style": "-", "width": 1.0},
                     {"kind": "h", "value": low, "color": "#d1342f", "style": "--", "width": 0.9},
                     {"kind": "h", "value": high, "color": "#d1342f", "style": "--", "width": 0.9}]
            charts.append(ChartSpec(f"Values in sheet order: {item['label']}", series=series,
                                    x_label=("Row of the sheet" if item["rows"] else "Order in the group") + f"  (lines: {what})",
                                    y_label="Value",
                                    reference_lines=lines, group="outlier_order", option=item["label"]))
    if opt.show_box_graph and judged:
        boxes = []
        for item in judged:
            v, f = item["v"], _fences(item["v"], 1.5)
            inside = v[(v >= f["low"]) & (v <= f["high"])]
            far = (v < f["far_low"]) | (v > f["far_high"])
            near = ((v < f["low"]) | (v > f["high"])) & ~far
            flagged = np.zeros(v.size, dtype=bool)
            flagged[item["flagged"]] = True
            boxes.append({"label": item["label"], "q1": f["q1"], "median": f["median"], "q3": f["q3"],
                          "low": float(inside.min()) if inside.size else f["q1"], "high": float(inside.max()) if inside.size else f["q3"],
                          "outside": v[near].tolist(), "far": v[far].tolist(), "flagged": v[flagged].tolist()})
        charts.append(ChartSpec("Box plot: whiskers to the last value within 1.5 × IQR", kind="boxes", y_label="Value",
                                extra={"boxes": boxes, "flag_name": f"Flagged by {method_name}"}))

    if charts and opt.show_order_graph and method not in ("fences", "robust"):
        notes.append("In the graph of values in sheet order, the dashed lines are the mean ± the critical G × SD of one Grubbs' "
                     "test on all the values: a guide to the eye. A rule that removes values in turn judges later ones by limits "
                     "that have moved in.")
    result = AnalysisResult(title="Any value out of line?", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.make_clean_copy:
        usable = [item for item in sets if item["rows"] is not None and item["label"] == item["group"]]
        if usable and opt.subcolumns == "average":
            n_rows = int(data.n_rows)
            x = np.asarray(data.x, dtype=float) if getattr(data, "x", None) is not None else np.array([])
            x_values = [float(x[i]) if i < x.size else float("nan") for i in range(n_rows)]
            columns = {}
            for item in usable:
                column = np.full(n_rows, np.nan)
                rows_ix = np.array(item["rows"], dtype=int) - 1
                column[rows_ix] = item["v"]
                column[rows_ix[item["flagged"]]] = np.nan
                columns[item["label"]] = column.tolist()
            result.new_sheet = NewSheetData("Without flagged values", "X", x_values, columns)
            notes.append("A copy of the data was made on a new sheet with the flagged values left blank"
                         + (" (side-by-side repeats are averaged in it)." if any(len(data.group_replicate_matrix(g)[0]) > 1 for g in groups) else ".")
                         + " The sheet you analysed is unchanged.")
        else:
            notes.append("No copy was made: a copy without the flagged values needs each value to be one row of the sheet "
                         "(repeats averaged, not pooled or judged separately).")
    return result
