"""Kaplan-Meier Survival Analysis. Each sheet group is read as time (the sheet's X
column) paired with that group's event-code column (configurable
death/censored codes) - the two-column convention most survival tools
use, one group per curve."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _common


@dataclass
class KaplanMeierOptions:
    death_code: float = 1.0
    censored_code: float = 0.0
    # the model
    compute_logrank: bool = True
    logrank_trend: bool = False
    compute_gehan_breslow: bool = False  # a second test on the same question only when asked for
    compute_tarone_ware: bool = False
    compute_peto_peto: bool = False
    tabulate: str = "survival"  # "survival" | "event"
    error_bar_mode: str = "ci"  # "none" | "se" | "ci"
    ci_style: str = "asymmetric"  # "asymmetric" | "symmetric"
    show_at_risk_table: bool = True
    show_time_summary: bool = True   # median, quartiles and mean survival time, with their intervals
    landmarks: str = ""              # times to report survival at ("12, 24, 36"); empty = chosen from the data
    show_ci_band: bool = True        # on the graph
    show_median_lines: bool = True   # on the graph
    significant_digits: int = 4
    # pairs to compare
    compare: str = "none"  # "none" | "all_pairs" | "vs_control"
    control_index: int = 0
    correction: str = "holm"  # "holm" | "fdr" | "none"
    report_adjusted_p: bool = True
    p_style: str = "number"
    # mean time event-free up to a horizon, compared between groups (Royston & Parmar 2013)
    compare_rmst: bool = False
    rmst_horizon: float = 0.0        # 0 = the longest time every group was followed


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)[::-1]
    m = len(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


def _km_curve(time: np.ndarray, event: np.ndarray, opt: KaplanMeierOptions) -> tuple[list, float | None]:
    order = np.argsort(time)
    time, event = time[order], event[order]
    unique_times = np.unique(time)
    n_at_risk = time.size
    survival = 1.0
    var_sum = 0.0  # running Σ d_i / (n_i (n_i - d_i)) for Greenwood's formula
    rows: list[list[str]] = []
    median_survival: float | None = None
    z = stats.norm.ppf(0.975)
    sig = opt.significant_digits

    for t in unique_times:
        at_this_time = time == t
        n_events = int(np.sum(event[at_this_time] == 1))
        n_censored = int(np.sum(event[at_this_time] == 0))
        if n_events > 0:
            survival *= 1 - n_events / n_at_risk
            if n_at_risk > n_events:
                var_sum += n_events / (n_at_risk * (n_at_risk - n_events))

        display = survival if opt.tabulate == "survival" else 1 - survival
        row = [f"{t:.{sig}g}", str(n_at_risk), str(n_events), str(n_censored), f"{display*100:.{sig}g}%"]

        if opt.error_bar_mode != "none" and n_events > 0:
            greenwood_var = (survival**2) * var_sum
            se = np.sqrt(max(greenwood_var, 0.0))
            if opt.error_bar_mode == "se":
                row.append(f"±{se*100:.{sig}g}%")
            elif opt.ci_style == "symmetric" or survival <= 0 or survival >= 1:
                lo, hi = max(0.0, survival - z * se), min(1.0, survival + z * se)
                row.append(f"{lo*100:.{sig}g}% to {hi*100:.{sig}g}%")
            else:
                g = np.log(-np.log(survival))
                se_g = se / abs(survival * np.log(survival))
                lo, hi = np.exp(-np.exp(g + z * se_g)), np.exp(-np.exp(g - z * se_g))
                row.append(f"{lo*100:.{sig}g}% to {hi*100:.{sig}g}%")
        elif opt.error_bar_mode != "none":
            row.append("-")

        rows.append(row)
        if median_survival is None and survival <= 0.5:
            median_survival = float(t)
        n_at_risk -= at_this_time.sum()

    return rows, median_survival


@dataclass
class _Estimate:
    """The Kaplan-Meier estimate of one group at each of its distinct times."""
    times: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    survival: np.ndarray
    se: np.ndarray          # Greenwood
    lower: np.ndarray
    upper: np.ndarray
    last: float             # longest follow-up


def _estimate(time: np.ndarray, event: np.ndarray, ci_style: str) -> _Estimate:
    from graphs.engine.stats import km_interval

    times = np.unique(time)
    at_risk = np.array([np.sum(time >= t) for t in times], dtype=float)
    events = np.array([np.sum((time == t) & (event == 1)) for t in times], dtype=float)
    survival = np.cumprod(1.0 - events / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(at_risk > events, events / (at_risk * (at_risk - events)), 0.0)
    se = np.sqrt(np.maximum(survival**2 * np.cumsum(term), 0.0))
    lower, upper = km_interval(survival, se, ci_style)
    return _Estimate(times, at_risk, events, survival, se, lower, upper, float(times[-1]))


def _first_time_at_or_below(values: np.ndarray, times: np.ndarray, level: float) -> float | None:
    """The first time the curve is at or under ``level`` - the rule the
    median uses - or None when it never gets there."""
    hit = np.flatnonzero(values <= level + 1e-12)
    return float(times[hit[0]]) if hit.size else None


def _percentile_time(est: _Estimate, share: float) -> tuple[float | None, float | None, float | None]:
    """The time by which ``share`` of the subjects have had the event, with
    the confidence limits of Brookmeyer and Crowley (1982): the times at
    which the lower and the upper limit of the curve cross the same level."""
    level = 1.0 - share
    return (_first_time_at_or_below(est.survival, est.times, level), _first_time_at_or_below(est.lower, est.times, level),
            _first_time_at_or_below(est.upper, est.times, level))


def _restricted_mean(est: _Estimate, tau: float) -> tuple[float, float]:
    """Mean survival time up to ``tau`` - the area under the curve from 0 to
    tau - and its standard error:
        Var = sum over event times t_i <= tau of A_i^2 d_i / (n_i (n_i - d_i)),
    A_i being the area under the curve from t_i to tau (Klein and Moeschberger,
    Survival Analysis, 2nd ed., section 4.5)."""
    use = (est.events > 0) & (est.times <= tau)
    knots = np.concatenate([[0.0], est.times[use], [tau]])
    height = np.concatenate([[1.0], est.survival[use]])       # the curve just after each knot
    pieces = height * np.diff(knots)
    mean = float(pieces.sum())
    after = np.cumsum(pieces[::-1])[::-1]                     # area from each knot to tau
    n, d = est.at_risk[use], est.events[use]
    ok = n > d
    variance = float(np.sum(after[1:][ok] ** 2 * d[ok] / (n[ok] * (n[ok] - d[ok]))))
    return mean, float(np.sqrt(variance))


def restricted_mean_contrast(first: tuple[float, float], second: tuple[float, float], z: float = 1.959964):
    """Two restricted means (each a (mean, SE) pair) compared: the
    difference second - first with its limits and P value, and the ratio
    second / first with limits found on the log scale."""
    (mean_a, se_a), (mean_b, se_b) = first, second
    diff, se_diff = mean_b - mean_a, float(np.hypot(se_a, se_b))
    p_value = float(2 * stats.norm.sf(abs(diff) / se_diff)) if se_diff > 0 else float("nan")
    if mean_a > 0 and mean_b > 0:
        log_ratio = float(np.log(mean_b / mean_a))
        se_log = float(np.hypot(se_a / mean_a, se_b / mean_b))
        ratio = (float(np.exp(log_ratio)), float(np.exp(log_ratio - z * se_log)), float(np.exp(log_ratio + z * se_log)))
    else:
        ratio = (float("nan"), float("nan"), float("nan"))
    return (diff, diff - z * se_diff, diff + z * se_diff, p_value), ratio


def _survival_at(est: _Estimate, t: float) -> tuple[float, float, float] | None:
    """Estimate and limits at time ``t`` (the curve is flat between its
    times), or None beyond the group's follow-up."""
    if t > est.last:
        return None
    before = np.flatnonzero(est.times <= t)
    if not before.size:
        return 1.0, 1.0, 1.0
    i = before[-1]
    return float(est.survival[i]), float(est.lower[i]), float(est.upper[i])


def _nice_times(longest: float, most: int = 6) -> list[float]:
    """Round times across the follow-up: 6, 12, 18 ... rather than 5.83, 11.7 ..."""
    if not np.isfinite(longest) or longest <= 0:
        return []
    raw = longest / most
    power = 10.0 ** np.floor(np.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    return [float(round(step * k, 10)) for k in range(1, most + 1) if step * k <= longest * (1 + 1e-9)]


def _landmark_times(text: str, longest: float) -> list[float]:
    """The times typed by the user ("12, 24 36"), or round ones from the data."""
    import re

    typed = []
    for token in re.split(r"[,;\s]+", (text or "").strip()):
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0 and value not in typed:
            typed.append(value)
    return sorted(typed) if typed else _nice_times(longest)


def _logrank_components(time: np.ndarray, event: np.ndarray, group_idx: np.ndarray, k: int, weighted) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed, expected, and covariance-matrix vectors/matrix for a
    k-sample (weighted) log-rank test, computed event-time by
    event-time over the pooled risk set - the same counting-process
    formula used for both the ordinary log-rank (weight=1) and the
    Gehan-Breslow (generalised Wilcoxon) test (weight = number at risk).

    ``weighted``: False or "logrank" (weight 1), True or "gehan" (number at
    risk n), "tarone" (square root of n; Tarone and Ware 1977) or "peto"
    (the pooled survival estimate prod(1 - d/(n + 1)) up to and including
    this time; Peto and Peto 1972)."""
    weight = {False: "logrank", True: "gehan"}.get(weighted, weighted)
    pooled = 1.0
    order = np.argsort(time)
    time, event, group_idx = time[order], event[order], group_idx[order]
    unique_times = np.unique(time[event == 1]) if np.any(event == 1) else np.array([])

    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    at_risk_mask = np.ones(time.size, dtype=bool)

    for t in unique_times:
        risk_mask = time >= t
        n_j = int(np.sum(risk_mask))
        if n_j < 2:
            continue
        event_here = risk_mask & (time == t) & (event == 1)
        d_j = int(np.sum(event_here))
        if d_j == 0:
            continue
        pooled *= 1.0 - d_j / (n_j + 1.0)
        w_j = {"logrank": 1.0, "gehan": float(n_j), "tarone": float(np.sqrt(n_j)), "peto": pooled}[weight]
        n_gj = np.array([np.sum(risk_mask & (group_idx == g)) for g in range(k)], dtype=float)
        d_gj = np.array([np.sum(event_here & (group_idx == g)) for g in range(k)], dtype=float)
        e_gj = d_j * n_gj / n_j
        O += w_j * d_gj
        E += w_j * e_gj
        if n_j > 1:
            factor = d_j * (n_j - d_j) / (n_j - 1) / n_j**2
            v = -factor * np.outer(n_gj, n_gj)
            np.fill_diagonal(v, factor * n_gj * (n_j - n_gj))
            V += (w_j**2) * v

    return O, E, V


def _k_sample_test(O: np.ndarray, E: np.ndarray, V: np.ndarray, k: int) -> tuple[float, float]:
    diff = O - E
    chi2 = float(diff @ np.linalg.pinv(V) @ diff)
    p = float(stats.chi2.sf(chi2, k - 1))
    return chi2, p


def _trend_test(O: np.ndarray, E: np.ndarray, V: np.ndarray, k: int) -> tuple[float, float]:
    scores = np.arange(k, dtype=float)
    num = float(scores @ (O - E))
    denom = float(scores @ V @ scores)
    z = num / np.sqrt(denom) if denom > 0 else 0.0
    chi2 = z**2
    return chi2, float(stats.chi2.sf(chi2, 1))


@register("kaplan_meier")
def run(data: SheetData, options: KaplanMeierOptions | None = None) -> AnalysisResult:
    opt = options or KaplanMeierOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")

    per_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in labels:
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.size == 0:
            continue
        event_raw = matrix[:, 0]  # row-aligned with X - the event code must stay paired with its own time
        n = min(data.x.size, event_raw.size)
        time, event_code = data.x[:n], event_raw[:n]
        mask = ~np.isnan(time) & np.isin(event_code, (opt.death_code, opt.censored_code))
        time, event_code = time[mask], event_code[mask]
        if time.size >= 2:
            per_group[label] = (time, (event_code == opt.death_code).astype(int))

    if not per_group:
        raise AnalysisError(f"No group has at least 2 subjects with a time and an event code matching Death/Event={opt.death_code:g} or Censored={opt.censored_code:g}.")

    tables: list[ResultTable] = []
    sig = opt.significant_digits
    k = len(per_group)
    labels_used = list(per_group)

    def num(value) -> str:
        return f"{value:.{sig}g}"

    def when(value) -> str:
        return num(value) if value is not None else "not reached"

    def span(low, high) -> str:
        return f"{when(low)} to {when(high)}"

    estimates = {label: _estimate(time, event, opt.ci_style) for label, (time, event) in per_group.items()}
    longest = max(est.last for est in estimates.values())
    tau = min(est.last for est in estimates.values())        # every group is followed at least this long
    landmarks = _landmark_times(opt.landmarks, longest)
    z = 1.959964

    # ── who was followed, and for how long ───────────────────────────────────
    def followed(time: np.ndarray, event: np.ndarray, name: str) -> list[str]:
        # median follow-up by the reverse Kaplan-Meier method: leaving early is the "event", so that
        # subjects who had the event early do not make the follow-up look short (Schemper and Smith 1996)
        reverse = _estimate(time, 1 - event, opt.ci_style)
        n, events = int(time.size), int(np.sum(event))
        follow_up = _first_time_at_or_below(reverse.survival, reverse.times, 0.5)
        return [name, str(n), str(events), str(n - events), f"{100.0 * (n - events) / n:.1f}",
                num(follow_up) if follow_up is not None else "-",      # too few left early to say
                num(float(time.max()))]

    subject_rows = [followed(time, event, label) for label, (time, event) in per_group.items()]
    if k > 1:
        subject_rows.append(followed(np.concatenate([t for t, _e in per_group.values()]),
                                     np.concatenate([e for _t, e in per_group.values()]), "All"))
    subjects_table = ResultTable("Subjects, events and censored",
                                 ["Group", "Subjects", "Events", "Censored", "Censored (%)", "Median follow-up",
                                  "Longest follow-up"], subject_rows)

    # ── survival time: median, quartiles, mean ───────────────────────────────
    medians: dict[str, tuple] = {}
    means: dict[str, tuple[float, float]] = {}
    time_rows = []
    for label, est in estimates.items():
        medians[label] = _percentile_time(est, 0.5)
        q25, q75 = _percentile_time(est, 0.25)[0], _percentile_time(est, 0.75)[0]
        means[label] = _restricted_mean(est, tau)
        mean, se = means[label]
        time_rows.append([label, when(medians[label][0]), span(medians[label][1], medians[label][2]), when(q25), when(q75),
                          num(mean), num(se), f"{num(max(0.0, mean - z * se))} to {num(min(tau, mean + z * se))}"])
    time_table = ResultTable("Survival time",
                             ["Group", "Median", "95% CI of the median", "25% had the event by", "75% had the event by",
                              f"Mean up to time {num(tau)}", "SE of the mean", "95% CI of the mean"], time_rows)

    # ── survival at chosen times ─────────────────────────────────────────────
    rising = opt.tabulate == "event"
    landmark_rows = []
    for label, est in estimates.items():
        time = per_group[label][0]
        for t in landmarks:
            at = _survival_at(est, t)
            if at is None:
                landmark_rows.append([label, num(t), "-", "beyond the follow-up", "0"])
                continue
            value, low, high = at
            if rising:
                value, low, high = 1 - value, 1 - high, 1 - low
            landmark_rows.append([label, num(t), f"{value * 100:.1f}", f"{low * 100:.1f} to {high * 100:.1f}",
                                  str(int(np.sum(time >= t)))])
    landmark_table = ResultTable("Had the event by chosen times" if rising else "Survival at chosen times",
                                 ["Group", "Time", "Had the event (%)" if rising else "Survival (%)", "95% CI", "At risk"],
                                 landmark_rows)

    # ── the curves themselves, time by time ──────────────────────────────────
    curve_tables: list[ResultTable] = []
    summaries = []
    for label, (time, event) in per_group.items():
        rows, _median = _km_curve(time, event, opt)
        headers = ["Time", "At risk", "Events", "Censored", "Had the event (%)" if rising else "Survival (%)"]
        if opt.error_bar_mode == "se":
            headers.append("SE")
        elif opt.error_bar_mode == "ci":
            headers.append("95% CI")
        curve_tables.append(ResultTable(title=f"Kaplan-Meier: '{label}' (n = {time.size}, {int(np.sum(event))} events)",
                                        headers=headers, rows=rows))
        median, low, high = medians[label]
        text = f"'{label}': median = {when(median)}"
        if median is not None and opt.show_time_summary:
            text += f" (95% CI {span(low, high)})"
        summaries.append(text)

    risk_times = [0.0] + [t for t in landmarks if t <= longest]
    at_risk = {"times": risk_times,
               "rows": [{"label": label, "counts": [int(np.sum(time >= t)) for t in risk_times]}
                        for label, (time, _event) in per_group.items()]}
    risk_table = ResultTable("Number at risk", ["Group"] + [f"Time {num(t)}" for t in risk_times],
                             [[row["label"]] + [str(c) for c in row["counts"]] for row in at_risk["rows"]])

    notes = ["Time is read from the sheet's X column; the event code is read from each group's Y column."]
    few = [f"{lbl} ({int(np.sum(e))} events)" for lbl, (_t, e) in per_group.items() if np.sum(e) < 5]
    if few:
        notes.insert(0, "Very few events: " + ", ".join(few) + ". Survival estimates and tests rest on the number "
                     "of events, not subjects - read them with caution.")
    summary = "; ".join(summaries) + "."

    group_list = list(per_group.items())
    if k >= 2:
        all_time = np.concatenate([t for t, _e in per_group.values()])
        all_event = np.concatenate([e for _t, e in per_group.values()])
        all_group = np.concatenate([np.full(t.size, i) for i, (t, _e) in enumerate(per_group.values())])

        effect_rows = []
        O = E = V = None
        if opt.compute_logrank:
            O, E, V = _logrank_components(all_time, all_event, all_group, k, weighted=False)
        if k == 2 and O is not None:
            # Mantel-Haenszel hazard ratio from the same log-rank sums:
            # ln HR = (O1 - E1) / V, SE = 1 / sqrt(V)
            v11 = float(V[0, 0]) if np.ndim(V) == 2 else float(V[0])
            if v11 > 0:
                ln_hr = float(O[0] - E[0]) / v11
                half = z / np.sqrt(v11)
                a_lbl, b_lbl = group_list[0][0], group_list[1][0]
                effect_rows.append([f"Hazard ratio ({a_lbl} / {b_lbl})", f"{np.exp(ln_hr):.{sig}g}",
                                    f"{np.exp(ln_hr - half):.{sig}g} to {np.exp(ln_hr + half):.{sig}g}", ""])
                summary += f" Hazard ratio ({a_lbl}/{b_lbl}) = {np.exp(ln_hr):.3g}."
        if k == 2 and opt.show_time_summary and not opt.compare_rmst:
            (mean_a, se_a), (mean_b, se_b) = means[labels_used[0]], means[labels_used[1]]
            se_diff = float(np.hypot(se_a, se_b))
            if se_diff > 0:
                diff = mean_a - mean_b
                p_diff = float(2 * stats.norm.sf(abs(diff) / se_diff))
                effect_rows.append([f"Difference in mean survival time up to {num(tau)} ({labels_used[0]} - {labels_used[1]})",
                                    num(diff), f"{num(diff - z * se_diff)} to {num(diff + z * se_diff)}",
                                    format_p(p_diff, opt.p_style)])
        if effect_rows:
            tables.append(ResultTable(title="Size of the difference", headers=["Effect", "Estimate", "95% CI", "P value"],
                                      rows=effect_rows))

        if opt.compare_rmst:
            horizon = float(opt.rmst_horizon) if opt.rmst_horizon > 0 else tau
            if horizon > tau:
                notes.append(f"The horizon asked for, {num(float(opt.rmst_horizon))}, is beyond the time every group was followed "
                             f"({num(tau)}); the comparison stops there, since a curve cannot be averaged past its own data.")
                horizon = tau
            at_horizon = {label: _restricted_mean(est, horizon) for label, est in estimates.items()}
            tables.append(ResultTable(
                f"Mean time event-free up to time {num(horizon)}",
                ["Group", "Mean time", "SE", "95% CI", f"Mean time lost (of {num(horizon)})"],
                [[label, num(mean), num(se), f"{num(max(0.0, mean - z * se))} to {num(min(horizon, mean + z * se))}",
                  num(horizon - mean)] for label, (mean, se) in at_horizon.items()]))
            if k == 2:
                rmst_pairs = [(0, 1)]
            elif opt.compare == "vs_control":
                control = max(0, min(opt.control_index, k - 1))
                rmst_pairs = [(control, j) for j in range(k) if j != control]
            else:
                rmst_pairs = list(combinations(range(k), 2))
            contrasts = [restricted_mean_contrast(at_horizon[labels_used[i]], at_horizon[labels_used[j]], z) for i, j in rmst_pairs]
            adjusted = _common.adjust([c[0][3] for c in contrasts], opt.correction) if len(contrasts) > 1 else None
            rows = []
            for index, ((i, j), (difference, ratio)) in enumerate(zip(rmst_pairs, contrasts)):
                row = [f"{labels_used[j]} − {labels_used[i]}", num(difference[0]), f"{num(difference[1])} to {num(difference[2])}",
                       format_p(difference[3], opt.p_style)]
                if adjusted is not None:
                    row.append(format_p(float(adjusted[index]), opt.p_style))
                row += [num(ratio[0]), f"{num(ratio[1])} to {num(ratio[2])}"]
                rows.append(row)
            tables.append(ResultTable(
                "Does one group stay event-free longer, on average?",
                ["Comparison", "Difference in mean time", "95% CI", "P value"]
                + ([f"P value ({ {'fdr': 'Benjamini-Hochberg', 'none': 'not adjusted'}.get(opt.correction, 'Holm') })"]
                   if adjusted is not None else []) + ["Ratio of mean times", "95% CI"], rows))
            notes.append(f"Mean time event-free is the area under each curve up to time {num(horizon)} (the restricted mean "
                         "survival time; Royston and Parmar 2013). Unlike a hazard ratio it assumes nothing about how the "
                         "curves relate, so it keeps its meaning when they cross. It answers for that horizon only: choose "
                         "the horizon from the question, before seeing the curves.")
            p_values = np.array([c[0][3] for c in contrasts], dtype=float)
            best = int(np.nanargmin(p_values)) if np.any(np.isfinite(p_values)) else 0
            summary += (f" Mean time event-free to {num(horizon)}: {labels_used[rmst_pairs[best][1]]} − "
                        f"{labels_used[rmst_pairs[best][0]]} = {num(contrasts[best][0][0])} "
                        f"(95% CI {num(contrasts[best][0][1])} to {num(contrasts[best][0][2])}).")

        # every test asks the same question - are the curves the same? - and differs in which times count most
        test_rows = []
        if O is not None:
            chi2, p = _k_sample_test(O, E, V, k)
            test_name = "Log-rank test"
            test_rows.append(["Log-rank" if k == 2 else "Log-rank", "every event the same",
                              f"{chi2:.4f}", str(k - 1), format_p(p, opt.p_style)])
            summary += f" {test_name} P = {p:.4g}."
        for wanted, weight, name, weighs in (
                (opt.compute_gehan_breslow, "gehan", "Gehan-Breslow (generalised Wilcoxon)", "early events (by the number still at risk)"),
                (opt.compute_tarone_ware, "tarone", "Tarone-Ware", "early events, less strongly (square root of the number at risk)"),
                (opt.compute_peto_peto, "peto", "Peto-Peto", "early events (by the share still event-free)")):
            if wanted:
                chi2_w, p_w = _k_sample_test(*_logrank_components(all_time, all_event, all_group, k, weighted=weight), k)
                test_rows.append([name, weighs, f"{chi2_w:.4f}", str(k - 1), format_p(p_w, opt.p_style)])
        if test_rows:
            tables.append(ResultTable(title="Are the curves the same?",
                                      headers=["Test", "Gives most weight to", "Chi-square", "df", "P value"], rows=test_rows))
            if len(test_rows) > 1:
                notes.append("The tests differ in which part of the follow-up counts most. Decide which to trust before "
                             "seeing the results; running several and reporting the smallest P value overstates the evidence.")
        if O is not None and k >= 3 and opt.logrank_trend:
            chi2_t, p_t = _trend_test(O, E, V, k)
            tables.append(ResultTable(title="Log-rank test for a trend across the groups", headers=["Chi-square", "df", "P value"], rows=[[f"{chi2_t:.4f}", "1", f"{p_t:.4g}"]]))
            notes.append("Trend test assumes groups are meaningfully ordered as listed on the sheet.")

        if opt.compare != "none":
            gk_labels = [lbl for lbl, _ in group_list]
            if opt.compare == "vs_control":
                ctrl = max(0, min(opt.control_index, k - 1))
                pairs = [(ctrl, j) for j in range(k) if j != ctrl]
            else:
                pairs = list(combinations(range(k), 2))
            raw_p, comp_rows = [], []
            for i, j in pairs:
                ti, ei = per_group[gk_labels[i]]
                tj, ej = per_group[gk_labels[j]]
                t2 = np.concatenate([ti, tj])
                e2 = np.concatenate([ei, ej])
                g2 = np.concatenate([np.zeros(ti.size, dtype=int), np.ones(tj.size, dtype=int)])
                O2, E2, V2 = _logrank_components(t2, e2, g2, 2, weighted=False)
                _chi2, p2 = _k_sample_test(O2, E2, V2, 2)
                raw_p.append(p2)
                comp_rows.append([f"{gk_labels[i]} vs {gk_labels[j]}", f"{p2:.4g}"])
            adjusted = _common.adjust(raw_p, opt.correction)
            method_label = {"fdr": "Benjamini-Hochberg", "none": "no correction"}.get(opt.correction, "Holm")
            headers = ["Comparison", "P value (raw)"]
            if opt.report_adjusted_p:
                headers += ["P value (adjusted)", "Significant (α = 0.05)"]
                for row, adj in zip(comp_rows, adjusted):
                    adj = float(adj)
                    row.append(f"{adj:.4g}")
                    row.append("Yes" if adj < 0.05 else "No")
            tables.append(ResultTable(
                title=f"Pairs of curves compared - log-rank tests ({method_label})",
                headers=headers,
                rows=comp_rows,
            ))

    # the comparison first, then who was followed and for how long, then the curves time by time
    tables.append(subjects_table)
    if opt.show_time_summary:
        tables.append(time_table)
        if landmark_rows:
            tables.append(landmark_table)
        interval = "log(-log) scale, so they stay within 0 and 100%" if opt.ci_style == "asymmetric" else "plain scale"
        notes += [
            "Median: the first time the curve is at or below 50%. Its confidence limits are the times at which the "
            f"limits of the curve cross 50% (Brookmeyer and Crowley 1982); curve limits use Greenwood's formula on the {interval}. "
            "'Not reached' means the curve, or its limit, stays above 50% to the end of the follow-up.",
            f"Mean survival time is the area under the curve up to time {num(tau)}"
            + (", the longest time every group was followed. It is a mean restricted to that time, not the mean of all "
               "survival times, which cannot be known while some subjects are censored." if k > 1 else
               ", the longest follow-up. It is a mean restricted to that time: the true mean cannot be known while some "
               "subjects are censored."),
            "Median follow-up is found by the reverse Kaplan-Meier method (leaving early counted as the event), so that "
            "early events do not make the follow-up look short.",
        ]
    if opt.show_at_risk_table and len(risk_times) > 1:
        tables.append(risk_table)

    from ..chart_helpers import km_charts

    charts = km_charts(per_group, first="event" if rising else "survival", ci_style=opt.ci_style, band=opt.show_ci_band,
                       median_lines=opt.show_median_lines,
                       at_risk=at_risk if opt.show_at_risk_table and len(risk_times) > 1 else None)
    notes.append("Above the graph, choose what to show: the share still event-free, the share who had the event, the cumulative "
                 "hazard (Nelson-Aalen estimate) or the log of the share event-free. Open symbols are censored subjects"
                 + (", the shaded band is the 95% confidence interval" if opt.show_ci_band else "")
                 + (" and dotted lines mark each median." if opt.show_median_lines else "."))
    return AnalysisResult(title="Survival curves", summary=summary, tables=tables + curve_tables, notes=notes,
                          charts=charts)
