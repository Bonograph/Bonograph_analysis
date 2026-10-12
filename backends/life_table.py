"""Survival summarised period by period: the life table (actuarial) method.

Where a survival curve steps down at every single event, a life table
divides the follow-up into intervals and says, for each, how many entered
it, how many left early, how many had the event, and what share came
through. It suits large studies, follow-up recorded in whole months or
years, and data that only exist as counts per interval.

Two ways to lay the data out:

- one row per subject, as for survival curves: X is the time, and each data
  set (a group) holds a code for what happened then;
- one row per interval: X is the time the interval starts, and each data
  set (a group) has two side-by-side columns - the number who had the event
  in the interval, then the number who left early in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg

MOST_INTERVALS = 200


@dataclass
class LifeTableOptions:
    layout: str = "auto"             # "auto" | "subjects" | "intervals"
    death_code: float = 1.0          # one row per subject: the code for the event ...
    censored_code: float = 0.0       # ... and for leaving early
    interval_width: float = 0.0      # one row per subject: 0 = chosen for you
    show_hazard: bool = True         # hazard rate and density of events in each interval
    compare: bool = True             # two groups or more: are their tables the same?
    graph_survival: bool = True
    graph_hazard: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def actuarial(starts: np.ndarray, widths: np.ndarray, events: np.ndarray, withdrawn: np.ndarray, level: float = 0.95) -> dict:
    """The life table for one group. ``widths`` may end with inf for a last
    interval with no upper end. Everyone is accounted for: the number
    entering the first interval is the total of the events and withdrawals.

    Those who leave early in an interval are counted as at risk for half of
    it (the actuarial assumption); standard errors follow Greenwood."""
    events, withdrawn = np.asarray(events, dtype=float), np.asarray(withdrawn, dtype=float)
    gone = events + withdrawn
    entering = gone.sum() - np.concatenate([[0.0], np.cumsum(gone)[:-1]])
    exposed = entering - withdrawn / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(exposed > 0, events / exposed, 0.0)
        p = 1 - q
        surv_end = np.cumprod(p)
        surv_start = np.concatenate([[1.0], surv_end[:-1]])
        piece = np.where((exposed > 0) & (p > 0), q / (exposed * p), 0.0)        # Greenwood's terms
        running = np.cumsum(piece)
        before = np.concatenate([[0.0], running[:-1]])
        se_surv = np.where(p > 0, surv_end * np.sqrt(running), 0.0)
        closed = np.isfinite(widths)
        density = np.where(closed, surv_start * q / widths, np.nan)
        se_density = np.where(closed & (q > 0) & (exposed > 0), density * np.sqrt(before + p / (exposed * q)), np.where(closed, 0.0, np.nan))
        hazard = np.where(closed, 2 * q / (widths * (1 + p)), np.nan)
        se_hazard = np.where(closed & (q > 0) & (exposed > 0),
                             hazard * np.sqrt(np.clip(1 - (widths * hazard / 2) ** 2, 0, None) / (exposed * q)), np.where(closed, 0.0, np.nan))
        # interval for the share still event-free: on the log(-log) scale, so it stays within 0 and 1
        z = stats.norm.ppf(0.5 + level / 2)
        inside = (surv_end > 0) & (surv_end < 1) & (se_surv > 0)
        spread = np.where(inside, z * se_surv / (surv_end * np.abs(np.log(np.where(inside, surv_end, 0.5)))), 0.0)
        low = np.where(inside, surv_end ** np.exp(spread), surv_end)
        high = np.where(inside, surv_end ** np.exp(-spread), surv_end)

    median = se_median = None
    reached = np.where(surv_end <= 0.5)[0]
    median_open = False
    if reached.size:
        i = int(reached[0])
        if closed[i] and surv_start[i] > surv_end[i]:
            median = float(starts[i] + widths[i] * (surv_start[i] - 0.5) / (surv_start[i] - surv_end[i]))
            if density[i] > 0 and exposed[0] > 0:
                se_median = float(1 / (2 * density[i] * np.sqrt(exposed[0])))
        else:
            median_open = True
    return {"entering": entering, "exposed": exposed, "q": q, "p": p, "surv_end": surv_end, "surv_start": surv_start, "se_surv": se_surv,
            "low": low, "high": high, "density": density, "se_density": se_density, "hazard": hazard, "se_hazard": se_hazard,
            "median": median, "se_median": se_median, "median_open": median_open, "events": events, "withdrawn": withdrawn}


def interval_tests(exposed: np.ndarray, events: np.ndarray) -> dict[str, tuple[float, int, float]]:
    """Log-rank and Wilcoxon (Gehan) tests worked from the intervals: groups
    in rows, intervals in columns, with the adjusted numbers at risk standing
    in for the risk sets. Returns {name: (chi-square, df, P)}."""
    k = exposed.shape[0]
    out = {}
    for name, power in (("Log-rank", 0), ("Wilcoxon (Gehan)", 1)):
        gap, V = np.zeros(k), np.zeros((k, k))
        for i in range(exposed.shape[1]):
            n_g, d_g = exposed[:, i], events[:, i]
            n, d = n_g.sum(), d_g.sum()
            if n <= 1 or d <= 0:
                continue
            w = n**power
            gap += w * (d_g - d * n_g / n)
            factor = d * (n - d) / (n - 1) / n**2
            v = -factor * np.outer(n_g, n_g)
            np.fill_diagonal(v, factor * n_g * (n - n_g))
            V += w**2 * v
        if not np.any(V):
            out[name] = (float("nan"), k - 1, float("nan"))
            continue
        chi2 = float(gap @ np.linalg.pinv(V) @ gap)
        out[name] = (chi2, k - 1, float(stats.chi2.sf(chi2, k - 1)))
    return out


def nice_width(longest: float, about: int = 10) -> float:
    """A round interval width that divides the follow-up into about ten parts."""
    raw = longest / about
    power = 10.0 ** np.floor(np.log10(raw))
    return float(next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw * (1 - 1e-9)))


def _is_interval_sheet(data: SheetData, opt: LifeTableOptions) -> bool:
    labels = data.group_labels()
    if not labels:
        return False
    x = np.asarray(data.x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.any(np.diff(x) <= 0):
        return False
    for label in labels:
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.shape[1] < 2:
            return False
        values = matrix[np.isfinite(matrix)]
        if values.size == 0 or np.any(values < 0) or np.any(np.abs(values - np.round(values)) > 1e-9):
            return False
    return True


def _read_intervals(data: SheetData):
    x = np.asarray(data.x, dtype=float)
    rows = np.where(np.isfinite(x))[0]
    if rows.size < 2:
        raise AnalysisError("For one row per interval, X needs the start time of at least 2 intervals.")
    starts = x[rows]
    if np.any(np.diff(starts) <= 0):
        raise AnalysisError("For one row per interval, the start times in X must rise from row to row.")
    widths = np.append(np.diff(starts), np.inf)
    groups = {}
    for label in data.group_labels():
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.size == 0:
            continue
        block = np.zeros((rows.size, 2))
        take = rows[rows < matrix.shape[0]]
        part = np.nan_to_num(matrix[take][:, : min(2, matrix.shape[1])], nan=0.0)
        block[: take.size, : part.shape[1]] = part
        if np.any(block < 0) or np.any(np.abs(block - np.round(block)) > 1e-9):
            raise AnalysisError(f"'{label}': the numbers who had the event and who left early must be whole numbers, zero or more.")
        if block.sum() >= 2:
            groups[label] = (block[:, 0], block[:, 1])
    if not groups:
        raise AnalysisError("No data set holds counts. Each needs two side-by-side columns: the number who had the event in each "
                            "interval, then the number who left early in it.")
    return starts, widths, groups


def _read_subjects(data: SheetData, opt: LifeTableOptions):
    per_group = {}
    for label in data.group_labels():
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.size == 0:
            continue
        code = matrix[:, 0]
        n = min(data.x.size, code.size)
        time, code = np.asarray(data.x[:n], dtype=float), code[:n]
        keep = np.isfinite(time) & np.isin(code, (opt.death_code, opt.censored_code))
        if keep.sum() >= 2:
            per_group[label] = (time[keep], (code[keep] == opt.death_code).astype(int))
    if not per_group:
        raise AnalysisError(f"No data set has at least 2 subjects with a time in X and a code of {opt.death_code:g} (event) or "
                            f"{opt.censored_code:g} (left early). For counts per interval, give each data set two side-by-side "
                            "columns: events, then those who left early.")
    longest = max(float(t.max()) for t, _e in per_group.values())
    if min(float(t.min()) for t, _e in per_group.values()) < 0:
        raise AnalysisError("Times cannot be negative.")
    if longest <= 0:
        raise AnalysisError("Every time is zero, so there is no follow-up to divide into intervals.")
    width = float(opt.interval_width) if opt.interval_width and opt.interval_width > 0 else nice_width(longest)
    count = int(np.floor(longest / width * (1 + 1e-12))) + 1
    chosen = not (opt.interval_width and opt.interval_width > 0)
    if count > MOST_INTERVALS:
        if chosen:
            width = longest / (MOST_INTERVALS - 1)
            count = MOST_INTERVALS
        else:
            raise AnalysisError(f"An interval width of {width:g} gives {count} intervals over a follow-up of {longest:g}. Use a wider "
                                f"interval (up to {MOST_INTERVALS} intervals).")
    starts = width * np.arange(count)
    widths = np.full(count, width)
    groups = {}
    for label, (time, event) in per_group.items():
        place = np.minimum(np.floor(time / width * (1 + 1e-12)).astype(int), count - 1)
        groups[label] = (np.bincount(place, weights=event, minlength=count).astype(float),
                         np.bincount(place, weights=1 - event, minlength=count).astype(float))
    return starts, widths, groups, width, chosen


@register("life_table")
def run(data: SheetData, options: LifeTableOptions | None = None) -> AnalysisResult:
    opt = options or LifeTableOptions()
    layout = opt.layout if opt.layout in ("subjects", "intervals") else ("intervals" if _is_interval_sheet(data, opt) else "subjects")
    notes: list[str] = []
    if layout == "intervals":
        starts, widths, groups = _read_intervals(data)
        how = "the intervals typed in X"
    else:
        starts, widths, groups, width, chosen = _read_subjects(data, opt)
        how = f"intervals of {width:g}" + (" (chosen for you; set the width in the settings)" if chosen else "")
    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"
    names = list(groups)
    k = len(names)

    def num(v) -> str:
        return f"{v:.{sig}g}" if v is not None and np.isfinite(v) else "-"

    def whole(v) -> str:
        return f"{v:g}"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    def span(i: int) -> str:
        return f"{starts[i]:g} to {starts[i] + widths[i]:g}" if np.isfinite(widths[i]) else f"{starts[i]:g} and over"

    tables: list[ResultTable] = []
    fitted = {name: actuarial(starts, widths, ev, wd, level) for name, (ev, wd) in groups.items()}

    # ---- one line per group
    overview = []
    for name in names:
        t = fitted[name]
        total, had, left = t["entering"][0], t["events"].sum(), t["withdrawn"].sum()
        if t["median"] is not None:
            z = stats.norm.ppf(0.5 + level / 2)
            med = num(t["median"])
            med_se = num(t["se_median"])
            med_ci = (f"{num(max(t['median'] - z * t['se_median'], 0.0))} to {num(t['median'] + z * t['se_median'])}"
                      if t["se_median"] is not None else "-")
        else:
            med = "in the open last interval" if t["median_open"] else "not reached"
            med_se = med_ci = "-"
        overview.append([name, whole(total), whole(had), whole(left), med, med_se, med_ci, f"{t['surv_end'][-1]:.1%}"])
    tables.append(ResultTable("Each group in brief", ["Group", "Subjects", "Had the event", "Left early", "Median time", "SE",
                                                      f"{pct} CI", "Still event-free at the end"], overview))

    # ---- are the groups the same?
    test_text = ""
    if k >= 2 and opt.compare:
        exposed = np.vstack([fitted[name]["exposed"] for name in names])
        events = np.vstack([fitted[name]["events"] for name in names])
        tests = interval_tests(exposed, events)
        tables.append(ResultTable("Are the groups' life tables the same?", ["Test", "Chi-square", "df", "P value"],
                                  [[f"{name} test, on the intervals", f"{c:.4f}" if c == c else "-", str(df), p_text(p)]
                                   for name, (c, df, p) in tests.items()]))
        c, df, p = tests["Log-rank"]
        c2, _df2, p2 = tests["Wilcoxon (Gehan)"]
        if p == p:
            test_text = (f" The groups {'differ' if p < opt.alpha else 'do not clearly differ'}: log-rank chi-square({df}) = {c:.3f}, "
                         f"P = {p:.4g} (Wilcoxon-Gehan: chi-square({df}) = {c2:.3f}, P = {p2:.4g}).")
            if (p < opt.alpha) != (p2 < opt.alpha):
                notes.append("The two tests fall on different sides of the threshold. They weigh the intervals differently - the "
                             "log-rank test all alike, the Wilcoxon (Gehan) test the early ones more - so this points to a difference "
                             "that is not steady over time. Decide which test to rely on before looking at the data, not after.")
        notes.append("Both tests compare the events seen in each interval with those expected if the groups shared one table. The "
                     "log-rank test weighs every interval alike; the Wilcoxon (Gehan) test weighs each by the number still at risk, "
                     "so it is more sensitive to early differences. They work from the interval counts, so they differ a little "
                     "from the same tests on exact times.")

    # ---- the tables themselves
    for name in names:
        t = fitted[name]
        last = int(np.max(np.where(t["entering"] > 0)[0])) if np.any(t["entering"] > 0) else 0
        headers = ["Interval", "Entered", "Left early", "At risk (adjusted)", "Events", "Had the event", "Came through",
                   "Event-free at the end", "SE", f"{pct} CI"]
        if opt.show_hazard:
            headers += ["Hazard rate", "SE", "Events per unit time (density)", "SE"]
        rows = []
        for i in range(last + 1):
            row = [span(i), whole(t["entering"][i]), whole(t["withdrawn"][i]), whole(t["exposed"][i]), whole(t["events"][i]),
                   f"{t['q'][i]:.4f}", f"{t['p'][i]:.4f}", f"{t['surv_end'][i]:.4f}", f"{t['se_surv'][i]:.4f}",
                   f"{t['low'][i]:.4f} to {t['high'][i]:.4f}"]
            if opt.show_hazard:
                row += [num(t["hazard"][i]), num(t["se_hazard"][i]), num(t["density"][i]), num(t["se_density"][i])]
            rows.append(row)
        tables.append(ResultTable(f"Life table: {name}" if k > 1 or name else "Life table", headers, rows))

    # ---- graphs
    charts = []
    if opt.graph_survival:
        series = []
        for j, name in enumerate(names):
            t = fitted[name]
            ends = np.where(np.isfinite(widths), starts + widths, np.nan)
            shown = np.isfinite(ends) & (t["entering"] > 0)
            xs = np.concatenate([[starts[0]], ends[shown]])
            ys = np.concatenate([[100.0], t["surv_end"][shown] * 100])
            series.append(ChartSeries(name if k > 1 else "", xs.tolist(), ys.tolist(), kind="line_markers", color=rg.SHADES[j % len(rg.SHADES)]))
        charts.append(ChartSpec("Share still event-free at the end of each interval", x_label="Time", y_label="Event-free (%)",
                                series=series, y_limits=(0.0, 102.0), reference_lines=[{"kind": "h", "value": 50.0}]))
    if opt.graph_hazard:
        series = []
        for j, name in enumerate(names):
            t = fitted[name]
            shown = np.isfinite(widths) & (t["exposed"] > 0)
            if shown.any():
                mid = (starts + np.where(np.isfinite(widths), widths, 0.0) / 2)[shown]
                series.append(ChartSeries(name if k > 1 else "", mid.tolist(), t["hazard"][shown].tolist(), kind="line_markers",
                                          color=rg.SHADES[j % len(rg.SHADES)]))
        if series:
            charts.append(ChartSpec("Hazard rate in each interval", x_label="Time (middle of the interval)", y_label="Events per subject at risk, per unit time",
                                    series=series))

    # ---- headline
    first = fitted[names[0]]
    parts = []
    for name in names[:4]:
        t = fitted[name]
        where = (f"median time {num(t['median'])}" if t["median"] is not None else
                 "half had the event only in the open last interval" if t["median_open"] else "median not reached")
        parts.append(f"{name + ': ' if k > 1 else ''}{whole(t['entering'][0])} subjects, {whole(t['events'].sum())} events, "
                     f"{t['surv_end'][-1]:.1%} event-free at the end, {where}")
    summary = f"Life table over {len(starts)} intervals ({how}). " + "; ".join(parts) + ("; ..." if k > 4 else "") + "." + test_text

    notes.insert(0, "A subject who leaves early in an interval is counted as at risk for half of it: 'At risk (adjusted)' is the "
                    "number who entered minus half of those who left early. 'Had the event' is the events over that number, and "
                    "'Event-free at the end' multiplies the shares that came through every interval so far.")
    notes.append("The hazard rate is the events per subject at risk per unit of time, taken at the middle of the interval; the "
                 "density is the share of all subjects having the event per unit of time. Neither can be worked out for a last "
                 "interval with no upper end.")
    notes.append("The median time is where the event-free share crosses 50%, read off by a straight line within that interval. "
                 "Standard errors of the event-free share are Greenwood's, with intervals on the log(-log) scale.")
    if layout == "subjects":
        notes.append("With a time for every subject, survival curves (Kaplan-Meier) use each event exactly and are the usual choice; "
                     "a life table is the better summary when there are very many subjects or the times were recorded coarsely.")
    if len(starts) < 3:
        notes.append(f"With only {len(starts)} interval{'s' if len(starts) != 1 else ''} the table says little, and a median read off "
                     "inside so wide an interval is a rough guess. Use narrower intervals.")
    thin = [name for name in names if fitted[name]["entering"][0] < 30]
    if thin:
        notes.append("A life table works with shares per interval, which are rough in small groups (" + ", ".join(thin[:4]) + "): "
                     "use wider intervals, or survival curves.")
    notes.append("References: Berkson and Gage 1950; Cutler and Ederer 1958; Gehan 1969 (hazard and density).")
    return AnalysisResult(title="Survival by intervals", summary=summary, tables=tables, notes=notes, charts=charts)
