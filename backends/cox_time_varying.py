"""Cox regression when something changes during follow-up.

Ordinary Cox regression gives each covariate one hazard ratio for the whole
of follow-up. Two things break that:

- an effect that fades or grows - a treatment that helps early and then
  stops helping. Here a covariate's hazard ratio may change with time: as a
  steady drift, with the logarithm of time, or in one step at a chosen time;
- a state that begins part-way - a transplant, a second drug, a relapse.
  One data set gives the time each subject's state switched on (blank =
  never); the subject counts as "off" before that time and "on" after it.

The sheet is read as for ordinary Cox regression: X is the time, one data
set says what happened, every other data set is a covariate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg

MOST_PAIRS = 3_000_000         # subjects at risk x event times the fit will take on (a few seconds, about 200 MB)
SHAPES = {"log": "the logarithm of time", "linear": "time", "step": "a step at one time"}


@dataclass
class CoxTimeVaryingOptions:
    event_index: int = 0               # which data set says what happened
    event_value: float = 1.0
    censored_value: float = 0.0
    ties_method: str = "efron"         # "efron" | "breslow"
    vary_effects: bool = True          # let covariates' hazard ratios change with time
    varying: list[str] = field(default_factory=list)   # which covariates; empty = all of them
    time_function: str = "log"         # "log" | "linear" | "step"
    step_time: float = 0.0             # "step": the time of the step; 0 = the median event time
    switch_index: int = -1             # the data set holding the time each subject's state switched on; -1 = none
    reference_levels: dict = field(default_factory=dict)
    auto_reference: dict = field(default_factory=dict)
    times: str = ""                    # report hazard ratios at these times ("6, 12, 24"); empty = chosen from the data
    show_fit: bool = True
    show_at_times: bool = True
    graph_effect: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


class RiskSets:
    """Every (event time, subject at risk then) pair, laid out once.

    A covariate that depends on time has a different value in each risk set,
    so the partial likelihood cannot be worked from one row per subject. Here
    a subject's linear predictor at event time k is

        fixed part + g(time k) × varying part + delta × (switched on before time k)

    which needs one number per pair rather than a whole row of covariates.
    The pairs are stored event time by event time, so sums over a risk set
    are sums over a stretch of an array; the few pairs in which the subject
    has the event are kept as a short list of places."""

    def __init__(self, time: np.ndarray, event: np.ndarray, switch: np.ndarray | None = None):
        n = time.size
        order = np.argsort(time, kind="stable")
        sorted_time = time[order]
        self.event_times = np.unique(time[event == 1])
        count = self.event_times.size
        first = np.searchsorted(sorted_time, self.event_times, side="left")
        sizes = n - first
        self.pairs = int(sizes.sum())
        self.starts = np.cumsum(sizes) - sizes                       # where each event time's risk set begins
        self.k = np.repeat(np.arange(count), sizes)
        place = np.arange(self.pairs) - np.repeat(self.starts, sizes) + np.repeat(first, sizes)
        self.i = order[place]
        self.dead = np.flatnonzero((event[self.i] == 1) & (time[self.i] == self.event_times[self.k]))
        self.dead_k = self.k[self.dead]
        self.deaths = np.bincount(self.dead_k, minlength=count).astype(float)
        self.n = n
        self.on = None
        if switch is not None:
            started = switch[self.i]
            self.on = (np.isfinite(started) & (started < self.event_times[self.k])).astype(float)
        counts = self.deaths.astype(int)
        self.tie_k = np.repeat(np.arange(count), counts)
        within = np.arange(self.tie_k.size) - np.repeat(np.cumsum(counts) - counts, counts)
        self.tie_share = within / np.repeat(self.deaths, counts)
        self._g = self._g_pair = None

    def loglik_and_slope(self, theta: np.ndarray, Zf: np.ndarray, Zv: np.ndarray, g: np.ndarray, efron: bool = True):
        """Log partial likelihood and its slope. ``theta`` = coefficients of
        the fixed columns, then of the varying ones, then (if any) the switch."""
        pf, pv = Zf.shape[1], Zv.shape[1]
        count = self.deaths.size
        eta = (Zf @ theta[:pf])[self.i] if pf else np.zeros(self.pairs)
        if pv:
            if self._g is not g:                                      # g(time) per pair, worked out once for a fit
                self._g, self._g_pair = g, g[self.k]
            eta += self._g_pair * (Zv @ theta[pf:pf + pv])[self.i]
        if self.on is not None:
            eta += theta[pf + pv] * self.on
        eta -= eta.max()                                              # a common shift cancels in every ratio
        risk = np.exp(eta)
        total = np.add.reduceat(risk, self.starts)
        risk_dead = risk[self.dead]
        dying = np.bincount(self.dead_k, weights=risk_dead, minlength=count)
        share = self.tie_share if efron else 0.0
        bottom = total[self.tie_k] - share * dying[self.tie_k]
        value = float(eta[self.dead].sum() - np.log(bottom).sum())
        whole = np.bincount(self.tie_k, weights=1 / bottom, minlength=count)
        left = risk                                                   # the buffer is reused: left = -risk × whole
        left *= whole[self.k]
        np.negative(left, out=left)
        if efron:
            part = np.bincount(self.tie_k, weights=share / bottom, minlength=count)
            left[self.dead] += 1.0 + risk_dead * part[self.dead_k]
        else:
            left[self.dead] += 1.0
        slope = np.empty(theta.size)
        if pf:
            slope[:pf] = Zf.T @ np.bincount(self.i, weights=left, minlength=self.n)
        if pv:
            slope[pf:pf + pv] = Zv.T @ np.bincount(self.i, weights=left * self._g_pair, minlength=self.n)
        if self.on is not None:
            slope[pf + pv] = float(left @ self.on)
        return value, slope


def time_shape(kind: str, times: np.ndarray, step: float) -> np.ndarray:
    """g(time): what a varying covariate is multiplied by."""
    if kind == "linear":
        return np.asarray(times, dtype=float)
    if kind == "step":
        return (np.asarray(times, dtype=float) > step).astype(float)
    return np.log(np.asarray(times, dtype=float))


def fit(sets: RiskSets, Zf: np.ndarray, Zv: np.ndarray, g: np.ndarray, efron: bool = True, with_covariance: bool = True):
    """Maximise the partial likelihood. Returns (theta, covariance, log partial
    likelihood, settled?); the covariance is None when it is not asked for."""
    p = Zf.shape[1] + Zv.shape[1] + (1 if sets.on is not None else 0)
    if p == 0:
        return np.zeros(0), np.zeros((0, 0)), sets.loglik_and_slope(np.zeros(0), Zf, Zv, g, efron)[0], True

    def objective(theta):
        value, slope = sets.loglik_and_slope(theta, Zf, Zv, g, efron)
        return -value, -slope

    def slope_only(theta):
        return sets.loglik_and_slope(theta, Zf, Zv, g, efron)[1]

    best = optimize.minimize(objective, np.zeros(p), jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    theta = best.x
    value, slope = sets.loglik_and_slope(theta, Zf, Zv, g, efron)
    settled = bool(best.success) or bool(np.max(np.abs(slope)) < 1e-4)
    if not with_covariance:
        return theta, None, value, settled
    H, fresh = rg.hessian_of(slope_only, theta), True
    for _ in range(4):                               # exact Newton steps to finish, while they still help
        if np.max(np.abs(slope)) < 1e-7:
            break
        if not fresh:
            H, fresh = rg.hessian_of(slope_only, theta), True
        try:
            step = np.linalg.solve(H, slope)
        except np.linalg.LinAlgError:
            break
        trial_value, trial_slope = sets.loglik_and_slope(theta - step, Zf, Zv, g, efron)
        if not np.isfinite(trial_value) or trial_value < value - 1e-9:
            break
        theta, value, slope = theta - step, trial_value, trial_slope
        fresh = bool(np.max(np.abs(step)) < 1e-4)    # after a step this small the curvature has not moved
    if not fresh:
        H = rg.hessian_of(slope_only, theta)
    settled = settled or bool(np.max(np.abs(slope)) < 1e-4)
    return theta, rg.inverse(-H), value, settled


def _typed_times(text: str) -> list[float]:
    out = []
    for token in re.split(r"[,;\s]+", (text or "").strip()):
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return sorted(out)


@register("cox_time_varying")
def run(data: SheetData, options: CoxTimeVaryingOptions | None = None) -> AnalysisResult:
    opt = options or CoxTimeVaryingOptions()
    names, matrix = data.first_replicate_matrix()
    if len(names) < 2:
        raise AnalysisError("This needs the time in X, one data set saying what happened, and at least one covariate or a data set "
                            "of switch-on times.")
    event_at = opt.event_index if 0 <= opt.event_index < len(names) else 0
    switch_at = opt.switch_index if 0 <= opt.switch_index < len(names) and opt.switch_index != event_at else -1
    covariates = [i for i in range(len(names)) if i not in (event_at, switch_at) and not np.all(np.isnan(matrix[:, i]))]
    n_all = min(data.x.size, matrix.shape[0])
    time_all = np.asarray(data.x[:n_all], dtype=float)
    matrix = matrix[:n_all]
    code = matrix[:, event_at]
    keep = np.isfinite(time_all) & np.isin(code, (opt.event_value, opt.censored_value))
    if covariates:
        keep &= ~np.any(np.isnan(matrix[:, covariates]), axis=1)
    touched = np.isfinite(time_all) | np.isfinite(code)
    dropped = int(touched.sum() - keep.sum())
    time = time_all[keep]
    event = (code[keep] == opt.event_value).astype(int)
    n = time.size
    if n < 6:
        raise AnalysisError(f"Only {n} rows have a time in X, a code of {opt.event_value:g} (event) or {opt.censored_value:g} "
                            f"(censored) in '{names[event_at]}', and every covariate filled in.")
    if event.sum() < 3:
        raise AnalysisError("At least 3 events (not censored subjects) are needed.")
    if np.any(time < 0):
        raise AnalysisError("Times cannot be negative.")
    raw = {names[i]: matrix[keep][:, i] for i in covariates}
    Zf, term_names, sources, categories = rg.code_terms(raw, opt.reference_levels, opt.auto_reference)
    switch = matrix[keep][:, switch_at] if switch_at >= 0 else None
    switch_name = names[switch_at] if switch_at >= 0 else ""
    notes: list[str] = []
    if switch is not None:
        during = np.isfinite(switch) & (switch < time)
        if not during.any():
            raise AnalysisError(f"No subject has a switch-on time in '{switch_name}' earlier than their own time in X, so there is "
                                "nobody in the 'on' state to compare.")
        if during.all() and np.all(switch[during] <= 0):
            notes.append(f"Everybody is 'on' from the start in '{switch_name}', so it cannot be told apart from the baseline.")
    kind = opt.time_function if opt.time_function in SHAPES else "log"
    wanted = [s for s in dict.fromkeys(sources) if not opt.varying or s in opt.varying] if opt.vary_effects else []
    vary_columns = [j for j, s in enumerate(sources) if s in wanted]
    if not vary_columns and switch is None:
        if not term_names:
            raise AnalysisError("There is no covariate that varies, and no data set of switch-on times.")
        notes.append("Nothing was asked to change with time, so this is an ordinary Cox regression.")
    event_times = np.unique(time[event == 1])
    step_at = float(opt.step_time) if opt.step_time and opt.step_time > 0 else float(np.median(time[event == 1]))
    if vary_columns:
        if kind == "log" and event_times.min() <= 0:
            raise AnalysisError("An effect that changes with the logarithm of time needs every event time to be above zero. Use a "
                                "steady drift or a step instead.")
        if kind == "step" and not (event_times.min() <= step_at < event_times.max()):
            raise AnalysisError(f"The step at time {step_at:g} needs events both up to it and after it; the events run from "
                                f"{event_times.min():g} to {event_times.max():g}.")
        if event_times.size < 3:
            raise AnalysisError("An effect can only be seen to change over time if events happen at 3 or more different times.")
    pairs = int(np.sum(n - np.searchsorted(np.sort(time), event_times, side="left")))
    if pairs > MOST_PAIRS:
        raise AnalysisError(f"With {n} subjects and {event_times.size} different event times, this model has to look at {pairs:,} "
                            "subject-by-event-time pairs, which is more than it can take on. Round the times to coarser units (whole "
                            "days or weeks instead of exact values) so that more events share a time.")
    pf, pv = Zf.shape[1], len(vary_columns)
    p = pf + pv + (1 if switch is not None else 0)
    if event.sum() < 2 * p or n < p + 4:
        raise AnalysisError(f"{int(event.sum())} events are too few for {p} coefficients. Let fewer effects change with time, or "
                            "leave covariates out.")
    if pf:
        rg.check_rank(rg.with_constant(Zf), term_names)

    # ---- the fit, on a common scale
    sets = RiskSets(time, event, switch)
    g_raw = time_shape(kind, sets.event_times, step_at) if pv else np.zeros(sets.event_times.size)
    weights = sets.deaths / sets.deaths.sum()
    g_centre = float(np.sum(weights * g_raw)) if pv else 0.0
    g_scale = float(np.sqrt(np.sum(weights * (g_raw - g_centre) ** 2))) if pv else 1.0
    if pv and g_scale <= 0:
        raise AnalysisError("Every event falls on the same side of the change in time, so a changing effect cannot be estimated.")
    scale = Zf.std(axis=0) if pf else np.ones(0)
    scale = np.where(scale > 0, scale, 1.0)
    Zs = Zf / scale if pf else Zf
    Zv = Zs[:, vary_columns] if pv else np.empty((n, 0))
    efron = opt.ties_method != "breslow"
    theta_s, cov_s, loglik, settled = fit(sets, Zs, Zv, (g_raw - g_centre) / g_scale, efron)
    loglik0 = sets.loglik_and_slope(np.zeros(p), Zs, Zv, (g_raw - g_centre) / g_scale, efron)[0]
    # back to the typed scale and to g as defined (not centred):  beta = b/s - sum(gamma × centre),  gamma = c/(s × g_scale)
    A = np.zeros((p, p))
    for j in range(pf):
        A[j, j] = 1 / scale[j]
    for v, j in enumerate(vary_columns):
        A[pf + v, pf + v] = 1 / (scale[j] * g_scale)
        A[j, pf + v] = -g_centre / (scale[j] * g_scale)
    if switch is not None:
        A[p - 1, p - 1] = 1.0
    theta = A @ theta_s
    cov = A @ cov_s @ A.T
    every_term = term_names + [f"{term_names[j]} (its change with time)" for j in vary_columns] + ([switch_name] if switch is not None else [])
    adrift = rg.runaway(theta_s, np.sqrt(np.clip(np.diag(cov_s), 0, None)), np.ones(p))
    if adrift:
        notes.append(rg.runaway_note([every_term[j] for j in adrift], "that term separates those who had the event from those still "
                                                                    "at risk completely"))
    elif not settled:
        notes.append("The fit did not settle: the estimates may be unreliable. This usually means too few events for the number of "
                     "coefficients.")

    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"
    zc = stats.norm.ppf(0.5 + level / 2)

    def num(v) -> str:
        return rg.clean_zero(f"{v:.{sig}g}") if np.isfinite(v) else "-"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    def line(label: str, w: np.ndarray, ratio_word: str = "") -> list[str]:
        est = float(w @ theta)
        se = float(np.sqrt(max(w @ cov @ w, 0.0)))
        z = est / se if se > 0 else float("nan")
        pv_ = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
        return [label, num(est), num(se), f"{z:.3f}" if np.isfinite(z) else "-", p_text(pv_), rg.ratio_text(rg.ratio(est), sig),
                rg.ratio_span(rg.ratio(est - zc * se), rg.ratio(est + zc * se), sig)]

    def unit(j: int) -> np.ndarray:
        w = np.zeros(p)
        w[j] = 1.0
        return w

    def effect_at(v: int, t: float) -> np.ndarray:
        """Weights giving a varying covariate's log hazard ratio at time t."""
        w = unit(vary_columns[v])
        w[pf + v] = float(time_shape(kind, np.array([t]), step_at)[0])
        return w

    middle = float(np.median(time[event == 1]))
    headers = ["Term", "Coefficient", "SE", "z", "P value", "Hazard ratio", f"{pct} CI"]
    steady = [line(term_names[j], unit(j)) for j in range(pf) if j not in vary_columns]
    if switch is not None:
        steady.append(line(f"{switch_name}: on, against off", unit(p - 1)))
    tables: list[ResultTable] = []
    if steady:
        tables.append(ResultTable("Effects that hold steady over time", headers, steady))

    change_p: list[tuple[float, str]] = []
    if pv:
        moving = []
        for v, j in enumerate(vary_columns):
            name = term_names[j]
            if kind == "step":
                moving.append(line(f"{name}, up to time {step_at:g}", effect_at(v, step_at)))
                moving.append(line(f"{name}, after time {step_at:g}", effect_at(v, step_at * 2 + 1)))
                change = line(f"{name}: after ÷ before", unit(pf + v))
            elif kind == "linear":
                moving.append(line(f"{name}, at time {middle:g}", effect_at(v, middle)))
                change = line(f"{name}: change for each unit of time", unit(pf + v))
            else:
                moving.append(line(f"{name}, at time {middle:g}", effect_at(v, middle)))
                change = line(f"{name}: change each time the time doubles", unit(pf + v) * np.log(2))
            moving.append(change)
            se_g = float(np.sqrt(max(cov[pf + v, pf + v], 0.0)))
            if se_g > 0:
                change_p.append((float(2 * stats.norm.sf(abs(theta[pf + v] / se_g))), name))
        tables.append(ResultTable(f"Effects allowed to change with {SHAPES[kind]}", headers, moving))

    # ---- hazard ratios at chosen times
    charts = []
    if pv and kind != "step":
        lo_t, hi_t = float(event_times.min()), float(event_times.max())
        typed = [t for t in _typed_times(opt.times) if (kind != "log" or t > 0)]
        chosen = typed or [float(q) for q in np.unique(np.round(np.quantile(time[event == 1], [0.1, 0.25, 0.5, 0.75, 0.9]), 6))]
        if opt.show_at_times:
            rows = []
            for v, j in enumerate(vary_columns):
                for t in chosen:
                    row = line(f"{term_names[j]}, at time {t:g}", effect_at(v, t))
                    rows.append([row[0], row[5], row[6], row[4]] + (["beyond the events seen"] if not lo_t <= t <= hi_t else [""]))
            tables.append(ResultTable("Hazard ratio at chosen times", ["Term", "Hazard ratio", f"{pct} CI", "P value (against 1)", ""], rows))
        if opt.graph_effect:
            grid = np.linspace(lo_t, hi_t, 80) if kind == "linear" else rg.ratio(np.linspace(np.log(lo_t), np.log(hi_t), 80))
            for v, j in enumerate(vary_columns[:8]):
                if j in adrift or pf + v in adrift:
                    continue                      # a ratio that has run away has no curve worth drawing
                est = np.array([effect_at(v, t) @ theta for t in grid])
                se = np.array([np.sqrt(max(effect_at(v, t) @ cov @ effect_at(v, t), 0.0)) for t in grid])
                charts.append(ChartSpec(f"Hazard ratio of {term_names[j]} over time", x_label="Time", y_label=f"Hazard ratio ({pct} band)",
                                        series=[ChartSeries("", grid.tolist(), rg.ratio(est - zc * se).tolist(), kind="band",
                                                            y_upper=rg.ratio(est + zc * se).tolist(), color="#1f5aa6"),
                                                ChartSeries("", grid.tolist(), rg.ratio(est).tolist(), kind="line", color="#1f5aa6")],
                                        reference_lines=[{"kind": "h", "value": 1.0}], y_log=True, group="hazard_over_time",
                                        option=term_names[j]))
    elif pv and opt.graph_effect:
        xs, ys, below, above, cats = [], [], [], [], []
        for v, j in enumerate(vary_columns[:6]):
            if j in adrift or pf + v in adrift:
                continue                          # a ratio that has run away has nothing worth drawing
            for label, t in ((f"up to {step_at:g}", step_at), (f"after {step_at:g}", step_at * 2 + 1)):
                w = effect_at(v, t)
                est, se = float(w @ theta), float(np.sqrt(max(w @ cov @ w, 0.0)))
                xs.append(float(len(xs)))
                ys.append(float(rg.ratio(est)))
                below.append(float(rg.ratio(est) - rg.ratio(est - zc * se)))
                above.append(float(rg.ratio(est + zc * se) - rg.ratio(est)))
                cats.append(f"{term_names[j]}\n{label}" if pv > 1 else label)
        if xs:
            charts.append(ChartSpec("Hazard ratio before and after the step", x_label="", y_label=f"Hazard ratio ({pct} CI)",
                                    series=[ChartSeries("", xs, ys, y_err=[below, above], color="#1f5aa6")], x_categories=cats, y_log=True,
                                    reference_lines=[{"kind": "h", "value": 1.0}]))

    # ---- the model as a whole
    whole = max(2 * (loglik - loglik0), 0.0)
    whole_p = float(stats.chi2.sf(whole, p)) if p else float("nan")
    drift_text = ""
    if opt.show_fit or pv:
        fit_rows = [["Subjects", str(n)], ["Events", str(int(event.sum()))], ["Different event times", str(event_times.size)],
                    ["−2 × log partial likelihood, no covariates", f"{-2 * loglik0:.3f}"],
                    ["−2 × log partial likelihood, this model", f"{-2 * loglik:.3f}"],
                    ["All terms together, against none", f"chi-square({p}) = {whole:.3f}, P = {p_text(whole_p)}"],
                    ["AIC", f"{2 * p - 2 * loglik:.3f}"]]
        if pv:
            _t, _c, loglik_steady, _ok = fit(sets, Zs, np.empty((n, 0)), np.zeros(sets.event_times.size), efron, with_covariance=False)
            drift = max(2 * (loglik - loglik_steady), 0.0)
            drift_p = float(stats.chi2.sf(drift, pv))
            fit_rows.append([f"Do the effects change with time at all? (against steady effects)", f"chi-square({pv}) = {drift:.3f}, P = {p_text(drift_p)}"])
            drift_text = (f" Letting the effect{'s' if pv != 1 else ''} change with {SHAPES[kind]} "
                          f"{'fits clearly better' if drift_p < opt.alpha else 'does not fit clearly better'} than steady effects: "
                          f"chi-square({pv}) = {drift:.3f}, P = {drift_p:.4g}.")
        if switch is not None:
            fit_rows.insert(2, [f"Subjects whose state ({switch_name}) switched on during follow-up",
                                f"{int(during.sum())} of {n}, at a median time of {np.median(switch[during]):g}"])
        if opt.show_fit:
            tables.append(ResultTable("The model as a whole", ["Measure", "Value"], fit_rows))

    # ---- headline
    summary = (f"Cox regression on {n} subjects with {int(event.sum())} events"
               f"{', effects free to change with ' + SHAPES[kind] if pv else ''}"
               f"{', and a state that switches on during follow-up' if switch is not None else ''}. All terms together: "
               f"chi-square({p}) = {whole:.3f}, P = {whole_p:.4g}.")
    summary += drift_text
    if change_p and min(change_p)[0] < opt.alpha:
        low_p, low_name = min(change_p)
        summary += f" Clearest change: {low_name} (P = {low_p:.4g})."
    if switch is not None:
        w = unit(p - 1)
        est, se = float(theta[p - 1]), float(np.sqrt(max(cov[p - 1, p - 1], 0.0)))
        if se > 0 and p - 1 not in adrift:
            summary += (f" {switch_name} on against off: hazard ratio {rg.ratio(est):.{sig}g} ({pct} CI {rg.ratio(est - zc * se):.{sig}g} to "
                        f"{rg.ratio(est + zc * se):.{sig}g}), P = {2 * stats.norm.sf(abs(est / se)):.4g}.")

    # ---- notes
    if dropped:
        notes.insert(0, f"{dropped} row{'s' if dropped != 1 else ''} without a time, a recognised code in '{names[event_at]}' or a "
                        f"value for every covariate {'were' if dropped != 1 else 'was'} left out.")
    if pv:
        how = {"log": "its logarithm of hazard ratio is a straight line against the logarithm of time, so the hazard ratio is "
                      "multiplied by a fixed amount each time the time doubles",
               "linear": "its hazard ratio is multiplied by a fixed amount for each unit of time",
               "step": f"it has one hazard ratio up to time {step_at:g} and another after it"}[kind]
        notes.append(f"A covariate allowed to change with time: {how}. The row for the change tests whether the effect is steady - "
                     "which is the proportional-hazards assumption for that covariate. A clear change means a single hazard ratio "
                     "for the whole follow-up would mislead.")
        if kind != "step":
            notes.append("Hazard ratios are most trustworthy at times where many events happened; towards the first and last "
                         "events the band widens.")
    if switch is not None:
        notes.append(f"'{switch_name}' gives the time a subject's state switched on. A subject is compared as 'off' at every event "
                     "before that time and as 'on' afterwards. Treating such subjects as 'on' from the start would credit the "
                     "state with the time they had to survive to reach it (immortal-time bias) and make it look protective.")
    if categories:
        notes.append("Read as categories (two to six distinct values): " + "; ".join(
            f"{key} - levels {', '.join(f'{v:g}' for v in lv)}, reference {ref:g}" for key, (lv, ref) in categories.items()) + ".")
    notes.append(f"Cox regression with time-dependent covariates (Cox 1972), fitted by maximising the partial likelihood over every "
                 f"risk set; tied event times by {'Efron' if efron else 'Breslow'}'s method; Wald intervals.")
    return AnalysisResult(title="Risk that changes over time", summary=summary, tables=tables, notes=notes, charts=charts)
