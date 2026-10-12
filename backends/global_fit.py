"""Fit one curve to several data sets at once, sharing some parameters.

Each data set is fitted with the same model. Any parameter can be one value
shared by all the data sets, or a value of its own in each. Sharing what the
data sets have in common (a top, a slope) makes the fit of what differs (an
EC50) more exact - and whether a parameter may be shared is itself tested:
the shared fit against the fit with nothing shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import nonlinear_regression as nlr


@dataclass
class GlobalFitOptions:
    model: str = "sigmoid_4pl"
    separate: list[str] = field(default_factory=list)   # parameters each data set has for itself; the rest are shared
    relative_weighting: bool = False
    test_each: bool = True          # for each separate parameter: does it really differ between the data sets?
    range_mode_start: str = "auto"
    range_start_x: float = 0.0
    range_mode_end: str = "auto"
    range_end_x: float = 0.0
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"
    data_sets: list[str] = field(default_factory=list)


def aicc(rss: float, n: int, parameters: int) -> float:
    """Akaike's criterion for a least-squares fit, corrected for small
    samples. One is added to the parameters for the scatter, which is also
    estimated (Burnham and Anderson 2002)."""
    k = parameters + 1
    if rss <= 0 or n - k - 1 <= 0:
        return float("nan")
    return float(n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def extra_ss_f(rss_simple: float, df_simple: int, rss_full: float, df_full: int) -> tuple[float, float]:
    """The extra sum-of-squares F test: is the fuller model's better fit more
    than its extra parameters would buy by chance? Returns (F, P)."""
    if df_simple <= df_full or df_full < 1 or rss_full <= 0:
        return float("nan"), float("nan")
    f = ((rss_simple - rss_full) / (df_simple - df_full)) / (rss_full / df_full)
    f = max(float(f), 0.0)
    return f, float(stats.f.sf(f, df_simple - df_full, df_full))


def fit_shared(model, sets, separate_index: list[int], start: np.ndarray, weighted: bool) -> dict:
    """Least squares over all the data sets together. ``start`` is one row of
    parameters per data set (their separate fits). Returns each data set's
    parameters and standard errors, the weighted residual sum of squares and
    its degrees of freedom."""
    m, k = len(sets), len(model.param_names)
    shared_index = [i for i in range(k) if i not in separate_index]
    n_shared = len(shared_index)
    size = n_shared + m * len(separate_index)

    def unpack(theta):
        params = np.empty((m, k))
        params[:, shared_index] = theta[:n_shared]
        params[:, separate_index] = theta[n_shared:].reshape(m, len(separate_index)) if separate_index else np.empty((m, 0))
        return params

    weights = [1.0 / np.where(y != 0, np.abs(y), 1.0) if weighted else np.ones(y.size) for _l, _x, y in sets]

    def residuals(theta):
        params = unpack(theta)
        out = []
        for (_label, x, y), p, w in zip(sets, params, weights):
            with np.errstate(all="ignore"):
                r = (model.fn(x, *p) - y) * w
            out.append(np.where(np.isfinite(r), r, 1e30))
        return np.concatenate(out)

    theta0 = np.concatenate([np.median(start[:, shared_index], axis=0), start[:, separate_index].ravel()])
    lower = np.full(size, -np.inf)
    if model.lower is not None:
        low = np.asarray(model.lower, dtype=float)
        lower = np.concatenate([low[shared_index], np.tile(low[separate_index], m)])
        theta0 = np.maximum(theta0, lower + 1e-9)
    scale = np.maximum(np.abs(theta0), 1e-8)
    sol = optimize.least_squares(residuals, theta0, bounds=(lower, np.full(size, np.inf)), x_scale=scale, max_nfev=600 * size)
    n = sum(y.size for _l, _x, y in sets)
    df = n - size
    rss = float((sol.fun**2).sum())
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * rss / max(df, 1)
    except np.linalg.LinAlgError:
        cov = np.full((size, size), np.nan)
    errors_theta = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"params": unpack(sol.x), "errors": unpack(errors_theta), "rss": rss, "df": df, "n": n, "size": size, "ok": bool(sol.success),
            "shared_index": shared_index}


def weighted_rss(model, x, y, params, weighted: bool) -> float:
    w = 1.0 / np.where(y != 0, np.abs(y), 1.0) if weighted else 1.0
    return float((((model.fn(x, *params) - y) * w) ** 2).sum())


@register("global_fit")
def run(data: SheetData, options: GlobalFitOptions | None = None) -> AnalysisResult:
    opt = options or GlobalFitOptions()
    model = nlr.MODELS.get(opt.model)
    if model is None:
        raise AnalysisError(f"There is no model called '{opt.model}'. Choose one from the list.")
    sets = [(label, x, y) for label, x, y in nlr._data_sets(data, opt) if x.size]
    if len(sets) < 2:
        raise AnalysisError("Fitting data sets together needs two or more of them, each with numbers in the X column and its own Y values.")
    names = list(model.param_names)
    k, m = len(names), len(sets)
    separate_index = [i for i, name in enumerate(names) if name in set(opt.separate)]
    sig = opt.significant_digits
    pct = f"{round(opt.confidence_level * 100)}%"

    def num(value) -> str:
        return "" if not np.isfinite(value) else f"{value:.{sig}g}"

    def p_text(p) -> str:
        return "" if not np.isfinite(p) else format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    own = [nlr._fit_one(model, label, x, y, opt) for label, x, y in sets]         # each data set by itself: the start, and the yardstick
    start = np.array([f.params for f in own])
    rss_own = sum(weighted_rss(model, x, y, f.params, opt.relative_weighting) for (_l, x, y), f in zip(sets, own))
    n = sum(y.size for _l, _x, y in sets)
    df_own = n - m * k
    if df_own < 1:
        raise AnalysisError(f"Too few points: {n} for {m * k} parameters when every data set is fitted by itself.")

    fit = fit_shared(model, sets, separate_index, start, opt.relative_weighting)
    t_crit = stats.t.ppf(0.5 + opt.confidence_level / 2, max(fit["df"], 1))
    shared_index = fit["shared_index"]
    tables: list[ResultTable] = []
    notes: list[str] = []

    if shared_index:
        tables.append(ResultTable("One value for all the data sets", ["Parameter", "Best fit", "Std. error", f"{pct} CI"],
                                  [[names[i], num(fit["params"][0, i]), num(fit["errors"][0, i]),
                                    f"{num(fit['params'][0, i] - t_crit * fit['errors'][0, i])} to {num(fit['params'][0, i] + t_crit * fit['errors'][0, i])}"]
                                   for i in shared_index]))
    if separate_index:
        tables.append(ResultTable("A value of its own in each data set", ["Parameter"] + [label for label, _x, _y in sets],
                                  [[names[i]] + [f"{num(fit['params'][s, i])} ± {num(fit['errors'][s, i])}" for s in range(m)] for i in separate_index]))
        tables.append(ResultTable(f"{pct} confidence intervals of those", ["Parameter"] + [label for label, _x, _y in sets],
                                  [[names[i]] + [f"{num(fit['params'][s, i] - t_crit * fit['errors'][s, i])} to "
                                                 f"{num(fit['params'][s, i] + t_crit * fit['errors'][s, i])}" for s in range(m)] for i in separate_index]))

    # ── may these parameters be shared? the shared fit against nothing shared ──
    f_share, p_share = extra_ss_f(fit["rss"], fit["df"], rss_own, df_own)
    a_shared, a_own = aicc(fit["rss"], n, fit["size"]), aicc(rss_own, n, m * k)
    fair = not (np.isfinite(p_share) and p_share < opt.alpha)
    if shared_index:
        delta = a_own - a_shared                       # above 0: the shared fit is the better by AICc
        prob_shared = 1.0 / (1.0 + np.exp(-0.5 * delta)) if np.isfinite(delta) else float("nan")
        tables.append(ResultTable("Is sharing fair to the data?", ["Fit", "Parameters", "Sum of squares", "df", "AICc", "Chance it is the better"],
                                  [["Shared, as set up", str(fit["size"]), num(fit["rss"]), str(fit["df"]), num(a_shared),
                                    "" if not np.isfinite(prob_shared) else f"{prob_shared:.1%}"],
                                   ["Nothing shared", str(m * k), num(rss_own), str(df_own), num(a_own),
                                    "" if not np.isfinite(prob_shared) else f"{1 - prob_shared:.1%}"]]))
        tables.append(ResultTable("The same question as a test", ["Test", "F", "df", "P value", "Reading"],
                                  [["Extra sum of squares", "" if not np.isfinite(f_share) else f"{f_share:.4f}", f"{fit['df'] - df_own}, {df_own}",
                                    p_text(p_share), "sharing is fair: the data sets do not need values of their own" if fair else
                                    "the shared fit is clearly worse: at least one shared parameter differs between the data sets"]]))

    # ── does each separate parameter really differ? fit again with it shared too ──
    differ = []
    if opt.test_each and separate_index:
        rows = []
        for i in separate_index:
            fewer = [j for j in separate_index if j != i]
            tied = fit_shared(model, sets, fewer, fit["params"], opt.relative_weighting)
            f_i, p_i = extra_ss_f(tied["rss"], tied["df"], fit["rss"], fit["df"])
            values = fit["params"][:, i]
            verdict = np.isfinite(p_i) and p_i < opt.alpha
            if verdict:
                differ.append(names[i])
            rows.append([names[i], f"{num(values.min())} to {num(values.max())}", "" if not np.isfinite(f_i) else f"{f_i:.4f}",
                         f"{tied['df'] - fit['df']}, {fit['df']}", p_text(p_i), "differs between the data sets" if verdict else "no clear difference"])
        tables.insert(2 if shared_index else 1, ResultTable("Does it differ between the data sets?",
                                                            ["Parameter", "Runs from", "F", "df", "P value", "Reading"], rows))

    fits = []
    rows = []
    for s, (label, x, y) in enumerate(sets):
        p = fit["params"][s]
        resid = y - model.fn(x, *p)
        ss_res, ss_tot = float((resid**2).sum()), float(((y - y.mean()) ** 2).sum())
        df_set = max(1, x.size - k)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fits.append(nlr._Fit(label, x, y, p, fit["errors"][s], r2, float(np.sqrt(ss_res / df_set)), df_set))
        rows.append([label, f"{r2:.4f}", f"{own[s].r_squared:.4f}", num(float(np.sqrt(ss_res / x.size))), str(x.size)])
    tables.append(ResultTable("How well each data set is fitted", ["Data set", "R² in the shared fit", "R² fitted by itself", "Typical residual", "Points"], rows))

    shared_names = [names[i] for i in shared_index]
    own_names = [names[i] for i in separate_index]
    summary = f"{model.label}, {m} data sets fitted together. "
    if shared_names and own_names:
        summary += f"Shared: {', '.join(shared_names)}. Own in each: {', '.join(own_names)}"
        first = separate_index[0]
        summary += " (" + names[first] + " = " + ", ".join(f"{num(fit['params'][s, first])} in {sets[s][0]}" for s in range(min(m, 4))) + ("…" if m > 4 else "") + "). "
    elif shared_names:
        summary += "Every parameter shared: one curve for all (" + ", ".join(f"{names[i]} = {num(fit['params'][0, i])}" for i in range(k)) + "). "
    else:
        summary += "Nothing shared: each data set fitted by itself. "
    if shared_names:
        summary += (f"Sharing is fair (P = {p_share:.4g})." if fair else f"Sharing makes the fit clearly worse (P = {p_share:.4g}).") if np.isfinite(p_share) else ""
    if differ:
        summary += f" Differs between the data sets: {', '.join(differ)}."

    if not fit["ok"]:
        notes.append("The fit stopped before it settled; treat its figures as rough.")
    notes += ["A shared parameter is estimated from all the data sets together, so it is known more exactly than from any one of "
              "them - and the parameters left separate gain from that. Share a parameter when the experiment says it should be the "
              "same (the same enzyme, the same maximal response), not because the test allows it.",
              "'Is sharing fair?' compares the fit as set up with one where nothing is shared. A small P value means some shared "
              "parameter truly differs between the data sets; the AICc comparison answers the same question without a threshold.",
              "'Does it differ?' fits once more with that one parameter shared as well: a small P value means the data sets need "
              "their own value of it. This is the usual way to ask whether two curves have the same EC50, Km or rate.",
              "The tests assume the scatter about the curve is normal and of the same size in every data set"
              + (" in proportion to Y." if opt.relative_weighting else "."),
              "Method: nonlinear least squares over all data sets at once; extra sum-of-squares F test after Bates and Watts "
              "(1988); AICc after Hurvich and Tsai (1989) and Burnham and Anderson (2002)."]
    if model.lower is not None:
        notes.append("Parameters were kept positive while fitting; a value sitting at 0 means the data could not pin it down.")
    charts = [nlr._curve_chart(model, fits), nlr._residual_chart(model, fits)]
    charts[0].title = "All data sets fitted together"
    return AnalysisResult(title="Fit data sets together", summary=summary.strip(), tables=tables, notes=notes, charts=charts)
