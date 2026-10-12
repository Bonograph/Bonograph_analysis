"""A yes/no outcome against one predictor (binary logistic regression).

The report is built from questions rather than from a list of statistics:

* how much the chance changes with X - the odds ratio for a step in X the
  user chooses, with a profile-likelihood interval (Wald on request);
* the X at which the chance reaches a level (50% by default), with Fieller's
  limits, which stay honest when the slope is poorly known;
* how well X tells yes from no - the area under the ROC curve with DeLong's
  interval, the cutoff that separates best, and a table at any cutoff;
* whether the predicted chances can be trusted - Brier score, the
  Hosmer-Lemeshow test and a calibration plot;
* whether X earns its place - likelihood-ratio test, information criteria
  and four R-squared-like measures;
* which rows pull the fit - leverage, residuals and Cook's distance;
* predictions for each row and for X values typed in.

When X splits the two outcomes completely the ordinary fit does not exist
(the slope runs off to infinity). Firth's penalised likelihood (Firth 1993;
Heinze & Schemper 2002) always gives a finite answer and is less biased in
small samples; it is used automatically in that case, or always on request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..registry import register

LEAST_ROWS = 8
MOST_LISTED = 500
BLUE, RED, GREY = "#4a90d9", "#d62728", "#8a8a96"


@dataclass
class LogisticRegressionOptions:
    # how the curve is fitted
    fit_method: str = "auto"            # "auto" (Firth's fit only when X splits the outcomes) | "ml" | "firth"
    interval_method: str = "profile"    # "profile" | "wald"
    confidence_level: float = 0.95
    significant_digits: int = 4
    # how much the chance changes
    odds_step: float = 1.0              # the odds ratio is given for a rise of this much in X
    # the X at which the chance reaches a level
    x_at_levels: bool = True
    levels: str = "50"                  # percentages, separated by commas
    # telling yes from no
    roc_area: bool = True
    roc_curve: bool = True
    best_cutoff: bool = False
    classification: bool = False
    cutoff: float = 0.5                 # call it "yes" at or above this predicted chance
    # can the predicted chances be trusted
    brier: bool = False
    calibration_test: bool = False
    calibration_plot: bool = False
    # does X earn its place
    likelihood_test: bool = True
    information: bool = False
    r2_tjur: bool = False
    r2_mcfadden: bool = False
    r2_nagelkerke: bool = False
    r2_cox_snell: bool = False
    # which rows pull the fit
    influence: bool = False
    influence_plot: bool = False
    # predictions
    predicted_probabilities: bool = False
    predict_at: str = ""                # X values typed in, separated by commas
    # the graph
    graph_curve: bool = True
    plot_confidence_bands: bool = False
    mark_levels: bool = False           # draw the X of each level on the graph
    min_x_mode: str = "auto"            # "auto" | "manual"
    min_x_value: float = 0.0
    max_x_mode: str = "auto"
    max_x_value: float = 0.0


# ── fitting ──────────────────────────────────────────────────────────────────

def _chance(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))


def _loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray, penalised: bool) -> float:
    eta = X @ beta
    value = float(np.sum(y * eta - np.logaddexp(0, eta)))
    if penalised:
        mu = _chance(eta)
        sign, logdet = np.linalg.slogdet(X.T @ (X * (mu * (1 - mu))[:, None]))
        value += 0.5 * logdet if sign > 0 else -1e12
    return value


def fit(X: np.ndarray, y: np.ndarray, penalised: bool = False, fixed: dict | None = None,
        start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Newton fit of the (optionally Firth-penalised) binomial likelihood.

    ``fixed`` holds coefficients at given values ({index: value}) for a
    profile likelihood. Returns (coefficients, their covariance, the
    log-likelihood reached - penalised if asked - and whether it converged).
    """
    fixed = fixed or {}
    free = [j for j in range(X.shape[1]) if j not in fixed]
    beta = np.zeros(X.shape[1]) if start is None else np.array(start, dtype=float)
    for j, value in fixed.items():
        beta[j] = value
    current = _loglik(beta, X, y, penalised)
    converged = not free
    for _ in range(200):
        if not free:
            break
        mu = _chance(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        info = X.T @ (X * w[:, None])
        try:
            inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(info)
        residual = y - mu
        if penalised:   # Firth's modified score: each row leans towards 1/2 by its leverage
            hat = w * np.einsum("ij,jk,ik->i", X, inv, X)
            residual = residual + hat * (0.5 - mu)
        score = X.T @ residual
        sub = np.ix_(free, free)
        try:
            step = np.linalg.solve(info[sub], score[free])
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info[sub]) @ score[free]
        size = float(np.max(np.abs(step)))
        if size > 5.0:               # a runaway step (separation under plain likelihood) is held back
            step *= 5.0 / size
        trial = beta.copy()
        for _half in range(30):
            trial[free] = beta[free] + step
            value = _loglik(trial, X, y, penalised)
            if value >= current - 1e-10:
                break
            step = step / 2
        moved = float(np.max(np.abs(trial - beta)))
        beta, current = trial, value
        if moved < 1e-9:
            converged = True
            break
    mu = _chance(X @ beta)
    info = X.T @ (X * np.clip(mu * (1 - mu), 1e-12, None)[:, None])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    return beta, cov, current, converged


def split_completely(x: np.ndarray, y: np.ndarray) -> bool:
    """True when a single cut of X puts every 0 on one side and every 1 on the
    other (touching at the cut counts): the ordinary fit then has no finite slope."""
    zeros, ones = x[y == 0], x[y == 1]
    return bool(zeros.max() <= ones.min() or ones.max() <= zeros.min())


def profile_limits(index: int, beta: np.ndarray, se: np.ndarray, X: np.ndarray, y: np.ndarray, top: float,
                   level: float, penalised: bool) -> tuple[float | None, float | None]:
    """The values of one coefficient at which the likelihood, with the other
    refitted, has fallen by half the chi-square critical value. None on a
    side where it never falls that far (the limit is then infinite)."""
    drop = stats.chi2.ppf(level, 1) / 2

    def gap(value: float) -> float:
        return fit(X, y, penalised, {index: value}, start=beta)[2] - (top - drop)

    step = max(float(se[index]), 1e-3)
    limits = []
    for direction in (-1.0, 1.0):
        inner, outer, found = 0.0, step, None
        for _ in range(40):
            if gap(beta[index] + direction * outer) < 0:
                found = optimize.brentq(gap, beta[index] + direction * inner, beta[index] + direction * outer, xtol=1e-9)
                break
            inner, outer = outer, outer * 2
            if outer > 1e6 * step:
                break
        limits.append(found)
    return limits[0], limits[1]


# ── pieces of the report ─────────────────────────────────────────────────────

def fieller(target: float, beta: np.ndarray, cov: np.ndarray, level: float) -> tuple[float, float, float, str]:
    """The X at which the linear predictor equals ``target``, with Fieller's
    limits: (estimate, lower, upper, remark). The limits are infinite when
    the slope is not clearly different from zero; the remark then says so."""
    a, b = float(beta[0]), float(beta[1])
    z2 = float(stats.norm.ppf(0.5 + level / 2)) ** 2
    estimate = (target - a) / b
    # (a + b x - target)^2 <= z^2 (Vaa + 2 x Vab + x^2 Vbb), as a quadratic in x
    qa = b * b - z2 * cov[1, 1]
    qb = 2 * (b * (a - target) - z2 * cov[0, 1])
    qc = (a - target) ** 2 - z2 * cov[0, 0]
    disc = qb * qb - 4 * qa * qc
    if qa <= 0 or disc < 0:
        return estimate, float("nan"), float("nan"), "no limits: the slope is not clearly different from zero"
    root = np.sqrt(disc)
    return estimate, float((-qb - root) / (2 * qa)), float((-qb + root) / (2 * qa)), ""


def delong(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Area under the ROC curve and DeLong's standard error (DeLong, DeLong &
    Clarke-Pearson 1988), by mid-ranks."""
    yes, no = scores[y == 1], scores[y == 0]
    m, n = yes.size, no.size
    rank_all = stats.rankdata(np.concatenate([yes, no]))
    v_yes = (rank_all[:m] - stats.rankdata(yes)) / n
    v_no = 1.0 - (rank_all[m:] - stats.rankdata(no)) / m
    area = float(np.mean(v_yes))
    variance = (np.var(v_yes, ddof=1) / m if m > 1 else 0.0) + (np.var(v_no, ddof=1) / n if n > 1 else 0.0)
    return area, float(np.sqrt(max(variance, 0.0)))


def area_limits(area: float, se: float, level: float) -> tuple[float, float]:
    """An interval for an area that cannot leave 0 to 1: Wald on the logit scale."""
    if not 0 < area < 1 or se <= 0:
        return area, area
    z = float(stats.norm.ppf(0.5 + level / 2))
    centre = np.log(area / (1 - area))
    half = z * se / (area * (1 - area))
    return float(_chance(centre - half)), float(_chance(centre + half))


def wilson(successes: float, total: float, level: float) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10):
    """Subjects sorted by predicted chance and cut into groups of equal size:
    (chi-square, degrees of freedom, P, [(mean predicted, share observed, n)])."""
    order = np.argsort(p, kind="stable")
    parts = [part for part in np.array_split(order, min(groups, max(3, y.size // 5))) if part.size]
    chi, table = 0.0, []
    for part in parts:
        seen, expected, size = float(np.sum(y[part])), float(np.sum(p[part])), part.size
        spread = expected * (1 - expected / size)
        if spread > 1e-12:
            chi += (seen - expected) ** 2 / spread
        table.append((expected / size, seen / size, size))
    df = max(len(parts) - 2, 1)
    return chi, df, float(stats.chi2.sf(chi, df)), table


def influence_measures(X: np.ndarray, y: np.ndarray, p: np.ndarray, cov: np.ndarray):
    """Leverage, standardised Pearson residual, deviance residual and Cook's distance of each row (Pregibon 1981)."""
    w = np.clip(p * (1 - p), 1e-12, None)
    leverage = np.clip(w * np.einsum("ij,jk,ik->i", X, cov, X), 0, 1 - 1e-9)
    pearson = (y - p) / np.sqrt(w)
    standard = pearson / np.sqrt(1 - leverage)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = np.sign(y - p) * np.sqrt(-2 * np.where(y == 1, np.log(np.clip(p, 1e-300, None)),
                                                          np.log(np.clip(1 - p, 1e-300, None))))
    cook = standard ** 2 * leverage / (X.shape[1] * (1 - leverage))
    return leverage, standard, deviance, cook


def _numbers(text: str, what: str, lo: float = -np.inf, hi: float = np.inf) -> list[float]:
    values = []
    for piece in str(text or "").replace(";", ",").split(","):
        piece = piece.strip().rstrip("%").strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            raise AnalysisError(f"'{piece}' is not a number. Write {what} as numbers separated by commas.") from None
        if not lo < value < hi:
            raise AnalysisError(f"{value:g} is outside the range allowed for {what}.")
        values.append(value)
    return values


def _span(lo, hi, sig: int, transform=None) -> str:
    def one(v, word):
        if v is None or not np.isfinite(v):
            return word
        return f"{(transform(v) if transform else v):.{sig}g}"

    return f"{one(lo, 'no lower limit')} to {one(hi, 'no upper limit')}"


# ── the analysis ─────────────────────────────────────────────────────────────

@register("logistic_regression")
def run(data: SheetData, options: LogisticRegressionOptions | None = None) -> AnalysisResult:
    from ..chart_helpers import roc_chart
    from ..ml_utils import roc_curve

    opt = options or LogisticRegressionOptions()
    labels = data.group_labels()
    if not labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    name = labels[0]
    x, y = data.xy_points(name)   # each outcome with the X of its own row
    if x.size < LEAST_ROWS:
        raise AnalysisError(f"This needs at least {LEAST_ROWS} rows with both an X and an outcome; found {x.size}.")
    codes = sorted(set(np.unique(y).tolist()))
    if not set(codes) <= {0.0, 1.0}:
        raise AnalysisError(f"The outcome must be coded 0 and 1; '{name}' holds {', '.join(f'{c:g}' for c in codes[:5])}.")
    if len(codes) < 2:
        raise AnalysisError(f"'{name}' holds only {codes[0]:g}. Both outcomes are needed to fit anything.")
    if np.ptp(x) == 0:
        raise AnalysisError("X is the same on every row, so it cannot explain the outcome.")
    if opt.odds_step <= 0:
        raise AnalysisError("The step in X for the odds ratio must be above zero.")
    level, sig = opt.confidence_level, opt.significant_digits
    pct = f"{level * 100:g}%"
    z = float(stats.norm.ppf(0.5 + level / 2))
    n = x.size
    events = int(np.sum(y))
    X = np.column_stack([np.ones(n), x])
    notes: list[str] = []

    split = split_completely(x, y)
    penalised = opt.fit_method == "firth" or (opt.fit_method == "auto" and split)
    beta, cov, top, converged = fit(X, y, penalised)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    intercept, slope = float(beta[0]), float(beta[1])
    null_beta, _null_cov, null_top, _ok = fit(X[:, :1], y, penalised)
    p = _chance(X @ beta)
    # plain log-likelihoods, for the measures that are defined on them
    ll = _loglik(beta, X, y, False)
    ll_null = _loglik(np.array([np.log(events / (n - events))]), X[:, :1], y, False)
    runaway = split and not penalised

    method = "Firth's penalised likelihood" if penalised else "maximum likelihood"
    if penalised and opt.fit_method == "auto":
        notes.append("X splits the two outcomes completely, so the ordinary fit has no finite slope. The fit shown is "
                     "by Firth's penalised likelihood, which always gives one. More data near the switch-over would "
                     "narrow the intervals.")
    if runaway:
        notes.append("X splits the two outcomes completely: under plain maximum likelihood the slope runs off to "
                     "infinity, so the estimate, its standard error and the odds ratio below mean nothing. Choose "
                     "Firth's fit, or collect data near the switch-over.")
    elif not converged:
        notes.append("The fit did not settle. Treat every number here with caution.")
    rarer = min(events, n - events)
    if rarer < 10 and not penalised:
        notes.append(f"Only {rarer} rows have the rarer outcome. Estimates from so few are unstable and lean away "
                     "from zero; Firth's fit is less biased.")

    # intervals for the two coefficients
    wald = [(beta[j] - z * se[j], beta[j] + z * se[j]) for j in range(2)]
    if opt.interval_method == "profile" and not runaway:
        limits = [profile_limits(j, beta, se, X, y, top, level, penalised) for j in range(2)]
        interval_name = "profile likelihood"
    else:
        limits, interval_name = wald, "Wald"
    lr = max(0.0, 2 * (top - null_top))
    lr_p = float(stats.chi2.sf(lr, 1))
    wald_p = 2 * stats.norm.sf(np.abs(beta / np.where(se > 0, se, np.nan)))

    tables = [ResultTable(
        title=f"logit(chance that {name} = 1) = intercept + slope · X",
        headers=["Parameter", "Estimate", "Std. error", f"{pct} CI", "P value"],
        rows=[["Intercept", f"{intercept:.{sig}g}", f"{se[0]:.{sig}g}", _span(*limits[0], sig), f"{wald_p[0]:.4g}"],
              ["Slope", f"{slope:.{sig}g}", f"{se[1]:.{sig}g}", _span(*limits[1], sig), f"{wald_p[1]:.4g}"]])]

    step = opt.odds_step
    ratio = float(np.exp(np.clip(step * slope, -700, 700)))
    ratio_span = _span(limits[1][0], limits[1][1], sig, lambda v: float(np.exp(np.clip(step * v, -700, 700))))
    # the same effect as a change in chance, at the middle of the data
    middle = float(np.median(x))
    chance_low, chance_high = float(_chance(intercept + slope * middle)), float(_chance(intercept + slope * (middle + step)))
    tables.append(ResultTable(
        title="How the chance changes with X",
        headers=["For a rise in X of", "Odds multiply by", f"{pct} CI", "For example, the chance goes"],
        rows=[[f"{step:g}", f"{ratio:.{sig}g}", ratio_span,
               f"from {chance_low * 100:.3g}% at X = {middle:.{sig}g} to {chance_high * 100:.3g}% at X = {middle + step:.{sig}g}"]]))

    marks = []
    if opt.x_at_levels:
        rows = []
        for percent in _numbers(opt.levels, "the chance levels, in percent", 0.0, 100.0) or [50.0]:
            share = percent / 100
            at, lo, hi, remark = fieller(float(np.log(share / (1 - share))), beta, cov, level)
            if slope < 0 and not remark:
                lo, hi = min(lo, hi), max(lo, hi)
            rows.append([f"{percent:g}%", f"{at:.{sig}g}", remark or f"{lo:.{sig}g} to {hi:.{sig}g}"])
            marks.append((percent, at))
        tables.append(ResultTable(title="The X at which the chance reaches a level",
                                  headers=["Chance", "X", f"{pct} CI (Fieller)"], rows=rows))
        if not float(x.min()) <= marks[0][1] <= float(x.max()):
            notes.append(f"The X for a chance of {marks[0][0]:g}% lies outside the X values on the sheet: it is an "
                         "extrapolation.")

    charts = []
    area = None
    if opt.roc_area or opt.roc_curve or opt.best_cutoff:
        direction = 1.0 if slope >= 0 else -1.0
        area, area_se = delong(direction * x, y)
        fpr, tpr, cut = roc_curve(y, p)
        if opt.roc_area:
            lo, hi = area_limits(area, area_se, level)
            against_chance = 2 * stats.norm.sf(abs(area - 0.5) / area_se) if area_se > 0 else float("nan")
            tables.append(ResultTable(
                title="How well X tells yes from no",
                headers=["Area under the ROC curve", "Std. error (DeLong)", f"{pct} CI", "P against 0.5 (no better than chance)"],
                rows=[[f"{area:.{sig}g}", f"{area_se:.{sig}g}", f"{lo:.{sig}g} to {hi:.{sig}g}", f"{against_chance:.4g}"]]))
        if opt.best_cutoff:
            best = int(np.argmax(tpr - fpr))
            chance_cut = float(np.clip(cut[best], 1e-12, 1 - 1e-12)) if np.isfinite(cut[best]) else 0.5
            x_cut = (np.log(chance_cut / (1 - chance_cut)) - intercept) / slope if slope != 0 else float("nan")
            sens, spec = float(tpr[best]), float(1 - fpr[best])
            s_lo, s_hi = wilson(sens * events, events, level)
            c_lo, c_hi = wilson(spec * (n - events), n - events, level)
            tables.append(ResultTable(
                title="The cutoff that separates best (largest sensitivity + specificity)",
                headers=["Call it yes when X is", "Predicted chance there", "Sensitivity", f"{pct} CI", "Specificity", f"{pct} CI"],
                rows=[[f"{'at least' if slope >= 0 else 'at most'} {x_cut:.{sig}g}", f"{chance_cut:.{sig}g}", f"{sens:.{sig}g}",
                       f"{s_lo:.{sig}g} to {s_hi:.{sig}g}", f"{spec:.{sig}g}", f"{c_lo:.{sig}g} to {c_hi:.{sig}g}"]]))
            notes.append("The best cutoff was chosen on the same rows it is judged on, so its sensitivity and "
                         "specificity flatter it. Check it on new data before relying on it.")
        if opt.roc_curve:
            listed = [[f"{t:.4g}" if np.isfinite(t) else "-", f"{f:.4g}", f"{tp:.4g}"] for t, f, tp in zip(cut, fpr, tpr)]
            tables.append(ResultTable(title=f"ROC curve (AUC = {area:.4f})",
                                      headers=["Predicted chance at or above", "False positive rate", "True positive rate"],
                                      rows=listed[:60]))
            charts.append(roc_chart("ROC curve", fpr, tpr, area))

    if opt.classification:
        if not 0 < opt.cutoff < 1:
            raise AnalysisError("The cutoff for calling a row 'yes' must lie between 0 and 1.")
        called = p >= opt.cutoff
        tp, fp = int(np.sum(called & (y == 1))), int(np.sum(called & (y == 0)))
        fn, tn = int(np.sum(~called & (y == 1))), int(np.sum(~called & (y == 0)))

        def share(top_, bottom):
            return f"{top_ / bottom:.{sig}g}" if bottom else "-"

        tables.append(ResultTable(
            title=f"Rows called yes at a predicted chance of {opt.cutoff:g} or more",
            headers=["", "Was yes", "Was no", "Share called right"],
            rows=[["Called yes", str(tp), str(fp), share(tp, tp + fp) + "  (positive predictive value)"],
                  ["Called no", str(fn), str(tn), share(tn, fn + tn) + "  (negative predictive value)"],
                  ["Share found", share(tp, tp + fn) + "  (sensitivity)", share(tn, fp + tn) + "  (specificity)",
                   share(tp + tn, n) + "  (overall)"]]))

    if opt.brier or opt.calibration_test or opt.calibration_plot:
        chi, df, hl_p, grouped = hosmer_lemeshow(y, p)
        rows = []
        if opt.brier:
            score = float(np.mean((p - y) ** 2))
            base = (events / n) * (1 - events / n)
            rows += [["Brier score (0 is perfect)", f"{score:.{sig}g}"],
                     ["Brier score with no predictor", f"{base:.{sig}g}"],
                     ["Share of that error removed by X", f"{(1 - score / base) * 100:.3g}%"]]
        if opt.calibration_test:
            rows.append([f"Hosmer-Lemeshow test, {len(grouped)} groups", f"chi-square = {chi:.4g}, df = {df}, P = {hl_p:.4g}"])
            if n < 50:
                notes.append("The Hosmer-Lemeshow test has little power below about 50 rows: a large P is not proof "
                             "that the predicted chances are right.")
        if rows:
            tables.append(ResultTable(title="Can the predicted chances be trusted?", headers=["Measure", "Value"], rows=rows))
        if opt.calibration_plot:
            predicted = [g[0] for g in grouped]
            charts.append(ChartSpec(
                title="Predicted chance against what happened",
                series=[ChartSeries("Groups of rows", predicted, [g[1] for g in grouped], kind="scatter", color=BLUE)],
                x_label="Mean predicted chance in the group", y_label="Share that were yes",
                x_limits=(-0.03, 1.03), y_limits=(-0.03, 1.03),
                reference_lines=[{"kind": "line", "points": [[0, 0], [1, 1]], "style": ":"}]))

    worth = []
    if opt.likelihood_test:
        worth.append([("Penalised l" if penalised else "L") + "ikelihood-ratio test against no predictor",
                      f"chi-square = {lr:.4g}, df = 1, P = {lr_p:.4g}"])
    if opt.information:
        for label_, k_, value in (("this model", 2, ll), ("no predictor", 1, ll_null)):
            aic = -2 * value + 2 * k_
            aicc = aic + 2 * k_ * (k_ + 1) / (n - k_ - 1)
            worth.append([f"AIC, AICc, BIC - {label_}", f"{aic:.{sig + 1}g}, {aicc:.{sig + 1}g}, {-2 * value + k_ * np.log(n):.{sig + 1}g}"])
        worth.append(["Deviance (-2 log-likelihood) - this model, no predictor", f"{-2 * ll:.{sig + 1}g}, {-2 * ll_null:.{sig + 1}g}"])
    cox_snell = 1 - np.exp(2 * (ll_null - ll) / n)
    ceiling = 1 - np.exp(2 * ll_null / n)
    for wanted, label_, value in (
            (opt.r2_tjur, "Tjur's R² (mean chance given to the yes rows, less that given to the no rows)",
             float(np.mean(p[y == 1]) - np.mean(p[y == 0]))),
            (opt.r2_mcfadden, "McFadden's R²", 1 - ll / ll_null),
            (opt.r2_nagelkerke, "Nagelkerke's R²", cox_snell / ceiling if ceiling > 0 else float("nan")),
            (opt.r2_cox_snell, "Cox and Snell's R²", cox_snell)):
        if wanted:
            worth.append([label_, f"{value:.{sig}g}"])
    if worth:
        tables.append(ResultTable(title="Does X earn its place?", headers=["Measure", "Value"], rows=worth))
        if any((opt.r2_tjur, opt.r2_mcfadden, opt.r2_nagelkerke, opt.r2_cox_snell)):
            notes.append("These R² values are not the share of variance explained that R² is for a straight "
                         "line, and they disagree with one another. Use one to compare models on the same data.")

    if opt.influence or opt.influence_plot:
        leverage, standard, deviance, cook = influence_measures(X, y, p, cov)
        if opt.influence:
            order = np.argsort(-cook)[:10]
            flagged = int(np.sum(cook > 4 / n))
            tables.append(ResultTable(
                title=f"The rows that pull the fit hardest ({flagged} with Cook's distance above 4/n = {4 / n:.3g})",
                headers=["X", "Outcome", "Predicted chance", "Standardised residual", "Deviance residual", "Leverage", "Cook's distance"],
                rows=[[f"{x[i]:.{sig}g}", f"{y[i]:g}", f"{p[i]:.{sig}g}", f"{standard[i]:.{sig}g}", f"{deviance[i]:.{sig}g}",
                       f"{leverage[i]:.{sig}g}", f"{cook[i]:.{sig}g}"] for i in order]))
        if opt.influence_plot:
            charts.append(ChartSpec(
                title="How hard each row pulls the fit",
                series=[ChartSeries("Rows", x.tolist(), cook.tolist(), kind="scatter", color=BLUE)],
                x_label="X", y_label="Cook's distance", reference_lines=[{"kind": "h", "value": 4 / n, "style": ":"}]))

    def chance_with_limits(at: np.ndarray):
        eta = intercept + slope * at
        half = z * np.sqrt(np.clip(cov[0, 0] + 2 * at * cov[0, 1] + at ** 2 * cov[1, 1], 0, None))
        return _chance(eta), _chance(eta - half), _chance(eta + half)

    typed = _numbers(opt.predict_at, "the X values to predict at")
    if typed:
        at = np.array(typed)
        mid, lo, hi = chance_with_limits(at)
        outside = [f"{v:g}" for v in typed if not float(x.min()) <= v <= float(x.max())]
        tables.append(ResultTable(title="Predicted chance at the X values asked for",
                                  headers=["X", "Predicted chance", f"{pct} CI"],
                                  rows=[[f"{a:g}", f"{m:.{sig}g}", f"{l:.{sig}g} to {h:.{sig}g}"] for a, m, l, h in zip(at, mid, lo, hi)]))
        if outside:
            notes.append(f"X = {', '.join(outside)} lies outside the X values on the sheet: those predictions are extrapolations.")
    if opt.predicted_probabilities:
        mid, lo, hi = chance_with_limits(x)
        tables.append(ResultTable(
            title="Predicted chance for each row",
            headers=["X", "Outcome", "Predicted chance", f"{pct} CI"],
            rows=[[f"{a:.4g}", f"{b:g}", f"{m:.{sig}g}", f"{l:.{sig}g} to {h:.{sig}g}"]
                  for a, b, m, l, h in zip(x, y, mid, lo, hi)][:MOST_LISTED]))

    if opt.graph_curve or opt.plot_confidence_bands:
        lo = opt.min_x_value if opt.min_x_mode == "manual" else float(np.min(x))
        hi = opt.max_x_value if opt.max_x_mode == "manual" else float(np.max(x))
        if hi <= lo:
            lo, hi = float(np.min(x)), float(np.max(x))
        grid = np.linspace(lo, hi, 200)
        mid, band_lo, band_hi = chance_with_limits(grid)
        series = []
        if opt.plot_confidence_bands:
            series.append(ChartSeries(f"{pct} confidence band", grid.tolist(), band_lo.tolist(), kind="band",
                                      y_upper=band_hi.tolist(), color=BLUE))
        series.append(ChartSeries(name, x.tolist(), y.tolist(), kind="scatter", color=BLUE))
        series.append(ChartSeries("Fitted chance", grid.tolist(), mid.tolist(), kind="line", color=RED))
        lines = [{"kind": "h", "value": 0.5, "style": ":"}]
        if opt.mark_levels:
            lines += [{"kind": "v", "value": float(at), "style": ":"} for _percent, at in marks if lo <= at <= hi]
        charts.append(ChartSpec(title="Probability of the outcome", series=series, x_label="X",
                                y_label=f"Chance that {name} = 1", y_limits=(-0.05, 1.05), reference_lines=lines))

    summary = (f"For every rise of {step:g} in X the odds of {name} = 1 multiply by {ratio:.{sig}g} "
               f"({pct} CI {ratio_span}); P = {lr_p:.4g}")
    if marks:
        percent, at = min(marks, key=lambda mark: abs(mark[0] - 50))   # the level nearest 50% speaks for the rest
        summary += f"; the chance is {percent:g}% at X = {at:.{sig}g}"
    if area is not None:
        summary += f"; AUC = {area:.4f}"
    summary += "."
    notes.append(f"'{name}' is the outcome (1 = it happened) and X the predictor; {n} rows, {events} with the outcome. "
                 f"Fitted by {method}; {interval_name} intervals for the slope and the odds ratio; the P in the summary "
                 "is from the likelihood-ratio test.")
    return AnalysisResult(title="Fit a yes/no outcome", summary=summary, tables=tables, notes=notes, charts=charts)
