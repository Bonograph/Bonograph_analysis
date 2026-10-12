"""Explain a count - events, cells, visits - from several predictors.

X and each data set are the variables, one subject on each row; one of them
is the count. Effects are rate ratios: what one more unit of a predictor
multiplies the expected count by. An exposure variable (time watched, area
searched, population) turns the counts into rates.

A Poisson count has a spread fixed by its mean. Real counts are usually more
spread out than that, which makes Poisson standard errors too small, so the
spread is checked and - unless told otherwise - a negative binomial model is
used when it is too wide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg

FAMILIES = {"poisson": "Poisson", "quasi": "quasi-Poisson", "negative_binomial": "negative binomial"}
PROFILE_LIMIT = 12       # more terms than this: Wald intervals, since each profile limit refits the model many times
MOST_COUNT_ROWS = 12     # rows of the seen-and-expected table before "or more"


@dataclass
class CountRegressionOptions:
    outcome_index: int = -1           # among X and the data sets; -1 = the last
    exposure_index: int = -1          # the variable holding each row's exposure; -1 = none
    family: str = "auto"              # "auto" | "poisson" | "quasi" | "negative_binomial"
    reference_levels: dict = field(default_factory=dict)
    auto_reference: dict = field(default_factory=dict)
    ci_approach: str = "profile"      # "profile" | "wald"
    test_each_term: bool = False      # each predictor tested by refitting without it
    show_fit: bool = True
    show_spread_check: bool = True
    show_count_table: bool = True
    show_predictions: bool = False    # fitted counts and residuals on a new sheet
    graph_observed_predicted: bool = True
    graph_residuals: bool = True
    graph_counts: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


# ── likelihoods ─────────────────────────────────────────────────────────────

def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sum(special.xlogy(y, mu) - mu - special.gammaln(y + 1)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    return float(2 * np.sum(special.xlogy(y, y / mu) - (y - mu)))


def nb_loglik(y: np.ndarray, mu: np.ndarray, theta: float) -> float:
    """Negative binomial with variance mu + mu²/theta."""
    return float(np.sum(special.gammaln(y + theta) - special.gammaln(theta) - special.gammaln(y + 1)
                        + theta * np.log(theta / (theta + mu)) + special.xlogy(y, mu / (theta + mu))))


def nb_deviance(y: np.ndarray, mu: np.ndarray, theta: float) -> float:
    return float(2 * np.sum(special.xlogy(y, y / mu) - (y + theta) * np.log((y + theta) / (mu + theta))))


# ── fitting ─────────────────────────────────────────────────────────────────

def fit_mean(X: np.ndarray, y: np.ndarray, offset: np.ndarray, theta: float | None = None, start: np.ndarray | None = None,
             rounds: int = 100, tolerance: float = 1e-10):
    """Log-linear model for the mean by iterated weighted least squares:
    Poisson when ``theta`` is None, else negative binomial with that theta
    held fixed. Returns (coefficients, their covariance, fitted means, converged)."""
    n, p = X.shape

    def badness(mu):
        return poisson_deviance(y, mu) if theta is None else nb_deviance(y, mu, theta)

    if start is None:
        beta = np.zeros(p)
        rate = max(float(y.sum()) / float(np.exp(offset).sum()), 1e-8)
        # with a constant in the first column the flat model is the natural start
        beta[0] = np.log(rate) if np.allclose(X[:, 0], 1.0) else 0.0
    else:
        beta = np.asarray(start, dtype=float).copy()
    eta = np.clip(X @ beta + offset, -30, 30)
    mu = np.exp(eta)
    old = badness(mu)
    converged = False
    for _ in range(rounds):
        weight = mu if theta is None else mu / (1 + mu / theta)
        working = eta - offset + (y - mu) / mu
        XtW = X.T * weight
        step_to = rg.solve_spd(XtW @ X, XtW @ working)
        new_beta = step_to
        for _half in range(30):                      # halve the step until the fit stops getting worse
            eta_new = np.clip(X @ new_beta + offset, -30, 30)
            mu_new = np.exp(eta_new)
            new = badness(mu_new)
            if np.isfinite(new) and new <= old + 1e-9 * (abs(old) + 1):
                break
            new_beta = (new_beta + beta) / 2
        change = abs(new - old) / (abs(new) + 0.1)
        beta, eta, mu, old = new_beta, eta_new, mu_new, new
        if change < tolerance:
            converged = True
            break
    weight = mu if theta is None else mu / (1 + mu / theta)
    cov = rg.inverse((X.T * weight) @ X)
    return beta, cov, mu, converged


def theta_ml(y: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """The theta that makes the negative binomial likelihood largest for
    these fitted means, and its standard error. (inf, nan) when the counts
    are no more spread out than Poisson counts, where the likelihood keeps
    rising as theta grows."""
    def score(log_theta: float) -> float:
        th = np.exp(log_theta)
        return float(np.sum(special.digamma(y + th) - special.digamma(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th)))

    low, high = np.log(1e-4), np.log(1e7)
    if score(high) > 0:                # still climbing at an enormous theta: Poisson
        return float("inf"), float("nan")
    if score(low) < 0:
        low = np.log(1e-8)
        if score(low) < 0:
            return float(np.exp(low)), float("nan")
    theta = float(np.exp(optimize.brentq(score, low, high, xtol=1e-12, rtol=1e-12)))
    information = float(np.sum(-special.polygamma(1, y + theta) + special.polygamma(1, theta) - 1 / theta
                               + 2 / (mu + theta) - (y + theta) / (mu + theta) ** 2))
    return theta, (float(np.sqrt(1 / information)) if information > 0 else float("nan"))


def fit_negative_binomial(X: np.ndarray, y: np.ndarray, offset: np.ndarray, start: np.ndarray | None = None, rounds: int = 60):
    """Coefficients and theta together, by fitting each in turn until neither
    moves. Returns (coefficients, covariance, means, theta, SE of theta, converged)."""
    beta, cov, mu, ok = fit_mean(X, y, offset, None, start)
    theta, se_theta = theta_ml(y, mu)
    if not np.isfinite(theta):
        return beta, cov, mu, theta, se_theta, ok
    last = nb_loglik(y, mu, theta)
    converged = False
    for _ in range(rounds):
        beta, cov, mu, ok = fit_mean(X, y, offset, theta, beta)
        theta, se_theta = theta_ml(y, mu)
        if not np.isfinite(theta):
            beta, cov, mu, ok = fit_mean(X, y, offset, None, beta)
            return beta, cov, mu, theta, se_theta, ok
        now = nb_loglik(y, mu, theta)
        if abs(now - last) < 1e-9 * (abs(now) + 1):
            converged = True
            break
        last = now
    beta, cov, mu, ok = fit_mean(X, y, offset, theta, beta)
    return beta, cov, mu, theta, se_theta, converged and ok


def dean_test(y: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Score test of a Poisson model against counts whose variance grows
    with the square of the mean (Dean and Lawless 1989): (z, one-sided P)."""
    bottom = np.sqrt(2 * np.sum(mu**2))
    if bottom <= 0:
        return float("nan"), float("nan")
    z = float(np.sum((y - mu) ** 2 - y) / bottom)
    return z, float(stats.norm.sf(z))


def count_bins(top: int, most: int = MOST_COUNT_ROWS) -> list[tuple[int, int]]:
    """The counts 0..top as (first, last) ranges: one count each while that
    gives a short table, wider ranges of equal width when it would not."""
    width = max(1, int(np.ceil((top + 1) / most)))
    return [(lo, min(lo + width - 1, top)) for lo in range(0, top + 1, width)]


def expected_counts(mu: np.ndarray, theta: float | None, bins: list[tuple[int, int]]) -> np.ndarray:
    """How many subjects the model expects in each range of counts, and above the last (final entry)."""
    if theta is None or not np.isfinite(theta):
        def below(k):                         # P(count <= k), summed over the subjects
            return float(stats.poisson.cdf(k, mu).sum()) if k >= 0 else 0.0
    else:
        def below(k):
            return float(stats.nbinom.cdf(k, theta, theta / (theta + mu)).sum()) if k >= 0 else 0.0
    each = [below(hi) - below(lo - 1) for lo, hi in bins]
    return np.array(each + [max(mu.size - below(bins[-1][1]), 0.0)])


def _profile_limits(j: int, X, y, offset, theta, beta, se, loglik_of, top: float, level: float):
    """The two values of coefficient j at which the log-likelihood, with every
    other coefficient refitted, has fallen by half the chi-square cut-off."""
    drop = stats.chi2.ppf(level, 1) / 2
    others = [k for k in range(X.shape[1]) if k != j]
    X_rest = X[:, others]

    def gap(value: float) -> float:
        shifted = offset + value * X[:, j]
        if others:
            _b, _c, mu, _ok = fit_mean(X_rest, y, shifted, theta, beta[others])
        else:
            mu = np.exp(np.clip(shifted, -30, 30))
        return loglik_of(mu) - (top - drop)

    limits = []
    for sign in (-1.0, 1.0):
        inner, reach, found = 0.0, max(se[j], 1e-4), None
        for _ in range(12):
            if gap(beta[j] + sign * reach) < 0:
                a, b = beta[j] + sign * inner, beta[j] + sign * reach
                found = optimize.brentq(gap, min(a, b), max(a, b), xtol=1e-7 * max(1.0, abs(beta[j])))
                break
            inner, reach = reach, reach * 2
        limits.append(found)
    return limits[0], limits[1]


@register("count_regression")
def run(data: SheetData, options: CountRegressionOptions | None = None) -> AnalysisResult:
    opt = options or CountRegressionOptions()
    family = opt.family if opt.family in ("auto", *FAMILIES) else "auto"
    labels, _matrix = data.full_variable_matrix()
    exposure_at = opt.exposure_index if 0 <= opt.exposure_index < len(labels) else -1
    design = rg.read(data, opt.outcome_index, opt.reference_levels, opt.auto_reference,
                     also=(exposure_at,) if exposure_at >= 0 else (), what="A count regression")
    y = design.y
    name = design.outcome
    if np.any(y < 0) or np.any(np.abs(y - np.round(y)) > 1e-9):
        raise AnalysisError(f"'{name}' is not a count: a count regression needs whole numbers, zero or more, as the outcome. "
                            "For a measurement, use the regression for a measured outcome.")
    if not design.names:
        raise AnalysisError("None of the predictors varies on the rows that are filled in.")
    X = rg.with_constant(design.Z)
    terms = ["Constant"] + design.names
    n, p = X.shape
    if n < p + 3:
        raise AnalysisError(f"There are {n} complete rows for {p} model terms: too few to fit. Add rows, or leave predictors out.")
    if y.sum() == 0:
        raise AnalysisError(f"Every count in '{name}' is zero, so there is nothing to explain.")
    rg.check_rank(X, design.names)
    exposure = design.extra.get(exposure_at)
    if exposure is not None:
        if np.any(exposure <= 0):
            raise AnalysisError(f"'{design.extra_names[exposure_at]}', the exposure, must be above zero on every row.")
        offset = np.log(exposure)
    else:
        offset = np.zeros(n)

    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"

    def num(v) -> str:
        return rg.clean_zero(f"{v:.{sig}g}") if np.isfinite(v) else "-"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    # ---- the Poisson fit comes first: every family starts from it, and the spread check needs it
    beta_p, cov_p, mu_p, ok_p = fit_mean(X, y, offset)
    pearson_p = float(np.sum((y - mu_p) ** 2 / mu_p))
    df_resid = n - p
    ratio = pearson_p / df_resid if df_resid > 0 else float("nan")
    z_dean, p_dean = dean_test(y, mu_p)
    notes: list[str] = []
    chosen_for_you = family == "auto"
    if chosen_for_you:
        family = "negative_binomial" if (np.isfinite(p_dean) and p_dean < opt.alpha and ratio > 1) else "poisson"

    theta = se_theta = None
    if family == "negative_binomial":
        beta, cov, mu, theta, se_theta, converged = fit_negative_binomial(X, y, offset, beta_p)
        if not np.isfinite(theta):
            notes.append("The counts are no more spread out than Poisson counts, so the negative binomial model comes down to the "
                         "Poisson model, which is what is reported.")
            family, theta, se_theta = "poisson", None, None
            beta, cov, mu, converged = beta_p, cov_p, mu_p, ok_p
    else:
        beta, cov, mu, converged = beta_p, cov_p, mu_p, ok_p
    if not converged:
        notes.append("The fit did not settle: the estimates may be unreliable. This usually means a predictor separates the zero "
                     "counts from the rest completely, or two predictors are nearly the same.")

    scale = ratio if family == "quasi" else 1.0
    se = np.sqrt(np.clip(np.diag(cov) * scale, 0, None))
    adrift = [j + 1 for j in rg.runaway(beta[1:], se[1:], design.Z.std(axis=0))]
    if adrift:
        notes.append(rg.runaway_note([terms[j] for j in adrift], "that term goes with nothing but zero counts (or separates the zero "
                                                               "counts from all the others)"))
    stat, p_values, lower, upper = rg.wald(beta, se, level, df_resid if family == "quasi" else None)
    if family == "negative_binomial":
        loglik, deviance = nb_loglik(y, mu, theta), nb_deviance(y, mu, theta)
        variance = mu + mu**2 / theta
    else:
        loglik, deviance = poisson_loglik(y, mu), poisson_deviance(y, mu)
        variance = mu * scale
    pearson_resid = (y - mu) / np.sqrt(variance)
    pearson = float(np.sum(pearson_resid**2))

    # ---- intervals: profile likelihood where there is a likelihood to profile
    interval_kind = "Wald"
    if opt.ci_approach == "profile" and family != "quasi" and p <= PROFILE_LIMIT and n * p <= 2_000_000 and converged and not adrift:
        def loglik_of(m):
            return nb_loglik(y, m, theta) if family == "negative_binomial" else poisson_loglik(y, m)

        got = 0
        for j in range(p):
            try:
                lo, hi = _profile_limits(j, X, y, offset, theta, beta, se, loglik_of, loglik, level)
            except (ValueError, FloatingPointError):
                lo = hi = None
            if lo is not None and hi is not None:
                lower[j], upper[j] = lo, hi
                got += 1
        interval_kind = "profile-likelihood" if got == p else ("profile-likelihood where it could be found, Wald elsewhere" if got else "Wald")

    # ---- the model with no predictors
    X0 = X[:, :1]
    if family == "negative_binomial":
        b0, _c0, mu0, theta0, _s0, _k0 = fit_negative_binomial(X0, y, offset)
        loglik0 = nb_loglik(y, mu0, theta0) if np.isfinite(theta0) else poisson_loglik(y, mu0)
        deviance0 = float("nan")
    else:
        b0, _c0, mu0, _k0 = fit_mean(X0, y, offset)
        loglik0, deviance0 = poisson_loglik(y, mu0), poisson_deviance(y, mu0)
    df_model = p - 1
    if family == "quasi":
        whole_stat = (deviance0 - deviance) / df_model / ratio if ratio > 0 else float("nan")
        whole_p = float(stats.f.sf(whole_stat, df_model, df_resid)) if np.isfinite(whole_stat) else float("nan")
        whole_text = f"F({df_model}, {df_resid}) = {whole_stat:.3f}"
    else:
        whole_stat = max(2 * (loglik - loglik0), 0.0)
        whole_p = float(stats.chi2.sf(whole_stat, df_model))
        whole_text = f"chi-square({df_model}) = {whole_stat:.3f}"

    # ---- tables
    what_ratio = "Rate ratio" if exposure is not None else "Rate ratio (count multiplied by)"
    stat_name = "t" if family == "quasi" else "z"
    rows = []
    for j, term in enumerate(terms):
        rows.append([term, num(beta[j]), num(se[j]), f"{stat[j]:.3f}" if np.isfinite(stat[j]) else "-", p_text(p_values[j]),
                     rg.ratio_text(rg.ratio(beta[j]), sig), rg.ratio_span(rg.ratio(lower[j]), rg.ratio(upper[j]), sig)])
    tables = [ResultTable(f"What each predictor does to the count ({FAMILIES[family]} model)",
                          ["Term", "Coefficient (log scale)", "SE", stat_name, "P value", what_ratio, f"{pct} CI"], rows)]

    if opt.test_each_term and len(design.predictors) >= 1:
        each = []
        for predictor in design.predictors:
            keep = [0] + [c + 1 for c in range(design.Z.shape[1]) if design.sources[c] != predictor]
            lost = p - len(keep)
            Xr = X[:, keep]
            if family == "negative_binomial":
                _b, _c, mur, thr, _s, _k = fit_negative_binomial(Xr, y, offset)
                llr = nb_loglik(y, mur, thr) if np.isfinite(thr) else poisson_loglik(y, mur)
                value = max(2 * (loglik - llr), 0.0)
                each.append([predictor, f"{value:.3f}", str(lost), p_text(float(stats.chi2.sf(value, lost)))])
            else:
                _b, _c, mur, _k = fit_mean(Xr, y, offset)
                gain = poisson_deviance(y, mur) - deviance
                if family == "quasi":
                    value = gain / lost / ratio
                    each.append([predictor, f"{value:.3f}", f"{lost}, {df_resid}", p_text(float(stats.f.sf(value, lost, df_resid)))])
                else:
                    each.append([predictor, f"{max(gain, 0.0):.3f}", str(lost), p_text(float(stats.chi2.sf(max(gain, 0.0), lost)))])
        tables.append(ResultTable("Each predictor, tested by fitting the model without it",
                                  ["Predictor", "F" if family == "quasi" else "Chi-square", "df", "P value"], each))

    if opt.show_fit:
        fit_rows = [["Rows used", str(n)], ["Model terms, with the constant", str(p)]]
        if family != "quasi":
            k = p + (1 if family == "negative_binomial" else 0)
            aic = 2 * k - 2 * loglik
            fit_rows += [["Log-likelihood", f"{loglik:.4f}"], ["AIC", f"{aic:.3f}"]]
            if n - k - 1 > 0:
                fit_rows.append(["AICc", f"{aic + 2 * k * (k + 1) / (n - k - 1):.3f}"])
            fit_rows.append(["BIC", f"{k * np.log(n) - 2 * loglik:.3f}"])
            fit_rows.append(["McFadden's R²", f"{1 - loglik / loglik0:.4f}" if loglik0 != 0 else "-"])
        fit_rows += [[f"Deviance (on {df_resid} df)", f"{deviance:.4f}"],
                     [f"Pearson chi-square (on {df_resid} df)", f"{pearson_p if family == 'quasi' else pearson:.4f}"]]
        if np.isfinite(deviance0):
            fit_rows.append([f"Deviance with no predictors (on {n - 1} df)", f"{deviance0:.4f}"])
            if deviance0 > 0:
                fit_rows.append(["Share of the deviance explained", f"{1 - deviance / deviance0:.1%}"])
        fit_rows.append(["All predictors together, against none", f"{whole_text}, P = {p_text(whole_p)}"])
        tables.append(ResultTable("How well the model fits", ["Measure", "Value"], fit_rows))

    if opt.show_spread_check:
        verdict = ("more spread than a Poisson count has" if np.isfinite(p_dean) and p_dean < opt.alpha and ratio > 1
                   else "about what a Poisson count has" if ratio >= 0.7 else "less spread than a Poisson count has")
        spread = [["Pearson chi-square ÷ df under the Poisson model (1 = Poisson spread)", f"{ratio:.3f}", ""],
                  ["Score test for extra spread (Dean-Lawless)", f"z = {z_dean:.3f}" if np.isfinite(z_dean) else "-", p_text(p_dean)]]
        if family == "negative_binomial":
            alpha_hat = 1 / theta
            if se_theta is not None and np.isfinite(se_theta):
                zc = stats.norm.ppf(0.5 + level / 2)
                half = zc * se_theta / theta
                spread.append([f"Extra spread, alpha (variance = mean + alpha × mean²), with {pct} CI",
                               f"{num(alpha_hat)} ({num(alpha_hat * rg.ratio(-half))} to {num(alpha_hat * rg.ratio(half))})", ""])
            else:
                spread.append(["Extra spread, alpha (variance = mean + alpha × mean²)", num(alpha_hat), ""])
            spread.append(["Theta (= 1 ÷ alpha)" + (", with its SE" if se_theta and np.isfinite(se_theta) else ""),
                           f"{num(theta)}" + (f" ({num(se_theta)})" if se_theta and np.isfinite(se_theta) else ""), ""])
            lr = max(2 * (loglik - poisson_loglik(y, mu_p)), 0.0)
            spread.append(["Negative binomial against Poisson, likelihood-ratio test", f"chi-square(1) = {lr:.3f}",
                           p_text(float(stats.chi2.sf(lr, 1) / 2))])
        elif family == "quasi":
            spread.append(["Standard errors multiplied by", f"{np.sqrt(ratio):.3f}", ""])
        spread.append(["Reading", verdict, ""])
        tables.append(ResultTable("Is the spread what a Poisson count would have?", ["Check", "Value", "P value"], spread))

    bins = count_bins(int(y.max()))
    seen = np.array([np.sum((y >= lo) & (y <= hi)) for lo, hi in bins], dtype=float)
    expect = expected_counts(mu, theta if family == "negative_binomial" else None, bins)
    bin_names = [str(lo) if lo == hi else f"{lo} to {hi}" for lo, hi in bins]
    if opt.show_count_table:
        under = "the negative binomial model" if family == "negative_binomial" else "the Poisson model"
        count_rows = [[bin_names[k], str(int(seen[k])), f"{expect[k]:.1f}"] for k in range(len(bins))]
        if expect[-1] >= 0.05:
            count_rows.append([f"above {bins[-1][1]}", "0", f"{expect[-1]:.1f}"])
        tables.append(ResultTable("How often each count was seen, and how often the model expects it",
                                  ["Count", "Rows with it", f"Expected under {under}"], count_rows))

    # ---- graphs
    charts = []
    if opt.graph_observed_predicted:
        gx, gy = rg.thin(mu, y)
        reach = float(max(mu.max(), y.max()))
        charts.append(ChartSpec("Counts seen against counts predicted", x_label="Predicted count", y_label=f"{name} (seen)",
                                series=[ChartSeries("", gx, gy)],
                                reference_lines=[{"kind": "path", "points": [[0.0, 0.0], [reach, reach]]}]))
    if opt.graph_residuals:
        gx, gr = rg.thin(mu, pearson_resid)
        charts.append(ChartSpec("Residuals against counts predicted", x_label="Predicted count", y_label="Pearson residual",
                                series=[ChartSeries("", gx, gr)], reference_lines=[{"kind": "h", "value": 0.0}]))
    if opt.graph_counts:
        ks = [float(k) for k in range(len(bins))]
        charts.append(ChartSpec("How often each count was seen and expected", x_label=name, y_label="Rows",
                                series=[ChartSeries("Seen", ks, seen.tolist(), kind="bar", color="#9db7d5"),
                                        ChartSeries("Expected", ks, expect[: len(bins)].tolist(), kind="line_markers", color="#d1342f")],
                                x_categories=[str(lo) if lo == hi else f"{lo}–{hi}" for lo, hi in bins]))

    # ---- headline
    slopes = [j for j in range(1, p) if np.isfinite(p_values[j]) and j not in adrift]
    lead = min(slopes, key=lambda j: p_values[j]) if slopes else None
    summary = (f"{name} was modelled as a {FAMILIES[family]} count on {len(design.predictors)} predictor"
               f"{'s' if len(design.predictors) != 1 else ''}, {n} rows. Together they "
               f"{'explain' if whole_p < opt.alpha else 'do not clearly explain'} the counts: {whole_text}, P = {whole_p:.4g}.")
    if lead is not None:
        summary += (f" Strongest: {terms[lead]}, rate ratio {rg.ratio(beta[lead]):.{sig}g} ({pct} CI "
                    f"{rg.ratio(lower[lead]):.{sig}g} to {rg.ratio(upper[lead]):.{sig}g}), P = {p_values[lead]:.4g}.")

    # ---- notes
    if design.dropped:
        notes.insert(0, f"{design.dropped} row{'s' if design.dropped != 1 else ''} with a blank in the outcome, a predictor or the "
                        f"exposure {'were' if design.dropped != 1 else 'was'} left out.")
    if chosen_for_you:
        notes.append(("The counts are more spread out than a Poisson model allows (see the spread check), so a negative binomial "
                      "model was fitted: it has one more parameter for the extra spread, and honest standard errors."
                      if family == "negative_binomial" else
                      "The spread of the counts is in line with a Poisson model, so that is what was fitted.")
                     + " Choose the model yourself in the settings if you decided on it beforehand.")
    elif family == "poisson" and np.isfinite(p_dean) and p_dean < opt.alpha and ratio > 1:
        notes.append(f"Caution: the counts are {ratio:.1f} times as spread out as a Poisson model allows. Its standard errors are "
                     "then too small and its P values too optimistic. A negative binomial or quasi-Poisson model is safer.")
    notes.append("A rate ratio is what the expected count is multiplied by when the predictor goes up by one unit, with the others "
                 "held fixed: 1.20 means 20% more, 0.80 means 20% fewer. For a predictor read as categories it compares that "
                 "level with the reference level.")
    if exposure is not None:
        notes.append(f"'{design.extra_names[exposure_at]}' is the exposure: the model is for the count per unit of it, and the "
                     "constant's rate ratio is the rate when every predictor is zero (or at its reference level).")
    else:
        notes.append("The constant's rate ratio is the expected count when every predictor is zero (or at its reference level). "
                     "If the rows were watched for different lengths of time, or cover different areas, give that as the exposure.")
    if design.categories:
        notes.append("Read as categories (two to six distinct values): " + "; ".join(
            f"{k} - levels {', '.join(f'{v:g}' for v in lv)}, reference {ref:g}" for k, (lv, ref) in design.categories.items()) + ".")
    how = {"poisson": "Poisson regression with a log link, fitted by maximum likelihood",
           "quasi": f"Poisson regression with a log link; standard errors are multiplied by {np.sqrt(ratio):.3f}, the square root of "
                    "the Pearson chi-square over its degrees of freedom, and tests use t and F (quasi-likelihood)",
           "negative_binomial": "Negative binomial regression (variance = mean + alpha × mean²) with a log link, fitted by maximum "
                                "likelihood; the test against Poisson halves the chi-square P value, since alpha cannot be negative"}
    notes.append(f"{how[family]}. Intervals for the coefficients: {interval_kind}.")
    zeros = float(np.sum(y == 0))
    zeros_expected = float(expected_counts(mu, theta if family == "negative_binomial" else None, [(0, 0)])[0])
    if zeros > 1.5 * zeros_expected and zeros - zeros_expected >= 5:
        notes.append(f"There are more zeros ({int(zeros)}) than the model expects ({zeros_expected:.1f}). If some subjects could never "
                     "have had an event, a count model alone will not describe them well.")
    notes.append("References: Nelder and Wedderburn 1972; Lawless 1987 (negative binomial); Dean and Lawless 1989 (score test for "
                 "extra spread).")

    result = AnalysisResult(title="Several predictors, a count", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.show_predictions:
        with np.errstate(divide="ignore", invalid="ignore"):
            if family == "negative_binomial":
                unit = 2 * (special.xlogy(y, y / mu) - (y + theta) * np.log((y + theta) / (mu + theta)))
            else:
                unit = 2 * (special.xlogy(y, y / mu) - (y - mu)) / scale
        dev_resid = np.sign(y - mu) * np.sqrt(np.clip(unit, 0, None))
        result.new_sheet = NewSheetData("Count model predictions", "Row", (design.rows + 1).astype(float).tolist(),
                                        {name: y.tolist(), "Predicted count": mu.tolist(), "Pearson residual": pearson_resid.tolist(),
                                         "Deviance residual": dev_resid.tolist()})
        notes.append("The predicted count and the residuals of every row used are on a new sheet, 'Count model predictions'.")
    return result
