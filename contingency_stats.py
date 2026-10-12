"""Confidence-interval methods for 2x2 tables of counts, matching what
are reported for the Chi-square / Fisher's exact analysis:

- Wilson/Brown        proportions (sensitivity, specificity, PPV, NPV)
- Newcombe/Wilson+CC  difference between two proportions
- Koopman score       relative risk
- Baptista-Pike       odds ratio (exact conditional; two-sided minimum-
                      likelihood mid-p test, inverted)

Everything here is pure numpy/scipy (no Qt) and written to stay accurate
for small counts, zero cells and one-row/one-column edge cases, where the
usual Wald/Woolf/Katz shortcuts break down. Limits that are genuinely
unbounded come back as 0.0 or ``inf`` rather than raising.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

_LOG_SPAN = 30.0  # search ln(ratio) over [-30, 30]
_MAX_EXACT_SUPPORT = 20000  # beyond this the exact enumeration gets slow


def _z(conf: float) -> float:
    return float(stats.norm.ppf(0.5 + conf / 2.0))


# ---------------------------------------------------------------------------
# Single proportion
# ---------------------------------------------------------------------------
def wilson(x: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Plain Wilson score interval."""
    z = _z(conf)
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_cc(x: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval with continuity correction (Newcombe 1998,
    method 3)."""
    z = _z(conf)
    p = x / n
    q = 1 - p
    lower = 0.0
    upper = 1.0
    if x > 0:
        root = z * z - 2 - 1 / n + 4 * p * (n * q + 1)
        lower = (2 * n * p + z * z - 1 - z * math.sqrt(max(root, 0.0))) / (2 * (n + z * z))
    if x < n:
        root = z * z + 2 - 1 / n + 4 * p * (n * q - 1)
        upper = (2 * n * p + z * z + 1 + z * math.sqrt(max(root, 0.0))) / (2 * (n + z * z))
    return max(0.0, lower), min(1.0, upper)


def wilson_brown(x: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Wilson's interval with Brown, Cai & DasGupta's (2001) boundary
    modification for small counts: when x (or n - x) is 0, 1 or 2 the
    Wilson limit is replaced by an exact Poisson-based one, which is what
    keeps the interval honest at the edges."""
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = 1.0 - conf
    lower, upper = wilson(x, n, conf)
    if x == 0:
        return 0.0, 1.0 - (alpha / 2.0) ** (1.0 / n)
    if x == n:
        return (alpha / 2.0) ** (1.0 / n), 1.0
    if n <= 50:
        if x in (1, 2):
            lower = 0.5 * float(stats.chi2.ppf(alpha, 2 * x)) / n
        if n - x in (1, 2):
            upper = 1.0 - 0.5 * float(stats.chi2.ppf(alpha, 2 * (n - x))) / n
    return lower, upper


# ---------------------------------------------------------------------------
# Difference between two proportions
# ---------------------------------------------------------------------------
def newcombe_cc_difference(x1: float, n1: float, x2: float, n2: float, conf: float = 0.95) -> tuple[float, float]:
    """Newcombe's hybrid score interval with continuity correction (1998,
    method 11) for p1 - p2."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_cc(x1, n1, conf)
    l2, u2 = wilson_cc(x2, n2, conf)
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, lower), min(1.0, upper)


# ---------------------------------------------------------------------------
# Relative risk - Koopman asymptotic score
# ---------------------------------------------------------------------------
def _koopman_z(phi: np.ndarray, x1: float, n1: float, x2: float, n2: float) -> np.ndarray:
    """Score statistic for H0: p1/p2 = phi, with p2 replaced by its
    constrained maximum-likelihood estimate (Koopman 1984)."""
    total = n1 + n2
    a = total * phi
    b = -(phi * (n1 + x2) + n2 + x1)
    c = x1 + x2
    disc = np.maximum(b * b - 4 * a * c, 0.0)
    p2 = (-b - np.sqrt(disc)) / (2 * a)
    p1 = phi * p2
    var = p1 * (1 - p1) / n1 + phi**2 * p2 * (1 - p2) / n2
    num = x1 / n1 - phi * x2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / np.sqrt(var)
    out = np.where(var > 0, out, np.where(num > 0, np.inf, np.where(num < 0, -np.inf, 0.0)))
    return out


def koopman_rr_ci(x1: float, n1: float, x2: float, n2: float, conf: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the relative risk (x1/n1) / (x2/n2). The
    limits are the values of the ratio whose score statistic equals the
    critical z - so they exist (and stay sensible) even when a count is
    zero, unlike the usual log-transform (Katz) interval."""
    if x1 == 0 and x2 == 0:
        return 0.0, math.inf  # nothing observed in either row: no information
    zc = _z(conf)
    grid = np.linspace(-_LOG_SPAN, _LOG_SPAN, 3001)
    zs = _koopman_z(np.exp(grid), x1, n1, x2, n2)  # decreasing in phi

    def _root(target: float) -> float | None:
        above = zs >= target
        if not above.any():
            return None  # never reaches the target: bound is at 0
        if above.all():
            return math.inf
        i = int(np.where(above)[0][-1])
        f = lambda lp: float(_koopman_z(np.array([math.exp(lp)]), x1, n1, x2, n2)[0]) - target
        return math.exp(optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14, maxiter=500))

    lower = 0.0 if x1 == 0 else (_root(zc) or 0.0)
    upper = math.inf if x2 == 0 else _root(-zc)
    if upper is None:
        upper = 0.0
    return lower, upper


# ---------------------------------------------------------------------------
# Odds ratio - Baptista-Pike (exact conditional, minimum-likelihood mid-p)
# ---------------------------------------------------------------------------
def _support(a: int, b: int, c: int, d: int) -> tuple[np.ndarray, int, int, int, int]:
    n1, n2, m1 = a + b, c + d, a + c
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return np.arange(lo, hi + 1), n1, n2, m1, lo


def _midp_minlike(psi: float, a: int, n1: int, n2: int, m1: int, ks: np.ndarray, ia: int) -> float:
    pmf = stats.nchypergeom_fisher.pmf(ks, n1 + n2, n1, m1, psi)
    if not np.all(np.isfinite(pmf)):
        return float("nan")
    obs = pmf[ia]
    tol = 1e-7 * obs
    strictly_less = pmf[pmf < obs - tol].sum()
    ties = pmf[np.abs(pmf - obs) <= tol].sum()
    return float(strictly_less + 0.5 * ties)


def is_whole(*values: float) -> bool:
    return all(abs(v - round(v)) < 1e-9 for v in values)


def baptista_pike_or_ci(a: float, b: float, c: float, d: float, conf: float = 0.95) -> tuple[float, float, bool]:
    """(lower, upper, exact) for the odds ratio ad/bc. ``exact`` is False
    only when the counts are so large that the exact enumeration would be
    slow and the Woolf log-odds interval is returned instead."""
    a, b, c, d = (int(round(v)) for v in (a, b, c, d))
    alpha = 1.0 - conf
    ks, n1, n2, m1, lo_k = _support(a, b, c, d)
    if n1 == 0 or n2 == 0 or m1 == 0 or (n1 + n2 - m1) == 0:
        return 0.0, math.inf, True  # an empty margin: the odds ratio is not identified
    if len(ks) == 1:
        return 0.0, math.inf, True
    if len(ks) > _MAX_EXACT_SUPPORT:
        aa, bb, cc, dd = (v + 0.5 for v in (a, b, c, d))
        se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        centre = math.log(aa * dd / (bb * cc))
        z = _z(conf)
        return math.exp(centre - z * se), math.exp(centre + z * se), False

    ia = a - lo_k
    grid = np.linspace(-_LOG_SPAN, _LOG_SPAN, 601)
    g = np.array([_midp_minlike(math.exp(x), a, n1, n2, m1, ks, ia) for x in grid]) - alpha
    g = np.nan_to_num(g, nan=-1.0)
    inside = np.where(g > 0)[0]
    if inside.size == 0:
        return 0.0, math.inf, True
    f = lambda lp: _midp_minlike(math.exp(lp), a, n1, n2, m1, ks, ia) - alpha
    first, last = int(inside[0]), int(inside[-1])
    lower = 0.0 if first == 0 else math.exp(optimize.brentq(f, grid[first - 1], grid[first], xtol=1e-13, maxiter=500))
    upper = math.inf if last == len(grid) - 1 else math.exp(optimize.brentq(f, grid[last], grid[last + 1], xtol=1e-13, maxiter=500))
    return lower, upper, True
