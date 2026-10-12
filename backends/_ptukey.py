"""Fast studentized range distribution (Tukey's q), for Tukey, Games-Howell
and the graph brackets.

scipy's studentized_range is exact but slow: about 15 ms for one P value and
160 ms for one critical value, so ten groups (45 pairs) took seconds. Here
the same integral is done on fixed Gauss-Legendre grids with numpy,
vectorised over every q at once:

    P(Q > q) = E_s[ H(q s) ],
    H(w) = k ∫ φ(z) [Φ(z)^(k-1) - (Φ(z) - Φ(z - w))^(k-1)] dz

with s = sqrt(χ²_ν / ν). H is written so the upper tail is computed
directly (no 1 - cdf cancellation), which keeps small P values accurate
too. The outer expectation uses Gauss-Legendre nodes in probability space
mapped through the χ² quantile function. Agreement with scipy is checked
in test/test_speed.py.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import special
from scipy.stats import chi2


@functools.lru_cache(maxsize=8)
def _legendre(nodes: int):
    return np.polynomial.legendre.leggauss(nodes)


def _grid(z_nodes: int, s_nodes: int):
    zn, zw = np.polynomial.legendre.leggauss(z_nodes)
    lo, hi = -8.5, 8.5
    z = 0.5 * (hi - lo) * zn + 0.5 * (hi + lo)
    weight = 0.5 * (hi - lo) * zw * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    pn, pw = np.polynomial.legendre.leggauss(s_nodes)
    return z, weight, special.ndtr(z), 0.5 * (pn + 1.0), 0.5 * pw


_FINE = _grid(200, 128)
_COARSE = _grid(64, 40)


def _upper(w: np.ndarray, k: int, grid) -> np.ndarray:
    """H(w): the infinite-df upper tail P(Q > w), computed without cancellation."""
    z, weight, cdf_z, _pn, _pw = grid
    w = np.asarray(w, dtype=float)
    a = special.ndtr(z[None, :] - w.ravel()[:, None])  # Φ(z - w)
    phi_z = np.maximum(cdf_z[None, :], 1e-300)
    ratio = np.clip(a / phi_z, 0.0, 1.0 - 1e-16)
    # Φ^(k-1) - (Φ - a)^(k-1) = -Φ^(k-1) * expm1((k-1) * log1p(-a/Φ))
    term = -(phi_z ** (k - 1)) * np.expm1((k - 1) * np.log1p(-ratio))
    out = k * (term @ weight)
    return np.clip(out, 0.0, 1.0).reshape(w.shape)


@functools.lru_cache(maxsize=512)
def _scales(df: float, s_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points and weights for s = sqrt(χ²/ν), on a log-s grid
    spanning the whole distribution - the far lower tail (small s) is what
    decides very small P values, so it has to be covered."""
    lo = np.sqrt(chi2.ppf(1e-14, df) / df)
    hi = np.sqrt(chi2.isf(1e-14, df) / df)
    un, uw = _legendre(s_nodes)
    u_lo, u_hi = np.log(lo), np.log(hi)
    u = 0.5 * (u_hi - u_lo) * un + 0.5 * (u_hi + u_lo)
    scale = np.exp(u)
    # density of s is 2 ν s f_χ²(ν s²); with u = ln s, ds = s du
    dens = 2.0 * df * scale * chi2.pdf(df * scale**2, df) * scale
    weights = 0.5 * (u_hi - u_lo) * uw * dens
    return scale, weights / weights.sum()


def _sf(q, k: int, df: float, grid) -> np.ndarray:
    q = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.ones_like(q)
    pos = q > 0
    if not pos.any():
        return out
    if not np.isfinite(df) or df > 1e5:
        out[pos] = _upper(q[pos], k, grid)
        return out
    s, pw = _scales(float(round(df, 6)), len(grid[4]))
    h = _upper(q[pos][:, None] * s[None, :], k, grid)  # len(q) x nodes
    out[pos] = h @ pw
    return np.clip(out, 0.0, 1.0)


def sf(q, k: int, df: float) -> np.ndarray:
    """P(Q > q) - the P value of an observed q."""
    return _sf(q, int(k), float(df), _FINE)


def cdf(q, k: int, df: float) -> np.ndarray:
    """P(Q <= q)."""
    return 1.0 - sf(q, k, df)


def _solve(p_upper: float, k: int, df: float, grid, x0: float, x1: float, steps: int) -> float:
    f0 = float(_sf(x0, k, df, grid)[0]) - p_upper
    f1 = float(_sf(x1, k, df, grid)[0]) - p_upper
    for _ in range(steps):
        if f1 == f0:
            break
        x2 = max(1e-6, x1 - f1 * (x1 - x0) / (f1 - f0))
        f2 = float(_sf(x2, k, df, grid)[0]) - p_upper
        x0, f0, x1, f1 = x1, f1, x2, f2
        if abs(f2) < 1e-12:
            break
    return x1


@functools.lru_cache(maxsize=2048)
def _ppf_cached(p: float, k: int, df: float) -> float:
    target = 1.0 - p  # work on the upper tail: accurate for p near 1
    # bracket on a coarse grid, converge there, then polish on the fine grid
    scan = np.geomspace(0.1, 5000.0, 30)
    vals = _sf(scan, k, df, _COARSE)
    j = int(np.searchsorted(-vals, -target))
    j = min(max(j, 1), len(scan) - 1)
    rough = _solve(target, k, df, _COARSE, float(scan[j - 1]), float(scan[j]), 30)
    return _solve(target, k, df, _FINE, rough * 0.999, rough, 4)


_DF_STEPS_PER_E = 40  # grid of df values, ~2.5% apart


def ppf(p: float, k: int, df: float) -> float:
    """The q with P(Q <= q) = p (the critical value for a confidence interval).

    q changes smoothly with df, so it is solved exactly on a fine grid of df
    values (about 2.5% apart, each cached) and read off with 4-point cubic
    interpolation in log(df) - many comparisons with slightly different df
    (as in Games-Howell) then share a handful of exact solutions."""
    if not np.isfinite(df) or df > 1e5:
        return _ppf_cached(float(p), int(k), float("inf"))
    u = np.log(df) * _DF_STEPS_PER_E
    base = int(np.floor(u))
    if abs(u - round(u)) < 1e-9:
        return _ppf_cached(float(p), int(k), float(np.exp(round(u) / _DF_STEPS_PER_E)))
    nodes = [base - 1, base, base + 1, base + 2]
    values = [_ppf_cached(float(p), int(k), float(np.exp(i / _DF_STEPS_PER_E))) for i in nodes]
    total = 0.0
    for a_i, v_a in zip(nodes, values):
        weight = 1.0
        for b_i in nodes:
            if b_i != a_i:
                weight *= (u - b_i) / (a_i - b_i)
        total += weight * v_a
    return float(total)
