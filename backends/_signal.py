"""Shared by the analyses that treat a data set as a trace - a spectrum, a
chromatogram, a recording: reading the trace off the sheet, estimating the
baseline under its peaks, and putting it on an even grid.
"""

from __future__ import annotations

import numpy as np

from ..data_source import SheetData


def traces(data: SheetData, least: int = 3) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, X, Y) for each data set: side-by-side repeats averaged, rows
    with a blank dropped, sorted by X. Data sets with fewer than ``least``
    points are left out."""
    x_all = np.asarray(data.x, dtype=float)
    out = []
    for label in data.group_labels():
        _reps, matrix = data.group_replicate_matrix(label)
        if not matrix.size:
            continue
        n = min(x_all.size, matrix.shape[0])
        with np.errstate(all="ignore"):
            y = np.nanmean(matrix[:n], axis=1) if matrix.shape[1] > 1 else matrix[:n, 0]
        x = x_all[:n]
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < least:
            continue
        order = np.argsort(x[keep], kind="stable")
        out.append((label, x[keep][order], y[keep][order]))
    return out


def als_baseline(y: np.ndarray, smoothness: float = 1e5, asymmetry: float = 0.01, rounds: int = 12) -> np.ndarray:
    """Baseline by asymmetric least squares (Eilers and Boelens 2005).

    A smooth curve z is fitted to y with weights that are small where y is
    above z and large where it is below, so the curve settles under the
    peaks. ``smoothness`` (lambda) is the penalty on the second difference of
    z - larger is stiffer; ``asymmetry`` (p) is the weight given to points
    above the curve."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        return np.full(n, y.min() if n else 0.0)
    second = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = smoothness * (second.T @ second)
    weights = np.ones(n)
    z = y
    for _ in range(rounds):
        system = sparse.diags(weights, 0, format="csc") + penalty
        z = spsolve(system, weights * y)
        new = np.where(y > z, asymmetry, 1 - asymmetry)
        if np.array_equal(new, weights):
            break
        weights = new
    return np.asarray(z, dtype=float)


def polynomial_baseline(x: np.ndarray, y: np.ndarray, degree: int = 3, rounds: int = 200) -> np.ndarray:
    """Baseline by a polynomial fitted again and again, each time to the
    lower of the data and the last fit, until it stops moving - so the peaks
    are cut away and the curve follows what lies under them (Lieber and
    Mahadevan-Jansen 2003)."""
    x = np.asarray(x, dtype=float)
    span = np.ptp(x) or 1.0
    u = (x - x.min()) / span * 2 - 1                 # on -1 to 1, so high powers stay well-behaved
    degree = int(max(0, min(degree, x.size - 1)))
    work = np.asarray(y, dtype=float).copy()
    fit = work
    for _ in range(rounds):
        fit = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyfit(u, work, degree))
        lower = np.minimum(work, fit)
        if np.allclose(lower, work, rtol=0, atol=1e-9 * (np.ptp(work) or 1.0)):
            break
        work = lower
    return fit


def ends_baseline(x: np.ndarray, y: np.ndarray, points: int = 5) -> np.ndarray:
    """A straight line from the mean of the first ``points`` values to the mean of the last."""
    points = int(max(1, min(points, x.size // 2 or 1)))
    x0, y0 = x[:points].mean(), y[:points].mean()
    x1, y1 = x[-points:].mean(), y[-points:].mean()
    if x1 == x0:
        return np.full(x.size, (y0 + y1) / 2)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def even_grid(x: np.ndarray, y: np.ndarray, tolerance: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """(X, Y, step, whether the points had to be moved). A Fourier transform
    needs equal steps in X; where the sheet's are not equal the trace is
    read off, by straight-line interpolation, at equal steps over the same range."""
    steps = np.diff(x)
    step = float(np.median(steps))
    if step <= 0:
        raise ValueError("X does not increase")
    if np.all(np.abs(steps - step) <= tolerance * step):
        return x, y, step, False
    grid = np.linspace(x[0], x[-1], x.size)
    return grid, np.interp(grid, x, y), float(grid[1] - grid[0]), True


def for_graph(*arrays, most: int = 4000):
    """Every k-th point of a long trace, as lists: enough for the eye, and
    light to draw and to save. The sheets keep every point."""
    step = max(1, int(np.ceil(len(arrays[0]) / most)))
    out = [np.asarray(a)[::step].tolist() for a in arrays]
    return out if len(out) > 1 else out[0]
