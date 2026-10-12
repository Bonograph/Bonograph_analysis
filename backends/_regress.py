"""Shared by the regressions that explain one outcome from several predictors
(a count, categories in an order, categories without one): reading the
variables off the sheet, turning the predictors into model terms, and the
small numerical pieces each of them needs.

The sheet is read the way the other multiple regressions read it: X and each
data set's first column are the variables, one subject on each row. A
predictor with two to six distinct values is read as categories and compared,
level by level, with a reference level; any other predictor is a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData

MOST_LEVELS = 6        # a predictor with this many distinct values or fewer is read as categories
MOST_OUTCOMES = 12     # an outcome with more distinct values than this is a measurement, not categories


@dataclass
class Design:
    outcome: str
    y: np.ndarray                       # the outcome, complete rows only
    Z: np.ndarray                       # the model terms without a constant, complete rows
    names: list[str]                    # one name per column of Z
    sources: list[str]                  # the predictor each column of Z came from
    predictors: list[str]
    raw: dict[str, np.ndarray]          # each predictor as typed, complete rows
    categories: dict[str, tuple[list[float], float]] = field(default_factory=dict)   # predictor -> (levels, reference)
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))        # sheet row of each complete row
    dropped: int = 0                    # rows left out for a blank
    extra: dict[int, np.ndarray] = field(default_factory=dict)                      # other variables asked for, complete rows
    extra_names: dict[int, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.y.size)

    def columns_of(self, predictor: str) -> list[int]:
        return [j for j, s in enumerate(self.sources) if s == predictor]


def read(data: SheetData, outcome_index: int = -1, reference_levels: dict | None = None,
         auto_reference: dict | None = None, also: tuple[int, ...] = (), what: str = "This analysis") -> Design:
    """The outcome, the predictors and any other variable asked for (``also``:
    places in the list of X and the data sets), on the rows where all of them
    are filled in."""
    labels, matrix = data.full_variable_matrix()
    if matrix.ndim != 2 or matrix.shape[0] == 0 or len(labels) < 2:
        raise AnalysisError(f"{what} needs an outcome and at least one predictor, one subject on each row.")
    has_values = ~np.all(np.isnan(matrix), axis=0)
    last = len(labels) - 1
    out = outcome_index if 0 <= outcome_index <= last else last
    if not has_values[out]:
        raise AnalysisError(f"'{labels[out]}', the outcome, holds no values.")
    also = tuple(i for i in also if 0 <= i <= last and i != out and has_values[i])
    predictors = [i for i in range(len(labels)) if i != out and i not in also and has_values[i]]
    if not predictors:
        raise AnalysisError(f"{what} needs at least one predictor with values besides the outcome.")
    used = [out, *predictors, *also]
    filled = ~np.any(np.isnan(matrix[:, used]), axis=1)
    touched = ~np.all(np.isnan(matrix[:, used]), axis=1)
    rows = np.where(filled)[0]
    block = matrix[filled]
    reference_levels, auto_reference = reference_levels or {}, auto_reference or {}

    raw = {labels[i]: block[:, i] for i in predictors}
    Z, names, sources, categories = code_terms(raw, reference_levels, auto_reference)
    kept = list(dict.fromkeys(sources))
    return Design(outcome=labels[out], y=block[:, out], Z=Z, names=names, sources=sources, predictors=kept,
                  raw={k: v for k, v in raw.items() if k in kept}, categories={k: v for k, v in categories.items() if k in kept},
                  rows=rows, dropped=int(touched.sum() - filled.sum()),
                  extra={i: block[:, i] for i in also}, extra_names={i: labels[i] for i in also})


def code_terms(raw: dict[str, np.ndarray], reference_levels: dict | None = None, auto_reference: dict | None = None):
    """Predictors (name -> values, no blanks) as model terms: (matrix, a name
    per column, the predictor each column came from, {predictor: (levels,
    reference)} for those read as categories). A predictor with two to six
    distinct values becomes one 0/1 column per level other than its
    reference; one that never varies is left out."""
    reference_levels, auto_reference = reference_levels or {}, auto_reference or {}
    columns, names, sources, categories = [], [], [], {}
    n = 0
    for label, values in raw.items():
        values = np.asarray(values, dtype=float)
        n = values.size
        levels = np.unique(values)
        if 2 <= levels.size <= MOST_LEVELS:
            reference = float(levels[0])
            if not auto_reference.get(label, True) and reference_levels.get(label) is not None:
                try:
                    chosen = float(reference_levels[label])
                    if chosen in levels:
                        reference = chosen
                except (TypeError, ValueError):
                    pass
            categories[label] = (levels.tolist(), reference)
            for level in levels:
                if level == reference:
                    continue
                columns.append((values == level).astype(float))
                names.append(f"{label} = {level:g} (vs {reference:g})")
                sources.append(label)
        elif levels.size >= 2:
            columns.append(values)
            names.append(label)
            sources.append(label)
    Z = np.column_stack(columns) if columns else np.empty((n, 0))
    return Z, names, sources, categories


def with_constant(Z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(Z.shape[0]), Z])


def check_rank(X: np.ndarray, names: list[str]) -> None:
    """Stop, with words the user can act on, when the terms cannot be told apart."""
    if X.shape[1] and np.linalg.matrix_rank(X) < X.shape[1]:
        raise AnalysisError("Some predictors carry the same information as others (one can be worked out exactly from the rest), "
                            "so their effects cannot be told apart. Leave one of them out: " + ", ".join(names[:8]) + ".")


def solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A x = b for a symmetric matrix, falling back to least squares when it is close to singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def inverse(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def hessian_of(gradient, at: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """The matrix of second derivatives, by central differences of an exact gradient."""
    p = at.size
    out = np.zeros((p, p))
    for j in range(p):
        h = step * max(1.0, abs(at[j]))
        up, down = at.copy(), at.copy()
        up[j] += h
        down[j] -= h
        out[:, j] = (gradient(up) - gradient(down)) / (2 * h)
    return (out + out.T) / 2


def wald(beta: np.ndarray, se: np.ndarray, level: float, df: float | None = None):
    """(statistic, two-sided P, lower, upper) for each coefficient: z, or t on ``df``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(se > 0, beta / se, np.nan)
    if df is None:
        p = 2 * stats.norm.sf(np.abs(stat))
        crit = stats.norm.ppf(0.5 + level / 2)
    else:
        p = 2 * stats.t.sf(np.abs(stat), df)
        crit = stats.t.ppf(0.5 + level / 2, df)
    return stat, p, beta - crit * se, beta + crit * se


def ratio(v):
    """exp(v) that cannot overflow: a coefficient that has run away prints as an enormous or a vanishing ratio."""
    return np.exp(np.clip(v, -700.0, 700.0))


def ratio_text(value: float, sig: int) -> str:
    """A ratio for a table: in words once it is too large or too small to mean anything as a number."""
    if not np.isfinite(value):
        return "-"
    if value > 1e8:
        return "very large"
    if value < 1e-8:
        return "close to 0"
    return clean_zero(f"{value:.{sig}g}")


def ratio_span(low: float, high: float, sig: int) -> str:
    if low < 1e-8 and high > 1e8:
        return "cannot be estimated"
    return f"{ratio_text(low, sig)} to {ratio_text(high, sig)}"


def runaway(beta, se, spread) -> list[int]:
    """Which coefficients have no finite best value: they, or their standard
    errors, have run off because the term predicts the outcome perfectly in
    these data. ``spread`` is each term's standard deviation, so that the
    test is on the effect of one standard deviation of the term."""
    beta, se, spread = (np.asarray(v, dtype=float) for v in (beta, se, spread))
    with np.errstate(invalid="ignore"):
        lost = (np.abs(beta) * spread > 10) | (se * spread > 100) | ~np.isfinite(se) | ~np.isfinite(beta)
    return [int(j) for j in np.where(lost)[0]]


def runaway_note(names: list[str], why: str) -> str:
    return (f"No trustworthy estimate for: {', '.join(names[:6])}{' ...' if len(names) > 6 else ''}. In these data {why}, so the "
            "coefficient has no finite best value: its ratio runs off towards zero or infinity, and the number printed means "
            "only 'very small' or 'very large'. Merge thin categories, leave the term out, or collect more data.")


def pseudo_r2(loglik: float, loglik_null: float, n: int) -> dict[str, float]:
    """McFadden's, Cox and Snell's and Nagelkerke's R-squared from two log-likelihoods."""
    mcfadden = 1 - loglik / loglik_null if loglik_null != 0 else float("nan")
    cox_snell = 1 - np.exp(2 * (loglik_null - loglik) / n)
    ceiling = 1 - np.exp(2 * loglik_null / n)
    return {"mcfadden": float(mcfadden), "cox_snell": float(cox_snell),
            "nagelkerke": float(cox_snell / ceiling) if ceiling > 0 else float("nan")}


def sweep(design: Design, predictor: str, points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Rows of model terms in which one predictor runs across its observed
    values and every other term sits at its mean - for drawing what the model
    says as that predictor changes. Returns (the predictor's values, the rows)."""
    base = design.Z.mean(axis=0)
    columns = design.columns_of(predictor)
    if predictor in design.categories:
        levels, reference = design.categories[predictor]
        xs = np.asarray(levels, dtype=float)
        rows = np.tile(base, (xs.size, 1))
        others = [lv for lv in levels if lv != reference]
        for r, level in enumerate(levels):
            for c, other in zip(columns, others):
                rows[r, c] = 1.0 if level == other else 0.0
        return xs, rows
    values = design.raw[predictor]
    xs = np.linspace(float(values.min()), float(values.max()), points)
    rows = np.tile(base, (points, 1))
    rows[:, columns[0]] = xs
    return xs, rows


def thin(*arrays, most: int = 4000):
    """Every k-th point of long arrays, as lists: enough for a graph."""
    step = max(1, int(np.ceil(len(arrays[0]) / most)))
    out = [np.asarray(a, dtype=float)[::step].tolist() for a in arrays]
    return out if len(out) > 1 else out[0]


def clean_zero(text: str) -> str:
    """'-0.000' reads as a mistake: a value that rounds to zero is written without a sign."""
    stripped = text.lstrip("-+")
    return stripped if stripped and float(stripped) == 0 and text.startswith("-") else text


# ── outcomes that are categories ────────────────────────────────────────────

SHADES = ["#1f5aa6", "#d1342f", "#2e8b57", "#b8860b", "#7b4fa3", "#0e8a8a", "#c2577a", "#5c6b7a", "#8c5a2b", "#3d9be9", "#6b8e23", "#a0522d"]


def categories_of(y: np.ndarray, name: str, least: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(the distinct values of a category outcome in rising order, each row's place among them)."""
    levels, codes = np.unique(y, return_inverse=True)
    if levels.size > MOST_OUTCOMES:
        raise AnalysisError(f"'{name}' has {levels.size} distinct values: that is a measurement rather than a set of categories. "
                            f"This analysis takes an outcome with up to {MOST_OUTCOMES} categories, each typed as a code such as 1, 2, 3.")
    if levels.size < least:
        raise AnalysisError(f"'{name}' holds only {levels.size} distinct value{'s' if levels.size != 1 else ''} on the rows that are "
                            f"filled in; it needs at least {least} categories.")
    return levels, codes.astype(int)


def level_names(levels: np.ndarray) -> list[str]:
    return [f"{v:g}" for v in levels]


def classification(codes: np.ndarray, picked: np.ndarray, names: list[str], outcome: str) -> tuple[ResultTable, float]:
    """Rows by the category seen and the category the model gives the highest chance: the table, and the share right."""
    k = len(names)
    grid = np.zeros((k, k), dtype=int)
    np.add.at(grid, (codes, picked), 1)
    rows = []
    for i in range(k):
        total = int(grid[i].sum())
        rows.append([f"Seen: {names[i]}", *[str(int(v)) for v in grid[i]], str(total), f"{grid[i, i] / total:.1%}" if total else "-"])
    right = float(np.trace(grid) / grid.sum())
    rows.append(["All rows", *[str(int(v)) for v in grid.sum(axis=0)], str(int(grid.sum())), f"{right:.1%}"])
    table = ResultTable(f"Rows by the {outcome} seen and the one the model picks",
                        ["", *[f"Picked: {nm}" for nm in names], "Rows", "Picked right"], rows)
    return table, right


def share_chart(names: list[str], seen: np.ndarray, predicted: np.ndarray, outcome: str) -> ChartSpec:
    xs = [float(i) for i in range(len(names))]
    return ChartSpec("Share in each category: seen and predicted", x_label=outcome, y_label="Rows (%)",
                     series=[ChartSeries("Seen", xs, (seen * 100).tolist(), kind="bar", color="#9db7d5"),
                             ChartSeries("Predicted", xs, (predicted * 100).tolist(), kind="line_markers", color="#d1342f")],
                     x_categories=names)


def probability_charts(design: Design, names: list[str], probabilities_at, group: str, most: int = 8) -> list[ChartSpec]:
    """One graph per predictor: the chance of each category as that predictor
    changes, every other term at its mean. ``probabilities_at(rows)`` gives
    rows x categories."""
    charts = []
    for predictor in design.predictors[:most]:
        xs, rows = sweep(design, predictor)
        probs = np.asarray(probabilities_at(rows)) * 100
        coded = predictor in design.categories
        series = [ChartSeries(names[j], xs.tolist(), probs[:, j].tolist(), kind="line_markers" if coded else "line",
                              color=SHADES[j % len(SHADES)]) for j in range(len(names))]
        charts.append(ChartSpec(f"Chance of each category as {predictor} changes", x_label=predictor, y_label="Chance (%)",
                                series=series, y_limits=(0.0, 100.0), group=group, option=predictor))
    return charts
