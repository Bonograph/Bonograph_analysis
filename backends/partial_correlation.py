"""Are two variables linked once others are held fixed?

Each data set is a variable, each row a subject. The partial correlation of
two of them is the correlation that is left after the straight-line effect
of the other variables has been taken out of both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from .. import chart_helpers


@dataclass
class PartialCorrelationOptions:
    first_index: int = 0
    second_index: int = 1
    ranks: bool = False            # Spearman: the same, on the ranks of every variable
    show_each_control: bool = True  # what holding each other variable fixed, one at a time, does
    show_matrix: bool = True        # every pair, each with all the others held fixed
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def partial_r(values: np.ndarray, i: int, j: int, controls: list[int]) -> float:
    """Correlation of columns i and j of ``values`` with the ``controls`` columns held fixed."""
    if not controls:
        return float(np.corrcoef(values[:, i], values[:, j])[0, 1])
    design = np.column_stack([np.ones(len(values)), values[:, controls]])
    resid_i = values[:, i] - design @ np.linalg.lstsq(design, values[:, i], rcond=None)[0]
    resid_j = values[:, j] - design @ np.linalg.lstsq(design, values[:, j], rcond=None)[0]
    si, sj = resid_i.std(), resid_j.std()
    if si < 1e-12 * max(1.0, values[:, i].std()) or sj < 1e-12 * max(1.0, values[:, j].std()):
        return float("nan")
    return float(np.corrcoef(resid_i, resid_j)[0, 1])


def r_test(r: float, n: int, held: int, level: float) -> tuple[float, float, int, tuple[float, float]]:
    """(t, P, df, confidence limits) of a correlation with ``held`` variables held fixed."""
    df = n - 2 - held
    if not np.isfinite(r) or df < 1:
        return float("nan"), float("nan"), df, (float("nan"), float("nan"))
    if abs(r) >= 1:
        return float("inf"), 0.0, df, (r, r)
    t = r * np.sqrt(df / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    if n - 3 - held < 1:
        return float(t), p, df, (float("nan"), float("nan"))
    half = stats.norm.ppf(0.5 + level / 2) / np.sqrt(n - 3 - held)          # Fisher's z
    return float(t), p, df, (float(np.tanh(np.arctanh(r) - half)), float(np.tanh(np.arctanh(r) + half)))


@register("partial_correlation")
def run(data: SheetData, options: PartialCorrelationOptions | None = None) -> AnalysisResult:
    opt = options or PartialCorrelationOptions()
    names, matrix = data.first_replicate_matrix()
    if len(names) < 3:
        raise AnalysisError("This needs three variables (data sets) or more: two to relate and at least one to hold fixed.")
    keep = np.isfinite(matrix).all(axis=1)
    dropped = int(np.isfinite(matrix).any(axis=1).sum() - keep.sum())
    values = matrix[keep]
    n, k = values.shape
    i, j = int(opt.first_index), int(opt.second_index)
    if not (0 <= i < k and 0 <= j < k) or i == j:
        raise AnalysisError("Choose two different variables to relate.")
    controls = [c for c in range(k) if c not in (i, j)]
    if n - 2 - len(controls) < 1:
        raise AnalysisError(f"Too few complete rows ({n}) for {len(controls)} variables held fixed: at least {len(controls) + 3} are needed.")
    if np.any(values.std(axis=0) == 0):
        flat = [names[c] for c in range(k) if values[:, c].std() == 0]
        raise AnalysisError("No variation in " + ", ".join(flat) + ": a correlation needs every variable to vary.")
    if opt.ranks:
        values = np.column_stack([stats.rankdata(values[:, c]) for c in range(k)])
    sig = opt.significant_digits
    level = opt.confidence_level
    pct = f"{round(level * 100)}%"
    kind = "Spearman" if opt.ranks else "Pearson"

    def r3(value) -> str:
        if not np.isfinite(value):
            return ""
        text = f"{value:.{max(3, sig - 1)}f}"
        return text.lstrip("-") if float(text) == 0 else text      # never "-0.000"

    def span(pair) -> str:
        return "" if not all(np.isfinite(pair)) else f"{r3(pair[0])} to {r3(pair[1])}"

    def p_text(p) -> str:
        return "" if not np.isfinite(p) else format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    def row(label: str, held: list[int]) -> list[str]:
        r = partial_r(values, i, j, held)
        t, p, df, ci = r_test(r, n, len(held), level)
        return [label, r3(r), span(ci), "" if not np.isfinite(t) else f"{t:.4f}", str(df), p_text(p)]

    plain = partial_r(values, i, j, [])
    partial = partial_r(values, i, j, controls)
    if not np.isfinite(partial):
        raise AnalysisError("One of the two variables is fixed exactly by the others, so nothing is left to correlate.")
    held_names = ", ".join(names[c] for c in controls)
    headers = ["Correlation", "r", f"{pct} CI", "t", "df", "P value"]
    tables = [ResultTable(f"{names[i]} and {names[j]}", headers,
                          [row("As measured (nothing held fixed)", []), row(f"With {held_names} held fixed", controls)])]

    # semipartial: the others taken out of one variable only - the unique share of the other's variance
    design = np.column_stack([np.ones(n), values[:, controls]])

    def residual(c: int) -> np.ndarray:
        return values[:, c] - design @ np.linalg.lstsq(design, values[:, c], rcond=None)[0]

    semi_j = float(np.corrcoef(values[:, i], residual(j))[0, 1])
    semi_i = float(np.corrcoef(values[:, j], residual(i))[0, 1])
    tables.append(ResultTable("What that amounts to", ["Measure", "Value", "In words"],
                              [["Partial r squared", r3(partial**2), f"Share of the variation {names[i]} has left, after the others, that {names[j]} accounts for"],
                               [f"Semipartial r ({names[j]} cleared of the others)", r3(semi_j),
                                f"Its square, {semi_j**2:.3f}, is the share of all the variation in {names[i]} that only {names[j]} explains"],
                               [f"Semipartial r ({names[i]} cleared of the others)", r3(semi_i),
                                f"Its square, {semi_i**2:.3f}, is the share of all the variation in {names[j]} that only {names[i]} explains"]]))

    if opt.show_each_control and len(controls) > 1:
        tables.append(ResultTable("Holding one variable fixed at a time", headers,
                                  [row(f"{names[c]} held fixed", [c]) for c in controls]))
    if opt.show_matrix:
        try:
            precision = np.linalg.inv(np.corrcoef(values, rowvar=False))
            scale = np.sqrt(np.diag(precision))
            every = -precision / np.outer(scale, scale)
            np.fill_diagonal(every, 1.0)
            tables.append(ResultTable("Every pair, with all the other variables held fixed", ["Variable"] + names,
                                      [[names[a]] + [r3(every[a, b]) for b in range(k)] for a in range(k)]))
        except np.linalg.LinAlgError:
            every = None
    else:
        every = None

    t, p, df, ci = r_test(partial, n, len(controls), level)
    change = abs(plain) - abs(partial)
    if np.sign(plain) != np.sign(partial) and abs(plain) > 0.1 and abs(partial) > 0.1:
        reading = "the link changes direction"
    elif change > 0.1:
        reading = "part of the link ran through the variables held fixed"
    elif change < -0.1:
        reading = "the variables held fixed were hiding part of the link"
    else:
        reading = "the variables held fixed make little difference"
    summary = (f"{kind} correlation of {names[i]} and {names[j]}: r = {plain:.3f} as measured, {partial:.3f} with {held_names} held fixed "
               f"({pct} CI {span(ci)}, P = {p:.4g}) - {reading}. {n} subjects.")

    notes = []
    if dropped:
        notes.append(f"{dropped} row{'s' if dropped != 1 else ''} with a missing value {'were' if dropped != 1 else 'was'} left out.")
    notes += ["A partial correlation is the correlation between what is left of each variable after the straight-line effect of the "
              "others has been taken out. It answers: among subjects alike in the variables held fixed, do these two still go together?",
              "Only straight-line effects are removed. A curved link with a variable held fixed stays in, and so does anything not "
              "measured: a partial correlation near zero does not prove there is no link, nor does one far from zero prove a direct one.",
              "Holding a variable fixed that is itself a result of the two (rather than a cause of them) can create a link that is "
              "not there. Choose what to hold fixed from what is known about the variables, not from the correlations.",
              ("Spearman: every variable is replaced by its ranks first, so the link need only be steadily rising or falling."
               if opt.ranks else "Pearson: assumes straight-line links and no extreme values; try ranks if in doubt."),
              f"The test has n - 2 - {len(controls)} = {df} degrees of freedom; the interval uses Fisher's z with n - 3 - {len(controls)}.",
              "To choose which data sets take part, tick them under 'Data sets to include'."]

    resid_i, resid_j = residual(i), residual(j)
    slope = float((resid_i * resid_j).sum() / (resid_i**2).sum()) if (resid_i**2).sum() > 0 else 0.0
    ends = [float(resid_i.min()), float(resid_i.max())]
    what = "ranks of " if opt.ranks else ""
    charts = [ChartSpec(f"What is left of each after {held_names} (partial r = {partial:.3f})",
                        series=[ChartSeries("", resid_i.tolist(), resid_j.tolist(), kind="scatter", color="#1f5aa6")],
                        x_label=f"{what}{names[i]}, others taken out", y_label=f"{what}{names[j]}, others taken out",
                        reference_lines=[{"kind": "path", "points": [[ends[0], slope * ends[0]], [ends[1], slope * ends[1]]],
                                          "color": "#d1342f", "style": "-", "width": 1.4}]),
              ChartSpec(f"As measured (r = {plain:.3f})",
                        series=[ChartSeries("", values[:, i].tolist(), values[:, j].tolist(), kind="scatter", color="#7b9fd0")],
                        x_label=what + names[i], y_label=what + names[j])]
    if every is not None:
        charts.append(chart_helpers.heatmap_chart("Every pair, with all the others held fixed", every, names, names, cmap="RdBu_r",
                                                  color_limits=(-1.0, 1.0)))
    return AnalysisResult(title="Two variables, others held fixed", summary=summary, tables=tables, notes=notes, charts=charts)
