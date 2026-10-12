"""How strongly measurements track each other: correlate every pair of Y data sets (a matrix), X vs. every Y
data set, or two specifically chosen data sets; Pearson or Spearman;
one- or two-tailed; a confidence interval on the coefficient."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class CorrelationOptions:
    mode: str = "x_vs_every_y"  # "matrix" | "x_vs_every_y" | "two_selected"
    first_var_index: int = 0  # for "two_selected": index into the [X, group1, group2, ...] list
    second_var_index: int = 1
    remove_incomplete_rows: bool = True  # matrix mode: drop a row if any Y is missing
    gaussian: bool = True  # True = Pearson, False = Spearman
    tails: str = "two"  # "two" | "positive" | "negative" ("one" = positive, kept for older sessions)
    confidence_level: float = 0.95
    significant_digits: int = 4
    create_heatmap: bool = False  # matrix mode only - adds a heat map chart of the matrix


def _correlate(x: np.ndarray, y: np.ndarray, opt: CorrelationOptions) -> tuple[float, float]:
    if opt.gaussian:
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)
    if opt.tails in ("one", "positive"):  # the hypothesis named in advance: a positive correlation
        p = p / 2 if r > 0 else 1 - p / 2
    elif opt.tails == "negative":
        p = p / 2 if r < 0 else 1 - p / 2
    return float(r), float(p)


def _fisher_ci(r: float, n: int, confidence_level: float) -> tuple[float, float]:
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    half = stats.norm.ppf(0.5 + confidence_level / 2) * se
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def _fmt(v: float, sig: int) -> str:
    return f"{v:.{sig}g}"


@register("correlation_analysis")
def run(data: SheetData, options: CorrelationOptions | None = None) -> AnalysisResult:
    opt = options or CorrelationOptions()
    labels, matrix = data.full_variable_matrix()
    if len(labels) < 2:
        raise AnalysisError("Correlation needs at least 2 variables (X plus a group column).")
    sig = opt.significant_digits
    method_name = "Pearson r" if opt.gaussian else "Spearman ρ"

    if opt.mode == "matrix":
        y_labels = labels[1:]
        y_only = matrix[:, 1:]
        if opt.remove_incomplete_rows:
            y_only = y_only[~np.any(np.isnan(y_only), axis=1)]
        if y_only.shape[0] < 4 or len(y_labels) < 2:
            raise AnalysisError("A correlation matrix needs at least 2 Y data sets and 4 complete rows.")
        n = len(y_labels)
        rows = []
        r_matrix = np.eye(n)
        for i in range(n):
            row = [y_labels[i]]
            for j in range(n):
                pair = y_only[:, [i, j]]
                pair = pair[~np.any(np.isnan(pair), axis=1)]
                r, _p = _correlate(pair[:, 0], pair[:, 1], opt) if pair.shape[0] >= 4 else (float("nan"), float("nan"))
                row.append(_fmt(r, sig) if i != j else "1")
                if i != j:
                    r_matrix[i, j] = r
            rows.append(row)
        table = ResultTable(title=f"{method_name} matrix", headers=["Data set"] + y_labels, rows=rows)
        summary = f"{n}x{n} correlation matrix across {y_only.shape[0]} rows."
        charts = []
        if opt.create_heatmap:
            from ..chart_helpers import heatmap_chart

            charts.append(heatmap_chart(f"{method_name} matrix", r_matrix, y_labels, y_labels, cmap="RdBu_r", color_limits=(-1.0, 1.0)))
        return AnalysisResult(title="Two variables together", summary=summary, tables=[table], charts=charts)

    if opt.mode == "two_selected":
        i = max(0, min(opt.first_var_index, len(labels) - 1))
        j = max(0, min(opt.second_var_index, len(labels) - 1))
        if i == j:
            raise AnalysisError("Choose two different data sets to correlate.")
        pairs = [(i, j)]
    else:  # x_vs_every_y
        pairs = [(0, j) for j in range(1, len(labels))]

    rows = []
    best = None
    for i, j in pairs:
        pair_data = matrix[:, [i, j]]
        complete = pair_data[~np.any(np.isnan(pair_data), axis=1)]
        if complete.shape[0] < 4:
            rows.append([f"{labels[i]} vs {labels[j]}", "-", "-", "-", "too few complete rows"])
            continue
        r, p = _correlate(complete[:, 0], complete[:, 1], opt)
        lo, hi = _fisher_ci(r, complete.shape[0], opt.confidence_level)
        rows.append([f"{labels[i]} vs {labels[j]}", str(complete.shape[0]), _fmt(r, sig), f"{_fmt(lo, sig)} to {_fmt(hi, sig)}", f"{p:.4g}"])
        if best is None or abs(r) > abs(best[1]):
            best = (f"{labels[i]} vs {labels[j]}", r, p)

    if best is None:
        raise AnalysisError("No pair had at least 4 complete rows to correlate.")

    pct = int(round(opt.confidence_level * 100))
    table = ResultTable(
        title=f"{method_name}, " + {"two": "two-sided", "negative": "one-sided (negative)"}.get(opt.tails, "one-sided (positive)"),
        headers=["Pair", "N", method_name, f"{pct}% CI", "P value"],
        rows=rows,
    )
    summary = f"{best[0]}: {method_name} = {_fmt(best[1], sig)}, p = {best[2]:.4g}."

    return AnalysisResult(title="Two variables together", summary=summary, tables=[table])
