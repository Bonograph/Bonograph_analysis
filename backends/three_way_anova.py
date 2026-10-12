"""Three-Way ANOVA (or Mixed-Effects Model) - tests three factors, their
three pairwise interactions, and the three-way interaction.

Like two_way_anova.py, a factor has to be encoded in each group's title
using a delimiter, but with three parts this time, e.g.
"Male-Low-Fasted", "Female-High-Fed" (splits into Sex x Dose x Feeding).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

_DELIMITERS = ["-", "_", "|", ":", "/"]


@dataclass
class ThreeWayAnovaOptions:
    delimiter: str = "auto"


def _split_labels(labels: list[str], delimiter: str) -> tuple[str, list[tuple[str, str, str]]] | None:
    candidates = _DELIMITERS if delimiter == "auto" else [delimiter]
    for d in candidates:
        parts = [label.split(d) for label in labels]
        if all(len(p) == 3 and all(s.strip() for s in p) for p in parts):
            return d, [tuple(s.strip() for s in p) for p in parts]
    return None


@register("three_way_anova")
def run(data: SheetData, options: ThreeWayAnovaOptions | None = None) -> AnalysisResult:
    opt = options or ThreeWayAnovaOptions()
    labels = data.group_labels()
    if len(labels) < 8:
        raise AnalysisError(
            "Three-way ANOVA needs at least 8 groups (a 2x2x2 design at "
            f"minimum); this sheet has {len(labels)}."
        )

    split = _split_labels(labels, opt.delimiter)
    if split is None:
        raise AnalysisError(
            "Couldn't find three factors in the group titles. Name each group "
            "'<FactorA><d><FactorB><d>Factor C>' (e.g. 'Male-Low-Fasted') using "
            "a consistent delimiter (-, _, |, : or /), then try again."
        )
    delimiter, triples = split
    a_levels = sorted({t[0] for t in triples})
    b_levels = sorted({t[1] for t in triples})
    c_levels = sorted({t[2] for t in triples})
    if min(len(a_levels), len(b_levels), len(c_levels)) < 2:
        raise AnalysisError(
            f"Split on '{delimiter}' found {len(a_levels)}, {len(b_levels)}, "
            f"{len(c_levels)} level(s) for the three factors - need 2+ of each."
        )

    cells: dict[tuple[str, str, str], np.ndarray] = {}
    for label, key in zip(labels, triples):
        values = data.group_values(label)
        if values.size:
            cells[key] = values

    missing = [
        f"{a}/{b}/{c}" for a in a_levels for b in b_levels for c in c_levels if (a, b, c) not in cells
    ]
    if missing:
        raise AnalysisError(
            "Three-way ANOVA needs every combination of the three factors to "
            f"have data; missing: {', '.join(missing)}."
        )

    cell_mean = {k: float(np.mean(v)) for k, v in cells.items()}
    cell_n = {k: v.size for k, v in cells.items()}
    n_h = len(cells) / sum(1 / n for n in cell_n.values())
    grand = float(np.mean(list(cell_mean.values())))

    mA = {a: np.mean([cell_mean[(a, b, c)] for b in b_levels for c in c_levels]) for a in a_levels}
    mB = {b: np.mean([cell_mean[(a, b, c)] for a in a_levels for c in c_levels]) for b in b_levels}
    mC = {c: np.mean([cell_mean[(a, b, c)] for a in a_levels for b in b_levels]) for c in c_levels}
    mAB = {(a, b): np.mean([cell_mean[(a, b, c)] for c in c_levels]) for a in a_levels for b in b_levels}
    mAC = {(a, c): np.mean([cell_mean[(a, b, c)] for b in b_levels]) for a in a_levels for c in c_levels}
    mBC = {(b, c): np.mean([cell_mean[(a, b, c)] for a in a_levels]) for b in b_levels for c in c_levels}

    a_n, b_n, c_n = len(a_levels), len(b_levels), len(c_levels)
    ss_a = b_n * c_n * n_h * sum((mA[a] - grand) ** 2 for a in a_levels)
    ss_b = a_n * c_n * n_h * sum((mB[b] - grand) ** 2 for b in b_levels)
    ss_c = a_n * b_n * n_h * sum((mC[c] - grand) ** 2 for c in c_levels)
    ss_ab = c_n * n_h * sum((mAB[(a, b)] - mA[a] - mB[b] + grand) ** 2 for a in a_levels for b in b_levels)
    ss_ac = b_n * n_h * sum((mAC[(a, c)] - mA[a] - mC[c] + grand) ** 2 for a in a_levels for c in c_levels)
    ss_bc = a_n * n_h * sum((mBC[(b, c)] - mB[b] - mC[c] + grand) ** 2 for b in b_levels for c in c_levels)
    ss_abc = n_h * sum(
        (cell_mean[(a, b, c)] - mAB[(a, b)] - mAC[(a, c)] - mBC[(b, c)] + mA[a] + mB[b] + mC[c] - grand) ** 2
        for a, b, c in product(a_levels, b_levels, c_levels)
    )
    ss_error = sum(float(np.sum((v - cell_mean[k]) ** 2)) for k, v in cells.items())
    df_error = sum(n - 1 for n in cell_n.values())
    if df_error < 1:
        raise AnalysisError("Need at least one cell with 2+ replicates to estimate error variance.")
    ms_error = ss_error / df_error

    def row(name, ss, df):
        ms = ss / df
        f = ms / ms_error if ms_error else float("inf")
        p = float(stats.f.sf(f, df, df_error))
        return [name, f"{ss:.4g}", str(df), f"{ms:.4g}", f"{f:.4f}", f"{p:.4g}"], p

    terms = [
        ("Factor A", ss_a, a_n - 1),
        ("Factor B", ss_b, b_n - 1),
        ("Factor C", ss_c, c_n - 1),
        ("A × B", ss_ab, (a_n - 1) * (b_n - 1)),
        ("A × C", ss_ac, (a_n - 1) * (c_n - 1)),
        ("B × C", ss_bc, (b_n - 1) * (c_n - 1)),
        ("A × B × C", ss_abc, (a_n - 1) * (b_n - 1) * (c_n - 1)),
    ]
    rows, p_values = [], {}
    for name, ss, df in terms:
        r, p = row(name, ss, df)
        rows.append(r)
        p_values[name] = p
    rows.append(["Residual", f"{ss_error:.4g}", str(df_error), f"{ms_error:.4g}", "-", "-"])

    anova_table = ResultTable(
        title="Three-way ANOVA (unweighted-means method)",
        headers=["Source", "SS", "df", "MS", "F", "P value"],
        rows=rows,
    )
    cell_table = ResultTable(
        title="Cell means",
        headers=["Factor A", "Factor B", "Factor C", "N", "Mean"],
        rows=[
            [a, b, c, str(cell_n[(a, b, c)]), f"{cell_mean[(a, b, c)]:.4g}"]
            for a, b, c in product(a_levels, b_levels, c_levels)
        ],
    )

    summary = (
        f"A p = {p_values['Factor A']:.4g}, B p = {p_values['Factor B']:.4g}, "
        f"C p = {p_values['Factor C']:.4g}, A×B×C p = {p_values['A × B × C']:.4g} "
        f"(factors split on '{delimiter}')."
    )
    notes = [
        f"Group titles were split on '{delimiter}' into three factors: "
        f"{{{', '.join(a_levels)}}} × {{{', '.join(b_levels)}}} × {{{', '.join(c_levels)}}}.",
        "Uses the unweighted-means method (harmonic mean cell size); true "
        "mixed-effects modeling for repeated measures isn't implemented.",
    ]

    return AnalysisResult(
        title="Three-Way ANOVA", summary=summary, tables=[cell_table, anova_table], notes=notes
    )
