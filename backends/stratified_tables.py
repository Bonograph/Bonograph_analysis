"""Is a link between two categories still there once a third is allowed for?

A link seen in a table that pools everybody can be created, hidden or even
reversed by a third factor - the clinic, the sex, the age band. Here the
table is kept apart for each level of that factor (each "layer") and the
layers are combined without mixing them.

Two ways to lay the data out:

- one row per layer and four data sets of counts - the four cells of that
  layer's 2×2 table, in the order row 1 & column 1, row 1 & column 2,
  row 2 & column 1, row 2 & column 2;
- one row per subject and three data sets of codes - the row factor, the
  column factor and the layer. Tables larger than 2×2 are welcome here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .. import contingency_stats as cs
from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register

MOST_LEVELS = 10      # categories of the row or the column factor
MOST_LAYERS = 60
EXACT_LAYERS = 10     # up to this many layers, and
EXACT_SUBJECTS = 200  # up to this many subjects in a table, its odds ratio gets an exact interval (slow beyond that)
SEPARATORS = (": ", " - ", " – ", ", ", " / ", " | ", ":", "/", "|")


@dataclass
class StratifiedTablesOptions:
    layout: str = "auto"              # "auto" | "counts" | "codes"
    row_index: int = 0                # "codes": the data sets holding the row factor, the column factor and the layer
    column_index: int = 1
    layer_index: int = 2
    continuity: bool = True           # 2×2: the Cochran-Mantel-Haenszel test with the continuity correction
    show_each_layer: bool = True
    show_common_risk: bool = False    # 2×2: pooled relative risk and risk difference as well as the odds ratio
    show_homogeneity: bool = True     # 2×2: is the odds ratio the same in every layer?
    show_pooled_table: bool = True    # what ignoring the layers would have given
    graph_forest: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


# ── 2×2×K ───────────────────────────────────────────────────────────────────

def cmh_test(a, b, c, d, continuity: bool = True) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel chi-square (1 df) and its P value for K 2×2 tables."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    N = a + b + c + d
    ok = N > 1
    a, b, c, d, N = a[ok], b[ok], c[ok], d[ok], N[ok]
    expected = (a + b) * (a + c) / N
    variance = (a + b) * (c + d) * (a + c) * (b + d) / (N**2 * (N - 1))
    total = float(variance.sum())
    if total <= 0:
        return float("nan"), float("nan")
    gap = abs(float((a - expected).sum()))
    if continuity:
        gap = max(gap - 0.5, 0.0)
    stat = gap**2 / total
    return float(stat), float(stats.chi2.sf(stat, 1))


def mh_odds_ratio(a, b, c, d, level: float = 0.95) -> tuple[float, float, float]:
    """Mantel-Haenszel common odds ratio with the Robins-Breslow-Greenland interval."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    N = a + b + c + d
    ok = N > 0
    a, b, c, d, N = a[ok], b[ok], c[ok], d[ok], N[ok]
    R, S = a * d / N, b * c / N
    sum_r, sum_s = float(R.sum()), float(S.sum())
    if sum_r == 0 and sum_s == 0:
        return float("nan"), float("nan"), float("nan")
    if sum_s == 0:
        return math.inf, float("nan"), math.inf
    if sum_r == 0:
        return 0.0, 0.0, float("nan")
    P, Q = (a + d) / N, (b + c) / N
    variance = float((P * R).sum() / (2 * sum_r**2) + (P * S + Q * R).sum() / (2 * sum_r * sum_s) + (Q * S).sum() / (2 * sum_s**2))
    z = stats.norm.ppf(0.5 + level / 2)
    estimate = sum_r / sum_s
    half = z * math.sqrt(variance)
    return estimate, estimate * math.exp(-half), estimate * math.exp(half)


def mh_risk_ratio(a, b, c, d, level: float = 0.95) -> tuple[float, float, float]:
    """Mantel-Haenszel common relative risk (row 1 against row 2) with the Greenland-Robins interval."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    N = a + b + c + d
    ok = N > 0
    a, b, c, d, N = a[ok], b[ok], c[ok], d[ok], N[ok]
    n1, n2 = a + b, c + d
    top, bottom = float((a * n2 / N).sum()), float((c * n1 / N).sum())
    if top == 0 or bottom == 0:
        return (math.inf if bottom == 0 and top > 0 else 0.0 if top == 0 and bottom > 0 else float("nan")), float("nan"), float("nan")
    variance = float(((n1 * n2 * (a + c) - a * c * N) / N**2).sum()) / (top * bottom)
    z = stats.norm.ppf(0.5 + level / 2)
    estimate = top / bottom
    half = z * math.sqrt(max(variance, 0.0))
    return estimate, estimate * math.exp(-half), estimate * math.exp(half)


def mh_risk_difference(a, b, c, d, level: float = 0.95) -> tuple[float, float, float]:
    """Mantel-Haenszel common risk difference (row 1 minus row 2) with the Greenland-Robins interval."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    n1, n2 = a + b, c + d
    N = n1 + n2
    ok = (n1 > 0) & (n2 > 0)
    a, b, c, d, n1, n2, N = a[ok], b[ok], c[ok], d[ok], n1[ok], n2[ok], N[ok]
    weight = float((n1 * n2 / N).sum())
    if weight == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(((a * n2 - c * n1) / N).sum()) / weight
    variance = float(((a * b * n2**3 + c * d * n1**3) / (n1 * n2 * N**2)).sum()) / weight**2
    z = stats.norm.ppf(0.5 + level / 2)
    half = z * math.sqrt(max(variance, 0.0))
    return estimate, estimate - half, estimate + half


def breslow_day(a, b, c, d, common: float) -> dict:
    """Breslow and Day's test that every layer has the same odds ratio, and
    Tarone's corrected form of it. Layers with an empty row or column say
    nothing about an odds ratio and are left out."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    n1, n2, m1, m2 = a + b, c + d, a + c, b + d
    ok = (n1 > 0) & (n2 > 0) & (m1 > 0) & (m2 > 0)
    a, n1, n2, m1 = a[ok], n1[ok], n2[ok], m1[ok]
    used = int(ok.sum())
    if used < 2 or not np.isfinite(common) or common <= 0:
        return {"df": max(used - 1, 0), "bd": float("nan"), "tarone": float("nan"), "p_bd": float("nan"), "p_tarone": float("nan"), "used": used}
    if abs(common - 1) < 1e-12:
        fitted = n1 * m1 / (n1 + n2)
    else:                               # the cell count that gives exactly the common odds ratio with these margins
        qa = 1 - common
        qb = n2 - m1 + common * (n1 + m1)
        qc = -common * n1 * m1
        root = np.sqrt(np.clip(qb**2 - 4 * qa * qc, 0, None))
        first, second = (-qb + root) / (2 * qa), (-qb - root) / (2 * qa)
        low, high = np.maximum(0, m1 - n2), np.minimum(n1, m1)
        fitted = np.where((first >= low - 1e-9) & (first <= high + 1e-9), first, second)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = 1 / (1 / fitted + 1 / (n1 - fitted) + 1 / (m1 - fitted) + 1 / (n2 - m1 + fitted))
    fine = np.isfinite(variance) & (variance > 0)
    if fine.sum() < 2:
        return {"df": max(used - 1, 0), "bd": float("nan"), "tarone": float("nan"), "p_bd": float("nan"), "p_tarone": float("nan"), "used": used}
    bd = float(np.sum((a[fine] - fitted[fine]) ** 2 / variance[fine]))
    tarone = bd - float((a[fine] - fitted[fine]).sum()) ** 2 / float(variance[fine].sum())
    df = int(fine.sum()) - 1
    return {"df": df, "bd": bd, "tarone": float(tarone), "p_bd": float(stats.chi2.sf(bd, df)), "p_tarone": float(stats.chi2.sf(tarone, df)),
            "used": int(fine.sum())}


def log_odds_interval(a: float, b: float, c: float, d: float, level: float) -> tuple[float, float]:
    """The usual large-sample interval for an odds ratio (Woolf); 0.5 is added to every cell when one is empty."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    half = stats.norm.ppf(0.5 + level / 2) * math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    centre = math.log(a * d / (b * c))
    return math.exp(centre - half), math.exp(centre + half)


def woolf_test(a, b, c, d) -> tuple[float, int, float, bool]:
    """Woolf's test that the odds ratio is the same in every layer:
    (chi-square, df, P, whether 0.5 was added to every cell because one was empty)."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    ok = (a + b > 0) & (c + d > 0) & (a + c > 0) & (b + d > 0)
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    if a.size < 2:
        return float("nan"), 0, float("nan"), False
    padded = bool(np.any(np.column_stack([a, b, c, d]) == 0))
    if padded:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log(a * d / (b * c))
    weight = 1 / (1 / a + 1 / b + 1 / c + 1 / d)
    centre = float((weight * log_or).sum() / weight.sum())
    stat = float((weight * (log_or - centre) ** 2).sum())
    return stat, a.size - 1, float(stats.chi2.sf(stat, a.size - 1)), padded


# ── any size ────────────────────────────────────────────────────────────────

def general_cmh(tables: np.ndarray) -> tuple[float, int, float]:
    """The Cochran-Mantel-Haenszel test of general association for K tables
    of any size (layers x rows x columns): (chi-square, df, P). For 2×2
    tables it is the test above without the continuity correction."""
    K, R, C = tables.shape
    q = (R - 1) * (C - 1)
    gap, V = np.zeros(q), np.zeros((q, q))
    for t in tables:
        N = t.sum()
        if N < 2:
            continue
        rows, cols = t.sum(axis=1), t.sum(axis=0)
        gap += (t - np.outer(rows, cols) / N)[: R - 1, : C - 1].ravel()
        row_part = (np.diag(rows) * N - np.outer(rows, rows))[: R - 1, : R - 1]
        col_part = (np.diag(cols) * N - np.outer(cols, cols))[: C - 1, : C - 1]
        V += np.kron(row_part, col_part) / (N**2 * (N - 1))
    rank = int(np.linalg.matrix_rank(V))
    if rank == 0:
        return float("nan"), 0, float("nan")
    stat = float(gap @ np.linalg.pinv(V) @ gap)
    return stat, rank, float(stats.chi2.sf(stat, rank))


def _split_titles(titles: list[str]):
    """Four data-set titles of the form "Drug: improved", "Drug: not",
    "Placebo: improved", "Placebo: not" give the names of the rows and the
    columns; anything else gives None."""
    for sep in SEPARATORS:
        parts = [t.split(sep, 1) for t in titles]
        if all(len(pp) == 2 and pp[0].strip() and pp[1].strip() for pp in parts):
            rows = [parts[0][0].strip(), parts[2][0].strip()]
            cols = [parts[0][1].strip(), parts[1][1].strip()]
            if (parts[1][0].strip() == rows[0] and parts[3][0].strip() == rows[1] and parts[2][1].strip() == cols[0]
                    and parts[3][1].strip() == cols[1] and rows[0] != rows[1] and cols[0] != cols[1]):
                return rows, cols
    return None


def _read_counts(data: SheetData):
    names, matrix = data.first_replicate_matrix()
    if len(names) < 4:
        raise AnalysisError("For counts, enter four data sets - the four cells of each layer's 2×2 table - with one layer on each row.")
    block = matrix[:, :4]
    filled = ~np.any(np.isnan(block), axis=1)
    block = block[filled]
    if block.shape[0] < 1:
        raise AnalysisError("No row has all four counts filled in.")
    if block.shape[0] > MOST_LAYERS:
        raise AnalysisError(f"There are {block.shape[0]} rows, and each row is read as one layer - too many (up to {MOST_LAYERS}). If each "
                            "row is a subject, choose 'A subject on each row' and say which data sets hold the row factor, the "
                            "column factor and the layer.")
    if np.any(block < 0):
        raise AnalysisError("Counts cannot be negative - check the sheet's values.")
    typed = list(data.row_names or [])
    layers = []
    for i in np.where(filled)[0]:
        title = typed[i].strip() if i < len(typed) and typed[i] else ""
        layers.append(title or f"Layer {len(layers) + 1}")
    split = _split_titles(names[:4])
    rows, cols = split if split else (["Row 1", "Row 2"], ["Column 1", "Column 2"])
    tables = block.reshape(-1, 2, 2)
    mapping = None if split else names[:4]
    return tables, layers, rows, cols, "the layers", "", "", mapping


def _read_codes(data: SheetData, opt: StratifiedTablesOptions):
    names, matrix = data.first_replicate_matrix()
    if len(names) < 3:
        raise AnalysisError("For one subject per row, enter three data sets of codes: the row factor, the column factor and the layer.")
    picks = [opt.row_index, opt.column_index, opt.layer_index]
    if any(not 0 <= i < len(names) for i in picks) or len(set(picks)) < 3:
        picks = [0, 1, 2]
    block = matrix[:, picks]
    filled = ~np.any(np.isnan(block), axis=1)
    block = block[filled]
    if block.shape[0] < 4:
        raise AnalysisError("Fewer than 4 rows have the row factor, the column factor and the layer all filled in.")
    levels = [np.unique(block[:, j]) for j in range(3)]
    titles = [names[i] for i in picks]
    for j, what in enumerate(("row factor", "column factor")):
        if levels[j].size > MOST_LEVELS:
            raise AnalysisError(f"'{titles[j]}', the {what}, has {levels[j].size} distinct values - too many to be categories "
                                f"(up to {MOST_LEVELS}).")
        if levels[j].size < 2:
            raise AnalysisError(f"'{titles[j]}', the {what}, has only one value, so there is nothing to compare.")
    if levels[2].size > MOST_LAYERS:
        raise AnalysisError(f"'{titles[2]}', the layer, has {levels[2].size} distinct values - too many to be layers (up to {MOST_LAYERS}).")
    index = [np.searchsorted(levels[j], block[:, j]) for j in range(3)]
    tables = np.zeros((levels[2].size, levels[0].size, levels[1].size))
    np.add.at(tables, (index[2], index[0], index[1]), 1)
    return (tables, [f"{titles[2]} = {v:g}" for v in levels[2]], [f"{titles[0]} = {v:g}" for v in levels[0]],
            [f"{titles[1]} = {v:g}" for v in levels[1]], titles[2], titles[0], titles[1], None)


def _looks_like_counts(data: SheetData) -> bool:
    """Four data sets on a modest number of rows are tables of counts, one
    layer per row. More rows than there could be layers means one subject
    per row."""
    names, matrix = data.first_replicate_matrix()
    if len(names) < 4:
        return False
    block = matrix[:, :4]
    block = block[~np.any(np.isnan(block), axis=1)]
    return bool(block.size and block.shape[0] <= MOST_LAYERS and np.all(block >= 0))


@register("stratified_tables")
def run(data: SheetData, options: StratifiedTablesOptions | None = None) -> AnalysisResult:
    opt = options or StratifiedTablesOptions()
    layout = opt.layout if opt.layout in ("counts", "codes") else ("counts" if _looks_like_counts(data) else "codes")
    tables, layers, row_names, col_names, layer_title, row_title, col_title, mapping = (
        _read_counts(data) if layout == "counts" else _read_codes(data, opt))
    K, R, C = tables.shape
    sig, level = opt.significant_digits, opt.confidence_level
    pct = f"{round(level * 100)}%"

    def num(v) -> str:
        if v != v:
            return "-"
        return "infinite" if math.isinf(v) else f"{v:.{sig}g}"

    def count(v) -> str:
        return f"{v:g}"

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    def between(lo, hi) -> str:
        return f"{num(lo)} to {num(hi)}" if lo == lo and hi == hi else "-"

    totals = tables.sum(axis=(1, 2))
    if totals.sum() < 4:
        raise AnalysisError("There are too few subjects in the tables to analyse.")
    pooled = tables.sum(axis=0)
    if np.any(pooled.sum(axis=0) == 0) or np.any(pooled.sum(axis=1) == 0):
        raise AnalysisError("A row or a column of the table is empty in every layer, so there is nothing to compare.")
    whole = cs.is_whole(*tables.ravel())
    notes: list[str] = []
    results: list[ResultTable] = []
    charts: list[ChartSpec] = []
    of_layers = f"{K} layer{'s' if K != 1 else ''}"

    if R == 2 and C == 2:
        a, b, c, d = tables[:, 0, 0], tables[:, 0, 1], tables[:, 1, 0], tables[:, 1, 1]
        stat, p_cmh = cmh_test(a, b, c, d, opt.continuity)
        if stat != stat:
            raise AnalysisError("No layer has subjects in both rows and both columns, so the layers cannot be combined.")
        common, lo, hi = mh_odds_ratio(a, b, c, d, level)
        test_name = "Cochran-Mantel-Haenszel test" + (" with continuity correction" if opt.continuity else "")
        results.append(ResultTable(f"Are {row_names[0]} and {col_names[0]} linked, allowing for {layer_title}?",
                                   ["Test", "Chi-square", "df", "P value"], [[test_name, f"{stat:.4f}", "1", p_text(p_cmh)]]))
        size = [[f"Odds ratio common to the layers (Mantel-Haenszel): odds of {col_names[0]} in {row_names[0]} against {row_names[1]}",
                 num(common), between(lo, hi), "Robins-Breslow-Greenland"]]
        if opt.show_common_risk:
            rr, rr_lo, rr_hi = mh_risk_ratio(a, b, c, d, level)
            rd, rd_lo, rd_hi = mh_risk_difference(a, b, c, d, level)
            size.append([f"Relative risk common to the layers: risk of {col_names[0]} in {row_names[0]} against {row_names[1]}",
                         num(rr), between(rr_lo, rr_hi), "Greenland-Robins"])
            size.append([f"Risk difference common to the layers ({row_names[0]} minus {row_names[1]}), percentage points",
                         f"{rd * 100:+.2f}" if rd == rd else "-", f"{rd_lo * 100:+.2f} to {rd_hi * 100:+.2f}" if rd_lo == rd_lo else "-",
                         "Greenland-Robins"])
        crude = (pooled[0, 0] * pooled[1, 1] / (pooled[0, 1] * pooled[1, 0])) if pooled[0, 1] * pooled[1, 0] > 0 else (
            math.inf if pooled[0, 0] * pooled[1, 1] > 0 else float("nan"))
        if opt.show_pooled_table:
            if whole and pooled.sum() <= EXACT_SUBJECTS:
                c_lo, c_hi, _exact = cs.baptista_pike_or_ci(*pooled.ravel(), level)
                by = "Baptista-Pike"
            else:
                c_lo, c_hi = log_odds_interval(*pooled.ravel(), level)
                by = "log odds (Woolf)"
            size.append(["Odds ratio if the layers are ignored (everybody in one table)", num(crude), between(c_lo, c_hi), by])
        results.append(ResultTable("Size of the link", ["Measure", "Estimate", f"{pct} CI", "Interval by"], size))

        same_text = ""
        if opt.show_homogeneity and K >= 2:
            bd = breslow_day(a, b, c, d, common)
            w_stat, w_df, w_p, padded = woolf_test(a, b, c, d)
            rows = []
            if bd["bd"] == bd["bd"]:
                rows.append(["Breslow-Day, with Tarone's correction", f"{bd['tarone']:.4f}", str(bd["df"]), p_text(bd["p_tarone"])])
                rows.append(["Breslow-Day, uncorrected", f"{bd['bd']:.4f}", str(bd["df"]), p_text(bd["p_bd"])])
            if w_stat == w_stat:
                rows.append(["Woolf" + (" (0.5 added to every cell: one was empty)" if padded else ""), f"{w_stat:.4f}", str(w_df), p_text(w_p)])
            if rows:
                results.append(ResultTable("Is the odds ratio the same in every layer?", ["Test", "Chi-square", "df", "P value"], rows))
                lead = bd["p_tarone"] if bd["bd"] == bd["bd"] else w_p
                if lead < opt.alpha:
                    same_text = (f" Caution: the odds ratio differs from layer to layer (P = {lead:.4g}), so a single common value "
                                 "hides part of the story - read the layers one by one.")
                    notes.append("When the odds ratio is not the same in every layer, the layer changes the size (or the direction) "
                                 "of the link itself - an interaction. The common odds ratio is then an average over layers that "
                                 "disagree; report the layers separately.")
                else:
                    same_text = f" The odds ratio is much the same in every layer (P = {lead:.4g})."
            if bd["used"] < K:
                notes.append(f"{K - bd['used']} layer{'s' if K - bd['used'] != 1 else ''} with an empty row or column cannot show an "
                             "odds ratio and take no part in the test of sameness.")

        each_or, each_lo, each_hi = [], [], []
        layer_rows = []
        exact_used = 0
        for k in range(K):
            ak, bk, ck, dk = a[k], b[k], c[k], d[k]
            ratio = (ak * dk / (bk * ck)) if bk * ck > 0 else (math.inf if ak * dk > 0 else float("nan"))
            informative = (ak + bk > 0) and (ck + dk > 0) and (ak + ck > 0) and (bk + dk > 0)
            if informative and whole and K <= EXACT_LAYERS and totals[k] <= EXACT_SUBJECTS:
                k_lo, k_hi, _exact = cs.baptista_pike_or_ci(ak, bk, ck, dk, level)
                exact_used += 1
            elif informative:
                k_lo, k_hi = log_odds_interval(ak, bk, ck, dk, level)
            else:
                k_lo = k_hi = float("nan")
            if informative and whole and totals[k] <= 1_000_000:
                p_k = float(stats.fisher_exact(np.array([[ak, bk], [ck, dk]]).astype(int))[1])
            elif informative:
                p_k = float(stats.chi2_contingency(np.array([[ak, bk], [ck, dk]]), correction=False)[1])
            else:
                p_k = float("nan")
            each_or.append(ratio)
            each_lo.append(k_lo)
            each_hi.append(k_hi)
            risk1 = f"{ak / (ak + bk):.1%}" if ak + bk > 0 else "-"
            risk2 = f"{ck / (ck + dk):.1%}" if ck + dk > 0 else "-"
            layer_rows.append([layers[k], count(ak), count(bk), count(ck), count(dk), risk1, risk2, num(ratio), between(k_lo, k_hi),
                               p_text(p_k) if p_k == p_k else "-"])
        if opt.show_each_layer:
            layer_rows.append(["All layers pooled", *[count(v) for v in pooled.ravel()],
                               f"{pooled[0, 0] / pooled[0].sum():.1%}", f"{pooled[1, 0] / pooled[1].sum():.1%}", num(crude), "", ""])
            results.append(ResultTable("Layer by layer", [
                "Layer", f"{row_names[0]}, {col_names[0]}", f"{row_names[0]}, {col_names[1]}", f"{row_names[1]}, {col_names[0]}",
                f"{row_names[1]}, {col_names[1]}", f"{col_names[0]} in {row_names[0]}", f"{col_names[0]} in {row_names[1]}",
                "Odds ratio", f"{pct} CI", "P value"], layer_rows))

        if opt.graph_forest:
            xs, ys, below, above = [], [], [], []
            for k in range(K):
                if each_or[k] == each_or[k] and 0 < each_or[k] < math.inf:
                    xs.append(float(k))
                    ys.append(float(each_or[k]))
                    lo_k = each_lo[k] if each_lo[k] == each_lo[k] and each_lo[k] > 0 else each_or[k]
                    hi_k = each_hi[k] if each_hi[k] == each_hi[k] and each_hi[k] < math.inf else each_or[k]
                    below.append(float(each_or[k] - lo_k))
                    above.append(float(hi_k - each_or[k]))
            series = []
            if xs:
                series.append(ChartSeries("Each layer", xs, ys, y_err=[below, above], color="#1f5aa6"))
            if 0 < common < math.inf and lo == lo and hi == hi:
                series.append(ChartSeries("All layers (Mantel-Haenszel)", [float(K)], [float(common)],
                                          y_err=[[float(common - lo)], [float(hi - common)]], color="#d1342f"))
            if series:
                charts.append(ChartSpec("Odds ratio in each layer, and common to all", x_label="", y_label=f"Odds ratio ({pct} CI)",
                                        series=series, x_categories=[*layers, "All layers"], y_log=True,
                                        reference_lines=[{"kind": "h", "value": 1.0}]))

        way = "" if common != common or common in (0.0, math.inf) else (
            f"the odds of {col_names[0]} are {common:.{sig}g} times as high in {row_names[0]} as in {row_names[1]} "
            f"({pct} CI {num(lo)} to {num(hi)})")
        summary = (f"Allowing for {layer_title} ({of_layers}, {count(totals.sum())} subjects), {row_names[0]} and {col_names[0]} are "
                   f"{'linked' if p_cmh < opt.alpha else 'not clearly linked'}: {test_name}, chi-square(1) = {stat:.3f}, P = {p_cmh:.4g}"
                   + (f"; {way}." if way else ".") + same_text)
        if crude == crude and common == common and 0 < crude < math.inf and 0 < common < math.inf:
            shift = abs(math.log(crude / common))
            if (crude - 1) * (common - 1) < 0 and shift > 0.1:
                summary += (f" Ignoring the layers would give an odds ratio of {crude:.{sig}g} - on the other side of 1 (Simpson's "
                            "paradox).")
                notes.append("The pooled table points the other way from the layers. That happens when the layers differ both in "
                             "how common the column outcome is and in how the rows are spread across them; the layered answer is "
                             "the one that compares like with like.")
            elif shift > math.log(1.1):
                summary += f" Ignoring the layers would give {crude:.{sig}g}, so the layers matter."
        notes.append("The Cochran-Mantel-Haenszel test asks whether the rows and columns are linked within the layers, never comparing "
                     "a subject in one layer with a subject in another. It is at its best when the link points the same way in every "
                     "layer; where layers pull in opposite directions they cancel, and the test of sameness shows it.")
        notes.append("The common odds ratio weights each layer by its size (Mantel and Haenszel 1959) and stays valid when layers are "
                     "many and small, as in a matched study; its interval is that of Robins, Breslow and Greenland 1986.")
        notes.append("Intervals for single layers: " + (
            "exact (Baptista-Pike)." if exact_used == K else
            "the usual log-odds interval (Woolf), with 0.5 added to every cell of a layer that has an empty one."
            if exact_used == 0 else
            f"exact (Baptista-Pike) for layers of up to {EXACT_SUBJECTS} subjects, the usual log-odds interval (Woolf) for the rest.")
                     + " P values for single layers are Fisher's exact test when the counts are whole.")
    else:
        stat, df, p_cmh = general_cmh(tables)
        if stat != stat:
            raise AnalysisError("No layer has enough subjects spread over the rows and columns to test.")
        results.append(ResultTable(f"Are {row_title or 'the rows'} and {col_title or 'the columns'} linked, allowing for {layer_title}?",
                                   ["Test", "Chi-square", "df", "P value"],
                                   [["Cochran-Mantel-Haenszel test of general association", f"{stat:.4f}", str(df), p_text(p_cmh)]]))
        layer_rows = []
        for k in range(K):
            t = tables[k]
            t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
            if t.shape[0] >= 2 and t.shape[1] >= 2:
                chi2, pv, dof, _e = stats.chi2_contingency(t, correction=False)
                v = math.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))
                layer_rows.append([layers[k], count(totals[k]), f"{chi2:.3f}", str(dof), p_text(float(pv)), f"{v:.3f}"])
            else:
                layer_rows.append([layers[k], count(totals[k]), "-", "-", "-", "-"])
        chi2, pv, dof, _e = stats.chi2_contingency(pooled, correction=False)
        v_all = math.sqrt(chi2 / (pooled.sum() * (min(pooled.shape) - 1)))
        if opt.show_each_layer:
            if opt.show_pooled_table:
                layer_rows.append(["All layers pooled (layers ignored)", count(pooled.sum()), f"{chi2:.3f}", str(dof), p_text(float(pv)), f"{v_all:.3f}"])
            results.append(ResultTable("Layer by layer (chi-square test in each)",
                                       ["Layer", "Subjects", "Chi-square", "df", "P value", "Cramér's V"], layer_rows))
            for k in range(min(K, 12)):
                results.append(ResultTable(f"Counts: {layers[k]}", ["", *col_names, "Total"],
                                           [[row_names[i], *[count(v) for v in tables[k, i]], count(tables[k, i].sum())] for i in range(R)]))
            if K > 12:
                notes.append(f"The counts of the first 12 of {K} layers are shown.")
        summary = (f"Allowing for {layer_title} ({of_layers}, {count(totals.sum())} subjects), {row_title or 'the rows'} and "
                   f"{col_title or 'the columns'} are {'linked' if p_cmh < opt.alpha else 'not clearly linked'}: Cochran-Mantel-Haenszel "
                   f"chi-square({df}) = {stat:.3f}, P = {p_cmh:.4g}. With the layers ignored the chi-square test gives P = {float(pv):.4g}.")
        notes.append("The test of general association asks whether rows and columns are linked within the layers, whatever form "
                     "the link takes. It does not use any order the categories may have.")
        if opt.graph_forest:
            xs = [float(k) for k in range(len(layer_rows) - (1 if opt.show_pooled_table and opt.show_each_layer else 0))]
            vs = [float(r[5]) if r[5] != "-" else float("nan") for r in layer_rows[: len(xs)]]
            keep = [i for i, v in enumerate(vs) if v == v]
            if keep:
                charts.append(ChartSpec("Strength of the link in each layer", x_label="", y_label="Cramér's V",
                                        series=[ChartSeries("", [xs[i] for i in keep], [vs[i] for i in keep], kind="bar", color="#1f5aa6")],
                                        x_categories=layers, y_limits=(0.0, 1.0)))

    if mapping:
        notes.insert(0, "The four data sets are read, in order, as the cells of each layer's table: " +
                     "; ".join(f"{t} = {r} & {cc}" for t, (r, cc) in zip(mapping, [(row_names[0], col_names[0]), (row_names[0], col_names[1]),
                                                                              (row_names[1], col_names[0]), (row_names[1], col_names[1])]))
                     + ". Title them like 'Drug: improved', 'Drug: not', 'Placebo: improved', 'Placebo: not' to have those names used.")
    thin = int(np.sum(totals < 5))
    if thin:
        notes.append(f"{thin} layer{'s have' if thin != 1 else ' has'} fewer than 5 subjects. Small layers are handled correctly by "
                     "the combined test, but their own odds ratios and P values mean little.")
    if not whole:
        notes.append("Some counts are not whole numbers; exact intervals and exact tests for single layers were replaced by "
                     "large-sample ones.")
    return AnalysisResult(title="Compare counts within layers", summary=summary, tables=results, notes=notes, charts=charts)
