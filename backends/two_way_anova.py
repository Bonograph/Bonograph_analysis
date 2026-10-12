"""Two-Way ANOVA (or Mixed Model).

Data-model mapping (our sheet has one explicit grouping level - the
sheet's groups/columns - plus, per group, replicate subcolumns and row
position): Factor A ("Column Factor") = the sheet's groups; Factor B
("Row Factor") = row position within each group. Subcolumn replicates
either (a) are ordinary within-cell replication (no matching), or
(b) - if the user says matched values are "stacked into a subcolumn" -
represent the same subject measured once per (row, column) cell, giving
a real two-way repeated-measures design; or (c) - if matched values are
"spread across a row" - each row is one subject, which demotes the row
dimension from a crossed factor to a subject index, so only Factor A is
tested (one-way repeated measures on the column factor).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _common, _ptukey


@dataclass
class TwoWayAnovaOptions:
    # the model
    include_interaction: bool = True
    matching: str = "none"  # "none" | "row_subject" | "subcolumn_subject"
    assume_sphericity: bool = False  # only used when matching != "none"; False = Geisser-Greenhouse correction
    # names of the factors
    column_factor_name: str = "Column Factor"
    row_factor_name: str = "Row Factor"
    subject_name: str = "Subject"
    # pairs to compare
    compare: str = "none"  # "none" | "column_means" | "row_means"
    comparison_scope: str = "all_pairs"  # "all_pairs" | "vs_control"
    control_index: int = 0
    # what to report
    correction: str = "auto"  # "auto" (Tukey) | "holm" | "fdr" | "none"
    swap_direction: bool = False
    show_group_means: bool = False
    show_goodness_of_fit: bool = False
    significant_digits: int = 4
    # checks on the residuals
    show_residual_plot: bool = False
    show_homoscedasticity_plot: bool = False
    show_qq_plot: bool = False
    diag_heteroscedastic: bool = False
    diag_normality: bool = False


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)[::-1]
    m = len(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank
        running_min = min(running_min, p[idx] * m / i)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0, 1)


def _greenhouse_geisser_epsilon(matrix: np.ndarray) -> float:
    """Box/Greenhouse-Geisser epsilon from a subjects x levels matrix -
    how far the within-subject covariance structure departs from
    sphericity (1.0 = perfectly spherical, no correction needed)."""
    k = matrix.shape[1]
    if k < 2 or matrix.shape[0] < 2:
        return 1.0
    cov = np.cov(matrix, rowvar=False, ddof=1)
    row_means = cov.mean(axis=1, keepdims=True)
    col_means = cov.mean(axis=0, keepdims=True)
    grand = cov.mean()
    dc = cov - row_means - col_means + grand
    denom = (k - 1) * np.sum(dc**2)
    if denom <= 0:
        return 1.0
    eps = float(np.sum(np.diag(dc)) ** 2 / denom)
    return min(1.0, max(1.0 / (k - 1), eps))


def _pairwise_holm_fdr(means: dict, ns: dict, pooled_var: float, df_error: int, keys: list, pairs: list[tuple], correction: str, sig: int) -> ResultTable:
    raw_p, rows = [], []
    for i, j in pairs:
        se = np.sqrt(pooled_var * (1 / ns[keys[i]] + 1 / ns[keys[j]]))
        t_stat = (means[keys[i]] - means[keys[j]]) / se if se else 0.0
        p = float(2 * stats.t.sf(abs(t_stat), df_error))
        raw_p.append(p)
        rows.append([f"{keys[i]} vs {keys[j]}", f"{means[keys[i]] - means[keys[j]]:.{sig}g}", f"{p:.4g}"])
    if correction in ("auto", "tukey") and len(keys) >= 2:
        # Tukey-Kramer on the marginal means, from the model's error term
        k = len(keys)
        adjusted = []
        for i, j in pairs:
            se = np.sqrt(pooled_var / 2 * (1 / ns[keys[i]] + 1 / ns[keys[j]]))
            q = abs(means[keys[i]] - means[keys[j]]) / se if se else np.inf
            adjusted.append(float(_ptukey.sf(q, k, df_error)[0]) if np.isfinite(q) else 0.0)
        method_label = "Tukey"
    else:
        adjusted = _common.adjust(raw_p, correction)
        method_label = {"fdr": "Benjamini-Hochberg", "none": "no correction"}.get(correction, "Holm")
    for row, adj in zip(rows, adjusted):
        adj = float(adj)
        row.append(f"{adj:.4g}")
        row.append("Yes" if adj < 0.05 else "No")
    return ResultTable(
        title=f"Which levels differ ({method_label})",
        headers=["Comparison", "Mean difference", "P value (raw)", "P value (adjusted)", "Significant (α = 0.05)"],
        rows=rows,
    )


def _effect_columns(idx: np.ndarray, levels: int) -> np.ndarray:
    """Sum-to-zero (effect) coding: levels - 1 columns, the last level -1."""
    cols = np.zeros((idx.size, levels - 1))
    for j in range(levels - 1):
        cols[:, j] = (idx == j).astype(float) - (idx == levels - 1).astype(float)
    return cols


def _rss(design: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(resid @ resid)


def _additive_rss(sums: np.ndarray, counts: np.ndarray, y: np.ndarray) -> float:
    """Residual SS of the additive model y = α_i + β_j, solved exactly from
    the cell totals: the larger factor is eliminated, leaving a system the
    size of the smaller one - so thousands of row levels stay cheap."""
    flip = sums.shape[0] > sums.shape[1]
    s_, n_ = (sums.T, counts.T) if flip else (sums, counts)
    row_n, col_n = n_.sum(axis=1), n_.sum(axis=0)
    row_s, col_s = s_.sum(axis=1), s_.sum(axis=0)
    w = n_ / col_n[None, :]
    lhs = np.diag(row_n) - w @ n_.T
    rhs = row_s - w @ col_s
    alpha = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    beta = (col_s - n_.T @ alpha) / col_n
    fitted = alpha[:, None] + beta[None, :]
    # RSS = Σy² - 2Σ fitted·S_ij + Σ n_ij fitted²
    total_sq = float(y @ y)
    return total_sq - 2.0 * float(np.sum(fitted * s_)) + float(np.sum(n_ * fitted**2))


def _type3_two_way(y: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, n_a: int, n_b: int,
                   interaction: bool) -> dict:
    """Type III sums of squares - exact for balanced data and still
    correct when cells hold different numbers of values.

    With the interaction, the cell-means form is used: for each effect,
    SS = (Cμ)' (C D C')⁻¹ (Cμ), with μ the cell means, D = diag(1/nᵢⱼ) and C
    that effect's contrasts - small matrices however many values there are.
    Without it, a linear model with effect coding (a + b - 1 columns)."""
    if interaction:
        cells = a_idx * n_b + b_idx
        counts = np.bincount(cells, minlength=n_a * n_b).astype(float)
        sums = np.bincount(cells, weights=y, minlength=n_a * n_b)
        mu = (sums / counts).reshape(n_a, n_b)
        resid = y - mu.ravel()[cells]
        rss = float(resid @ resid)
        d = (1.0 / counts).reshape(n_a, n_b)
        out = {"rss": rss, "df_error": int(y.size - n_a * n_b)}
        # main effects: unweighted marginal means, each against the last level.
        # cov = diag(v[:-1]) + v[-1]·J, inverted in closed form (Sherman-Morrison)
        for name, means, var in (("A", mu.mean(axis=1), d.sum(axis=1) / n_b**2),
                                 ("B", mu.mean(axis=0), d.sum(axis=0) / n_a**2)):
            est = means[:-1] - means[-1]
            inv = 1.0 / var[:-1]
            last = var[-1]
            ss = float(np.sum(est**2 * inv) - last * np.sum(est * inv) ** 2 / (1.0 + last * np.sum(inv)))
            out[name] = (ss, means.size - 1)
        # interaction: what the cell means explain beyond the additive model
        # (for the highest-order term Type III equals this reduction)
        out["AB"] = (max(0.0, _additive_rss(sums.reshape(n_a, n_b), counts.reshape(n_a, n_b), y) - rss),
                     (n_a - 1) * (n_b - 1))
        return out

    # main effects only: each factor's SS is what it adds beyond the other
    # (Type II = Type III here), from cell totals - no design matrix needed
    cells = a_idx * n_b + b_idx
    counts = np.bincount(cells, minlength=n_a * n_b).astype(float).reshape(n_a, n_b)
    sums = np.bincount(cells, weights=y, minlength=n_a * n_b).reshape(n_a, n_b)
    rss_full = _additive_rss(sums, counts, y)
    total_sq = float(y @ y)

    def one_factor_rss(totals, ns):
        keep = ns > 0
        return total_sq - float(np.sum(totals[keep] ** 2 / ns[keep]))

    rss_b_only = one_factor_rss(sums.sum(axis=0), counts.sum(axis=0))
    rss_a_only = one_factor_rss(sums.sum(axis=1), counts.sum(axis=1))
    return {"rss": rss_full, "df_error": int(y.size - (n_a + n_b - 1)),
            "A": (max(0.0, rss_b_only - rss_full), n_a - 1),
            "B": (max(0.0, rss_a_only - rss_full), n_b - 1)}


@register("two_way_anova")
def run(data: SheetData, options: TwoWayAnovaOptions | None = None) -> AnalysisResult:
    opt = options or TwoWayAnovaOptions()
    a_labels = data.group_labels()
    if len(a_labels) < 2:
        raise AnalysisError(f"Two-way ANOVA needs at least 2 groups (Factor A levels); this sheet has {len(a_labels)}.")

    cell_matrices = {label: data.group_replicate_matrix(label)[1] for label in a_labels}
    n_rows = min(m.shape[0] for m in cell_matrices.values() if m.size)
    if n_rows < 2:
        raise AnalysisError(f"Two-way ANOVA needs at least 2 rows (Factor B levels); this sheet has {n_rows}.")
    n_reps = min((m.shape[1] for m in cell_matrices.values() if m.size), default=1)
    names = [str(n).strip() for n in (getattr(data, "row_names", None) or [])][:n_rows]
    b_labels = ([n if n else str(i + 1) for i, n in enumerate(names)] + [str(i + 1) for i in range(len(names), n_rows)])
    if len(set(b_labels)) != len(b_labels):
        b_labels = [str(i + 1) for i in range(n_rows)]
    sig = opt.significant_digits
    col_name, row_name, subj_name = opt.column_factor_name, opt.row_factor_name, opt.subject_name

    matching = opt.matching
    include_interaction = opt.include_interaction
    forced_note = None
    if n_reps < 2 and matching != "none":
        matching = "none"
        forced_note = "Repeated-measures matching needs replicate subcolumns representing subjects; this sheet has none, so an ordinary (non-repeated) analysis was run instead."
    if n_reps < 2 and include_interaction:
        include_interaction = False
        forced_note = (forced_note + " " if forced_note else "") + "With one value per cell there's no residual left to test an interaction against, so only the main-effects model was fit."

    tables: list[ResultTable] = []
    charts = []
    notes: list[str] = []
    if forced_note:
        notes.append(forced_note)

    if matching == "row_subject":
        # Each row is one subject; the row dimension is demoted from a
        # crossed factor to a subject index, so only Factor A is tested,
        # as a one-way repeated-measures ANOVA (subject = row).
        matrix = np.column_stack([np.nanmean(cell_matrices[a][:n_rows, :], axis=1) for a in a_labels])
        grand = float(np.mean(matrix))
        subj_means = np.mean(matrix, axis=1)
        a_means_arr = np.mean(matrix, axis=0)
        ss_a = n_rows * float(np.sum((a_means_arr - grand) ** 2))
        ss_subject = len(a_labels) * float(np.sum((subj_means - grand) ** 2))
        ss_total = float(np.sum((matrix - grand) ** 2))
        ss_error = ss_total - ss_a - ss_subject
        df_a, df_subject = len(a_labels) - 1, n_rows - 1
        df_error = df_a * df_subject
        eps = 1.0 if opt.assume_sphericity else _greenhouse_geisser_epsilon(matrix)
        ms_a, ms_error = ss_a / df_a, ss_error / df_error if df_error else float("nan")
        f_a = ms_a / ms_error if ms_error else float("inf")
        p_a = float(stats.f.sf(f_a, df_a * eps, df_error * eps))
        tables.append(ResultTable(
            title=f"One-way repeated-measures ANOVA on {col_name} (subject = row)" + ("" if opt.assume_sphericity else f" - Geisser-Greenhouse ε = {eps:.3f}"),
            headers=["Source", "SS", "df", "MS", "F", "P value"],
            rows=[[col_name, f"{ss_a:.4g}", f"{df_a}", f"{ms_a:.4g}", f"{f_a:.4f}", f"{p_a:.4g}"]],
        ))
        summary = f"{col_name} p = {p_a:.4g} (one-way repeated-measures ANOVA, subject = row; {row_name}/interaction not tested in this matching mode)."
        notes.append(f"'{matching}' matching treats each row as one {subj_name.lower()}, so {row_name} no longer serves as a separate crossed factor.")
        return AnalysisResult(title="Two-Way ANOVA", summary=summary, tables=tables, notes=notes, charts=charts)

    # ---- build the (row x column) cell grid, one value or one subject-vector per cell ----
    if matching == "subcolumn_subject":
        cube = np.stack([cell_matrices[a][:n_rows, :n_reps] for a in a_labels], axis=1)  # subjects x A x B  (rows=subject axis via reps)
        cube = np.transpose(cube, (2, 1, 0))  # subject, a, b  -> want subject x a x b; matrix currently rows(b) x a x reps(subject)
        # cube shape now: (subject, a, b)
        n_s, n_a, n_b = cube.shape
        grand = float(np.mean(cube))
        subj_means = np.mean(cube, axis=(1, 2))
        a_means_arr = np.mean(cube, axis=(0, 2))
        b_means_arr = np.mean(cube, axis=(0, 1))
        ab_means = np.mean(cube, axis=0)  # a x b
        sa_means = np.mean(cube, axis=2)  # subject x a
        sb_means = np.mean(cube, axis=1)  # subject x b

        ss_subject = n_a * n_b * float(np.sum((subj_means - grand) ** 2))
        ss_a = n_s * n_b * float(np.sum((a_means_arr - grand) ** 2))
        ss_b = n_s * n_a * float(np.sum((b_means_arr - grand) ** 2))
        ss_ab = n_s * float(np.sum((ab_means - a_means_arr[:, None] - b_means_arr[None, :] + grand) ** 2))
        ss_total = float(np.sum((cube - grand) ** 2))
        ss_sa = n_b * float(np.sum((sa_means - subj_means[:, None] - a_means_arr[None, :] + grand) ** 2))
        ss_sb = n_a * float(np.sum((sb_means - subj_means[:, None] - b_means_arr[None, :] + grand) ** 2))
        ss_resid = ss_total - ss_subject - ss_a - ss_b - ss_ab - ss_sa - ss_sb

        df_subject, df_a, df_b = n_s - 1, n_a - 1, n_b - 1
        df_ab = df_a * df_b
        df_sa, df_sb = df_subject * df_a, df_subject * df_b
        df_resid = df_subject * df_ab

        eps_a = 1.0 if opt.assume_sphericity else _greenhouse_geisser_epsilon(sa_means)
        eps_b = 1.0 if opt.assume_sphericity else _greenhouse_geisser_epsilon(sb_means)
        eps_ab = 1.0 if opt.assume_sphericity else _greenhouse_geisser_epsilon(cube.reshape(n_s, n_a * n_b))

        ms_a, ms_sa = ss_a / df_a, ss_sa / df_sa if df_sa else float("nan")
        ms_b, ms_sb = ss_b / df_b, ss_sb / df_sb if df_sb else float("nan")
        ms_ab, ms_resid = (ss_ab / df_ab if df_ab else float("nan")), (ss_resid / df_resid if df_resid else float("nan"))

        f_a = ms_a / ms_sa if ms_sa else float("inf")
        f_b = ms_b / ms_sb if ms_sb else float("inf")
        f_ab = ms_ab / ms_resid if (ms_resid and include_interaction) else float("nan")
        p_a = float(stats.f.sf(f_a, df_a * eps_a, df_sa * eps_a))
        p_b = float(stats.f.sf(f_b, df_b * eps_b, df_sb * eps_b))
        p_ab = float(stats.f.sf(f_ab, df_ab * eps_ab, df_resid * eps_ab)) if include_interaction and df_resid else float("nan")

        rows = [
            [col_name, f"{ss_a:.4g}", str(df_a), f"{ms_a:.4g}", f"{f_a:.4f}", f"{p_a:.4g}"],
            [row_name, f"{ss_b:.4g}", str(df_b), f"{ms_b:.4g}", f"{f_b:.4f}", f"{p_b:.4g}"],
        ]
        if include_interaction:
            rows.append([f"{col_name} × {row_name}", f"{ss_ab:.4g}", str(df_ab), f"{ms_ab:.4g}", f"{f_ab:.4f}", f"{p_ab:.4g}"])
        rows.append([subj_name, f"{ss_subject:.4g}", str(df_subject), f"{ss_subject/df_subject:.4g}", "-", "-"])
        eps_note = "" if opt.assume_sphericity else f" (Geisser-Greenhouse ε: {col_name}={eps_a:.3f}, {row_name}={eps_b:.3f}" + (f", interaction={eps_ab:.3f})" if include_interaction else ")")
        tables.append(ResultTable(title=f"Two-way repeated-measures ANOVA{eps_note}", headers=["Source", "SS", "df", "MS", "F", "P value"], rows=rows))

        a_means, b_means = dict(zip(a_labels, a_means_arr.tolist())), dict(zip(b_labels, b_means_arr.tolist()))
        a_ns, b_ns = {a: n_s * n_b for a in a_labels}, {b: n_s * n_a for b in b_labels}
        pooled_var_a, df_error_for_mc, effective_ms = ms_sa, df_sa, ms_sa
        summary = f"{col_name} p = {p_a:.4g}; {row_name} p = {p_b:.4g}" + (f"; interaction p = {p_ab:.4g}" if include_interaction else "") + " (two-way repeated-measures ANOVA)."
        resid_values = cube.flatten().tolist()
        resid_predicted = np.broadcast_to(ab_means, cube.shape).flatten().tolist()

    else:  # matching == "none": ordinary two-way ANOVA with replication (or main-effects-only if no replicates)
        n_a, n_b = len(a_labels), len(b_labels)
        y_list, ai, bi, cell_values = [], [], [], {}
        for ia, a in enumerate(a_labels):
            for r, b in enumerate(b_labels):
                vals = np.asarray(cell_matrices[a][r, :], dtype=float)
                vals = vals[np.isfinite(vals)]
                cell_values[(a, b)] = vals
                y_list.extend(vals)
                ai.extend([ia] * vals.size)
                bi.extend([r] * vals.size)
        if n_a * n_b > 1_000_000:
            raise AnalysisError(
                f"This sheet makes {n_a} × {n_b} = {n_a * n_b:,} cells (every row is a level of the row factor). "
                "Two-way ANOVA is for a handful of levels per factor - if the rows are repeats rather than a "
                "factor, use 'Three groups or more' instead.")
        empty = [f"{a} / {b}" for (a, b), v in cell_values.items() if v.size == 0]
        if empty:
            raise AnalysisError("Every combination of the two factors needs at least one value; these are empty: "
                                + ", ".join(empty[:6]) + ("…" if len(empty) > 6 else ""))
        counts = [v.size for v in cell_values.values()]
        if include_interaction and max(counts) < 2:
            include_interaction = False
        y = np.asarray(y_list, dtype=float)
        fit = _type3_two_way(y, np.asarray(ai), np.asarray(bi), n_a, n_b, include_interaction)
        unbalanced = len(set(counts)) > 1
        if unbalanced:
            notes.append("The cells hold different numbers of values, so Type III sums of squares were used "
                         "(each effect adjusted for the others).")

        cell_means = {k: float(v.mean()) for k, v in cell_values.items()}
        a_means_d = {a: float(np.mean([cell_means[(a, b)] for b in b_labels])) for a in a_labels}
        b_means_d = {b: float(np.mean([cell_means[(a, b)] for a in a_labels])) for b in b_labels}
        ss_a, df_a = fit["A"]
        ss_b, df_b = fit["B"]
        ss_error, df_error_final = fit["rss"], fit["df_error"]
        ms_error = ss_error / df_error_final if df_error_final else float("nan")
        ms_a, ms_b = ss_a / df_a, ss_b / df_b
        f_a = ms_a / ms_error if ms_error else float("inf")
        f_b = ms_b / ms_error if ms_error else float("inf")
        p_a = float(stats.f.sf(f_a, df_a, df_error_final)) if df_error_final else float("nan")
        p_b = float(stats.f.sf(f_b, df_b, df_error_final)) if df_error_final else float("nan")

        def partial(ss):
            return f"{ss / (ss + ss_error):.3f}" if ss + ss_error > 0 else "–"

        rows = [
            [col_name, f"{ss_a:.4g}", str(df_a), f"{ms_a:.4g}", f"{f_a:.4f}", f"{p_a:.4g}", partial(ss_a)],
            [row_name, f"{ss_b:.4g}", str(df_b), f"{ms_b:.4g}", f"{f_b:.4f}", f"{p_b:.4g}", partial(ss_b)],
        ]
        if include_interaction:
            ss_ab, df_ab = fit["AB"]
            ms_ab = ss_ab / df_ab
            f_ab = ms_ab / ms_error if ms_error else float("inf")
            p_ab = float(stats.f.sf(f_ab, df_ab, df_error_final)) if df_error_final else float("nan")
            rows.insert(0, [f"{col_name} × {row_name} (interaction)", f"{ss_ab:.4g}", str(df_ab), f"{ms_ab:.4g}",
                            f"{f_ab:.4f}", f"{p_ab:.4g}", partial(ss_ab)])
            if p_ab < 0.05:
                notes.insert(0, "The interaction is significant: the effect of one factor depends on the other, "
                             "so each main effect on its own can mislead - look at the cell means.")
        rows.append(["Residual", f"{ss_error:.4g}", str(df_error_final), f"{ms_error:.4g}", "–", "–", ""])
        title = "Two-way ANOVA with interaction" if include_interaction else "Two-way ANOVA, main effects only"
        tables.append(ResultTable(title=title, headers=["Source", "SS", "df", "MS", "F", "P value", "Partial η²"],
                                  rows=rows))
        if opt.show_goodness_of_fit:
            every = np.concatenate([v for v in cell_values.values() if v.size])
            ss_tot = float(np.sum((every - every.mean()) ** 2))
            n_all = every.size
            df_model = n_all - 1 - df_error_final
            r2 = 1 - ss_error / ss_tot if ss_tot > 0 else float("nan")
            adj = 1 - (1 - r2) * (n_all - 1) / df_error_final if df_error_final > 0 and r2 == r2 else float("nan")
            tables.append(ResultTable(title="Goodness of fit", headers=["R²", "Adjusted R²", "Model df", "N"],
                                      rows=[[f"{r2:.4f}", f"{adj:.4f}", str(df_model), str(n_all)]]))
        n_reps = max(counts)

        a_means, b_means = a_means_d, b_means_d
        a_ns = {a: sum(cell_values[(a, b)].size for b in b_labels) for a in a_labels}
        b_ns = {b: sum(cell_values[(a, b)].size for a in a_labels) for b in b_labels}
        pooled_var_a, df_error_for_mc, effective_ms = ms_error, df_error_final, ms_error
        p_ab_summary = f"; interaction p = {p_ab:.4g}" if include_interaction else ""
        summary = f"{col_name} p = {p_a:.4g}; {row_name} p = {p_b:.4g}{p_ab_summary} ({title})."
        resid_values = [float(v) for b in b_labels for a in a_labels for v in cell_values[(a, b)]]
        resid_predicted = [cell_means[(a, b)] for b in b_labels for a in a_labels for _ in cell_values[(a, b)]]

    need_residuals = opt.show_residual_plot or opt.show_homoscedasticity_plot or opt.show_qq_plot or opt.diag_heteroscedastic or opt.diag_normality
    if need_residuals:
        values_arr = np.array(resid_values)
        predicted_arr = np.array(resid_predicted)
        residuals = values_arr - predicted_arr

        from ..chart_helpers import qq_chart, residual_chart

        if opt.show_residual_plot:
            charts.append(residual_chart("Residuals against fitted values", predicted_arr, residuals))
            tables.append(ResultTable(
                title="Residuals against fitted values, point by point",
                headers=["Predicted Y", "Residual"],
                rows=[[f"{p:.4g}", f"{r:.4g}"] for p, r in zip(predicted_arr, residuals)][:200],
            ))
        if opt.show_homoscedasticity_plot:
            charts.append(residual_chart("Size of the residuals against fitted values", predicted_arr, residuals, absolute=True))
            tables.append(ResultTable(
                title="Size of each residual against its fitted value",
                headers=["Predicted Y", "Abs(Residual)"],
                rows=[[f"{p:.4g}", f"{abs(r):.4g}"] for p, r in zip(predicted_arr, residuals)][:200],
            ))
        if opt.show_qq_plot and residuals.size >= 3:
            order = np.argsort(residuals)
            sorted_resid = residuals[order]
            n = sorted_resid.size
            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            charts.append(qq_chart("Q–Q plot of the residuals", theoretical, sorted_resid, "Actual residual"))
            tables.append(ResultTable(
                title="Residuals against the normal quantiles they should match",
                headers=["Theoretical quantile", "Actual residual"],
                rows=[[f"{t:.4g}", f"{r:.4g}"] for t, r in zip(theoretical, sorted_resid)][:200],
            ))
        if opt.diag_heteroscedastic and residuals.size >= 3:
            rho, p_spear = stats.spearmanr(predicted_arr, np.abs(residuals))
            tables.append(ResultTable(
                title="Spearman's rank correlation test for heteroscedasticity",
                headers=["Test", "Statistic", "P value"],
                rows=[["Spearman correlation (Predicted Y vs |Residual|)", f"rho = {rho:.4f}", f"{p_spear:.4g}"]],
            ))
            notes.append("Tests whether cells with larger predicted values tend to have larger |residuals|; a small P value suggests heteroscedasticity.")
        if opt.diag_normality and residuals.size >= 3:
            tables.append(ResultTable(title="Are the residuals normal?", headers=["Test", "Statistic", "P value"],
                                      rows=_common.normality_rows(residuals)))

    if opt.show_group_means:
        mean_rows = [[a, f"{a_means[a]:.{sig}g}"] for a in a_labels] + [[b, f"{b_means[b]:.{sig}g}"] for b in b_labels]
        tables.append(ResultTable(title=f"{col_name}/{row_name} means", headers=["Level", "Mean"], rows=mean_rows))

    if opt.compare != "none":
        if opt.compare == "column_means":
            means, ns, keys = a_means, a_ns, a_labels
        else:
            means, ns, keys = b_means, b_ns, b_labels
        if opt.comparison_scope == "vs_control":
            ctrl = max(0, min(opt.control_index, len(keys) - 1))
            pairs = [(ctrl, j) for j in range(len(keys)) if j != ctrl]
        else:
            pairs = list(combinations(range(len(keys)), 2))
        if opt.swap_direction:
            pairs = [(j, i) for i, j in pairs]
        tables.append(_pairwise_holm_fdr(means, ns, pooled_var_a, int(df_error_for_mc) or 1, keys, pairs, opt.correction, sig))


    return AnalysisResult(title="Two-Way ANOVA", summary=summary, tables=tables, notes=notes, charts=charts)
