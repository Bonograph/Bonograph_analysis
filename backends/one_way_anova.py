"""Three groups or more: does any of them differ, and which ones?

One overall test, chosen from the design and the assumptions:

- ordinary one-way ANOVA (equal scatter assumed);
- Welch's ANOVA, with the Brown-Forsythe ANOVA beside it (scatter may differ;
  the default);
- Kruskal-Wallis (ranks - nothing assumed about shape);
- repeated-measures ANOVA or Friedman, when every row is one subject.

Follow-up comparisons run by default, with the method that suits the test:
Tukey or Dunnett after an ordinary ANOVA, Games-Howell after Welch's, Dunn
after Kruskal-Wallis, and paired tests with Holm's adjustment after a
repeated-measures test. Results come in the shared order: effect size,
the test, which groups differ, each group, then the assumption checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _common, _ptukey


@dataclass
class OneWayAnovaOptions:
    repeated_measures: bool = False
    distribution: str = "gaussian"  # "gaussian" | "lognormal" | "nonparametric"
    assume_equal_variance: bool = False  # False -> Welch's and Brown-Forsythe ANOVA
    post_hoc: str = "all_pairs"  # "none" | "all_pairs" | "vs_control" | "linear_trend"
    control_index: int = 0
    # "auto" | "holm" | "tukey" | "dunnett" | "fdr" | "none" (see _common.resolve_correction)
    correction: str = "auto"
    show_descriptive_stats: bool = False
    check_assumptions: bool = True
    show_residual_plot: bool = False
    show_homoscedasticity_plot: bool = False
    show_qq_plot: bool = False
    show_heatmap_plot: bool = False
    diag_heteroscedastic: bool = False
    diag_normality: bool = False
    confidence_level: float = 0.95
    significant_digits: int = 4
    p_style: str = "number"


# kept for older callers
_holm = _common.holm
_benjamini_hochberg = _common.benjamini_hochberg


def _welch_anova(arrays: list[np.ndarray]) -> tuple[float, float, float]:
    k = len(arrays)
    ni = np.array([a.size for a in arrays])
    mi = np.array([np.mean(a) for a in arrays])
    vi = np.array([np.var(a, ddof=1) for a in arrays])
    wi = ni / vi
    grand = float(np.sum(wi * mi) / np.sum(wi))
    num = np.sum(wi * (mi - grand) ** 2) / (k - 1)
    denom_term = np.sum((1 - wi / np.sum(wi)) ** 2 / (ni - 1))
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * denom_term
    f_stat = num / denom
    df2 = (k**2 - 1) / (3 * denom_term)
    return float(f_stat), float(df2), float(stats.f.sf(f_stat, k - 1, df2))


def _brown_forsythe_anova(arrays: list[np.ndarray]) -> tuple[float, int, float, float]:
    """Brown-Forsythe ANOVA for equal *means* without equal scatter:
    F* = Σ nᵢ(x̄ᵢ − x̄)² / Σ (1 − nᵢ/N) sᵢ², with Satterthwaite's df.
    (Not to be confused with the Brown-Forsythe test of equal spread.)"""
    ni = np.array([a.size for a in arrays], dtype=float)
    mi = np.array([np.mean(a) for a in arrays])
    vi = np.array([np.var(a, ddof=1) for a in arrays])
    n = ni.sum()
    grand = float(np.sum(ni * mi) / n)
    weights = (1 - ni / n) * vi
    f_stat = float(np.sum(ni * (mi - grand) ** 2) / np.sum(weights))
    c = weights / np.sum(weights)
    df2 = float(1 / np.sum(c**2 / (ni - 1)))
    df1 = len(arrays) - 1
    return f_stat, df1, df2, float(stats.f.sf(f_stat, df1, df2))


def _spread_test(arrays: list[np.ndarray]) -> tuple[float, int, int, float]:
    """Brown-Forsythe test of equal spread: ANOVA on each value's distance
    from its group's median."""
    f_stat, p = stats.levene(*arrays, center="median")
    return float(f_stat), len(arrays) - 1, sum(a.size for a in arrays) - len(arrays), float(p)


_brown_forsythe = _spread_test  # older name


def _tukey_hsd(means: list[float], ns: list[int], mse: float, df_error: int) -> list[list]:
    from itertools import combinations

    rows = []
    k = len(means)
    for i, j in combinations(range(k), 2):
        diff = means[i] - means[j]
        se = np.sqrt(mse / 2 * (1 / ns[i] + 1 / ns[j]))
        q = abs(diff) / se if se else 0.0
        p = float(_ptukey.sf(q, k, df_error)[0]) if se else 1.0
        rows.append([i, j, diff, p])
    return rows


def _p_phrase(text: str) -> str:
    t = text.strip()
    if t.upper().startswith("P"):
        return t
    return f"P {t}" if t.startswith(("<", ">")) else f"P = {t}"


@register("one_way_anova")
def run(data: SheetData, options: OneWayAnovaOptions | None = None) -> AnalysisResult:
    from ..p_values import format_p

    opt = options or OneWayAnovaOptions()
    groups = data.all_group_arrays()
    labels = [label for label, values in groups.items() if values.size >= 2]
    if len(labels) < 3:
        raise AnalysisError(
            "Comparing three or more groups needs at least 3 groups with 2+ values each; "
            f"this sheet has {len(labels)}. Use 'Two groups' if you only have two.")
    sig = opt.significant_digits
    unmatched_rows = 0
    if opt.repeated_measures:
        # every row is one subject: keep the rows with a value under every condition, where they
        # are, so that a blank drops that subject instead of pairing the values below it with
        # other subjects' values
        rows = data.row_matrix(labels)
        whole = ~np.any(np.isnan(rows), axis=1)
        unmatched_rows = int(np.sum(~np.all(np.isnan(rows), axis=1) & ~whole))
        if whole.sum() < 2:
            raise AnalysisError("Repeated measures need at least 2 subjects (rows) with a value in every group.")
        groups = {label: rows[whole, j] for j, label in enumerate(labels)}
    raw_arrays = [groups[label] for label in labels]
    arrays = raw_arrays

    if opt.distribution == "lognormal":
        if any(np.any(a <= 0) for a in arrays):
            raise AnalysisError("'Normal once logged' needs every value to be above 0.")
        arrays = [np.log(a) for a in arrays]

    def p_text(p):
        return format_p(p, opt.p_style)

    tables: list[ResultTable] = []
    charts = []
    notes: list[str] = []
    effect_rows: list[list[str]] = []
    k = len(arrays)
    n_total = sum(a.size for a in arrays)
    matched = opt.repeated_measures
    ranks = opt.distribution == "nonparametric"
    if unmatched_rows:
        notes.append(f"{unmatched_rows} row{'s' if unmatched_rows != 1 else ''} with a blank in at least one group "
                     f"{'were' if unmatched_rows != 1 else 'was'} left out: with repeated measures a subject counts only when it "
                     "has a value in every group.")

    # ---------------------------------------------------------------- the overall test
    if matched:
        n_common = min(a.size for a in arrays)
        if n_common < 2:
            raise AnalysisError("Repeated measures need at least 2 subjects (rows) with a value in every group.")
        matrix = np.column_stack([a[:n_common] for a in arrays])
        if ranks:
            stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])
            test_name, stat_text, df_text = "Friedman test", f"χ² = {stat:.4g}", str(k - 1)
            kendall_w = float(stat / (n_common * (k - 1)))
            effect_rows.append(["Kendall's W (agreement across subjects)", f"{kendall_w:.3f}",
                                _common.magnitude(kendall_w, (0.1, 0.3, 0.5))])
        else:
            grand = float(np.mean(matrix))
            ss_between = n_common * float(np.sum((matrix.mean(axis=0) - grand) ** 2))
            ss_subjects = k * float(np.sum((matrix.mean(axis=1) - grand) ** 2))
            ss_error = float(np.sum((matrix - grand) ** 2)) - ss_between - ss_subjects
            df_between, df_error = k - 1, (k - 1) * (n_common - 1)
            stat = (ss_between / df_between) / (ss_error / df_error) if ss_error > 0 else float("inf")
            p = float(stats.f.sf(stat, df_between, df_error))
            test_name, stat_text, df_text = "Repeated-measures ANOVA", f"F = {stat:.4g}", f"{df_between}, {df_error}"
            partial = ss_between / (ss_between + ss_error) if ss_between + ss_error else float("nan")
            effect_rows.append(["Partial η² (share of within-subject variation explained)", f"{partial:.3f}",
                                _common.magnitude(np.sqrt(partial / (1 - partial)) if partial < 1 else np.inf,
                                                  (0.1, 0.25, 0.4))])
    elif ranks:
        stat, p = stats.kruskal(*arrays)
        test_name, stat_text, df_text = "Kruskal-Wallis test", f"H = {stat:.4g}", str(k - 1)
        eps = float(stat / (n_total - 1))
        effect_rows.append(["ε² (share of the ranking explained by group)", f"{eps:.3f}",
                            _common.magnitude(np.sqrt(eps), (0.1, 0.3, 0.5))])
    elif not opt.assume_equal_variance:
        stat, df2, p = _welch_anova(arrays)
        test_name, stat_text, df_text = "Welch's ANOVA", f"F = {stat:.4g}", f"{k - 1}, {df2:.4g}"
        bf_stat, bf_df1, bf_df2, bf_p = _brown_forsythe_anova(arrays)
    else:
        stat, p = stats.f_oneway(*arrays)
        test_name, stat_text, df_text = "Ordinary one-way ANOVA", f"F = {stat:.4g}", f"{k - 1}, {n_total - k}"

    if not matched and not ranks:
        eta, omega = _common.eta_omega(arrays)
        effect_rows.append(["ω² (share of the variation explained by group; less biased)", f"{omega:.3f}",
                            _common.magnitude(np.sqrt(omega / (1 - omega)) if omega < 1 else np.inf,
                                              (0.1, 0.25, 0.4))])
        effect_rows.append(["η² (share of the variation explained by group)", f"{eta:.3f}", ""])

    if opt.distribution == "lognormal":
        test_name += " on logged values"
    test_rows = [[test_name, stat_text, df_text, p_text(p), "Yes" if p < 0.05 else "No"]]
    if not matched and not ranks and not opt.assume_equal_variance:
        test_rows.append(["Brown-Forsythe ANOVA", f"F* = {bf_stat:.4g}", f"{bf_df1}, {bf_df2:.4g}", p_text(bf_p),
                          "Yes" if bf_p < 0.05 else "No"])

    tables.append(ResultTable("Size of the effect", ["Effect size", "Estimate", "Size"], effect_rows))
    tables.append(ResultTable("Do the groups differ?", ["Test", "Statistic", "df", "P value", "Significant (α = 0.05)"],
                              test_rows))

    # ---------------------------------------------------------------- which groups differ
    follow_up_text = ""
    if opt.post_hoc == "linear_trend" and not matched:
        means = np.array([float(np.mean(a)) for a in arrays])
        ns = np.array([a.size for a in arrays])
        pooled_var = sum((n - 1) * np.var(a, ddof=1) for n, a in zip(ns, arrays)) / (n_total - k)
        order_idx = np.arange(k)
        slope, intercept = np.polyfit(order_idx, means, 1, w=np.sqrt(ns))
        fitted = slope * order_idx + intercept
        ss_trend = float(np.sum(ns * (fitted - np.average(means, weights=ns)) ** 2))
        f_trend = ss_trend / pooled_var if pooled_var else float("inf")
        p_trend = float(stats.f.sf(f_trend, 1, n_total - k))
        tables.append(ResultTable("Is there a trend across the groups (in the order they sit in)?",
                                  ["Slope per group", "F", "P value"],
                                  [[f"{slope:.{sig}g}", f"{f_trend:.4g}", p_text(p_trend)]]))
        follow_up_text = "A test for linear trend across the groups, in their order on the sheet."
    elif opt.post_hoc in ("all_pairs", "vs_control"):
        method = _common.resolve_correction(opt.correction, opt.post_hoc, opt.distribution,
                                            opt.assume_equal_variance, paired=matched)
        if matched:
            n_common = min(a.size for a in arrays)
            fitted = [a[:n_common] for a in arrays]
        else:
            fitted = arrays
        table, follow_up_text = _common.pairwise(fitted, labels, opt.post_hoc, opt.control_index, method,
                                                 opt.confidence_level, paired=matched, ranks=ranks)
        if opt.distribution == "lognormal":
            table.title += " - differences on the log scale"
        tables.append(table)

    # ---------------------------------------------------------------- each group
    group_rows = []
    for label, v in zip(labels, raw_arrays):
        group_rows.append([label, str(v.size), f"{np.mean(v):.{sig}g}", f"{np.std(v, ddof=1):.{sig}g}",
                           f"{np.std(v, ddof=1) / np.sqrt(v.size):.{sig}g}", f"{np.median(v):.{sig}g}"])
    tables.append(ResultTable("Each group", ["Group", "N", "Mean", "SD", "SEM", "Median"], group_rows))
    if opt.show_descriptive_stats:
        desc_rows = []
        for label, v in zip(labels, raw_arrays):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            desc_rows.append([label, f"{np.min(v):.{sig}g}", f"{q1:.{sig}g}", f"{med:.{sig}g}", f"{q3:.{sig}g}",
                              f"{np.max(v):.{sig}g}"])
        tables.append(ResultTable("Spread of each group", ["Group", "Min", "Q1", "Median", "Q3", "Max"], desc_rows))

    # ---------------------------------------------------------------- checks
    if opt.check_assumptions:
        check_table, advice = _common.assumption_checks(
            dict(zip(labels, arrays)), opt.distribution,
            None if (matched or ranks) else opt.assume_equal_variance)
        tables.append(check_table)
        notes.extend("Checks: " + line for line in advice)

    need_residuals = (opt.show_residual_plot or opt.show_homoscedasticity_plot or opt.show_qq_plot
                      or opt.show_heatmap_plot or opt.diag_heteroscedastic or opt.diag_normality)
    if need_residuals:
        from ..chart_helpers import heatmap_chart, qq_chart, residual_chart, unavailable_chart

        group_means = [float(np.mean(a)) for a in arrays]
        all_residuals = np.concatenate([a - m for a, m in zip(arrays, group_means)])
        group_col = [lbl for lbl, a in zip(labels, arrays) for _ in range(a.size)]
        predicted = np.concatenate([np.full(a.size, m) for a, m in zip(arrays, group_means)])

        if opt.show_residual_plot:
            charts.append(residual_chart("Residuals against fitted values", predicted, all_residuals, groups=group_col))
            tables.append(ResultTable("Residuals against fitted values, point by point",
                                      ["Group", "Predicted Y", "Residual"],
                                      [[g, f"{pv:.4g}", f"{r:.4g}"] for g, pv, r in
                                       zip(group_col, predicted, all_residuals)][:200]))
        if opt.show_homoscedasticity_plot:
            charts.append(residual_chart("Size of the residuals against fitted values", predicted, all_residuals, groups=group_col,
                                         absolute=True))
            tables.append(ResultTable("Size of each residual against its fitted value",
                                      ["Group", "Predicted Y", "Abs(Residual)"],
                                      [[g, f"{pv:.4g}", f"{abs(r):.4g}"] for g, pv, r in
                                       zip(group_col, predicted, all_residuals)][:200]))
        if opt.show_qq_plot and all_residuals.size >= 3:
            sorted_resid = np.sort(all_residuals)
            n = sorted_resid.size
            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            charts.append(qq_chart("Q–Q plot of the residuals", theoretical, sorted_resid, "Actual residual"))
            tables.append(ResultTable("Residuals against the normal quantiles they should match",
                                      ["Theoretical quantile", "Actual residual"],
                                      [[f"{t:.4g}", f"{r:.4g}"] for t, r in zip(theoretical, sorted_resid)][:200]))
        if opt.show_heatmap_plot:
            if len({a.size for a in arrays}) == 1:
                matrix = np.column_stack(arrays)
                charts.append(heatmap_chart("Heat map (row x group values)", matrix,
                                            [str(i + 1) for i in range(matrix.shape[0])], labels))
                tables.append(ResultTable("The values behind the heat map", ["Row"] + labels,
                                          [[str(i + 1)] + [f"{v:.4g}" for v in row]
                                           for i, row in enumerate(matrix)][:200]))
            else:
                notes.append("The table behind the heat map needs groups of equal size; these differ.")
                charts.append(unavailable_chart(
                    "Heat map (row x group values)",
                    "A heat map needs the same number of values in every group, so the rows line up. "
                    "These groups have " + ", ".join(f"{lbl}: {a.size}" for lbl, a in zip(labels, arrays)) + "."))
        if opt.diag_heteroscedastic:
            sp_stat, sp_df1, sp_df2, sp_p = _spread_test(arrays)
            try:
                bart_stat, bart_p = stats.bartlett(*arrays)
            except ValueError:
                bart_stat, bart_p = float("nan"), float("nan")
            tables.append(ResultTable(
                "Does the scatter differ between groups?", ["Test", "Statistic", "P value"],
                [["Brown-Forsythe (spread)", f"F = {sp_stat:.4f}", f"{sp_p:.4g}"],
                 ["Bartlett's test", f"χ² = {bart_stat:.4f}" if np.isfinite(bart_stat) else "n/a",
                  f"{bart_p:.4g}" if np.isfinite(bart_p) else "n/a"]]))
        if opt.diag_normality and all_residuals.size >= 3:
            tables.append(ResultTable("Are the residuals normal?", ["Test", "Statistic", "P value"],
                                      _common.normality_rows(all_residuals)))

    small = _common.small_n_note({label: v.size for label, v in zip(labels, raw_arrays)})
    if small:
        notes.insert(0, small)
    notes.append("What was run: " + test_name + "." + (" " + follow_up_text if follow_up_text else ""))
    if opt.post_hoc == "none":
        notes.append("A significant overall test says at least one group differs, not which - turn on "
                     "follow-up tests to find out.")

    summary = f"{k} groups: {test_name}, {stat_text} ({df_text}), {_p_phrase(p_text(p))}"
    if effect_rows:
        summary += f"; {effect_rows[0][0].split(' (')[0]} = {effect_rows[0][1]}"
        if effect_rows[0][2]:
            summary += f" ({effect_rows[0][2]})"
    summary += "."
    return AnalysisResult(title="Three groups or more", summary=summary, tables=tables, notes=notes, charts=charts)
