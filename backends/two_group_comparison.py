"""Two groups: is the difference between them real?

Runs exactly one test - the one the design and the assumptions call for:
Welch's or Student's t test, the paired t test, the same on logged values
(for skewed, positive data), or Mann-Whitney / Wilcoxon when nothing is
assumed about the shape. Showing a second test beside it only invites
picking whichever P looks better.

Results come in the order every comparison uses: the effect with its
confidence interval (and a standardised effect size), the test, a summary
of each group, then the checks on the test's assumptions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _common


@dataclass
class TwoGroupOptions:
    paired: bool = False
    distribution: str = "gaussian"  # "gaussian" | "lognormal" | "nonparametric"
    assume_equal_variance: bool = False  # False = Welch's t test (the safer default)
    # for (A - B): "two-sided" | "less" (B bigger) | "greater" (A bigger)
    alternative: str = "two-sided"
    confidence_level: float = 0.95
    report_diff_as: str = "b_minus_a"  # "b_minus_a" | "a_minus_b"
    show_descriptive_stats: bool = False
    aicc_compare: bool = False
    # checks
    check_assumptions: bool = True
    show_residual_table: bool = False
    show_homoscedasticity_table: bool = False
    run_normality_diagnostics: bool = False
    significant_digits: int = 4
    p_style: str = "number"
    # are the groups close enough to count as the same (two one-sided tests)
    equivalence: bool = False
    equivalence_margin: float = 0.0     # the largest difference that would not matter, in the units measured
    equivalence_ratio: float = 1.25     # for values compared as a ratio: the same if between 1/this and this
    # the answer without the formulas
    bootstrap: bool = False
    permutation: bool = False
    resamples: int = 9999
    resample_seed: int = 20261009       # fixed, so that a run can be repeated exactly


def equivalence_test(a: np.ndarray, b: np.ndarray, margin: float, opt: "TwoGroupOptions"):
    """Two one-sided tests (Schuirmann 1987) that the difference A - B lies
    inside -margin to +margin: (P that it is below -margin, P that it is
    above +margin). Equivalence is shown when both are small, so the larger
    of the two is the P value of the procedure. ``a`` and ``b`` are on the
    scale the test works on (logs for values compared as a ratio)."""
    if opt.distribution == "nonparametric":
        if opt.paired:
            low = stats.wilcoxon(a + margin, b, alternative="greater").pvalue
            high = stats.wilcoxon(a - margin, b, alternative="less").pvalue
        else:
            low = stats.mannwhitneyu(a + margin, b, alternative="greater").pvalue
            high = stats.mannwhitneyu(a - margin, b, alternative="less").pvalue
    elif opt.paired:
        low = stats.ttest_rel(a + margin, b, alternative="greater").pvalue
        high = stats.ttest_rel(a - margin, b, alternative="less").pvalue
    else:
        low = stats.ttest_ind(a + margin, b, equal_var=opt.assume_equal_variance, alternative="greater").pvalue
        high = stats.ttest_ind(a - margin, b, equal_var=opt.assume_equal_variance, alternative="less").pvalue
    return float(low), float(high)


def _centre(opt: "TwoGroupOptions"):
    return np.median if opt.distribution == "nonparametric" else np.mean


def resampled(a: np.ndarray, b: np.ndarray, opt: "TwoGroupOptions", confidence: float):
    """The difference A - B by resampling: its bias-corrected bootstrap
    interval (Efron 1987) and a permutation P value. Means are compared, or
    medians when no shape is assumed; ``a`` and ``b`` are on the scale the
    test works on. Returns (difference, (lower, upper) or None, P or None,
    name of the interval)."""
    centre = _centre(opt)
    rng = np.random.default_rng(opt.resample_seed)
    draws = int(np.clip(opt.resamples, 199, 200_000))
    if opt.paired:
        data = (a - b,)

        def statistic(d, axis=-1):
            return centre(d, axis=axis)
    else:
        data = (a, b)

        def statistic(first, second, axis=-1):
            return centre(first, axis=axis) - centre(second, axis=axis)

    observed = float(statistic(*data))
    interval, name = None, "bias-corrected and accelerated (BCa)"
    if opt.bootstrap:
        for method in ("BCa", "percentile"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = stats.bootstrap(data, statistic, n_resamples=draws, confidence_level=confidence, method=method,
                                          vectorized=True, random_state=rng)
                low, high = float(got.confidence_interval.low), float(got.confidence_interval.high)
                if np.isfinite(low) and np.isfinite(high):
                    interval = (low, high)
                    name = "bias-corrected and accelerated (BCa)" if method == "BCa" else "percentile"
                    break
            except Exception:  # noqa: BLE001 - try the plainer interval
                continue
    p_value = None
    if opt.permutation:
        got = stats.permutation_test(data, statistic, permutation_type="samples" if opt.paired else "independent",
                                     n_resamples=draws, alternative=opt.alternative, vectorized=True, random_state=rng)
        p_value = float(got.pvalue)
    return observed, interval, p_value, name


def _paired_rows(data: SheetData, a_label: str, b_label: str) -> tuple[np.ndarray, np.ndarray]:
    """Row-aligned values for a paired test: each row's replicates are
    averaged, and a row counts only when both groups have a value in it -
    so a blank cell drops that one pair instead of shifting every pair
    below it."""
    def rows(label):
        _reps, matrix = data.group_replicate_matrix(label)
        if matrix.size == 0:
            return np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-blank row is simply NaN
            return np.nanmean(matrix, axis=1)

    a, b = rows(a_label), rows(b_label)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def _aicc_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """AICc of 'one shared SD' against 'an SD for each group' (lower is
    better), from the normal log-likelihood of each model."""
    n = a.size + b.size
    ss_a, ss_b = np.sum((a - a.mean()) ** 2), np.sum((b - b.mean()) ** 2)

    def aicc(log_lik, k):
        return 2 * k - 2 * log_lik + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    shared = -n / 2 * (np.log(2 * np.pi) + np.log((ss_a + ss_b) / n) + 1)
    separate = (-a.size / 2 * (np.log(2 * np.pi) + np.log(ss_a / a.size) + 1)
                - b.size / 2 * (np.log(2 * np.pi) + np.log(ss_b / b.size) + 1))
    return aicc(shared, 3), aicc(separate, 4)


@register("two_group_comparison")
def run(data: SheetData, options: TwoGroupOptions | None = None) -> AnalysisResult:
    from ..p_values import format_p

    opt = options or TwoGroupOptions()
    groups = data.all_group_arrays()
    labels = [label for label, values in groups.items() if values.size]
    if len(labels) < 2:
        raise AnalysisError("This test compares two groups, but the sheet has "
                            f"{len(labels)} group(s) with data. Add a second group column.")
    a_label, b_label = labels[0], labels[1]

    if opt.paired:
        a, b = _paired_rows(data, a_label, b_label)
        if a.size < 2:
            raise AnalysisError("A paired test needs at least 2 rows with a value in both groups.")
    else:
        a, b = groups[a_label], groups[b_label]
        if a.size < 2 or b.size < 2:
            raise AnalysisError("Each group needs at least 2 values to compare.")

    log_scale = opt.distribution == "lognormal"
    if log_scale and (np.any(a <= 0) or np.any(b <= 0)):
        raise AnalysisError("'Normal once logged' needs every value to be above 0. Choose another option "
                            "for data with zeros or negatives.")
    a_fit, b_fit = (np.log(a), np.log(b)) if log_scale else (a, b)

    sig = opt.significant_digits
    conf = opt.confidence_level
    pct = int(round(conf * 100))

    def num(v):
        return f"{v:.{sig}g}"

    def p_text(p):
        return format_p(p, opt.p_style)

    # the direction differences are reported in
    b_first = opt.report_diff_as != "a_minus_b"
    first, second = (b_label, a_label) if b_first else (a_label, b_label)
    sign = -1.0 if b_first else 1.0  # applied to (A - B)

    tables: list[ResultTable] = []
    notes: list[str] = []

    # ---------------------------------------------------------------- the test
    if opt.distribution == "nonparametric":
        if opt.paired:
            res = stats.wilcoxon(a, b, alternative=opt.alternative)
            test_name, stat_text, df_text = "Wilcoxon matched-pairs signed rank test", f"W = {res.statistic:.4g}", "–"
        else:
            res = stats.mannwhitneyu(a, b, alternative=opt.alternative)
            test_name, stat_text, df_text = "Mann-Whitney test", f"U = {res.statistic:.4g}", "–"
        p = float(res.pvalue)
        est, lo, hi = _common.hodges_lehmann(a, b, paired=opt.paired, confidence=conf)
        est, lo, hi = sign * est, *sorted((sign * lo, sign * hi))
        r = sign * _common.rank_biserial(a, b, paired=opt.paired)
        effect_rows = [
            [f"Median difference ({first} − {second}), Hodges-Lehmann", num(est), f"{num(lo)} to {num(hi)}", ""],
            ["Rank-biserial correlation", f"{r:.3f}", "–", _common.magnitude(r, (0.1, 0.3, 0.5))],
        ]
        headline = f"Median difference ({first} − {second}) = {num(est)} ({pct}% CI {num(lo)} to {num(hi)})"
    else:
        if opt.paired:
            res = stats.ttest_rel(a_fit, b_fit, alternative=opt.alternative)
            test_name = "Paired t test" + (" on logged values" if log_scale else "")
        else:
            res = stats.ttest_ind(a_fit, b_fit, equal_var=opt.assume_equal_variance, alternative=opt.alternative)
            test_name = ("Student's t test" if opt.assume_equal_variance else "Welch's t test") + (
                " on logged values" if log_scale else "")
        p = float(res.pvalue)
        stat_text, df_text = f"t = {sign * res.statistic:.4g}", f"{res.df:.4g}"  # t for the reported direction
        ci = res.confidence_interval(confidence_level=conf)  # for (A - B) on the fitted scale
        diff = float(np.mean(a_fit - b_fit) if opt.paired else np.mean(a_fit) - np.mean(b_fit))
        if opt.alternative == "two-sided":
            ci_lo, ci_hi = float(ci.low), float(ci.high)
        else:  # a one-sided test's interval has an open end; report the two-sided one alongside
            two = (stats.ttest_rel(a_fit, b_fit) if opt.paired
                   else stats.ttest_ind(a_fit, b_fit, equal_var=opt.assume_equal_variance))
            two_ci = two.confidence_interval(confidence_level=conf)
            ci_lo, ci_hi = float(two_ci.low), float(two_ci.high)
        g, g_lo, g_hi = _common.hedges_g(a_fit, b_fit, paired=opt.paired, confidence=conf)
        g, g_lo, g_hi = sign * g, *sorted((sign * g_lo, sign * g_hi))
        if log_scale:
            ratio = np.exp(sign * diff)
            r_lo, r_hi = sorted((np.exp(sign * ci_lo), np.exp(sign * ci_hi)))
            effect_rows = [[f"Ratio of geometric means ({first} / {second})", num(ratio), f"{num(r_lo)} to {num(r_hi)}", ""]]
            headline = f"{first} / {second} = {num(ratio)} ({pct}% CI {num(r_lo)} to {num(r_hi)})"
        else:
            d = sign * diff
            d_lo, d_hi = sorted((sign * ci_lo, sign * ci_hi))
            label = "Mean of the differences" if opt.paired else "Difference between means"
            effect_rows = [[f"{label} ({first} − {second})", num(d), f"{num(d_lo)} to {num(d_hi)}", ""]]
            headline = f"{first} − {second} = {num(d)} ({pct}% CI {num(d_lo)} to {num(d_hi)})"
        effect_rows.append(["Hedges' g (standardised difference)", f"{g:.3f}", f"{g_lo:.3f} to {g_hi:.3f}",
                            _common.magnitude(g, (0.2, 0.5, 0.8))])

    tables.append(ResultTable("Size of the difference", ["Effect", "Estimate", f"{pct}% CI", "Size"], effect_rows))
    side = {"two-sided": "two-sided", "less": f"one-sided: {b_label} bigger", "greater": f"one-sided: {a_label} bigger"}
    tables.append(ResultTable("Is it real?", ["Test", "Statistic", "df", "P value", "Significant (α = 0.05)"],
                              [[f"{test_name} ({side[opt.alternative]})", stat_text, df_text, p_text(p),
                                "Yes" if p < 0.05 else "No"]]))

    def summary_row(label, v):
        sem = np.std(v, ddof=1) / np.sqrt(v.size)
        return [label, str(v.size), num(np.mean(v)), num(np.std(v, ddof=1)), num(sem), num(np.median(v))]

    group_rows = [summary_row(a_label, a), summary_row(b_label, b)]
    if log_scale:
        for row, v in zip(group_rows, (a, b)):
            row.append(num(stats.gmean(v)))
    tables.append(ResultTable("Each group", ["Group", "N", "Mean", "SD", "SEM", "Median"]
                              + (["Geometric mean"] if log_scale else []), group_rows))

    if opt.show_descriptive_stats:
        desc_rows = []
        for lbl, v in ((a_label, a), (b_label, b)):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            desc_rows.append([lbl, num(np.min(v)), num(q1), num(med), num(q3), num(np.max(v))])
        tables.append(ResultTable("Spread of each group", ["Group", "Min", "Q1", "Median", "Q3", "Max"], desc_rows))

    # ---------------------------------------------------------------- close enough to count as the same?
    if opt.equivalence:
        margin = float(np.log(opt.equivalence_ratio)) if log_scale else float(opt.equivalence_margin)
        if log_scale and opt.equivalence_ratio <= 1:
            notes.append("The equivalence test was not run: the ratio that still counts as the same must be above 1.")
        elif margin <= 0:
            notes.append("The equivalence test was not run: give the largest difference that would not matter. It is a "
                         "judgement about the science, to be fixed before the data are seen.")
        else:
            p_low, p_high = equivalence_test(a_fit, b_fit, margin, opt)
            if sign < 0:      # reported the other way round, below and above change places
                p_low, p_high = p_high, p_low
            alpha = 1 - conf
            inner = 1 - 2 * alpha     # the interval that matches two one-sided tests at alpha
            if opt.distribution == "nonparametric":
                _est, in_lo, in_hi = _common.hodges_lehmann(a, b, paired=opt.paired, confidence=inner)
            else:
                two = (stats.ttest_rel(a_fit, b_fit) if opt.paired
                       else stats.ttest_ind(a_fit, b_fit, equal_var=opt.assume_equal_variance))
                both = two.confidence_interval(confidence_level=inner)
                in_lo, in_hi = float(both.low), float(both.high)
            in_lo, in_hi = sorted((sign * in_lo, sign * in_hi))
            if log_scale:
                in_lo, in_hi = float(np.exp(in_lo)), float(np.exp(in_hi))
                zone = f"{num(1 / opt.equivalence_ratio)} to {num(opt.equivalence_ratio)} (as a ratio)"
            else:
                zone = f"−{num(margin)} to +{num(margin)}"
            worst = max(p_low, p_high)
            tables.append(ResultTable(
                "Are the two groups close enough to count as the same?",
                ["Counts as the same", f"{inner * 100:g}% CI of the difference", "P, against lying below the zone",
                 "P, against lying above the zone", "P value (the larger)", "Reading"],
                [[zone, f"{num(in_lo)} to {num(in_hi)}", p_text(p_low), p_text(p_high), p_text(worst),
                  "The difference lies inside the zone: equivalent" if worst < alpha else
                  "Not shown to be equivalent"]]))
            notes.append("Equivalence was tested by two one-sided tests (Schuirmann 1987). A large P in the ordinary test does "
                         "not show two groups are the same; only this test, with a zone fixed in advance, can.")
            if worst >= alpha and p >= alpha:
                notes.append("Neither a difference nor equivalence was shown: the data are too few, or too scattered, to say.")

    # ---------------------------------------------------------------- the answer without the formulas
    if opt.bootstrap or opt.permutation:
        observed, interval, perm_p, interval_name = resampled(a_fit, b_fit, opt, conf)
        observed = sign * observed
        if interval is not None:
            interval = tuple(sorted((sign * interval[0], sign * interval[1])))
        if log_scale:
            observed = float(np.exp(observed))
            interval = tuple(float(np.exp(v)) for v in interval) if interval is not None else None
        what = (("Median of the differences" if opt.paired else "Difference between medians")
                if opt.distribution == "nonparametric" else
                ("Ratio of geometric means" if log_scale else
                 ("Mean of the differences" if opt.paired else "Difference between means")))
        tables.append(ResultTable(
            "The same question, by resampling the data",
            ["Effect", "Estimate"] + ([f"{pct}% bootstrap CI"] if opt.bootstrap else [])
            + (["Permutation P value"] if opt.permutation else []),
            [[f"{what} ({first} {'/' if log_scale else '−'} {second})", num(observed)]
             + ([f"{num(interval[0])} to {num(interval[1])}" if interval is not None else "n/a"] if opt.bootstrap else [])
             + ([p_text(perm_p)] if opt.permutation else [])]))
        draws = int(np.clip(opt.resamples, 199, 200_000))
        parts = []
        if opt.bootstrap:
            parts.append(f"the bootstrap interval is {interval_name}, from {draws} resamples of the data with replacement")
        if opt.permutation:
            parts.append(f"the permutation P value comes from {'flipping the sign of each pair' if opt.paired else 'reshuffling the group labels'}"
                         f" (up to {draws} times; every arrangement when there are fewer)")
        notes.append("Resampling: " + "; ".join(parts) + ". No shape is assumed for the values. The random numbers are "
                     "seeded, so the same data give the same answer.")
        if opt.distribution == "nonparametric":
            notes.append("By resampling, medians are compared; the Hodges-Lehmann estimate above is the median of all "
                         "pairwise differences, which is not the same thing and need not agree exactly.")

    # ---------------------------------------------------------------- checks
    if opt.check_assumptions:
        check_table, advice = _common.assumption_checks(
            {a_label: a_fit, b_label: b_fit}, opt.distribution,
            None if (opt.paired or opt.distribution == "nonparametric") else opt.assume_equal_variance,
            paired_diffs=(a_fit - b_fit) if opt.paired else None)
        tables.append(check_table)
        notes.extend("Checks: " + line for line in advice)

    if opt.aicc_compare and opt.distribution != "nonparametric" and not opt.paired:
        shared, separate = _aicc_pair(a_fit, b_fit)
        tables.append(ResultTable("Shared or separate scatter? (AICc, lower is better)", ["Model", "AICc"],
                                  [["One SD for both groups", f"{shared:.4g}"],
                                   ["An SD for each group", f"{separate:.4g}"]]))
        notes.append("AICc prefers " + ("one shared SD." if shared < separate else "a separate SD for each group."))

    if opt.show_residual_table or opt.show_homoscedasticity_table or opt.run_normality_diagnostics:
        residuals = np.concatenate([a_fit - a_fit.mean(), b_fit - b_fit.mean()])
        if opt.show_residual_table:
            group_col = [a_label] * a.size + [b_label] * b.size
            values = np.concatenate([a, b])
            tables.append(ResultTable("Residuals (value − group mean)", ["Group", "Value", "Residual"],
                                      [[g_, num(v), num(r_)] for g_, v, r_ in zip(group_col, values, residuals)]))
        if opt.show_homoscedasticity_table:
            tables.append(ResultTable(
                "Scatter of each group (should look similar if the SDs are equal)",
                ["Group", "Mean |residual|", "Max |residual|"],
                [[lbl, num(np.mean(np.abs(v - v.mean()))), num(np.max(np.abs(v - v.mean())))]
                 for lbl, v in ((a_label, a_fit), (b_label, b_fit))]))
        if opt.run_normality_diagnostics and residuals.size >= 3:
            tables.append(ResultTable("Are the residuals normal?", ["Test", "Statistic", "P value"],
                                      _common.normality_rows(residuals)))

    # ---------------------------------------------------------------- notes
    small = _common.small_n_note({a_label: a.size, b_label: b.size})
    if small:
        notes.insert(0, small)
    if len(labels) > 2:
        notes.append(f"Only the first two groups ({a_label}, {b_label}) were compared; the sheet has "
                     f"{len(labels)}. Use 'Three groups or more' to compare them all.")
    if opt.paired:
        notes.append("Rows were paired by position; a row missing either value was left out.")
    if opt.alternative != "two-sided":
        notes.append("One-sided test: the interval shown is the two-sided one, for reference.")

    summary = f"{headline}; {test_name}, {_p_phrase(p_text(p))}."
    return AnalysisResult(title="Two groups", summary=summary, tables=tables, notes=notes)


def _p_phrase(text: str) -> str:
    """'P = 0.012' whether the style printed '0.012', 'P=0.012' or '<0.001'."""
    t = text.strip()
    if t.upper().startswith("P"):
        return t
    if t.startswith(("<", ">")):
        return f"P {t}"
    return f"P = {t}"
