"""Chi-Square (and Fisher's exact) test. Treats each sheet row as a category and each
group's first column as that category's count, building a contingency
table (rows = categories, columns = groups).

For a 2x2 table the effect-size confidence intervals use the same methods
the published methods in ``analysis/contingency_stats.py``: Koopman score for
relative risk, Newcombe/Wilson with continuity correction for the
difference between proportions, Baptista-Pike for the odds ratio, and
Wilson/Brown for sensitivity, specificity and the predictive values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .. import contingency_stats as cs
from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _common

ALPHA = 0.05  # significance threshold for the "statistically significant?" line



def _p(value, opt) -> str:
    """One P value, written the way the user asked for (analysis/p_values.py)."""
    from ..p_values import format_p

    return format_p(value, getattr(opt, "p_style", "number"),
                    digits=getattr(opt, "significant_digits", 4),
                    alpha=getattr(opt, "alpha", 0.05))

# the labels this analysis's results use
RR = "Relative risk, row 1 vs row 2"
RR_REVERSED = "Relative risk, row 2 vs row 1"
RISK_DIFFERENCE = "Risk difference, row 1 − row 2"
NNT = "Number needed to treat"
OR = "Odds ratio, row 1 vs row 2"
OR_REVERSED = "Odds ratio, row 2 vs row 1"
SHARE_OF_ROW = "Each cell as a share of its row"
SHARE_OF_COLUMN = "Each cell as a share of its column"
SHARE_OF_TABLE = "Each cell as a share of the whole table"


@dataclass
class ChiSquareOptions:
    # which test
    # effect sizes are reported by default - a P value alone says nothing about how big the link is
    report_relative_risk: bool = True
    report_risk_difference: bool = True
    report_odds_ratio: bool = True
    report_sensitivity_specificity: bool = False
    # "auto" = Fisher's exact when any expected count is under 5 (or for any
    # 2x2 table), chi-square otherwise
    p_method: str = "auto"  # "auto" | "fisher" | "yates" | "chi_square" | "trend"
    # what to report
    tails: str = "two"  # "one" | "two"
    confidence_level: float = 0.95
    significant_digits: int = 4
    p_style: str = "number"  # how P values are written (analysis/p_values.py)


def _cochran_armitage_trend(table: np.ndarray) -> tuple[float, float]:
    """Chi-square test for trend across ordered rows of an Rx2 table -
    tests whether the proportion in column 1 changes linearly with row
    order (row scores 0, 1, 2, ...). Rows with no observations carry no
    information and are skipped."""
    n_total = table.sum(axis=1)
    keep = n_total > 0
    n_events = table[keep, 0]
    n_total = n_total[keep]
    scores = np.arange(table.shape[0], dtype=float)[keep]
    n = n_total.sum()
    p_bar = n_events.sum() / n
    mean_score = np.average(scores, weights=n_total)
    num = np.sum(n_total * (scores - mean_score) * (n_events / n_total - p_bar))
    denom = p_bar * (1 - p_bar) * np.sum(n_total * (scores - mean_score) ** 2)
    z = num / np.sqrt(denom) if denom > 0 else 0.0
    chi2 = z**2
    p = float(stats.chi2.sf(chi2, 1))
    return float(chi2), p


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_p(p: float) -> str:
    """A P value printed to four decimals, with >0.9999 and
    <0.0001 at the extremes."""
    if p != p:
        return "n/a"
    if p > 0.9999:
        return ">0.9999"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _p_summary(p: float) -> str:
    if p != p:
        return "n/a"
    if p >= 0.05:
        return "ns"
    if p >= 0.01:
        return "*"
    if p >= 0.001:
        return "**"
    if p >= 0.0001:
        return "***"
    return "****"


def _num(v: float, sig: int) -> str:
    if v != v:
        return "Undefined"
    if math.isinf(v):
        return "Infinity"
    return f"{v:.{sig}g}"


def _ci(lo: float, hi: float, sig: int) -> str:
    return f"{_num(lo, sig)} to {_num(hi, sig)}"


def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _ratio(numer: float, denom: float) -> float:
    if denom == 0:
        return math.nan if numer == 0 else math.inf
    return numer / denom


def _reciprocal(v: float) -> float:
    if v != v:
        return math.nan
    if v == 0:
        return math.inf
    return 0.0 if math.isinf(v) else 1.0 / v


def _recip_ci(lo: float, hi: float) -> tuple[float, float]:
    """The interval of 1/ratio from the interval of a ratio (0 <-> infinity)."""
    return (0.0 if math.isinf(hi) else 1.0 / hi), (math.inf if lo == 0 else 1.0 / lo)


def _row_labels(data: SheetData, row_mask: np.ndarray) -> list[str]:
    """Row titles typed into the sheet's first column, falling back to
    "Row N" for any row left untitled."""
    names = list(data.row_names or [])
    out = []
    for idx in np.where(row_mask)[0]:
        title = names[idx].strip() if idx < len(names) and names[idx] else ""
        out.append(title or f"Row {len(out) + 1}")
    return out


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------
@register("chi_square_test")
def run(data: SheetData, options: ChiSquareOptions | None = None) -> AnalysisResult:
    opt = options or ChiSquareOptions()
    labels, matrix = data.first_replicate_matrix()
    if matrix.shape[1] < 2:
        raise AnalysisError(f"This test needs at least 2 group columns of counts; found {matrix.shape[1]}.")

    row_mask = ~np.any(np.isnan(matrix), axis=1)
    table_data = matrix[row_mask]
    if table_data.shape[0] < 2:
        raise AnalysisError("Need at least 2 complete rows (categories) of counts.")
    if np.any(table_data < 0):
        raise AnalysisError("Counts can't be negative - check the sheet's values.")
    if np.any(table_data > 1e12) or not np.isfinite(table_data.sum()):
        raise AnalysisError("These numbers are too large to be counts of subjects. This analysis needs the counts themselves.")
    if np.any(table_data.sum(axis=0) == 0) or np.any(table_data.sum(axis=1) == 0):
        raise AnalysisError(
            "A row or column of the table adds up to zero, so there is nothing to compare - "
            "remove it or enter its counts."
        )

    row_labels = _row_labels(data, row_mask)
    is_2x2 = table_data.shape == (2, 2)
    whole_counts = cs.is_whole(*table_data.ravel())
    sig = opt.significant_digits
    conf = opt.confidence_level
    pct_conf = int(round(conf * 100))
    one_sided = opt.tails == "one"
    notes = ["Each group's first replicate column is read as a column of category counts; rows with any missing value are excluded."]

    row_totals = table_data.sum(axis=1)
    col_totals = table_data.sum(axis=0)
    grand = table_data.sum()
    expected = np.outer(row_totals, col_totals) / grand

    # ---- P value ----------------------------------------------------------
    chi2 = float("nan")
    dof = int((table_data.shape[0] - 1) * (table_data.shape[1] - 1))
    method = opt.p_method
    auto_reason = ""
    if method == "auto":
        small_expected = bool(np.any(expected < 5))
        exact_ok = whole_counts and (is_2x2 or (grand <= 2000 and table_data.size <= 20))
        if exact_ok and (is_2x2 or small_expected):
            method = "fisher"
            auto_reason = ("this is a 2x2 table, where the exact test is always available" if is_2x2 and not small_expected
                           else "some expected counts are below 5")
        else:
            method = "chi_square"
            auto_reason = ("every expected count is 5 or more" if not small_expected
                           else "the table is too large for the exact test")
    if method == "trend":
        if table_data.shape[1] != 2:
            raise AnalysisError("Chi-square test for trend needs exactly 2 columns (e.g. event / no-event) across ordered rows.")
        chi2, p = _cochran_armitage_trend(table_data)
        dof = 1
        test_name = "Chi-square test for trend"
    elif method == "chi_square":
        chi2, p, dof, _e = stats.chi2_contingency(table_data, correction=False)
        test_name = "Chi-square test"
    elif method == "yates":
        chi2, p, dof, _e = stats.chi2_contingency(table_data, correction=is_2x2)
        test_name = "Yates' continuity corrected chi-square test" if is_2x2 else "Chi-square test (Yates' correction only applies to 2x2 tables)"
    else:  # fisher
        if is_2x2:
            if not whole_counts:
                raise AnalysisError("Fisher's exact test needs whole-number counts. Choose the chi-square test for fractional data.")
            a, b, c, d = (int(v) for v in table_data.ravel())
            two_sided = float(stats.fisher_exact(table_data, alternative="two-sided")[1])
            if one_sided:
                direction = "greater" if a * d >= b * c else "less"
                p = float(stats.fisher_exact(table_data, alternative=direction)[1])
            else:
                p = two_sided
            dof = 1
            test_name = "Fisher's exact test"
        elif whole_counts:
            chi2, _p_chi, dof, _e = stats.chi2_contingency(table_data, correction=False)
            p, how = _common.fisher_rxc(table_data.astype(int))
            test_name = f"Fisher's exact test (Freeman-Halton, {how})"
            if one_sided:
                notes.append("One-sided P values are only defined for 2x2 tables (or the trend test); the two-sided P value is shown.")
                one_sided = False
        else:
            chi2, p, dof, _e = stats.chi2_contingency(table_data, correction=False)
            test_name = "Chi-square test (the exact test needs whole-number counts)"
            if one_sided:
                notes.append("One-sided P values are only defined for 2x2 tables (or the trend test); the two-sided P value is shown.")
                one_sided = False
    if auto_reason:
        notes.append(f"{test_name.split(' (')[0]} was chosen because {auto_reason}.")

    # Chi-square statistics are two-sided by construction; for a 2x2 (df = 1)
    # the one-sided P is half of it, in the direction the data lean.
    if one_sided and method != "fisher":
        if is_2x2 or opt.p_method == "trend":
            p = p / 2.0
        else:
            notes.append("One-sided P values are only defined for 2x2 tables (or the trend test); the two-sided P value is shown.")
            one_sided = False
    if method != "fisher":  # a chi-square approximation was used
        low = int(np.sum(expected < 5))
        # Cochran's rule: unreliable if any expected count is under 1, or
        # more than 20% are under 5 (any at all, for a 2x2 table).
        if low and (is_2x2 or low / expected.size > 0.2 or expected.min() < 1):
            advice = " Fisher's exact test is the safer choice for a 2x2 table." if is_2x2 else ""
            notes.append(
                f"{low} of {expected.size} cells have an expected count below 5 (smallest {expected.min():.3g}), "
                f"so the chi-square approximation may be unreliable.{advice}"
            )

    # the test, on one row: what ran, its statistic, P, the stars, the sides, and the verdict
    tails_label = "One-sided" if one_sided else "Two-sided"
    p_table = ResultTable(
        title="The test",
        headers=["Test", "Chi-square (df)", "P value", "Stars", "Sides", f"Significant at α = {ALPHA:g}"],
        rows=[[test_name, f"{chi2:.4f} ({dof})" if chi2 == chi2 else "–", _p(p, opt), _p_summary(p),
               tails_label, "Yes" if p < ALPHA else "No"]])

    # ---- Effect sizes (2x2 only) -------------------------------------------
    effect_tables: list[ResultTable] = []
    wants_effects = any((opt.report_relative_risk, opt.report_risk_difference, opt.report_odds_ratio, opt.report_sensitivity_specificity))
    if is_2x2 and wants_effects:
        a, b, c, d = (float(v) for v in table_data.ravel())
        n1, n2 = a + b, c + d  # row totals
        m1, m2 = a + c, b + d  # column totals
        p1, p2 = a / n1, c / n2
        rows: list[list[str]] = []

        if opt.report_relative_risk:
            rr = _ratio(p1, p2)
            lo, hi = cs.koopman_rr_ci(a, n1, c, n2, conf)
            rlo, rhi = _recip_ci(lo, hi)
            rows.append([RR, _num(rr, sig), _ci(lo, hi, sig), "Koopman score"])
            rows.append([RR_REVERSED, _num(_reciprocal(rr), sig), _ci(rlo, rhi, sig), "Koopman score"])

        if opt.report_risk_difference:
            diff = p1 - p2
            lo, hi = cs.newcombe_cc_difference(a, n1, c, n2, conf)
            nnt = 1.0 / abs(diff) if diff else math.inf
            rows.append([RISK_DIFFERENCE, _num(diff, sig), _ci(lo, hi, sig), "Newcombe (Wilson, continuity-corrected)"])
            rows.append([NNT, _num(nnt, sig) if diff else "Undefined", "", "1 / risk difference"])
            if lo < 0 < hi:
                notes.append("The interval for the difference between proportions includes 0, so the NNT is not meaningfully different from 'infinite'.")

        if opt.report_odds_ratio:
            odds = _ratio(a * d, b * c)
            if whole_counts:
                lo, hi, exact = cs.baptista_pike_or_ci(a, b, c, d, conf)
                ci_text = _ci(lo, hi, sig)
                rlo, rhi = _recip_ci(lo, hi)
                recip_ci_text = _ci(rlo, rhi, sig)
                if not exact:
                    notes.append("The counts are too large for the exact odds-ratio interval; the Woolf log-odds interval is shown instead.")
            else:
                ci_text = recip_ci_text = "n/a (counts must be whole numbers)"
            method = ("Baptista-Pike (mid-p)" if exact else "Woolf (log odds)") if whole_counts else "n/a"
            rows.append([OR, _num(odds, sig), ci_text, method])
            rows.append([OR_REVERSED, _num(_reciprocal(odds), sig), recip_ci_text, method])

        if opt.report_sensitivity_specificity:
            sens, spec, ppv, npv = a / m1, d / m2, a / n1, d / n2
            wb = "Wilson, exact bounds for small counts (Brown)"
            rows.append(["Sensitivity", _num(sens, sig), _ci(*cs.wilson_brown(a, m1, conf), sig), wb])
            rows.append(["Specificity", _num(spec, sig), _ci(*cs.wilson_brown(d, m2, conf), sig), wb])
            rows.append(["Positive predictive value", _num(ppv, sig), _ci(*cs.wilson_brown(a, n1, conf), sig), wb])
            rows.append(["Negative predictive value", _num(npv, sig), _ci(*cs.wilson_brown(d, n2, conf), sig), wb])
            rows.append(["Positive likelihood ratio", _num(_ratio(sens, 1.0 - spec), sig), "", "sensitivity / (1 − specificity)"])

        effect_tables.append(ResultTable(title="Size of the effect",
                                         headers=["Measure", "Value", f"{pct_conf}% CI", "Interval by"], rows=rows))

    # ---- strength of association, any table --------------------------------
    chi2_plain = float(stats.chi2_contingency(table_data, correction=False)[0])
    k_min = min(table_data.shape) - 1
    cramers_v = float(np.sqrt(chi2_plain / (grand * k_min))) if grand and k_min else float("nan")
    cuts = {1: (0.1, 0.3, 0.5), 2: (0.07, 0.21, 0.35)}.get(k_min, (0.06, 0.17, 0.29))
    strength = ResultTable(title="Strength of association", headers=["Measure", "Value", "Size"],
                           rows=[["Cramér's V" + (" (= |phi| for a 2x2 table)" if is_2x2 else ""),
                                  _num(cramers_v, 3), _common.magnitude(cramers_v, cuts)]])

    # ---- Data tables ---------------------------------------------------------
    headers = [""] + list(labels) + ["Total"]
    analyzed_rows = [[name] + [f"{v:g}" for v in row] + [f"{tot:g}"] for name, row, tot in zip(row_labels, table_data, row_totals)]
    analyzed_rows.append(["Total"] + [f"{v:g}" for v in col_totals] + [f"{grand:g}"])
    analyzed = ResultTable(title="Counts analysed, with totals", headers=headers, rows=analyzed_rows)

    def _pct_table(title: str, values: np.ndarray) -> ResultTable:
        rows_ = [[name] + [_pct(v) for v in row] for name, row in zip(row_labels, values)]
        return ResultTable(title=title, headers=[""] + list(labels), rows=rows_)

    pct_tables = [
        _pct_table(SHARE_OF_ROW, table_data / row_totals[:, None]),
        _pct_table(SHARE_OF_COLUMN, table_data / col_totals[None, :]),
        _pct_table(SHARE_OF_TABLE, table_data / grand),
    ]

    summary = (f"{table_data.shape[0]}×{table_data.shape[1]} table; {test_name.split(' (')[0]} P = {_p(p, opt)}; "
               f"Cramér's V = {cramers_v:.2f}.")
    return AnalysisResult(
        title="Compare counts",
        summary=summary,
        tables=effect_tables + [strength, p_table, analyzed] + pct_tables,
        notes=notes,
    )
