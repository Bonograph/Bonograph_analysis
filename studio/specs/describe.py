"""Describe the data: summaries down columns or across rows, the shape of the
values, values out of line, and batches of P values."""

from __future__ import annotations

from ...backends.batch_pvalue_analysis import BatchPValueOptions
from ...backends.descriptive_statistics import DescriptiveStatsOptions
from ...backends.frequency_distribution import FrequencyDistributionOptions
from ...backends.normality_testing import NormalityOptions
from ...backends.outlier_detection import OutlierOptions
from ...backends.variance_tests import VarianceOptions
from ...backends.row_wise_statistics import RowStatisticsOptions
from ..spec import Ask, Chips, Choose, Heading, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch
from .common import confidence, join, repeats
from .sketches import grid_glyph

_REPEATS = [("average", "average each row's repeats"), ("separate", "treat each repeat column on its own"),
            ("pooled", "pool every value together")]

# ── each column ─────────────────────────────────────────────────────────────

_STAT_NAMES = {
    "mean_sd_sem": "mean, SD and SEM", "geometric_mean": "geometric mean", "harmonic_mean": "harmonic mean",
    "quadratic_mean": "quadratic mean", "min_max_range": "minimum, maximum and range",
    "quartiles": "median and quartiles", "coefficient_of_variation": "coefficient of variation",
    "skewness_kurtosis": "skewness and kurtosis", "column_sum": "total",
}


def _summary_methods(s, ctx) -> str:
    chosen = [name for key, name in _STAT_NAMES.items() if s.get(key)]
    if s.get("percentile"):
        chosen.append(f"{s['percentile_value']:g}th percentile")
    cis = [w for key, w in (("ci_mean", "mean"), ("ci_median", "median"), ("ci_geometric_mean", "geometric mean"),
                            ("ci_harmonic_mean", "harmonic mean")) if s.get(key)]
    text = f"Each data set is summarised by its {join(chosen) or 'values'}"
    if cis:
        text += f", with {confidence()} confidence intervals for the {join(cis)}"
    return text + "."


COLUMNS = Spec(
    "descriptive_statistics", DescriptiveStatsOptions,
    [
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("What is a typical value?",
            "The mean suits values that scatter evenly; the median, skewed ones; the geometric mean, values that "
            "multiply - ratios, titres, concentrations.",
            pieces=[("mean_sd_sem", "Mean, with SD and SEM"), ("quartiles", "Median and quartiles"),
                    ("geometric_mean", "Geometric mean"),
                    ("trimmed_mean", "Trimmed mean", "The mean with a share of the values cut from each end: a mean that "
                                                      "a few odd values cannot drag."),
                    ("harmonic_mean", "Harmonic mean", "For rates averaged over equal distances or amounts."),
                    ("quadratic_mean", "Quadratic mean (RMS)", "For values that swing either side of zero.")],
            first=["mean_sd_sem", "quartiles"],
            inside=[Sentence(["Trim", Num("trim_percent", 0, 49.9, 1, 58, " %"), "from each end"],
                             when=lambda s, c: s["trimmed_mean"])]),
        Ask("How sure is that typical value?",
            "An interval that would hold the true value in the stated share of repeated studies.",
            pieces=[("ci_mean", "Interval for the mean"), ("ci_median", "Interval for the median"),
                    ("ci_geometric_mean", "Interval for the geometric mean"),
                    ("ci_harmonic_mean", "Interval for the harmonic mean"),
                    ("ci_sd", "Interval for the SD", "Trusts a normal shape - far more than the interval for the mean does.")],
            first=["ci_mean"]),
        Ask("How far do the values spread?",
            "From the lowest to the highest, relative to the mean, or up to a point you choose.",
            pieces=[("min_max_range", "Smallest, largest and range"),
                    ("coefficient_of_variation", "Coefficient of variation", "The SD as a share of the mean."),
                    ("robust_spread", "Sturdy measures: IQR, MAD and robust SD",
                     "Spread judged from the middle of the data, so a few odd values do not inflate it."),
                    ("geometric_sd", "Geometric SD factor",
                     "For values that multiply: the geometric mean times and divided by this spans about two-thirds of them."),
                    ("percentile", "A percentile of my choosing")],
            first=["min_max_range"],
            inside=[Sentence(["Give the", Num("percentile_value", 0, 100, 2, 60), "th percentile"],
                             when=lambda s, c: s["percentile"])]),
        Ask("Is the distribution lopsided or heavy-tailed?",
            "Skewness is lean to one side; kurtosis is weight in the tails. Both are zero for a normal distribution.",
            pieces=[("skewness_kurtosis", "Skewness and kurtosis")]),
        Ask("What do the values add up to?", "The total of each column.", pieces=[("column_sum", "Total")]),
        Row("Side-by-side repeats", Pick("subcolumn_mode", _REPEATS), when=lambda s, c: repeats(c)),
    ],
    methods=_summary_methods,
)

# ── each row ────────────────────────────────────────────────────────────────

ROWS = Spec(
    "row_wise_statistics", RowStatisticsOptions,
    [
        Heading("Which values make up a row's summary?"),
        Choose("scope", [Opt("per_column", "Each data set's own", "Its side-by-side repeats on that row.",
                             glyph=grid_glyph("each")),
                         Opt("whole_row", "The whole row", "Every value on the row, whichever data set it sits in.",
                             glyph=grid_glyph("row"))]),
        Heading("Summarise them by the"),
        Choose("central_stat", [Opt("means", "Mean"), Opt("medians", "Median"), Opt("geometric means", "Geometric mean")],
               style="segmented"),
        Heading("Together with"),
        Choose("companion_stat", [Opt("sd_n", "SD"), Opt("sem_n", "SEM"), Opt("ci_n", "Confidence interval"),
                                  Opt("min_max", "Lowest and highest")], style="segmented"),
        Note("How many values the row held is always given beside them.", tone="faint"),
        Switch("create_new_sheet", "Also put the row summaries on a new sheet", "Ready to graph."),
    ],
    methods=lambda s, c: (f"Each row was summarised by its {s['central_stat'].rstrip('s')}, "
                          f"{'inside each data set' if s['scope'] == 'per_column' else 'across all data sets'}."),
)

# ── histogram ───────────────────────────────────────────────────────────────


def _histogram_sheet(panel) -> None:
    rng = panel.ctx.y_range()
    if rng is not None:
        lo, hi = rng
        panel.suggest({"first_bin_center": round(lo, 6), "last_bin_center": round(hi, 6)})


def _histogram_finish(s, ctx) -> None:
    own = bool(s.pop("_own_range", False))
    s["first_bin_center_auto"] = not own
    s["last_bin_center_auto"] = not own


HISTOGRAM = Spec(
    "frequency_distribution", FrequencyDistributionOptions,
    [
        Heading("Count"),
        Choose("create_cumulative", [
            Opt(False, "How many fall in each bin", "A histogram.", phrase="frequency distribution"),
            Opt(True, "How many fall at or below each bin", "A cumulative distribution - climbs to the total.",
                phrase="cumulative distribution"),
        ]),
        Sentence(["and give them as", Pick("tabulate", [("count", "counts"), ("fraction", "fractions of each data set"),
                                                       ("percent", "percentages of each data set")])]),
        Heading("Bins"),
        Choose("bin_width_mode", [
            Opt("auto", "Automatic", "Width chosen from the spread and the number of values."),
            Opt("fixed", "A width I choose", "The same width for every data set, so they line up."),
            Opt("exact", "One per distinct value", "For whole numbers such as counts or scores."),
        ], columns=3),
        Sentence(["Each bin", Num("bin_width", 1e-9, 1e12, 6, 80), "wide"], when=lambda s, c: s["bin_width_mode"] == "fixed"),
        Switch("_own_range", "Set where the bins start and stop"),
        Sentence(["First bin centred on", Num("first_bin_center", decimals=6, width=80), ", last on",
                  Num("last_bin_center", decimals=6, width=80)], when=lambda s, c: s["_own_range"]),
        Row("Side-by-side repeats", Pick("replicate_mode", [("means", "bin each row's mean"),
                                                             ("each", "bin every repeat")]),
            when=lambda s, c: repeats(c)),
        Heading("Show"),
        Chips([("", [("graph_results", "Draw it"), ("new_sheet", "Put the table on a new sheet")])]),
        Row("Draw it as", Pick("graph_kind", [("bars", "bars"), ("line", "a line")]),
            when=lambda s, c: s["graph_results"] and not s["create_cumulative"]),
    ],
    virtual={"_own_range": False},
    finish=_histogram_finish,
    on_sheet=_histogram_sheet,
    methods=lambda s, c: "Values were binned into a " + ("cumulative distribution." if s["create_cumulative"]
                                                          else "frequency distribution."),
)

# ── normal? ─────────────────────────────────────────────────────────────────

_NORMALITY_TESTS = (("method_shapiro", "Shapiro-Wilk"), ("method_dagostino", "D'Agostino-Pearson"),
                    ("method_anderson", "Anderson-Darling"), ("method_ks", "Kolmogorov-Smirnov (Lilliefors)"))


def _normality_methods(s, ctx) -> str:
    shapes = join([w for k, w in (("test_normal", "normal"), ("test_lognormal", "lognormal")) if s[k]])
    tests = join([w for k, w in _NORMALITY_TESTS if s[k]])
    return f"Each data set was tested for a {shapes or 'normal'} shape with the {tests or 'chosen'} test(s)."


NORMAL = Spec(
    "normality_testing", NormalityOptions,
    [
        Heading("What do you want to know?"),
        Ask("Could the values come from a normal distribution?",
            "The bell shape that t tests, ANOVA and least squares assume.", pieces=[("test_normal", "Test for normality")]),
        Ask("Could they come from a lognormal one?",
            "Values that are bell-shaped once logged: common for concentrations, titres and times.",
            pieces=[("test_lognormal", "Test for lognormality")]),
        Ask("Which of the two fits better?",
            "The two shapes compared by AICc, as how many times likelier one is than the other.",
            pieces=[("compute_relative_likelihood", "Normal against lognormal")]),
        Heading("Judge it with"),
        Chips([("Tests", list(_NORMALITY_TESTS)), ("Graph", [("create_qq_plot", "Q–Q plot")])]),
        Note("Small samples hide real departures and large ones flag trivial ones - the Q–Q plot is often the "
             "better guide."),
        Row("Side-by-side repeats", Pick("subcolumns", _REPEATS), when=lambda s, c: repeats(c)),
    ],
    methods=_normality_methods,
)

# ── out of line ─────────────────────────────────────────────────────────────

_OUTLIER_TESTS = {"esd": "the generalized ESD test", "grubbs": "Grubbs' test",
                  "iterative_grubbs": "Grubbs' test, repeated", "fences": "the box-plot rule",
                  "robust": "robust z-scores"}


def _outlier_methods(s, c) -> str:
    method = s["method"]
    if method == "fences":
        return (f"Outliers were sought with the box-plot rule (Tukey): values more than {s['fence_factor']:g} interquartile ranges "
                "beyond a quartile.")
    if method == "robust":
        return (f"Outliers were sought with robust z-scores (Iglewicz and Hoaglin): values whose score, "
                f"0.6745 × (value - median) / MAD, was beyond {s['robust_cut']:g}.")
    return f"Outliers were sought with {_OUTLIER_TESTS[method]} at α = {s['q_percent'] / 100:g}."

OUTLIERS = Spec(
    "outlier_detection", OutlierOptions,
    [
        Heading("Look for"),
        Choose("method", [
            Opt("esd", "Any number of outliers", "Generalized ESD (Rosner) - several at once, without one hiding another.",
                phrase="the generalized ESD test"),
            Opt("grubbs", "At most one", "Grubbs' test on the most extreme value.", phrase="Grubbs' test"),
            Opt("iterative_grubbs", "Several, one at a time", "Grubbs' test repeated - each removal shifts the next "
                                                             "test, so use with care.",
                phrase="Grubbs' test, repeated"),
            Opt("fences", "Beyond the box-plot limits", "Tukey's rule on the quartiles - no normal distribution assumed; "
                                                        "a screen, not a test.", phrase="the box-plot rule"),
            Opt("robust", "Far from the median", "Robust z-scores from the median and its deviation - one wild value "
                                                 "cannot hide another.", phrase="robust z-scores"),
        ], columns=3),
        Sentence(["Accept a", Num("q_percent", 0.01, 20, 2, 58, " %"), "chance of flagging a value that is fine"],
                 when=lambda s, c: s["method"] not in ("fences", "robust")),
        Sentence(["Flag values more than", Num("fence_factor", 0.5, 10, 2, 58), "interquartile ranges beyond a quartile"],
                 when=lambda s, c: s["method"] == "fences"),
        Sentence(["Flag values whose robust z-score is beyond", Num("robust_cut", 1, 10, 2, 58)],
                 when=lambda s, c: s["method"] == "robust"),
        Heading("What else do you want to know?"),
        Ask("Why was a value flagged - or not?",
            "The working of the rule for each group: its statistic, the value it had to beat, and the limits.",
            pieces=[("show_details", "The working of the rule")]),
        Ask("Where do the flagged values sit?",
            "Every value against its row with the flagged ones in red, and the values at each end of every group.",
            pieces=[("show_order_graph", "Values in sheet order, with the limits"), ("show_box_graph", "Box plot"),
                    ("show_ends", "The five lowest and highest values")],
            first=["show_order_graph"]),
        Ask("Would leaving them out change the picture?",
            "Mean, SD and median of each group, with the flagged values and without.",
            pieces=[("show_effect", "With and without")]),
        Switch("make_clean_copy", "Make a copy without the flagged values",
               "A new sheet with the flagged values left blank. The sheet you analyse is never changed."),
        Note("Nothing is removed from the sheet: flagged values are listed, with their rows, for you to judge."),
        Row("Side-by-side repeats", Pick("subcolumns", _REPEATS), when=lambda s, c: repeats(c)),
    ],
    methods=_outlier_methods,
)

# ── the same spread? ────────────────────────────────────────────────────────

_SPREAD_TESTS = (("brown_forsythe", "the Brown-Forsythe test"), ("levene", "Levene's test"), ("bartlett", "Bartlett's test"),
                 ("fligner", "the Fligner-Killeen test"))


def _spread_methods(s, c) -> str:
    tests = [name for key, name in _SPREAD_TESTS if s[key]]
    if s["f_test"] and c.groups == 2:
        tests.append("the F test of two variances")
    if not tests:
        return "No test of equal spread was chosen."
    return "Equality of variances was checked with " + (", ".join(tests[:-1]) + " and " + tests[-1] if len(tests) > 1 else tests[0]) + "."


SPREAD = Spec(
    "variance_tests", VarianceOptions,
    [
        Heading("Judge it with"),
        Chips([("", [("brown_forsythe", "Brown-Forsythe", "Distances from each group's median. Sturdy against skew and odd values: "
                      "the one to use by default."),
                     ("levene", "Levene", "Distances from each group's mean - the original form."),
                     ("bartlett", "Bartlett", "The most power when the values are normal; misled when they are not."),
                     ("fligner", "Fligner-Killeen", "On ranks, for clearly non-normal values."),
                     ("f_test", "F test of two variances", "Two groups only; trusts a normal shape.")])]),
        Note("Welch's t test and Welch's ANOVA do not need equal spreads, so this check is only needed when you want the "
             "ordinary forms - or when the spread itself is the question."),
        Row("Side-by-side repeats", Pick("subcolumns", [("average", "average each row's repeats"), ("pooled", "pool every value")]),
            when=lambda s, c: repeats(c)),
    ],
    methods=_spread_methods,
)

# ── many P values ───────────────────────────────────────────────────────────

P_VALUES = Spec(
    "batch_pvalue_analysis", BatchPValueOptions,
    [
        Heading("The column holds P values worked out elsewhere, one per row. What matters more?"),
        Choose("approach", [
            Opt("fdr", "Few false leads among the finds", "Limit the share of discoveries that are false - "
                                                         "for screening many results.", phrase="false discovery rate"),
            Opt("significance", "No false positive at all", "Limit the chance of even one - when every claim "
                                                           "must stand.", phrase="family-wise error"),
        ]),
        Sentence(["Use", Pick("fdr_method", [("two_stage_bky", "the two-stage method (Benjamini, Krieger, Yekutieli)"),
                                             ("benjamini_hochberg", "Benjamini-Hochberg"),
                                             ("benjamini_yekutieli", "Benjamini-Yekutieli (any dependence)")]),
                  "and accept up to", Num("desired_fdr_percent", 0.01, 50, 2, 58, " %"), "false discoveries"],
                 when=lambda s, c: s["approach"] == "fdr"),
        Sentence(["Adjust with", Pick("sig_method", [("holm_sidak", "Holm-Šidák"), ("bonferroni", "Bonferroni"),
                                                     ("none", "no adjustment")])],
                 when=lambda s, c: s["approach"] == "significance"),
        Heading("Also report"),
        Chips([("", [("graph_ranked", "Graph of the P values in rank order")])]),
    ],
    methods=lambda s, c: ("P values were screened controlling the false discovery rate at "
                          f"{s['desired_fdr_percent']:g}%." if s["approach"] == "fdr" else
                          "P values were adjusted to control the family-wise error rate."),
)

SPECS = [SPREAD, COLUMNS, ROWS, HISTOGRAM, NORMAL, OUTLIERS, P_VALUES]
