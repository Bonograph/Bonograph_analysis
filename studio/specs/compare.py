"""Compare groups: two groups, three or more, one against a value, factorial
designs, pairs one at a time, and subsamples nested within subjects."""

from __future__ import annotations

from ...backends.ancova import AncovaOptions
from ...backends.effect_sizes import EffectSizeOptions
from ...backends.mixed_model import MixedModelOptions
from ...backends.multiple_comparisons import MultipleComparisonsOptions
from ...backends.nested_one_way_anova import NestedOneWayAnovaOptions
from ...backends.nested_t_test import NestedTTestOptions
from ...backends.one_sample_test import OneSampleOptions
from ...backends.one_way_anova import OneWayAnovaOptions
from ...backends.power_sample_size import PowerOptions
from ...backends.three_way_anova import ThreeWayAnovaOptions
from ...backends.two_group_comparison import TwoGroupOptions
from ...backends.two_way_anova import TwoWayAnovaOptions
from ..spec import (Ask, Checks, Chips, Choose, Drawer, Heading, Int, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch,
                    Txt)
from .common import CORRECTIONS, a_b, columns, confidence, correction_pick, paired, repeats, shape_cards, skewed

# ── two groups ──────────────────────────────────────────────────────────────


def _two_group_test(s) -> str:
    if s["distribution"] == "nonparametric":
        return "Wilcoxon matched-pairs signed-rank test" if s["paired"] else "Mann-Whitney test"
    logged = " on logarithms" if s["distribution"] == "lognormal" else ""
    if s["paired"]:
        return "paired t test" + logged
    return ("Student's t test" if s["assume_equal_variance"] else "Welch's t test") + logged


def _two_group_methods(s, ctx) -> str:
    a, b = a_b(ctx)
    side = {"two-sided": "two-sided", "less": f"one-sided, {b} greater", "greater": f"one-sided, {a} greater"}
    what = {"gaussian": "difference between means", "lognormal": "ratio of geometric means",
            "nonparametric": "median difference (Hodges-Lehmann)"}[s["distribution"]]
    text = (f"{a} and {b} were compared using {_two_group_test(s)} ({side[s['alternative']]}), "
            f"reporting the {what} with its {confidence()} confidence interval.")
    if s.get("equivalence"):
        zone = (f"a ratio between 1/{s['equivalence_ratio']:g} and {s['equivalence_ratio']:g}" if s["distribution"] == "lognormal"
                else f"a difference within ±{s['equivalence_margin']:g}")
        text += f" Equivalence was tested by two one-sided tests, taking {zone} as equivalent."
    if s.get("bootstrap") or s.get("permutation"):
        parts = (["a BCa bootstrap confidence interval"] if s.get("bootstrap") else []) + (
            ["a permutation test"] if s.get("permutation") else [])
        text += f" The comparison was repeated by resampling ({' and '.join(parts)}, {s['resamples']} resamples)."
    return text


def _sides(ctx):
    a, b = a_b(ctx)
    return [("two-sided", "in either direction"), ("less", f"only where {b} is larger"),
            ("greater", f"only where {a} is larger")]


def _diff_ways(ctx):
    a, b = a_b(ctx)
    return [("b_minus_a", f"{b} − {a}"), ("a_minus_b", f"{a} − {b}")]


TWO_GROUPS = Spec(
    "two_group_comparison", TwoGroupOptions,
    [
        Heading("How were the values collected?"),
        Choose("paired", [
            Opt(False, "Separate subjects", "Every value comes from a different subject, animal or sample.",
                phrase="unpaired"),
            Opt(True, "Paired", "Each row is one subject measured twice - before and after, left and right.",
                suggest=paired, phrase="paired"),
        ]),
        Heading("What shape are the values?"),
        Choose("distribution", shape_cards(), columns=3),
        Heading("Can the spread differ between the groups?",
                when=lambda s, c: not s["paired"] and s["distribution"] != "nonparametric"),
        Choose("assume_equal_variance", [
            Opt(False, "It may differ", "Welch's test - safe either way, and the recommended default."),
            Opt(True, "The spreads match", "Student's test - slightly more power when the spreads really match."),
        ], when=lambda s, c: not s["paired"] and s["distribution"] != "nonparametric"),
        Heading("What counts as a difference?"),
        Sentence(["Look for a difference", Pick("alternative", _sides), "and report it as",
                  Pick("report_diff_as", _diff_ways)]),
        Note("Test one direction only if it was fixed before the data were seen - a difference the other "
             "way can then never count.", tone="warn", when=lambda s, c: s["alternative"] != "two-sided"),
        Heading("What else do you want to know?"),
        Ask("What does each group look like?", "Its size, mean, SD and median, beside the test.",
            pieces=[("show_descriptive_stats", "Summary of each group")]),
        Ask("Are the assumptions of the test met?",
            "A plain verdict on the shape of the values and their spread, with the numbers behind it.",
            pieces=[("check_assumptions", "Verdict on shape and spread"), ("run_normality_diagnostics", "Normality of the residuals"),
                    ("show_homoscedasticity_table", "Spread of each group"), ("show_residual_table", "The residuals themselves")],
            first=["check_assumptions"]),
        Ask("Do the two groups share one spread?",
            "A model with one spread against a model with two, compared by AICc.",
            pieces=[("aicc_compare", "Same against different spread")]),
        Ask("Are the two groups close enough to count as the same?",
            "A large P above does not show that. This does: two one-sided tests that the difference lies inside a "
            "zone you fix in advance.",
            pieces=[("equivalence", "Equivalence test")],
            inside=[Sentence(["They count as the same if the difference is within ±", Num("equivalence_margin", 0, 1e12, 6, 84)],
                             when=lambda s, c: s["distribution"] != "lognormal"),
                    Sentence(["They count as the same if the ratio lies between 1 ÷", Num("equivalence_ratio", 1.0001, 1e6, 4, 70),
                              "and that number"], when=lambda s, c: s["distribution"] == "lognormal"),
                    Note("Choose the zone from the science, before looking at the data: the largest difference that would "
                         "change nothing you do.", tone="faint")]),
        Ask("Does the answer hold without the formulas?",
            "The data resampled thousands of times: no bell shape assumed, no formula trusted.",
            pieces=[("bootstrap", "Bootstrap interval for the difference"), ("permutation", "Permutation test")],
            inside=[Sentence(["Use", Int("resamples", 199, 200000, 78), "resamples"])]),
    ],
    methods=_two_group_methods,
)

# ── three or more groups ────────────────────────────────────────────────────


def _anova_test(s) -> str:
    if s["repeated_measures"]:
        test = "Friedman test" if s["distribution"] == "nonparametric" else "repeated-measures ANOVA"
    elif s["distribution"] == "nonparametric":
        test = "Kruskal-Wallis test"
    else:
        test = "ordinary one-way ANOVA" if s["assume_equal_variance"] else "Welch's ANOVA"
    if s["distribution"] == "lognormal":
        test += " on logarithms"
    return test


def _anova_methods(s, ctx) -> str:
    follow = {"all_pairs": "every pair of groups was then compared",
              "vs_control": f"each group was then compared with {ctx.name(s['control_index'])}",
              "linear_trend": "a linear trend across the groups was then tested",
              "none": ""}[s["post_hoc"]]
    adjust = (f", adjusting for multiple comparisons with {CORRECTIONS[s['correction']]}"
              if s["post_hoc"] in ("all_pairs", "vs_control") else "")
    first = f"The groups were compared using {_anova_test(s)}"
    return first + (f"; {follow}{adjust}." if follow else ".")


ONE_WAY = Spec(
    "one_way_anova", OneWayAnovaOptions,
    [
        Heading("How were the values collected?"),
        Choose("repeated_measures", [
            Opt(False, "Separate subjects", "Each group holds different subjects."),
            Opt(True, "The same subjects throughout", "One subject per row, measured under every condition."),
        ]),
        Heading("What shape are the values?"),
        Choose("distribution", shape_cards(), columns=3),
        Heading("Can the spread differ between groups?",
                when=lambda s, c: not s["repeated_measures"] and s["distribution"] != "nonparametric"),
        Choose("assume_equal_variance", [
            Opt(False, "It may differ", "Welch's ANOVA, with the Brown-Forsythe ANOVA alongside - safe either way."),
            Opt(True, "The spreads match", "Ordinary ANOVA - a little more power when the spreads really match."),
        ], when=lambda s, c: not s["repeated_measures"] and s["distribution"] != "nonparametric"),
        Heading("Then look at"),
        Choose("post_hoc", [
            Opt("all_pairs", "Every pair", "Find the pairs that differ."),
            Opt("vs_control", "Each against a control", "Fewer comparisons, so more power for each."),
            Opt("linear_trend", "A trend", "Do values rise or fall across the groups in their order?"),
            Opt("none", "Nothing more", "Just the overall test."),
        ], columns=2),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: s["post_hoc"] == "vs_control"),
        Row("Keep false positives in check with", correction_pick(),
            when=lambda s, c: s["post_hoc"] in ("all_pairs", "vs_control")),
        Heading("What else do you want to know?"),
        Ask("What does each group look like?", "Its size, mean, SD and median - and the values as a heat map.",
            pieces=[("show_descriptive_stats", "Summary of each group"), ("show_heatmap_plot", "Heat map of the values")],
            first=["show_descriptive_stats"]),
        Ask("Are the assumptions of the test met?",
            "A plain verdict on the shape of the values and their spread, with the plots and tests behind it.",
            pieces=[("check_assumptions", "Verdict on shape and spread"), ("show_residual_plot", "Plot of the residuals"),
                    ("show_qq_plot", "Q–Q plot of the residuals"), ("show_homoscedasticity_plot", "Spread against fitted value"),
                    ("diag_normality", "Normality of the residuals"), ("diag_heteroscedastic", "Tests of equal spread")],
            first=["check_assumptions"]),
    ],
    methods=_anova_methods,
)

# ── one group against a value ───────────────────────────────────────────────


def _one_sample_methods(s, ctx) -> str:
    test = {"t_test": "one-sample t test", "ratio_t_test": "one-sample t test on logarithms",
            "wilcoxon": "Wilcoxon signed-rank test"}[s["test"]]
    what = {"t_test": "mean", "ratio_t_test": "geometric mean", "wilcoxon": "median"}[s["test"]]
    return f"The {what} of each data set was compared with {s['hypothetical_value']:g} using the {test}."


ONE_SAMPLE = Spec(
    "one_sample_test", OneSampleOptions,
    [
        Heading("Compare with what?"),
        Sentence(["Is the typical value different from", Num("hypothetical_value", decimals=6, width=90), "?"]),
        Heading("Judge it by"),
        Choose("test", [
            Opt("t_test", "The mean", "t test - for roughly bell-shaped values."),
            Opt("ratio_t_test", "The geometric mean", "t test on logarithms - skewed positive values; "
                                                       "the value compared with must be above 0.",
                suggest=skewed),
            Opt("wilcoxon", "The median", "Signed-rank test - no assumption about shape."),
        ], columns=3),
        Row("A value exactly equal to it is", Pick("tie_handling", [("ignore", "left out"),
                                                                    ("pratt", "ranked, then dropped (Pratt)")]),
            when=lambda s, c: s["test"] == "wilcoxon"),
        Switch("calc_discrepancy_ci", "How far from the value, with a confidence interval",
               "The median difference (Hodges-Lehmann) and its interval.",
               when=lambda s, c: s["test"] == "wilcoxon"),
        Row("Side-by-side repeats", Pick("subcolumns", [("average", "average each row's repeats"),
                                                         ("separate", "test each repeat column"),
                                                         ("pooled", "pool every value")]),
            when=lambda s, c: repeats(c)),
    ],
    methods=_one_sample_methods,
)

# ── two factors ─────────────────────────────────────────────────────────────


def _two_way_finish(s, ctx) -> None:
    s["control_index"] = max(0, int(s.pop("_control_level", 1)) - 1)
    for key, fallback in (("column_factor_name", "Column factor"), ("row_factor_name", "Row factor"),
                          ("subject_name", "Subject")):
        s[key] = (s.get(key) or "").strip() or fallback


def _two_way_sheet(panel) -> None:
    if not repeats(panel.ctx):
        panel.suggest({"include_interaction": False})
        if panel.state.get("matching") == "subcolumn_subject":
            panel.state["matching"] = "none"


def _two_way_methods(s, ctx) -> str:
    design = {"none": "two-way ANOVA", "subcolumn_subject": "two-way repeated-measures ANOVA",
              "row_subject": "repeated-measures ANOVA with the columns as the repeated factor"}[s["matching"]]
    model = "including their interaction" if s["include_interaction"] else "main effects only"
    text = (f"{s['column_factor_name'] or 'Column factor'} and {s['row_factor_name'] or 'Row factor'} were "
            f"analysed by {design} (Type III sums of squares, {model})")
    if s["matching"] != "none" and not s["assume_sphericity"]:
        text += ", with the Geisser-Greenhouse correction"
    return text + "."


TWO_WAY = Spec(
    "two_way_anova", TwoWayAnovaOptions,
    [
        Heading("How were the subjects measured?"),
        Choose("matching", [
            Opt("none", "Separate subjects in every cell", "The usual layout: nothing is measured twice."),
            Opt("subcolumn_subject", "Each repeat column is a subject",
                "Followed down the rows - e.g. one animal per column, weighed each day.", available=repeats),
            Opt("row_subject", "Each row is a subject", "Measured once under every column."),
        ], columns=3),
        Heading("Can the factors interact?"),
        Choose("include_interaction", [
            Opt(True, "Test the interaction", "Does one factor's effect depend on the other? Needs repeats in "
                                              "each cell.", available=repeats),
            Opt(False, "Main effects only", "Each factor's effect, assumed the same at every level of the other."),
        ]),
        Choose("assume_sphericity", [
            Opt(False, "Correct for uneven variability", "Geisser-Greenhouse - the safe choice for repeated measures."),
            Opt(True, "Sphericity holds", "Only when differences between conditions are equally variable."),
        ], when=lambda s, c: s["matching"] != "none"),
        Heading("Name the factors", "The names label every table and graph."),
        Sentence(["The columns are levels of", Txt("column_factor_name", 130, "Column factor"),
                  ", the rows are levels of", Txt("row_factor_name", 130, "Row factor")]),
        Sentence(["and each subject is a", Txt("subject_name", 110, "Subject")],
                 when=lambda s, c: s["matching"] != "none"),
        Heading("Then compare"),
        Choose("compare", [
            Opt("none", "Nothing more", "Just the ANOVA table."),
            Opt("column_means", "The column factor's levels", "Averaged over the rows."),
            Opt("row_means", "The row factor's levels", "Averaged over the columns."),
        ], columns=3),
        Sentence(["Compare", Pick("comparison_scope", [("all_pairs", "every pair of levels"),
                                                       ("vs_control", "each level against level")]),
                  Int("_control_level", 1, 100, live=lambda s, c: s["comparison_scope"] == "vs_control"),
                  ", adjusting with", correction_pick(keys=("auto", "holm", "fdr", "none"))],
                 when=lambda s, c: s["compare"] != "none"),
        Switch("swap_direction", "Subtract the other way round", "B − A instead of A − B.",
               when=lambda s, c: s["compare"] != "none"),
        Heading("What else do you want to know?"),
        Ask("What is the mean at each level?", "The mean of every level of each factor, and of every cell.",
            pieces=[("show_group_means", "Mean of every level")]),
        Ask("How much of the variation does the model explain?", "R² of the whole model.",
            pieces=[("show_goodness_of_fit", "R² of the whole model")]),
        Ask("Are the assumptions of the test met?",
            "The leftovers should scatter evenly and roughly bell-shaped. These show whether they do.",
            pieces=[("show_residual_plot", "Plot of the residuals"), ("show_qq_plot", "Q–Q plot of the residuals"),
                    ("show_homoscedasticity_plot", "Spread against fitted value"),
                    ("diag_normality", "Normality of the residuals"), ("diag_heteroscedastic", "Does spread grow with the mean?")],
            first=["show_residual_plot", "show_qq_plot"]),
    ],
    virtual={"_control_level": 1},
    finish=_two_way_finish,
    on_sheet=_two_way_sheet,
    methods=_two_way_methods,
    start={"column_factor_name": "", "row_factor_name": "", "subject_name": ""},
)

# ── three factors ───────────────────────────────────────────────────────────

THREE_WAY = Spec(
    "three_way_anova", ThreeWayAnovaOptions,
    [
        Heading("Where are the three factors?",
                "Name each data set after its three levels - Male-Low-Fed, Female-High-Fasted - and the "
                "names are split into the three factors."),
        Row("Split the names at", Pick("delimiter", [("auto", "whatever separator they share"), ("-", "–  dash"),
                                                     ("_", "_  underscore"), ("|", "|  bar"), (":", ":  colon"),
                                                     ("/", "/  slash")])),
        Note(lambda s, c: (f"These names split into {c.shape.title_factors} factors."
                           if c.shape is not None and c.shape.title_factors else
                           "The names on this sheet don't split into three factors yet."),
             when=lambda s, c: c.shape is not None),
    ],
    methods=lambda s, c: "The three factors were analysed by three-way ANOVA with all interactions.",
)

# ── pairs one at a time ─────────────────────────────────────────────────────

PAIRS = Spec(
    "multiple_comparisons", MultipleComparisonsOptions,
    [
        Heading("Test each pair with", "No overall test first: use this when the pairs themselves are the "
                                       "question."),
        Choose("distribution", [
            Opt("gaussian", "Welch t tests", "Means; the spread may differ between groups."),
            Opt("nonparametric", "Mann-Whitney tests", "Ranks; no assumption about shape.", suggest=skewed),
        ]),
        Heading("Which pairs?"),
        Choose("scope", [Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against a control")],
               style="segmented"),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: s["scope"] == "vs_control"),
        Row("Keep false positives in check with", correction_pick(keys=("holm", "fdr", "none"))),
    ],
    methods=lambda s, c: (f"Pairs of groups were compared with {'Welch' if s['distribution'] == 'gaussian' else 'Mann-Whitney'} "
                          f"tests, adjusted with {CORRECTIONS[s['correction']]}."),
)

# ── subsamples inside subjects ─────────────────────────────────────────────

_ZERO = Row("If a subject-level variance comes out as zero",
            Pick("zero_variance_mode", [("keep", "keep it in the model"), ("simplify", "fit the simpler model"),
                                        ("full", "allow a negative estimate")]))
_NESTED_CHECKS = Ask("Are the assumptions of the test met?",
                     "The leftovers should scatter evenly and roughly bell-shaped. These show whether they do.",
                     pieces=[("show_residual_plot", "Plot of the residuals"), ("show_qq_plot", "Q–Q plot of the residuals"),
                             ("show_homoscedasticity_plot", "Spread against fitted value")])
_NESTED_FIT = Ask("Does it matter which subject a value came from?",
                  "How the variation divides between subjects and repeat measurements, and how well the model fits.",
                  pieces=[("report_goodness_of_fit", "Goodness of fit")])

NESTED_T = Spec(
    "nested_t_test", NestedTTestOptions,
    [
        Heading("How the data are laid out",
                "Each data set holds several subjects - one repeat column each - and every subject was measured "
                "more than once. The subject, not the single measurement, is what gets compared."),
        Sentence(["Report the difference as", Pick("report_diff_as", lambda c: [(k, t) for k, t in reversed(_diff_ways(c))])]),
        Heading("What else do you want to know?"),
        Ask("What size is the difference, and how sure is it?", "The difference between the two groups drawn with its interval.",
            pieces=[("graph_ci", "Graph of the difference with its interval")]),
        _NESTED_FIT,
        _NESTED_CHECKS,
        Drawer("Fine-tune", [_ZERO]),
    ],
    methods=lambda s, c: (f"{c.name(0)} and {c.name(1)} were compared with a nested t test (a mixed model with "
                          "subject as a random effect)."),
)

NESTED_ANOVA = Spec(
    "nested_one_way_anova", NestedOneWayAnovaOptions,
    [
        Heading("How the data are laid out",
                "Each data set holds several subjects - one repeat column each - measured more than once."),
        Heading("Then look at"),
        Choose("post_hoc", [Opt("none", "Nothing more", "Just the overall test."),
                            Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against a control")], columns=3),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: s["post_hoc"] == "vs_control"),
        Row("Keep false positives in check with", correction_pick(keys=("holm", "fdr", "none")),
            when=lambda s, c: s["post_hoc"] != "none"),
        Switch("swap_direction", "Subtract the other way round", "B − A instead of A − B.",
               when=lambda s, c: s["post_hoc"] != "none"),
        Heading("What else do you want to know?"),
        Ask("Where does each group sit?", "Each group's mean drawn with its interval.",
            pieces=[("graph_grand_mean_ci", "Each group's mean with its interval")]),
        Ask("What size is each difference, and how sure is it?",
            "Each compared pair: its difference with an interval, and a P value adjusted for the number of pairs.",
            pieces=[("graph_comparison_ci", "Graph of each difference with its interval"),
                    ("report_adjusted_p", "Adjusted P for each comparison")]),
        _NESTED_FIT,
        _NESTED_CHECKS,
        Drawer("Fine-tune", [_ZERO]),
    ],
    methods=lambda s, c: "The groups were compared with a nested one-way ANOVA (subject as a random effect).",
)

# ── how large ───────────────────────────────────────────────────────────────

_PAIR_WAYS = [Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against a control")]
_KEEP_HONEST = [("holm", "Holm's step-down method"), ("fdr", "a false-discovery-rate limit"), ("none", "no adjustment")]


def _effect_methods(s, ctx) -> str:
    several = ctx.groups >= 3
    text = ("Effect sizes are given as Hedges' g " + ("(from the paired differences) " if s["paired"] else "")
            + f"with {confidence()} confidence intervals from the noncentral t distribution")
    if several and not s["paired"]:
        text += ", and as omega squared for the design as a whole"
    if s["show_rank_based"]:
        text += "; the probability of superiority and the rank-biserial correlation are given as shape-free measures"
    return text + "."


EFFECT = Spec(
    "effect_sizes", EffectSizeOptions,
    [
        Heading("How were the values collected?"),
        Choose("paired", [
            Opt(False, "Separate subjects", "Every value comes from a different subject, animal or sample.", phrase="unpaired"),
            Opt(True, "Paired", "Each row is one subject measured under every condition.", suggest=paired, phrase="paired"),
        ]),
        Heading("Which pairs?", when=lambda s, c: c.groups >= 3),
        Choose("compare", _PAIR_WAYS, style="segmented", when=lambda s, c: c.groups >= 3),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: c.groups >= 3 and s["compare"] == "vs_control"),
        Heading("What else do you want to know?"),
        Ask("How often is a value from one group above a value from the other?",
            "A measure that assumes no shape: that probability, and the rank-biserial correlation.",
            pieces=[("show_rank_based", "Measures that assume no shape")]),
        Ask("Is that small, medium or large?", "Cohen's rough guide beside each value. A guide, not a rule: what "
                                                "counts as large depends on the field.",
            pieces=[("show_reading", "Cohen's guide")]),
        Note("An effect size says how big a difference is; a P value only how sure you can be it is not zero. "
             "Report both, with the interval."),
    ],
    methods=_effect_methods,
)

# ── allowing for a covariate ────────────────────────────────────────────────


def _ancova_methods(s, ctx) -> str:
    text = "Groups were compared by analysis of covariance with the X variable as covariate (Type III sums of squares)"
    if s["test_slopes"]:
        text += ", after testing that the groups share one slope"
    if s["compare"] != "none" and ctx.groups >= 2:
        how = {"holm": "Holm's method", "fdr": "the Benjamini-Hochberg method", "none": "no adjustment"}[s["correction"]]
        text += f"; adjusted means were compared pairwise with {how}"
    return text + "."


ANCOVA = Spec(
    "ancova", AncovaOptions,
    [
        Note("X is the covariate - a baseline, an age, a dose. Each data set holds one group's outcome at that X. "
             "A group's rows may be its own: blanks in the other data sets are fine."),
        Heading("First check"),
        Switch("test_slopes", "That the groups share one slope",
               "The comparison assumes the covariate acts alike in every group; this tests it and shows each group's own slope."),
        Heading("Then compare pairs of groups", when=lambda s, c: c.groups >= 3),
        Choose("compare", [Opt("none", "No")] + _PAIR_WAYS, style="segmented", when=lambda s, c: c.groups >= 3),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: c.groups >= 3 and s["compare"] == "vs_control"),
        Row("Keep false positives in check with", Pick("correction", _KEEP_HONEST),
            when=lambda s, c: c.groups >= 3 and s["compare"] != "none"),
        Heading("What else do you want to know?"),
        Ask("What did allowing for the covariate change?",
            "The same comparison with the covariate ignored, side by side with the adjusted one.",
            pieces=[("show_unadjusted", "With and without the covariate")]),
        Sentence(["Compare the groups at covariate =", Txt("at_value", width=110, placeholder="its mean")]),
    ],
    methods=_ancova_methods,
)

# ── repeated measures with gaps ─────────────────────────────────────────────


def _mixed_methods(s, ctx) -> str:
    text = ("Conditions were compared with a mixed-effects model fitted by restricted maximum likelihood, with condition as a "
            "fixed effect and subject as a random intercept, so that subjects with missing values were kept")
    if s["compare"] != "none":
        how = {"holm": "Holm's method", "fdr": "the Benjamini-Hochberg method", "none": "no adjustment"}[s["correction"]]
        text += f"; pairs of conditions were compared with {how}"
    return text + "."


MIXED = Spec(
    "mixed_model", MixedModelOptions,
    [
        Note("Each row is one subject, each data set a condition or time point. Leave a cell blank where a value is missing: "
             "the subject's other values are still used."),
        Heading("Then compare pairs of conditions"),
        Choose("compare", [Opt("none", "No"), Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against one"),
                           Opt("consecutive", "Each with the one before")], style="segmented"),
        Row("Compare each with", Pick("control_index", columns), when=lambda s, c: s["compare"] == "vs_control"),
        Row("Keep false positives in check with", Pick("correction", _KEEP_HONEST), when=lambda s, c: s["compare"] != "none"),
        Heading("What else do you want to know?"),
        Ask("What did each subject do?", "A line for every subject across the conditions, gaps and all.",
            pieces=[("show_subjects_graph", "Each subject across the conditions")]),
    ],
    methods=_mixed_methods,
)

# ── how many subjects ───────────────────────────────────────────────────────

_MEANS = ("two_means", "paired_means", "one_mean", "anova")


def _power_methods(s, ctx) -> str:
    design = {"two_means": "a two-sample t test", "paired_means": "a paired t test", "one_mean": "a one-sample t test",
              "anova": f"a one-way ANOVA of {s['groups']} groups", "two_proportions": "a comparison of two proportions",
              "correlation": "a test of a correlation"}[s["design"]]
    sides = "" if s["design"] == "anova" else (", two-sided" if s["two_sided"] else ", one-sided")
    if s["design"] in _MEANS:
        effect = f"a difference of {s['difference']:g} with an SD of {s['sd']:g}"
    elif s["design"] == "two_proportions":
        effect = f"proportions of {s['p1']:g} and {s['p2']:g}"
    else:
        effect = f"a correlation of {s['r']:g}"
    if s["solve_for"] == "n":
        return f"The sample size was chosen to give {s['power']:.0%} power to detect {effect} with {design} at α = {s['alpha']:g}{sides}."
    if s["solve_for"] == "power":
        return f"With {s['n']} subjects per group, the power of {design} to detect {effect} at α = {s['alpha']:g}{sides} was calculated."
    return f"The smallest effect detectable with {s['power']:.0%} power by {design} with {s['n']} subjects per group at α = {s['alpha']:g}{sides} was calculated."


POWER = Spec(
    "power_sample_size", PowerOptions,
    [
        Heading("What kind of study?"),
        Choose("design", [
            Opt("two_means", "Two separate groups", "Means compared with a t test."),
            Opt("paired_means", "The same subjects twice", "Before and after, left and right: a paired t test."),
            Opt("one_mean", "One group against a value", "A one-sample t test."),
            Opt("anova", "Three groups or more", "Means compared with a one-way ANOVA."),
            Opt("two_proportions", "Two proportions", "A yes/no outcome in two separate groups."),
            Opt("correlation", "A correlation", "Is a correlation different from zero?"),
        ], columns=3),
        Heading("Work out"),
        Choose("solve_for", [Opt("n", "The number of subjects"), Opt("power", "The power I have"), Opt("effect", "The effect I can find")],
               style="segmented"),
        Heading("The effect worth finding", when=lambda s, c: s["solve_for"] != "effect"),
        Sentence(["A difference of", Num("difference", -1e9, 1e9, 4, 80), "between the means"],
                 when=lambda s, c: s["design"] in ("two_means", "paired_means", "one_mean") and s["solve_for"] != "effect"),
        Sentence(["Group means up to", Num("difference", -1e9, 1e9, 4, 80), "apart, in", Int("groups", 3, 50, 50), "groups"],
                 when=lambda s, c: s["design"] == "anova" and s["solve_for"] != "effect"),
        Sentence(["In", Int("groups", 3, 50, 50), "groups"], when=lambda s, c: s["design"] == "anova" and s["solve_for"] == "effect"),
        Sentence(["The standard deviation within a group is", Num("sd", 1e-9, 1e9, 4, 80)],
                 when=lambda s, c: s["design"] in ("two_means", "one_mean", "anova")),
        Sentence(["The standard deviation of the differences is", Num("sd", 1e-9, 1e9, 4, 80)], when=lambda s, c: s["design"] == "paired_means"),
        Sentence(["Proportions of", Num("p1", 0.001, 0.999, 3, 62), "in one group and", Num("p2", 0, 1, 3, 62), "in the other"],
                 when=lambda s, c: s["design"] == "two_proportions" and s["solve_for"] != "effect"),
        Sentence(["The proportion in the first group is", Num("p1", 0.001, 0.999, 3, 62)],
                 when=lambda s, c: s["design"] == "two_proportions" and s["solve_for"] == "effect"),
        Sentence(["A correlation of", Num("r", -0.999, 0.999, 3, 62)], when=lambda s, c: s["design"] == "correlation" and s["solve_for"] != "effect"),
        Heading("How sure to be"),
        Sentence(["Find it with", Num("power", 0.5, 0.999, 3, 62), "power"], when=lambda s, c: s["solve_for"] != "power"),
        Sentence(["With", Int("n", 2, 10_000_000, 80), "subjects per group"],
                 when=lambda s, c: s["solve_for"] != "n" and s["design"] in ("two_means", "anova", "two_proportions")),
        Sentence(["With", Int("n", 2, 10_000_000, 80), "subjects"],
                 when=lambda s, c: s["solve_for"] != "n" and s["design"] not in ("two_means", "anova", "two_proportions")),
        Sentence(["Accept a", Num("alpha", 0.0001, 0.2, 4, 66), "chance of a false positive"]),
        Choose("two_sided", [Opt(True, "A difference either way", "Two-sided - the usual choice."),
                             Opt(False, "One direction only", "One-sided - only if the other direction could never count.")],
               when=lambda s, c: s["design"] != "anova"),
        Drawer("Fine-tune", [
            Sentence(["The second group is", Num("ratio", 0.05, 20, 2, 58), "times the size of the first"],
                     when=lambda s, c: s["design"] in ("two_means", "two_proportions")),
            Sentence(["Allow for", Num("dropout_percent", 0, 94, 1, 58, " %"), "of subjects being lost"]),
        ]),
        Note("Nothing is read from the sheet: this is for planning, before the data exist."),
    ],
    methods=_power_methods,
    own={"alpha"},
)

SPECS = [TWO_GROUPS, ONE_WAY, ONE_SAMPLE, TWO_WAY, THREE_WAY, PAIRS, NESTED_T, NESTED_ANOVA, EFFECT, ANCOVA, MIXED, POWER]
