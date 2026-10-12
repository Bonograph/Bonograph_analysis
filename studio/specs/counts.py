"""Counts and proportions: are two classifications linked, do counts match
expectations, and each value's share of a total."""

from __future__ import annotations

from ...backends.chi_square_test import ChiSquareOptions
from ...backends.goodness_of_fit import GoodnessOfFitOptions
from ...backends.paired_proportions import PairedProportionsOptions
from ...backends.percentage_of_total import PercentageOfTotalOptions
from ...backends.stratified_tables import StratifiedTablesOptions
from ..spec import Ask, Chips, Choose, Custom, Heading, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch
from .common import columns, confidence, join
from .editors import expected_table, is_2x2, observed_counts
from .sketches import grid_glyph


def _chi_sheet(panel) -> None:
    panel.ctx.extra["2x2"] = is_2x2(panel.ctx)
    if not panel.ctx.extra["2x2"] and panel.state.get("p_method") == "yates":
        panel.state["p_method"] = "auto"


def _chi_finish(s, ctx) -> None:
    square = ctx.extra.get("2x2", ctx.sheet is None)
    for key in ("report_relative_risk", "report_risk_difference", "report_odds_ratio", "report_sensitivity_specificity"):
        s[key] = bool(s[key]) and square
    s["tails"] = "two"  # a direction read off the data isn't a one-sided test


def _chi_methods(s, ctx) -> str:
    test = {"auto": "Fisher's exact test (small counts or a 2×2 table) or the chi-square test",
            "fisher": "Fisher's exact test", "chi_square": "the chi-square test", "trend": "the chi-square test for trend",
            "yates": "the chi-square test with Yates' correction"}[s["p_method"]]
    effects = [w for k, w in (("report_relative_risk", "relative risk"), ("report_risk_difference", "the risk difference"),
                              ("report_odds_ratio", "the odds ratio"),
                              ("report_sensitivity_specificity", "sensitivity and specificity")) if s[k]]
    text = f"The association was tested with {test}"
    if effects and ctx.extra.get("2x2", False):
        text += f"; {join(effects)} are given with {confidence()} confidence intervals"
    return text + "."


LINK = Spec(
    "chi_square_test", ChiSquareOptions,
    [
        Heading("Which test?"),
        Choose("p_method", [
            Opt("auto", "Choose for me", "Fisher's exact test when any expected count is small or the table is 2×2; "
                                         "chi-square otherwise."),
            Opt("fisher", "Fisher's exact test", "Exact at any sample size; for bigger tables it can be slow."),
            Opt("chi_square", "Chi-square test", "Fine once every expected count is 5 or more."),
            Opt("trend", "Chi-square test for trend", "Rows in a meaningful order, two outcome columns."),
            Opt("yates", "Chi-square with Yates' correction", "2×2 only; errs on the cautious side.",
                available=lambda c: c.extra.get("2x2", True)),
        ], columns=2),
        Heading("How strong is the link?", "For 2×2 tables: rows are the groups, columns the outcomes. The right measure "
                                        "depends on how the study was run.", when=lambda s, c: c.extra.get("2x2", True)),
        Ask("A trial or a cohort: how much does the risk change?",
            "Subjects were grouped first and the outcome counted afterwards, so risks can be compared directly.",
            pieces=[("report_relative_risk", "Relative risk"),
                    ("report_risk_difference", "Difference in risk, and the number needed to treat")],
            when=lambda s, c: c.extra.get("2x2", True)),
        Ask("A case-control study: how do the odds compare?",
            "Subjects were chosen by outcome, so risks cannot be compared - but the odds ratio still can.",
            pieces=[("report_odds_ratio", "Odds ratio")], when=lambda s, c: c.extra.get("2x2", True)),
        Ask("A diagnostic test against the truth: how good is the test?",
            "How often it finds the condition, how often it clears those without it, and what a result means.",
            pieces=[("report_sensitivity_specificity", "Sensitivity, specificity and predictive values")],
            when=lambda s, c: c.extra.get("2x2", True)),
        Note("Cramér's V, the strength of the association, is reported for every table.", tone="faint"),
    ],
    finish=_chi_finish,
    on_sheet=_chi_sheet,
    methods=_chi_methods,
)

# ── counts against expectations ─────────────────────────────────────────────

EXPECTED = Spec(
    "goodness_of_fit", GoodnessOfFitOptions,
    [
        Note("Works on counts of events or items - not percentages, means or other scaled numbers."),
        Row("Counts are in", Pick("dataset_index", columns)),
        Heading("What did you expect?"),
        Choose("expected_as_percent", [Opt(False, "Relative amounts - 9, 3, 3, 1 means 9:3:3:1"),
                                       Opt(True, "Percentages adding up to 100")], style="segmented"),
        Custom(lambda p: expected_table()),
        Heading("With exactly two rows", when=lambda s, c: len(observed_counts(c, int(s["dataset_index"] or 0))) == 2),
        Choose("two_row_method", [Opt("binomial", "Exact binomial test", "With a Wilson interval for the proportion."),
                                  Opt("chi_square", "Chi-square test")],
               when=lambda s, c: len(observed_counts(c, int(s["dataset_index"] or 0))) == 2),
    ],
    methods=lambda s, c: "Observed counts were compared with the expected proportions by a chi-square goodness-of-fit test.",
)

# ── shares of a total ───────────────────────────────────────────────────────

SHARES = Spec(
    "percentage_of_total", PercentageOfTotalOptions,
    [
        Heading("What should add up to the whole?"),
        Choose("divide_by", [Opt("column", "Each column", "Shares within a column.", glyph=grid_glyph("column")),
                             Opt("row", "Each row", "Shares within a row.", glyph=grid_glyph("row")),
                             Opt("grand", "The whole table", "Every cell against the grand total.", glyph=grid_glyph("all")),
                             Opt("all", "All three", "Side by side.", glyph=grid_glyph("three"))], columns=4),
        Heading("Write each share"),
        Choose("present_as", [Opt("percentages", "Out of 100"), Opt("fractions", "Out of 1")], style="segmented"),
        Chips([("", [("calculate_ci", "With a confidence interval",
                      "Only meaningful for counts - whole numbers as counted, not values already scaled.")])]),
        Choose("ci_method", [Opt("wilson", "Wilson score - the usual choice"),
                             Opt("clopper_pearson", "Clopper-Pearson - exact, and wider")], style="segmented",
               when=lambda s, c: s["calculate_ci"]),
    ],
    methods=lambda s, c: f"Each value was expressed as a {s['present_as'][:-1]} of its {s['divide_by']} total.",
)

# ── yes/no on the same subjects ─────────────────────────────────────────────


def _paired_yes_no_methods(s, ctx) -> str:
    if ctx.groups <= 2:
        return ("The change in proportion between the two paired measurements was tested with McNemar's test ("
                + ("exact binomial" if s["exact"] else "chi-square with continuity correction") + ").")
    text = "Proportions across the paired measurements were compared with Cochran's Q test"
    if s["compare"] != "none":
        how = {"holm": "Holm's method", "fdr": "the Benjamini-Hochberg method", "none": "no adjustment"}[s["correction"]]
        text += f", followed by pairwise McNemar tests with {how}"
    return text + "."


PAIRED_YES_NO = Spec(
    "paired_proportions", PairedProportionsOptions,
    [
        Note("Each row is one subject; each data set holds a code for yes or no under one condition. Rows with any other "
             "value are left out."),
        Sentence(["Yes is coded", Num("yes_code", decimals=4, width=54), "; no", Num("no_code", decimals=4, width=54)]),
        Heading("Judge a pair by"),
        Choose("exact", [Opt(True, "The exact test", "Right at any size - the default."),
                         Opt(False, "Chi-square", "The classical form, with continuity correction.")]),
        Heading("Then compare pairs of data sets", when=lambda s, c: c.groups >= 3),
        Choose("compare", [Opt("none", "No"), Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against one")],
               style="segmented", when=lambda s, c: c.groups >= 3),
        Row("Compare each with", Pick("control_index", columns), when=lambda s, c: c.groups >= 3 and s["compare"] == "vs_control"),
        Row("Keep false positives in check with", Pick("correction", [("holm", "Holm's step-down method"),
                                                                       ("fdr", "a false-discovery-rate limit"),
                                                                       ("none", "no adjustment")]),
            when=lambda s, c: c.groups >= 3 and s["compare"] != "none"),
    ],
    methods=_paired_yes_no_methods,
)

# ── counts within layers ────────────────────────────────────────────────────


def _as_codes(s, ctx) -> bool:
    if s["layout"] != "auto":
        return s["layout"] == "codes"
    return ctx.groups == 3 or (ctx.shape is not None and ctx.shape.rows > 60)


def _layers_sheet(panel) -> None:
    n = max(panel.ctx.groups, 3)
    picks = [panel.state.get(k, i) for i, k in enumerate(("row_index", "column_index", "layer_index"))]
    if any(not 0 <= v < n for v in picks) or len(set(picks)) < 3:
        panel.state.update(row_index=0, column_index=1, layer_index=2)


def _layers_methods(s, ctx) -> str:
    text = ("The association was tested within layers by the Cochran-Mantel-Haenszel test"
            + (" with continuity correction" if s["continuity"] else "")
            + f"; for 2×2 tables the Mantel-Haenszel common odds ratio is given with a {confidence()} confidence interval "
              "(Robins-Breslow-Greenland)")
    if s["show_homogeneity"]:
        text += ", and the sameness of the odds ratios was tested with the Breslow-Day test with Tarone's correction"
    return text + "."


LAYERS = Spec(
    "stratified_tables", StratifiedTablesOptions,
    [
        Heading("How is the sheet laid out?"),
        Choose("layout", [
            Opt("auto", "Work it out", "Four data sets of whole counts are read as tables; three as codes."),
            Opt("counts", "A table on each row", "Each row is a layer; four data sets hold the counts of its 2×2 table."),
            Opt("codes", "A subject on each row", "Three data sets of codes: the row factor, the column factor, the layer."),
        ], columns=3),
        Note("The four data sets are read in this order: row 1 & column 1, row 1 & column 2, row 2 & column 1, row 2 & "
             "column 2. Titles such as 'Drug: improved', 'Drug: not', 'Placebo: improved', 'Placebo: not' name the rows and "
             "the columns; row titles name the layers.", when=lambda s, c: not _as_codes(s, c)),
        Row("Row factor", Pick("row_index", columns), when=_as_codes),
        Row("Column factor", Pick("column_index", columns), when=_as_codes),
        Row("Layer", Pick("layer_index", columns), when=_as_codes),
        Note("Choose three different data sets.", tone="warn",
             when=lambda s, c: _as_codes(s, c) and len({s["row_index"], s["column_index"], s["layer_index"]}) < 3),
        Heading("What else do you want to know?"),
        Ask("What does each layer say on its own?", "The link inside every layer, as a table and as a graph of the layers.",
            pieces=[("show_each_layer", "Each layer on its own"), ("graph_forest", "Graph of the layers")]),
        Ask("Would ignoring the layers change the answer?",
            "The same table with the layers added together. If it disagrees, the layers were hiding or faking the link.",
            pieces=[("show_pooled_table", "The table with layers ignored")]),
        Ask("Is the link the same in every layer?",
            "For 2×2 tables: the Breslow-Day and Woolf tests. If it differs, one combined odds ratio describes no layer well.",
            pieces=[("show_homogeneity", "Breslow-Day and Woolf tests")]),
        Ask("A trial or a cohort: how much does the risk change?",
            "For 2×2 tables: the relative risk and risk difference combined across the layers.",
            pieces=[("show_common_risk", "Common relative risk and risk difference")]),
        Switch("continuity", "Continuity correction in the combined test", "For 2×2 tables: the classical, slightly cautious form."),
    ],
    on_sheet=_layers_sheet,
    methods=_layers_methods,
)

SPECS = [LINK, LAYERS, EXPECTED, PAIRED_YES_NO, SHARES]
