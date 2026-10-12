"""Find relationships: correlation, and models with several predictors."""

from __future__ import annotations

from ...backends.correlation_analysis import CorrelationOptions
from ...backends.count_regression import CountRegressionOptions
from ...backends.multinomial_regression import MultinomialRegressionOptions
from ...backends.multiple_linear_regression import MultipleLinearRegressionOptions
from ...backends.multiple_logistic_regression import MultipleLogisticRegressionOptions
from ...backends.ordinal_regression import OrdinalRegressionOptions
from ...backends.partial_correlation import PartialCorrelationOptions
from ..spec import Ask, Checks, Chips, Choose, Custom, Drawer, Heading, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch, Txt
from .common import confidence, join, skewed
from .models import all_variables, drop_editor, levels_of, outcome_of, reference_editor, scenario_editor

# ── correlation ─────────────────────────────────────────────────────────────


def _variables(ctx):
    return [(i, label) for i, label in enumerate(["X"] + (ctx.names or ["Set A", "Set B", "Set C"]))]


def _correlation_methods(s, ctx) -> str:
    coefficient = "Pearson's r" if s["gaussian"] else "Spearman's rank correlation"
    which = {"x_vs_every_y": "X and each data set", "matrix": "every pair of data sets",
             "two_selected": "the two chosen variables"}[s["mode"]]
    return f"{coefficient} was calculated between {which}, with {confidence()} confidence intervals."


CORRELATION = Spec(
    "correlation_analysis", CorrelationOptions,
    [
        Heading("Which pairs?"),
        Choose("mode", [
            Opt("x_vs_every_y", "X with each data set", "One correlation per data set."),
            Opt("matrix", "Every pair of data sets", "A correlation matrix."),
            Opt("two_selected", "Two I choose", "Any two variables, X included."),
        ], columns=3),
        Sentence(["Correlate", Pick("first_var_index", _variables), "with", Pick("second_var_index", _variables)],
                 when=lambda s, c: s["mode"] == "two_selected"),
        Switch("remove_incomplete_rows", "Leave out a whole row when any value is missing",
               "So every coefficient uses the same rows.", when=lambda s, c: s["mode"] == "matrix"),
        Heading("What can you assume about the values?"),
        Choose("gaussian", [
            Opt(True, "Both roughly bell-shaped", "Pearson's r - a straight-line relationship.", phrase="Pearson"),
            Opt(False, "Nothing about their shape", "Spearman's rank correlation - any steadily rising or falling "
                                                    "relationship.", suggest=skewed, phrase="Spearman"),
        ]),
        Sentence(["Count", Pick("tails", [("two", "a correlation of either sign"),
                                          ("positive", "only a positive correlation"),
                                          ("negative", "only a negative correlation")])]),
        Note("Test one sign only if it was fixed before the data were seen.", tone="warn",
             when=lambda s, c: s["tails"] != "two"),
        Chips([("", [("create_heatmap", "Heat map of the matrix")])], when=lambda s, c: s["mode"] == "matrix"),
    ],
    on_sheet=lambda p: p.suggest({"second_var_index": min(2, len(p.ctx.names))}),
    methods=_correlation_methods,
)

# ── several predictors, a measured outcome ──────────────────────────────────


def _outcome_options(ctx):
    return [(i, label) for i, label in enumerate(all_variables(ctx))]


def _predictors(panel):
    variables = all_variables(panel.ctx)
    outcome = outcome_of(panel.state, variables)
    return [v for v in variables if v != outcome]


def _formula(s, ctx) -> str:
    variables = all_variables(ctx)
    outcome = outcome_of(s, variables)
    terms = (["intercept"] if s.get("include_intercept", True) else []) + [v for v in variables if v != outcome]
    if s.get("include_two_way"):
        terms.append("every two-way interaction")
    if s.get("include_three_way"):
        terms.append("every three-way interaction")
    return f"The model:  {outcome}  ~  " + " + ".join(terms)


def _outcome_sheet(panel) -> None:
    variables = all_variables(panel.ctx)
    if panel.state.get("outcome_index", -1) < 0 or panel.state["outcome_index"] >= len(variables):
        panel.state["outcome_index"] = len(variables) - 1


_TERMS = Chips([("Terms", [("include_intercept", "Intercept"), ("include_two_way", "Two-way interactions"),
                           ("include_three_way", "Three-way interactions")])])
_EACH = [("show_ci", "With a confidence interval"), ("show_se", "With its standard error"),
         ("run_significance_test", "Tested by refitting without it", "A test that does not lean on the standard error.")]
_TANGLED = Ask("Are the predictors tangled together?",
               "When two predictors rise and fall together, the model cannot say which one is doing the work.",
               pieces=[("show_multicollinearity", "How much each is explained by the others (VIF)"),
                       ("show_correlation_matrix", "Correlations between the predictors")])
_RESIDUAL_PIECES = [("show_residual_plot", "Plot of the residuals"), ("show_homoscedasticity_plot", "Spread against fitted value"),
                    ("show_qq_plot", "Q–Q plot of the residuals"), ("show_residual_vs_order", "Residuals in row order"),
                    ("normality_shapiro", "Shapiro-Wilk"), ("normality_dagostino", "D'Agostino-Pearson"),
                    ("normality_anderson", "Anderson-Darling"), ("normality_ks", "Kolmogorov-Smirnov")]


def _mlr_finish(s, ctx) -> None:
    s["ci_approach"] = "wald" if s["regression_type"] == "poisson" else "profile"
    if not s.pop("_use_scenarios", False):
        s["scenarios"] = []
    variables = all_variables(ctx)
    if s["outcome_index"] < 0 or s["outcome_index"] >= len(variables):
        s["outcome_index"] = len(variables) - 1


MULTIPLE_LINEAR = Spec(
    "multiple_linear_regression", MultipleLinearRegressionOptions,
    [
        Heading("What are you explaining?"),
        Row("Outcome", Pick("outcome_index", _outcome_options)),
        Choose("regression_type", [
            Opt("least_squares", "A measurement", "Least squares - the outcome can take any value."),
            Opt("poisson", "A count", "Poisson regression - whole numbers of events or objects. The 'A count' mode above "
                                      "adds rate ratios, an exposure and a check of the spread."),
        ]),
        Note(_formula, tone="accent"),
        _TERMS,
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("What does each predictor do?",
            "Its coefficient: the change in the outcome for a rise of one in that predictor, the others held where they are.",
            always=True, pieces=_EACH),
        _TANGLED,
        Ask("How well does the model fit?",
            "How much of the outcome it accounts for, and how far the points lie from it.",
            pieces=[("show_r2", "R²"), ("show_adj_r2", "Adjusted R²", "R², docked for every predictor added."),
                    ("show_multiple_r", "Multiple R"), ("show_syx", "Residual SD"), ("show_rmse", "RMSE"),
                    ("show_sum_of_square", "Sum of squares"), ("show_aicc", "AICc")],
            first=["show_r2", "show_adj_r2", "show_syx"]),
        Ask("Can the fit be trusted?",
            "Least squares assumes the leftovers scatter evenly and roughly bell-shaped. These show whether they do.",
            pieces=_RESIDUAL_PIECES, first=["show_residual_plot", "show_qq_plot", "normality_shapiro"]),
        Ask("Would a simpler model do as well?",
            "An F test of the full model against one with some predictors left out.",
            pieces=[("compare_models", "Compare with a simpler model")],
            inside=[Custom(lambda p: drop_editor(_predictors, "dropped_predictors"))]),
        Ask("What does it predict?",
            "The outcome the model expects, for rows that lack one or for values you type.",
            pieces=[("interpolate_missing_rows", "For rows whose outcome is blank", "Every predictor must be filled in."),
                    ("_use_scenarios", "For values I type in"),
                    ("report_ci_for_interpolated", "With a confidence interval")],
            first=["interpolate_missing_rows", "report_ci_for_interpolated"],
            inside=[Custom(lambda p: scenario_editor(_predictors, "scenarios"), when=lambda s, c: s["_use_scenarios"])]),
        Drawer("Categories", [Custom(lambda p: reference_editor(_predictors))]),
        Drawer("Weights", [
            Choose("weighting", [Opt("none", "Equal"), Opt("inv_y", "1/Y"), Opt("inv_y2", "1/Y²"), Opt("inv_yk", "1/Yᴷ")],
                   style="segmented"),
            Sentence(["with K =", Num("weighting_k", 0, 10, 3, 58)], when=lambda s, c: s["weighting"] == "inv_yk"),
            Note("Weight when the scatter grows with the outcome: 1/Y for Poisson-like scatter, 1/Y² when the SD "
                 "is a constant share of Y.", tone="faint"),
        ]),
    ],
    virtual={"_use_scenarios": False},
    finish=_mlr_finish,
    on_sheet=_outcome_sheet,
    methods=lambda s, c: (f"{outcome_of(s, all_variables(c))} was modelled by "
                          f"{'Poisson' if s['regression_type'] == 'poisson' else 'multiple linear (least-squares)'} "
                          f"regression on {join([v for v in all_variables(c) if v != outcome_of(s, all_variables(c))])}."),
)

# ── several predictors, a yes/no outcome ────────────────────────────────────


def _logit_sheet(panel) -> None:
    _outcome_sheet(panel)
    variables = all_variables(panel.ctx)
    outcome = outcome_of(panel.state, variables)
    levels = levels_of(panel.ctx).get(outcome)
    if levels and len(levels) == 2:
        panel.suggest({"negative_value": float(levels[0]), "positive_value": float(levels[1])})


def _logit_finish(s, ctx) -> None:
    variables = all_variables(ctx)
    if s["outcome_index"] < 0 or s["outcome_index"] >= len(variables):
        s["outcome_index"] = len(variables) - 1
    s["negative_label"] = s.get("negative_label") or f"{s['negative_value']:g}"
    s["positive_label"] = s.get("positive_label") or f"{s['positive_value']:g}"


MULTIPLE_LOGISTIC = Spec(
    "multiple_logistic_regression", MultipleLogisticRegressionOptions,
    [
        Heading("What are you explaining?"),
        Row("Yes/no outcome", Pick("outcome_index", _outcome_options)),
        Sentence(["“Yes” is coded", Num("positive_value", decimals=4, width=54), "and “no”",
                  Num("negative_value", decimals=4, width=54)]),
        Note(_formula, tone="accent"),
        _TERMS,
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How does each predictor change the odds?",
            "Its odds ratio: what a rise of one in that predictor does to the odds of yes, the others held where they are.",
            always=True, pieces=_EACH),
        _TANGLED,
        Ask("How well does the model tell yes from no?",
            "One number from 0.5 (no better than a coin) to 1 (perfect), and how many rows it calls right.",
            pieces=[("gof_roc", "Area under the ROC curve"), ("graph_roc", "The ROC curve"),
                    ("gof_classification_table", "Rows called right and wrong at a cutoff"),
                    ("graph_proportion_correct", "Share called right at every cutoff")],
            first=["gof_roc", "graph_roc"],
            inside=[Sentence(["Call a row “yes” above a predicted probability of", Num("classification_cutoff", 0, 1, 3, 58)],
                             when=lambda s, c: s["gof_classification_table"])]),
        Ask("Can the predicted chances be trusted?",
            "Whether a predicted 30% comes true about 30% of the time.",
            pieces=[("gof_brier", "Brier score", "The average squared gap between predicted chance and outcome."),
                    ("gof_hosmer_lemeshow", "Hosmer-Lemeshow test"),
                    ("graph_calibration", "Graph of predicted against what happened"),
                    ("graph_predicted_vs_observed", "Each row's predicted chance, by its outcome")],
            first=["gof_brier", "gof_hosmer_lemeshow", "graph_calibration"]),
        Ask("Which rows pull the fit?",
            "The rows that would change the answer most if they were left out, by their row on the sheet.",
            pieces=[("show_influence", "Table of the ten most influential rows"), ("graph_influence", "Graph of every row's pull")]),
        Ask("Do the predictors earn their place?",
            "A test against a model with no predictors, and measures for comparing models.",
            pieces=[("gof_lrt", "Likelihood-ratio test"), ("show_aicc", "AICc"),
                    ("show_loglik_both", "Log-likelihood, with and without predictors"),
                    ("show_deviance_both", "Deviance, with and without predictors"),
                    ("pseudo_r2_tjur", "Tjur's R²"), ("pseudo_r2_mcfadden", "McFadden's R²"),
                    ("pseudo_r2_nagelkerke", "Nagelkerke's R²"), ("pseudo_r2_coxsnell", "Cox and Snell's R²")],
            first=["gof_lrt", "show_aicc"]),
        Ask("Would a simpler model do as well?",
            "A likelihood-ratio test and AIC: does the full model earn its extra terms?",
            pieces=[("compare_models", "Compare with a simpler model")],
            inside=[Custom(lambda p: drop_editor(_predictors, "dropped_predictors"))]),
        Ask("What does it predict?", "The chance of yes that the model gives each row.",
            pieces=[("gof_predicted_prob", "Predicted probability for each row")]),
        Ask("Show me the curve", "The S-curve of chance against a predictor - drawn when the model has one measured predictor.",
            pieces=[("graph_logistic_plot", "The curve"), ("plot_confidence_bands", "Its confidence band")],
            first=["graph_logistic_plot"]),
        Drawer("Categories", [Custom(lambda p: reference_editor(_predictors))]),
        Drawer("Fine-tune", [
            Row("Fit by", Pick("fit_method", [("auto", "maximum likelihood - Firth's fit if a predictor splits yes from no"),
                                              ("firth", "Firth's penalised likelihood, always"),
                                              ("ml", "maximum likelihood, always")]),
                "Firth's fit is less biased when one outcome is rare, and has an answer when a predictor splits them completely."),
            Row("Coefficient intervals by", Pick("ci_approach", [("profile", "profile likelihood (can be lopsided)"),
                                                                 ("wald", "Wald (always symmetric)")])),
            Sentence(["Name “yes”", Txt("positive_label", 110, "1"), "and “no”", Txt("negative_label", 110, "0"),
                      "in the results"]),
        ]),
    ],
    finish=_logit_finish,
    on_sheet=_logit_sheet,
    start={"positive_label": "", "negative_label": ""},
    methods=lambda s, c: (f"{outcome_of(s, all_variables(c))} was modelled by multiple logistic regression; "
                          f"effects are reported as odds ratios with {confidence()} confidence intervals."),
)

# ── others held fixed ───────────────────────────────────────────────────────


def _partial_methods(s, ctx) -> str:
    a, b = ctx.name(int(s["first_index"])), ctx.name(int(s["second_index"]))
    kind = "Spearman" if s["ranks"] else "Pearson"
    return (f"The {kind} partial correlation between {a} and {b} was calculated with the other variables held fixed, "
            f"with a {confidence()} confidence interval by Fisher's z transformation.")


PARTIAL = Spec(
    "partial_correlation", PartialCorrelationOptions,
    [
        Heading("Which two variables?"),
        Sentence(["Relate", Pick("first_index", lambda c: c.column_options()), "to", Pick("second_index", lambda c: c.column_options())]),
        Note("Every other data set ticked under 'Data sets to include' is held fixed. Untick the ones that should take no part."),
        Note("Choose two different variables.", tone="warn", when=lambda s, c: s["first_index"] == s["second_index"]),
        Heading("What kind of link?"),
        Choose("ranks", [
            Opt(False, "A straight line", "Pearson - for measurements without extreme values.", phrase="Pearson"),
            Opt(True, "Any steady rise or fall", "Spearman - on ranks; robust to skew and outliers.", suggest=skewed, phrase="Spearman"),
        ]),
        Heading("What else do you want to know?"),
        Ask("Which of the others is doing the work?",
            "The same correlation with one of the other variables held fixed at a time.",
            pieces=[("show_each_control", "One variable held fixed at a time")]),
        Ask("What about every other pair?",
            "The whole table: each pair of variables with all the rest held fixed.",
            pieces=[("show_matrix", "Every pair, others held fixed")]),
    ],
    methods=_partial_methods,
)

# ── several predictors, a count ─────────────────────────────────────────────


def _exposure_options(ctx):
    return [(-1, "nothing - every row was watched alike")] + _outcome_options(ctx)


def _exposure_of(state, variables) -> str | None:
    i = state.get("exposure_index", -1)
    return variables[i] if i is not None and 0 <= i < len(variables) and variables[i] != outcome_of(state, variables) else None


def _count_predictors(panel):
    variables = all_variables(panel.ctx)
    left_out = (outcome_of(panel.state, variables), _exposure_of(panel.state, variables))
    return [v for v in variables if v not in left_out]


def _filled(ctx) -> list[str]:
    """The variables that hold any values - an X column left empty takes no part in a model."""
    variables = all_variables(ctx)
    if ctx.sheet is None:
        return variables
    try:
        labels, matrix = ctx.sheet.full_variable_matrix()
        return [label for i, label in enumerate(labels) if (matrix[:, i] == matrix[:, i]).any()]
    except Exception:  # noqa: BLE001
        return variables


def _count_formula(s, ctx) -> str:
    variables = all_variables(ctx)
    outcome, exposure = outcome_of(s, variables), _exposure_of(s, variables)
    terms = [v for v in _filled(ctx) if v not in (outcome, exposure)]
    left = f"{outcome} per unit of {exposure}" if exposure else outcome
    return f"The model:  {left}  ~  " + (" + ".join(terms) or "the other variables")


def _plain_formula(s, ctx) -> str:
    outcome = outcome_of(s, all_variables(ctx))
    return f"The model:  {outcome}  ~  " + (" + ".join(v for v in _filled(ctx) if v != outcome) or "the other variables")


def _clamp_outcome(s, ctx) -> None:
    variables = all_variables(ctx)
    if s["outcome_index"] < 0 or s["outcome_index"] >= len(variables):
        s["outcome_index"] = len(variables) - 1


def _count_methods(s, ctx) -> str:
    variables = all_variables(ctx)
    family = {"auto": "Poisson regression, or negative binomial regression where the counts were more spread out than a Poisson "
                      "model allows (Dean-Lawless score test)",
              "poisson": "Poisson regression", "negative_binomial": "negative binomial regression",
              "quasi": "quasi-Poisson regression"}[s["family"]]
    exposure = _exposure_of(s, variables)
    text = f"{outcome_of(s, variables)} was modelled by {family} with a log link"
    if exposure:
        text += f", with {exposure} as the exposure"
    return text + f"; effects are rate ratios with {confidence()} confidence intervals."


COUNT = Spec(
    "count_regression", CountRegressionOptions,
    [
        Heading("What are you explaining?"),
        Row("The count", Pick("outcome_index", _outcome_options)),
        Row("Exposure", Pick("exposure_index", _exposure_options)),
        Note("Give an exposure when the rows were watched for different lengths of time, or cover different areas or "
             "populations. The model is then for the count per unit of it - a rate.", tone="faint"),
        Note("The exposure cannot be the count itself.", tone="warn",
             when=lambda s, c: s["exposure_index"] >= 0 and all_variables(c)[min(s["exposure_index"], len(all_variables(c)) - 1)]
             == outcome_of(s, all_variables(c))),
        Note(_count_formula, tone="accent"),
        Heading("How are the counts spread?"),
        Choose("family", [
            Opt("auto", "Check, and choose for me", "Poisson - unless the counts are more spread out than that allows, when "
                                                    "a negative binomial model is fitted instead."),
            Opt("poisson", "Poisson", "Spread equal to the mean: events that happen independently, at a steady rate."),
            Opt("negative_binomial", "Negative binomial", "More spread than Poisson, as counts from subjects who differ "
                                                          "usually are."),
            Opt("quasi", "Quasi-Poisson", "Poisson estimates, with standard errors widened to match the spread seen."),
        ], columns=2),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("Does each predictor matter?",
            "Each one tested by refitting the model without it - the test to read first.",
            pieces=[("test_each_term", "Test each predictor")]),
        Ask("Are the counts spread as the model expects?",
            "A Poisson count has a spread equal to its mean. More than that, and its P values are too small.",
            pieces=[("show_spread_check", "Check of the spread"),
                    ("show_count_table", "Counts seen and expected", "Shows up too many zeros, for one."),
                    ("graph_counts", "Graph of counts seen and expected")],
            first=["show_spread_check"]),
        Ask("How well does the model fit?",
            "Its deviance and information criteria, and what it predicts against what was counted.",
            pieces=[("show_fit", "Measures of fit"), ("graph_observed_predicted", "Graph of seen against predicted"),
                    ("graph_residuals", "Graph of the residuals")]),
        Ask("What does it predict?", "The count the model expects on every row, with the leftover, on a new sheet.",
            pieces=[("show_predictions", "Predicted counts and residuals")]),
        Drawer("Categories", [Custom(lambda p: reference_editor(_count_predictors))]),
        Drawer("Fine-tune", [Row("Coefficient intervals by", Pick("ci_approach", [
            ("profile", "profile likelihood (can be lopsided)"), ("wald", "Wald (always symmetric)")]))]),
    ],
    finish=_clamp_outcome,
    on_sheet=_outcome_sheet,
    methods=_count_methods,
)

# ── several predictors, categories ──────────────────────────────────────────


def _category_note(s, ctx) -> str:
    variables = all_variables(ctx)
    outcome = outcome_of(s, variables)
    levels = None
    if ctx.sheet is not None:
        try:
            labels, matrix = ctx.sheet.full_variable_matrix()
            values = matrix[:, labels.index(outcome)]
            levels = sorted(set(values[values == values].tolist()))
        except Exception:  # noqa: BLE001
            levels = None
    if levels is None:
        return "The outcome holds a code for each category: 1, 2, 3 ..."
    if len(levels) > 12:
        return f"{outcome} has {len(levels)} distinct values - that is a measurement, not a set of categories."
    return f"{outcome} has {len(levels)} categories: " + ", ".join(f"{v:g}" for v in levels) + "."


def _too_many(s, ctx) -> bool:
    return "a measurement" in _category_note(s, ctx)


_EACH_MATTERS = Ask("Does each predictor matter?",
                    "Each one tested by refitting the model without it - the test to read first.",
                    pieces=[("test_each_term", "Test each predictor")])
_CATEGORY_FIT = Ask("How well does the model fit?",
                    "Measures of fit, how many rows it places in the right category, and the shares it predicts.",
                    pieces=[("show_fit", "Measures of fit"), ("show_classification", "Rows placed right and wrong"),
                            ("graph_shares", "Graph of shares seen and predicted")])
_CATEGORY_SHIFT = Ask("How do the chances shift with a predictor?",
                      "A graph of the chance of each category as one predictor runs across its range.",
                      pieces=[("graph_probabilities", "Chance of each category against a predictor")])
_CATEGORY_PREDICT = Ask("What does it predict?", "The chance of each category for every row, on a new sheet.",
                        pieces=[("show_predictions", "Each row's chances")])

ORDINAL = Spec(
    "ordinal_regression", OrdinalRegressionOptions,
    [
        Heading("What are you explaining?"),
        Row("Ordered outcome", Pick("outcome_index", _outcome_options)),
        Note(_category_note, tone="faint", when=lambda s, c: not _too_many(s, c)),
        Note(_category_note, tone="warn", when=_too_many),
        Note("The codes must rise with the order - 1 for the lowest category, upwards.", tone="faint"),
        Note(_plain_formula, tone="accent"),
        Heading("Which model?"),
        Choose("link", [
            Opt("logit", "Proportional odds", "Effects are odds ratios for being in a higher category - the usual choice.",
                phrase="proportional-odds"),
            Opt("probit", "Probit", "Effects on an underlying bell-shaped scale; no odds ratios.", phrase="cumulative probit"),
        ]),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        _EACH_MATTERS,
        Ask("Does a predictor act alike all the way up the scale?",
            "The model assumes one odds ratio per predictor, whichever two neighbouring categories are compared. "
            "The Brant test checks it.",
            pieces=[("show_parallel_check", "Brant test")]),
        Ask("Where do the categories divide?", "The cut points on the model's underlying scale.",
            pieces=[("show_cut_points", "Cut points")]),
        _CATEGORY_FIT, _CATEGORY_SHIFT, _CATEGORY_PREDICT,
        Drawer("Categories", [Custom(lambda p: reference_editor(_predictors))]),
    ],
    finish=_clamp_outcome,
    on_sheet=_outcome_sheet,
    methods=lambda s, c: (f"{outcome_of(s, all_variables(c))} was modelled by ordinal regression "
                          f"({'proportional-odds, cumulative logit' if s['link'] == 'logit' else 'cumulative probit'} model)"
                          + (f"; effects are odds ratios with {confidence()} confidence intervals, and the proportional-odds "
                             "assumption was checked with the Brant test." if s["link"] == "logit" and s["show_parallel_check"]
                             else f", with {confidence()} confidence intervals.")),
)

MULTINOMIAL = Spec(
    "multinomial_regression", MultinomialRegressionOptions,
    [
        Heading("What are you explaining?"),
        Row("Outcome", Pick("outcome_index", _outcome_options)),
        Note(_category_note, tone="faint", when=lambda s, c: not _too_many(s, c)),
        Note(_category_note, tone="warn", when=_too_many),
        Row("Compare every category with", Pick("reference", [("first", "the one with the lowest code"),
                                                             ("last", "the one with the highest code"),
                                                             ("largest", "the commonest one")])),
        Note(_plain_formula, tone="accent"),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        _EACH_MATTERS, _CATEGORY_FIT, _CATEGORY_SHIFT, _CATEGORY_PREDICT,
        Drawer("Categories", [Custom(lambda p: reference_editor(_predictors))]),
    ],
    finish=_clamp_outcome,
    on_sheet=_outcome_sheet,
    methods=lambda s, c: (f"{outcome_of(s, all_variables(c))} was modelled by multinomial logistic regression; effects are odds "
                          f"ratios against a reference category, with {confidence()} confidence intervals, and each predictor "
                          "was tested by a likelihood-ratio test."),
)

SPECS = [CORRELATION, MULTIPLE_LINEAR, MULTIPLE_LOGISTIC, COUNT, ORDINAL, MULTINOMIAL, PARTIAL]
