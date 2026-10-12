"""Time to event: survival curves, and what changes the risk."""

from __future__ import annotations

from ...backends.cox_regression import CoxRegressionOptions
from ...backends.cox_time_varying import CoxTimeVaryingOptions
from ...backends.kaplan_meier import KaplanMeierOptions
from ...backends.life_table import LifeTableOptions
from ..spec import Ask, Chips, Choose, Custom, Drawer, Heading, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch, Txt
from .common import columns, confidence, join
from .models import drop_editor, reference_editor, scenario_editor, subset_editor

_CODES = Sentence(["An event is coded", Num("death_code", decimals=4, width=54), "; leaving early (censored)",
                   Num("censored_code", decimals=4, width=54)])


def _km_methods(s, ctx) -> str:
    tests = [w for k, w in (("compute_logrank", "the log-rank test"), ("compute_gehan_breslow", "the Gehan-Breslow (generalised Wilcoxon) test"),
                            ("compute_tarone_ware", "the Tarone-Ware test"), ("compute_peto_peto", "the Peto-Peto test"),
                            ("logrank_trend", "the log-rank test for trend")) if s.get(k)]
    text = "Survival was estimated by the Kaplan-Meier method"
    if tests:
        text += f" and the curves compared with {join(tests)}"
    text += "."
    if s.get("show_time_summary", True):
        scale = "log(-log)" if s.get("ci_style", "asymmetric") == "asymmetric" else "plain"
        text += (f" Median survival times are given with 95% confidence intervals (Brookmeyer-Crowley, Greenwood variance, {scale} "
                 "scale), mean survival times as restricted means, and median follow-up by the reverse Kaplan-Meier method.")
    if s.get("compare_rmst") and ctx.groups != 1:
        horizon = f"time {s['rmst_horizon']:g}" if s.get("rmst_horizon", 0) > 0 else "the longest time every group was followed"
        text += f" Groups were also compared by the difference and ratio of restricted mean survival times up to {horizon}."
    return text


SURVIVAL = Spec(
    "kaplan_meier", KaplanMeierOptions,
    [
        Note("X is the time. Each data set holds a code saying what happened at that time; rows with any other "
             "code are left out."),
        _CODES,
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("Do the curves differ?",
            "Each test weighs the follow-up differently. Choose by where a difference is expected - and before seeing the data.",
            pieces=[("compute_logrank", "Log-rank", "Every time point counts alike; gives a hazard ratio for two groups. "
                                                     "Strongest when one curve stays under the other throughout."),
                    ("compute_gehan_breslow", "Gehan-Breslow (generalised Wilcoxon)", "Early events count for more."),
                    ("compute_tarone_ware", "Tarone-Ware", "Between the two: some extra weight to early events."),
                    ("compute_peto_peto", "Peto-Peto", "Early events count for more, and steadier than Gehan-Breslow when "
                                                       "the groups are censored differently."),
                    ("logrank_trend", "A trend across the groups", "Three groups or more, in a meaningful order.")],
            first=["compute_logrank"]),
        Heading("Then compare pairs of curves", when=lambda s, c: c.groups != 2),
        Choose("compare", [Opt("none", "No"), Opt("all_pairs", "Every pair"), Opt("vs_control", "Each against a control")],
               style="segmented", when=lambda s, c: c.groups != 2),
        Row("Control group", Pick("control_index", columns), when=lambda s, c: s["compare"] == "vs_control"),
        Row("Keep false positives in check with", Pick("correction", [("holm", "Holm's step-down method"),
                                                                       ("fdr", "a false-discovery-rate limit"),
                                                                       ("none", "no adjustment")]),
            when=lambda s, c: s["compare"] != "none"),
        Ask("How long until the event?",
            "The median time, its quartiles and the mean, each with an interval; the median follow-up; and the share "
            "still event-free at times you name.",
            pieces=[("show_time_summary", "Times and landmarks")],
            inside=[Sentence(["Give the share at times", Txt("landmarks", width=170, placeholder="chosen for you, or: 12, 24, 36")])]),
        Ask("Does one group stay event-free longer, on average?",
            "The mean time without the event up to a horizon you choose, compared between the groups. It assumes nothing "
            "about how the curves relate, so it still means something when they cross.",
            pieces=[("compare_rmst", "Difference and ratio of mean times")],
            inside=[Sentence(["Up to time", Num("rmst_horizon", 0, 1e12, 6, 84), "(0 = the longest time every group was followed)"])],
            when=lambda s, c: c.groups != 1),
        Ask("How many were still being followed?",
            "The number at risk as time goes on - as a table, and under the time axis of the graph.",
            pieces=[("show_at_risk_table", "Number still at risk")]),
        Heading("The curves"),
        Sentence(["Draw the share", Pick("tabulate", [("survival", "still event-free"), ("event", "who had the event")]),
                  ", with", Pick("error_bar_mode", [("ci", "confidence intervals"), ("se", "standard errors"),
                                                   ("none", "no error bars")])]),
        Chips([("", [("show_ci_band", "Shade the interval as a band"), ("show_median_lines", "Mark each median")])]),
        Drawer("Fine-tune", [Row("Intervals", Pick("ci_style", [("asymmetric", "asymmetric - stay within 0 and 100% (usual)"),
                                                               ("symmetric", "symmetric")]))]),
    ],
    methods=_km_methods,
)

# ── Cox regression ──────────────────────────────────────────────────────────


def _event_label(s, ctx) -> str:
    return ctx.name(int(s.get("event_index", 0) or 0))


def _covariates(panel):
    event = _event_label(panel.state, panel.ctx)
    return [n for n in (panel.ctx.names or ["Set A", "Set B", "Set C"]) if n != event]


def _cox_finish(s, ctx) -> None:
    if not s.pop("_use_scenarios", False):
        s["scenarios"] = []


RISK = Spec(
    "cox_regression", CoxRegressionOptions,
    [
        Note("X holds each subject's time. One data set records what happened; every other data set is a covariate."),
        Row("What happened is in", Pick("event_index", columns)),
        Sentence(["An event is coded", Num("event_value", decimals=4, width=54), "; censored",
                  Num("censored_value", decimals=4, width=54)]),
        Note(lambda s, c: f"The model:  (time, {_event_label(s, c)})  ~  " +
             (" + ".join(n for n in (c.names or []) if n != _event_label(s, c)) or "every other data set"),
             tone="accent"),
        Row("Tied times handled by", Pick("ties_method", [("efron", "Efron's method (usual)"), ("breslow", "Breslow's method")])),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How does each covariate change the risk?",
            "Its hazard ratio: what a rise of one in that covariate does to the rate of events, the others held where they are.",
            always=True,
            pieces=[("show_ci", "With a confidence interval"), ("show_se", "With its standard error"),
                    ("run_significance_test", "Tested by refitting without it", "A test that does not lean on the standard error.")]),
        Ask("Are the covariates tangled together?",
            "When two covariates rise and fall together, the model cannot say which one carries the risk.",
            pieces=[("show_multicollinearity", "How much each is explained by the others (VIF)"),
                    ("show_covariance_matrix", "Covariance of the coefficients")]),
        Ask("Does the model explain anything?",
            "Three tests of all the covariates together, and how well the model ranks who has the event first.",
            pieces=[("gof_lrt", "Likelihood-ratio test"), ("gof_wald", "Wald test"), ("gof_score", "Score test"),
                    ("gof_concordance", "Harrell's C", "The share of pairs of subjects the model ranks in the right order.")],
            first=["gof_lrt", "gof_concordance"]),
        Ask("Does each effect hold steady over time?",
            "A Cox model assumes a hazard ratio stays the same throughout follow-up. If it does not, the single ratio is "
            "an average that describes no moment.",
            pieces=[("check_schoenfeld_ph", "Test of each covariate"),
                    ("check_schoenfeld_vs_time", "Schoenfeld residuals against time"),
                    ("check_loglog_survival", "log(−log survival) curves for categories")],
            first=["check_schoenfeld_ph"]),
        Ask("Is the shape of each effect right, and does any subject stand out?",
            "Residual plots: a bend says a covariate does not act in a straight line; a point far from the rest is a "
            "subject the model fits badly.",
            pieces=[("check_martingale_vs_covariate", "Martingale residuals against each covariate"),
                    ("check_martingale_vs_linear", "Martingale residuals against risk score"),
                    ("check_deviance_vs_covariate", "Deviance residuals against each covariate"),
                    ("check_deviance_vs_linear", "Deviance residuals against risk score"),
                    ("check_coxsnell", "Cox-Snell residuals - the model as a whole")],
            first=["check_martingale_vs_covariate", "check_deviance_vs_linear"]),
        Ask("Would a simpler model do as well?",
            "A likelihood-ratio test and AIC of the full model against one with covariates left out.",
            pieces=[("compare_models", "Compare with a simpler model")],
            inside=[Custom(lambda p: drop_editor(_covariates, "dropped_covariates"))]),
        Ask("How does it compare with other models?",
            "Numbers that only mean something beside the same numbers from another model of the same data.",
            pieces=[("show_aic", "AIC"), ("show_partial_loglik", "Partial log-likelihood"),
                    ("show_neg2ll", "−2 × log-likelihood"), ("show_pseudo_r2", "Pseudo R²")]),
        Ask("What does it predict?",
            "The relative risk of a subject - for rows without a time or outcome, or for covariate values you type.",
            pieces=[("predict_missing_rows", "For subjects with no time or outcome", "Every covariate must be filled in."),
                    ("_use_scenarios", "For covariate values I type in")],
            inside=[Custom(lambda p: scenario_editor(_covariates, "scenarios"), when=lambda s, c: s["_use_scenarios"])]),
        Drawer("Predicted survival curves", [
            Note("One curve per row, for the covariate values you give - e.g. a treated and an untreated patient of "
                 "the same age."),
            Custom(lambda p: scenario_editor(_covariates, "graph_defs", named=True, baseline=True, noun="curve")),
        ]),
        Drawer("Categories", [Custom(lambda p: reference_editor(_covariates, include_x=False))]),
        Drawer("Fine-tune", [Row("Coefficient intervals by", Pick("ci_approach", [
            ("profile", "profile likelihood (can be lopsided)"), ("wald", "Wald (always symmetric)")]))]),
    ],
    virtual={"_use_scenarios": False},
    finish=_cox_finish,
    methods=lambda s, c: ("Time to event was modelled by Cox proportional hazards regression "
                          f"({'Efron' if s['ties_method'] == 'efron' else 'Breslow'} ties); effects are hazard ratios "
                          f"with {confidence()} confidence intervals."),
)

# ── survival by intervals ───────────────────────────────────────────────────


def _by_subject(s, ctx) -> bool:
    if s["layout"] != "auto":
        return s["layout"] == "subjects"
    return not (ctx.shape is not None and ctx.shape.replicates >= 2 and ctx.shape.counts)


LIFE = Spec(
    "life_table", LifeTableOptions,
    [
        Heading("How is the sheet laid out?"),
        Choose("layout", [
            Opt("auto", "Work it out"),
            Opt("subjects", "A subject on each row", "X is the time; each data set holds a code for what happened."),
            Opt("intervals", "An interval on each row", "X is where the interval starts; each data set has two columns side "
                                                        "by side - events, then those who left early."),
        ], columns=3),
        Sentence(["An event is coded", Num("death_code", decimals=4, width=54), "; leaving early (censored)",
                  Num("censored_code", decimals=4, width=54)], when=_by_subject),
        Sentence(["Intervals of width", Num("interval_width", 0, 1e9, 4, 74), "(0 = chosen for you)"], when=_by_subject),
        Note("Everybody is accounted for: the number entering the first interval is the total of the events and of those "
             "who left early. Put subjects still being followed at the end as leaving early in the last interval.",
             tone="faint", when=lambda s, c: not _by_subject(s, c)),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How many are still event-free as time goes on?",
            "The share that reaches the end of each interval without the event, with its standard error.",
            always=True, pieces=[("graph_survival", "Graph of the share still event-free")]),
        Ask("When is the risk highest?",
            "The hazard rate - the risk per unit of time among those who got that far - interval by interval.",
            pieces=[("show_hazard", "Hazard rate and density in each interval"), ("graph_hazard", "Graph of the hazard rate")]),
        Ask("Do the groups differ?",
            "Log-rank and Wilcoxon (Gehan) tests on the intervals.",
            pieces=[("compare", "Test whether the groups differ")]),
    ],
    methods=lambda s, c: ("Survival was summarised by the life-table (actuarial) method, with Greenwood standard errors"
                          + ("; groups were compared with log-rank and Wilcoxon (Gehan) tests on the intervals."
                             if s["compare"] and c.groups >= 2 else ".")),
)

# ── risk that changes over time ─────────────────────────────────────────────


def _tv_covariates(panel):
    names = panel.ctx.names or ["Set A", "Set B", "Set C"]
    skip = {panel.state.get("event_index", 0), panel.state.get("switch_index", -1)}
    return [n for i, n in enumerate(names) if i not in skip]


def _tv_gaps(s, ctx) -> list[str]:
    """Covariates with a blank on a row that has a time: those rows will be left out."""
    if ctx.sheet is None:
        return []
    try:
        labels, matrix = ctx.sheet.first_replicate_matrix()
        timed = ctx.sheet.x[: matrix.shape[0]] == ctx.sheet.x[: matrix.shape[0]]
        rows = matrix[: timed.size][timed]
    except Exception:  # noqa: BLE001
        return []
    skip = {s.get("event_index", 0), s.get("switch_index", -1)}
    return [label for i, label in enumerate(labels) if i not in skip and (rows[:, i] != rows[:, i]).any() and (rows[:, i] == rows[:, i]).any()]


def _tv_methods(s, ctx) -> str:
    text = ("Time to event was modelled by Cox regression with time-dependent covariates "
            f"({'Efron' if s['ties_method'] == 'efron' else 'Breslow'} ties)")
    if s["vary_effects"]:
        how = {"log": "the logarithm of time", "linear": "time", "step": "a step at a chosen time"}[s["time_function"]]
        text += f"; hazard ratios were allowed to change with {how}"
    if s["switch_index"] >= 0:
        text += f"; {ctx.name(int(s['switch_index']))} entered as a state that switches on during follow-up"
    return text + f". Hazard ratios are given with {confidence()} confidence intervals."


RISK_OVER_TIME = Spec(
    "cox_time_varying", CoxTimeVaryingOptions,
    [
        Note("X holds each subject's time. One data set records what happened; the others are covariates."),
        Row("What happened is in", Pick("event_index", columns)),
        Sentence(["An event is coded", Num("event_value", decimals=4, width=54), "; censored",
                  Num("censored_value", decimals=4, width=54)]),
        Heading("An effect that fades or grows"),
        Switch("vary_effects", "Let hazard ratios change during follow-up", "Instead of one hazard ratio for the whole of it."),
        Custom(lambda p: subset_editor(_tv_covariates, "varying", "Covariates whose effect may change"),
               when=lambda s, c: s["vary_effects"]),
        Choose("time_function", [
            Opt("log", "Smoothly, fastest early", "With the logarithm of time - the usual choice."),
            Opt("linear", "Steadily", "By the same factor for each unit of time."),
            Opt("step", "In one step", "One hazard ratio up to a chosen time, another after it."),
        ], columns=3, when=lambda s, c: s["vary_effects"]),
        Sentence(["The step is at time", Num("step_time", 0, 1e9, 4, 74), "(0 = the median event time)"],
                 when=lambda s, c: s["vary_effects"] and s["time_function"] == "step"),
        Sentence(["Report hazard ratios at times", Txt("times", width=170, placeholder="chosen for you, or: 6, 12, 24")],
                 when=lambda s, c: s["vary_effects"] and s["time_function"] != "step"),
        Heading("A state that begins part-way"),
        Row("Start times are in", Pick("switch_index", lambda c: c.column_options([(-1, "no data set - nothing begins part-way")]))),
        Note("A data set holding the time each subject's treatment, transplant or relapse began, blank for those it never "
             "did. They count as 'off' before that time and 'on' after it.", tone="faint"),
        Note("Choose a different data set from the one that records what happened.", tone="warn",
             when=lambda s, c: s["switch_index"] == s["event_index"]),
        Note(lambda s, c: (", ".join(_tv_gaps(s, c)) + (" has" if len(_tv_gaps(s, c)) == 1 else " have") + " blanks. As a covariate, "
                           "every row with a blank is left out; if it holds start times, choose it above."),
             tone="warn", when=lambda s, c: bool(_tv_gaps(s, c))),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How does the hazard ratio move over time?",
            "The ratio at times through follow-up, and the whole course of it as a graph with its band.",
            pieces=[("show_at_times", "Hazard ratios at chosen times"), ("graph_effect", "Graph of the hazard ratio over time")]),
        Ask("Does the model explain anything?",
            "Tests of all the covariates together, and how well the model ranks who has the event first.",
            pieces=[("show_fit", "The model as a whole")]),
        Row("Tied times handled by", Pick("ties_method", [("efron", "Efron's method (usual)"), ("breslow", "Breslow's method")])),
        Drawer("Categories", [Custom(lambda p: reference_editor(_tv_covariates, include_x=False))]),
    ],
    methods=_tv_methods,
)

SPECS = [SURVIVAL, LIFE, RISK, RISK_OVER_TIME]
