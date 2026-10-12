"""Fit a curve: nonlinear models, straight lines, yes/no outcomes, standard
curves, and working with a trace (its area, slope and trend)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...backends.area_under_curve import AreaUnderCurveOptions
from ...backends.baseline_spectrum import BaselineOptions
from ...backends.compare_models import NESTED, CompareModelsOptions
from ...backends.curve_smoothing_diff_integrate import CurveOpsOptions
from ...backends.deming_regression import DemingRegressionOptions
from ...backends.fourier_filter import FourierOptions
from ...backends.global_fit import GlobalFitOptions
from ...backends.linear_regression import LinearRegressionOptions
from ...backends.logistic_regression import LogisticRegressionOptions
from ...backends.nonlinear_regression import MODELS, NonlinearRegressionOptions
from ...backends.peak_fitting import PeakOptions
from ...backends.spline_lowess_smoothing import LOWESS_DEFAULT, LOWESS_PRESETS, SplineLowessOptions
from ...backends.standard_curve_interpolation import StandardCurveOptions
from .. import kit as sk
from ..spec import Ask, Chips, Choose, Custom, Drawer, Heading, Int, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch, Txt
from .common import confidence, repeats
from .sketches import area_scene, sketch, slope_scene


def _x_limits(panel) -> None:
    """Start any X limits at the sheet's own range, so switching them on never empties the fit."""
    rng = panel.ctx.x_range()
    if rng is not None:
        panel.suggest({"range_start_x": rng[0], "range_end_x": rng[1]})


def _limit_finish(s, ctx) -> None:
    own = bool(s.pop("_limit", False))
    s["range_mode_start"] = s["range_mode_end"] = "manual" if own else "auto"


_LIMIT = [
    Switch("_limit", "Fit only part of the X range", "Points outside it stay on the graph but not in the fit."),
    Sentence(["Fit X from", Num("range_start_x", decimals=6, width=80), "to", Num("range_end_x", decimals=6, width=80)],
             when=lambda s, c: s["_limit"]),
]
_REPEAT_CARDS = Choose("replicate_handling", [
    Opt("average", "Fit each point's mean", "The safe choice when repeats are technical - the same sample again."),
    Opt("individual", "Fit every repeat", "Only when each repeat is an independent experiment; otherwise intervals "
                                          "come out too narrow."),
], when=lambda s, c: repeats(c))

# ── a curve ─────────────────────────────────────────────────────────────────

_last_model = ["michaelis_menten"]


def _picker(panel) -> QWidget:
    from ...equation_picker import EquationPicker, FitModelCatalog

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    key = panel.state.get("model")
    start = key if key in MODELS and key != "quadratic" else _last_model[0]
    panel.state["model"] = start
    picker = EquationPicker(start, catalog=FitModelCatalog())
    picker.setMinimumHeight(360)
    lay.addWidget(picker)

    def chosen(k: str) -> None:
        if k in MODELS:
            _last_model[0] = k
            panel.assign({"model": k})

    picker.chosen.connect(chosen)
    box.refresh = lambda p: None
    return box


def _curve_methods(s, ctx) -> str:
    model = MODELS.get(s["model"])
    name = model.label if model is not None else s["model"].replace("_", " ")
    weight = "weighted by 1/Y²" if s["relative_weighting"] else "unweighted"
    return (f"The data were fitted to the model \"{name}\" by nonlinear least squares ({weight}); parameters are "
            f"reported with {confidence()} confidence intervals.")


CURVE = Spec(
    "nonlinear_regression", NonlinearRegressionOptions,
    [
        Heading("Which curve?", "Browse by family, or search by name or parameter (EC50, Km, Kd...)."),
        Custom(_picker),
        Heading("How does the scatter behave?"),
        Choose("relative_weighting", [
            Opt(False, "The same everywhere", "Every point counts equally - ordinary least squares."),
            Opt(True, "It grows with Y", "Weight each point by 1/Y² - common in enzyme and binding assays."),
        ]),
        Drawer("Fine-tune", _LIMIT),
    ],
    virtual={"_limit": False},
    finish=_limit_finish,
    on_sheet=_x_limits,
    methods=_curve_methods,
)

# ── a straight line ─────────────────────────────────────────────────────────


def _line_sheet(panel) -> None:
    _x_limits(panel)
    rng = panel.ctx.x_range()
    if rng is not None:
        panel.suggest({"ci_y_at_x_value": round((rng[0] + rng[1]) / 2, 6)})
    y = panel.ctx.y_range()
    if y is not None:
        panel.suggest({"ci_x_at_y_value": round((y[0] + y[1]) / 2, 6)})


LINE = Spec(
    "linear_regression", LinearRegressionOptions,
    [
        _REPEAT_CARDS,
        Heading("Must the line pass through a point?"),
        Switch("force_point", "Force it through a point",
               "Only when theory demands it - often the origin. R² then means something different."),
        Sentence(["Through X =", Num("force_x", decimals=6, width=70), ", Y =", Num("force_y", decimals=6, width=70)],
                 when=lambda s, c: s["force_point"]),
        Heading("What else do you want to know?"),
        Ask("Is a straight line the right shape?",
            "Points that wander above the line and then below it say the truth is curved. With repeats at the same X, "
            "the scatter about the line can also be weighed against the scatter among the repeats.",
            pieces=[("runs_test", "Runs test"),
                    ("lack_of_fit", "Lack-of-fit test", "Needs some X values to be repeated - side-by-side repeats count."),
                    ("residual_table", "Residuals against fitted values")],
            first=["runs_test", "residual_table"]),
        Ask("Which points pull the line?",
            "The points that would move the line most if they were left out, and whether the furthest one is an outlier.",
            pieces=[("influence", "Table of the ten most influential points"), ("influence_plot", "Graph of every point's pull")]),
        Ask("Would a line that ignores odd points agree?",
            "The Theil-Sen line: the middle one of the slopes between every pair of points. A few wild points cannot move it.",
            pieces=[("robust_line", "Robust line, drawn beside the least-squares one")]),
        Ask("What does the line predict?",
            "A value read off the line in either direction, with its interval. For X at a Y the limits are Fieller's, "
            "which are exact where a symmetric interval is not.",
            pieces=[("ci_y_at_x", "Y at an X I choose"), ("ci_x_at_y", "X at a Y I choose"),
                    ("interpolate_unknowns", "X for every Y in the second data set")],
            first=["ci_y_at_x"],
            inside=[Sentence(["Y at X =", Num("ci_y_at_x_value", decimals=6, width=80)], when=lambda s, c: s["ci_y_at_x"]),
                    Sentence(["X at Y =", Num("ci_x_at_y_value", decimals=6, width=80)], when=lambda s, c: s["ci_x_at_y"])]),
        Ask("Show me the line", "The points with the fitted line through them.",
            pieces=[("graph", "The data with the line"), ("show_confidence_band", "A shaded band around it")],
            first=["graph"],
            inside=[Row("The band shows", Pick("band_type", [("confidence", "where the true line probably lies"),
                                                              ("prediction", "where a new point would probably fall")]),
                        when=lambda s, c: s["show_confidence_band"])]),
        Drawer("Fine-tune", _LIMIT),
    ],
    virtual={"_limit": False},
    finish=_limit_finish,
    on_sheet=_line_sheet,
    methods=lambda s, c: ("A straight line was fitted by least squares" +
                          (f", forced through ({s['force_x']:g}, {s['force_y']:g})" if s["force_point"] else "") +
                          f"; slope and intercept are given with {confidence()} confidence intervals." +
                          (" Linearity was assessed by a lack-of-fit F test against the scatter among repeats." if s["lack_of_fit"] else "") +
                          (" Influential points were identified by Cook's distance and studentised residuals." if s["influence"] else "") +
                          (" A Theil-Sen line was fitted as a robust check." if s["robust_line"] and not s["force_point"] else "")),
)

DEMING = Spec(
    "deming_regression", DemingRegressionOptions,
    [
        _REPEAT_CARDS,
        Heading("How do the two measurement errors compare?"),
        Choose("equal_error", [
            Opt(True, "About the same", "Two methods in the same units and of similar precision - the usual case."),
            Opt(False, "Different, and I know each", "Give each method's error SD; only their ratio matters."),
        ]),
        Sentence(["Error SD of X", Num("sd_x", 1e-12, 1e12, 6, 70), ", of Y", Num("sd_y", 1e-12, 1e12, 6, 70)],
                 when=lambda s, c: not s["equal_error"]),
        Heading("What else do you want to know?"),
        Ask("What does the line predict?", "Y read off the line at an X you choose.",
            pieces=[("ci_y_at_x", "Y at an X I choose")],
            inside=[Sentence(["Y at X =", Num("ci_y_at_x_value", decimals=6, width=80)])]),
        Ask("Show me the line", "The points with the fitted line, and the line of identity (Y = X) for comparison.",
            pieces=[("graph", "The data with both lines")]),
        Drawer("Fine-tune", _LIMIT),
    ],
    virtual={"_limit": False},
    finish=_limit_finish,
    on_sheet=_line_sheet,
    methods=lambda s, c: ("The line was fitted by Deming regression, allowing for measurement error in X and Y" +
                          (" (equal error variances)." if s["equal_error"] else
                           f" (error SDs {s['sd_x']:g} and {s['sd_y']:g}).")),
)

# ── a yes/no outcome ────────────────────────────────────────────────────────


def _logistic_finish(s, ctx) -> None:
    own = bool(s.pop("_extend", False))
    s["min_x_mode"] = s["max_x_mode"] = "manual" if own else "auto"
    if not s.pop("_predict_typed", False):
        s["predict_at"] = ""
    if not s["graph_curve"]:
        s["plot_confidence_bands"] = s["mark_levels"] = False


def _logistic_sheet(panel) -> None:
    rng = panel.ctx.x_range()
    if rng is not None:
        panel.suggest({"min_x_value": rng[0], "max_x_value": rng[1]})
        span = rng[1] - rng[0]
        if span > 0:   # a step of about a tenth of the range, as a round number
            panel.suggest({"odds_step": float(f"{10 ** round(math.log10(span / 10)):g}")})


def _logistic_methods(s, ctx) -> str:
    fit = {"auto": "maximum likelihood (Firth's penalised likelihood if the predictor separated the outcomes)",
           "ml": "maximum likelihood", "firth": "Firth's penalised likelihood"}[s["fit_method"]]
    text = (f"The probability of the outcome was modelled by binary logistic regression on X, fitted by {fit}. The "
            f"effect is reported as the odds ratio for a rise of {s['odds_step']:g} in X with its {confidence()} "
            f"{'profile-likelihood' if s['interval_method'] == 'profile' else 'Wald'} confidence interval.")
    if s["x_at_levels"]:
        text += " The X giving a stated probability is reported with Fieller's confidence limits."
    if s["roc_area"]:
        text += " Discrimination was summarised by the area under the ROC curve (DeLong's standard error)."
    return text


YES_NO = Spec(
    "logistic_regression", LogisticRegressionOptions,
    [
        Note("Code the outcome 1 when it happened and 0 when it didn't; X is the predictor."),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How much does the chance change with X?",
            "The odds ratio and its interval - for a step in X that means something to you, such as 10 years or one dose.",
            always=True,
            inside=[Sentence(["Give it for a rise of", Num("odds_step", 1e-12, 1e12, 6, 80), "in X"])]),
        Ask("At what X does the chance reach a level?",
            "The X for a 50% chance, or any level you name, with limits that stay honest when the slope is uncertain.",
            pieces=[("x_at_levels", "The X at each level")],
            inside=[Sentence(["Chance levels, in percent:", Txt("levels", 150, "50  or  10, 50, 90")])]),
        Ask("How well does X tell yes from no?",
            "One number from 0.5 (no better than a coin) to 1 (perfect), and where to draw the line.",
            pieces=[("roc_area", "Area under the ROC curve, with its interval"), ("roc_curve", "The ROC curve"),
                    ("best_cutoff", "The cutoff that separates best", "The X with the largest sensitivity + specificity."),
                    ("classification", "Rows called right and wrong at a cutoff")],
            first=["roc_area", "roc_curve"],
            inside=[Sentence(["Call a row yes at a predicted chance of", Num("cutoff", 0.001, 0.999, 3, 64), "or more"],
                             when=lambda s, c: s["classification"])]),
        Ask("Can the predicted chances be trusted?",
            "Whether a predicted 30% comes true about 30% of the time.",
            pieces=[("brier", "Brier score", "The average squared gap between predicted chance and outcome."),
                    ("calibration_test", "Hosmer-Lemeshow test"),
                    ("calibration_plot", "Graph of predicted against what happened")]),
        Ask("Does X earn its place?",
            "A test against a model with no predictor, and measures for comparing models.",
            pieces=[("likelihood_test", "Likelihood-ratio test"), ("information", "AIC, AICc, BIC and deviance"),
                    ("r2_tjur", "Tjur's R²"), ("r2_mcfadden", "McFadden's R²"), ("r2_nagelkerke", "Nagelkerke's R²"),
                    ("r2_cox_snell", "Cox and Snell's R²")],
            first=["likelihood_test"]),
        Ask("Which rows pull the fit?",
            "The rows that would change the answer most if they were left out.",
            pieces=[("influence", "Table of the ten most influential rows"), ("influence_plot", "Graph of every row's pull")]),
        Ask("What does it predict?",
            "The predicted chance, with an interval, for the rows you have or for X values you name.",
            pieces=[("predicted_probabilities", "For each row on the sheet"), ("_predict_typed", "For X values I type")],
            inside=[Sentence(["At X =", Txt("predict_at", 190, "for example 5, 12.5, 20")],
                             when=lambda s, c: s["_predict_typed"])]),
        Ask("Show me the curve",
            "The fitted S-curve drawn over the 0 and 1 points.",
            pieces=[("graph_curve", "The curve"), ("plot_confidence_bands", "Its confidence band"),
                    ("mark_levels", "Lines at the chance levels")],
            first=["graph_curve"],
            inside=[Switch("_extend", "Draw it beyond the data", "Extrapolation - read it with care."),
                    Sentence(["From X =", Num("min_x_value", decimals=6, width=80), "to", Num("max_x_value", decimals=6, width=80)],
                             when=lambda s, c: s["_extend"])]),
        Drawer("Fine-tune", [
            Row("Fit by", Pick("fit_method", [("auto", "maximum likelihood - Firth's fit if X splits yes from no"),
                                              ("firth", "Firth's penalised likelihood, always"),
                                              ("ml", "maximum likelihood, always")]),
                "Firth's fit is less biased when one outcome is rare, and has an answer when X splits them completely."),
            Row("Intervals by", Pick("interval_method", [("profile", "profile likelihood (careful; the usual choice)"),
                                                          ("wald", "Wald (quick, always symmetric)")])),
        ]),
    ],
    virtual={"_extend": False, "_predict_typed": False},
    finish=_logistic_finish,
    on_sheet=_logistic_sheet,
    methods=_logistic_methods,
)

# ── a standard curve ────────────────────────────────────────────────────────


def _sig(x, lo=0.05, hi=0.95, mid=0.5, k=10.0):
    return lo + (hi - lo) / (1 + math.exp(-k * (x - mid)))


_STANDARD_SHAPES = [
    ("4pl_x", "Sigmoid", "Bottom, top, midpoint and steepness.", lambda x: _sig(x)),
    ("4pl_logx", "Sigmoid on log X", "The same, with concentrations spread on a log scale.", lambda x: _sig(x, k=6)),
    ("5pl_x", "Lopsided sigmoid", "A fifth term for a curve steeper at one end.",
     lambda x: 0.05 + 0.9 / (1 + math.exp(-9 * (x - 0.45))) ** 0.45),
    ("hyperbola", "Saturating", "Rises and levels off - binding-style.", lambda x: x / (0.2 + x)),
    ("line", "Straight line", "When the standards fall on a line.", lambda x: x),
    ("semilog", "Line against log X", "Straight once X is logged.", lambda x: math.log10(1 + 9 * x)),
    ("quadratic", "Gentle bend", "A second-order polynomial.", lambda x: 0.2 + 0.4 * x + 0.6 * x * x),
    ("cubic", "S-bend", "A cubic curve.", lambda x: 0.5 + 1.2 * (x - 0.5) ** 3 + 0.1 * (x - 0.5)),
]

STANDARD = Spec(
    "standard_curve_interpolation", StandardCurveOptions,
    [
        Note("Standards go in the first data set (known X, measured Y); each unknown's reading goes in the next, "
             "and gets a value read off the curve."),
        Heading("Shape of the standard curve"),
        Choose("model", [Opt(k, t, b, glyph=sk.curve_glyph(f)) for k, t, b, f in _STANDARD_SHAPES], columns=2),
        Heading("A standard that looks wrong"),
        Choose("outlier_handling", [
            Opt("none", "Keep every standard", "Each counts the same."),
            Opt("robust", "Count far-off ones for less", "A robust fit - nothing is removed."),
            Opt("eliminate", "Leave clear outliers out", "Tested on the residuals, then refitted without them."),
            Opt("report", "Only flag them", "The fit uses every standard; suspects are listed."),
        ], columns=2),
        Sentence(["Accept a", Num("outlier_q", 0.01, 20, 2, 58, " %"), "chance of flagging a good standard"],
                 when=lambda s, c: s["outlier_handling"] in ("eliminate", "report")),
        Heading("Also"),
        Chips([("", [("relative_weighting", "Weights of 1/Y²"), ("report_ci", "An interval for each unknown")])]),
    ],
    methods=lambda s, c: (f"Unknowns were read off a {dict((k, t.lower()) for k, t, *_ in _STANDARD_SHAPES)[s['model']]} "
                          "standard curve fitted to the standards" +
                          (f", with {confidence()} confidence intervals." if s["report_ci"] else ".")),
)

# ── a trace: area ───────────────────────────────────────────────────────────


_HEIGHT_WORDS = [("percent", "of the tallest one"), ("y_units", "in Y units")]


def _area_sheet(panel) -> None:
    rng = panel.ctx.y_range()
    if rng is not None:
        panel.suggest({"min_height_units": float(f"{(rng[1] - rng[0]) / 10:.3g}")})


AREA = Spec(
    "area_under_curve", AreaUnderCurveOptions,
    [
        Heading("See what will be measured", "The shaded part of the trace is its area. It follows every choice below."),
        Custom(lambda p: sketch(area_scene)),
        Heading("Measure the area from"),
        Choose("baseline_mode", [Opt("fixed", "A level I set"), Opt("mean_ends", "The trace's own ends")],
               style="segmented"),
        Sentence(["The level is Y =", Num("baseline_y", decimals=6, width=80)], when=lambda s, c: s["baseline_mode"] == "fixed"),
        Sentence(["Average the first", Int("mean_first_n", 0, 10000), "points and the last", Int("mean_last_n", 0, 10000)],
                 when=lambda s, c: s["baseline_mode"] == "mean_ends"),
        Heading("Bumps too small to count"),
        Sentence(["Leave out any bump lower than", Num("min_height_percent", 0, 100, 2, 58, " %"),
                  Pick("min_height_mode", _HEIGHT_WORDS)], when=lambda s, c: s["min_height_mode"] == "percent"),
        Sentence(["Leave out any bump lower than", Num("min_height_units", 0, 1e12, 6, 80),
                  Pick("min_height_mode", _HEIGHT_WORDS)], when=lambda s, c: s["min_height_mode"] == "y_units"),
        Chips([("", [("min_width_enabled", "Also leave out very narrow ones", "So noise is not counted as extra bumps."),
                     ("allow_below_baseline", "Count dips under the level as negative area")])]),
        Sentence(["Narrow means fewer than", Int("min_width_points", 1, 10000), "points wide"],
                 when=lambda s, c: s["min_width_enabled"]),
    ],
    on_sheet=_area_sheet,
    methods=lambda s, c: "Areas were calculated by the trapezoid rule, peak by peak, above the chosen baseline.",
)

# ── a trace: slope and running total ────────────────────────────────────────

_SMOOTH = {"light": (2, 2), "medium": (4, 2), "strong": (9, 2)}   # neighbours on each side, polynomial degree


def _slope_finish(s, ctx) -> None:
    level = s.pop("_smoothness")
    s["smooth"] = level != "off"
    if level in _SMOOTH:
        s["smooth_neighbors"], s["smooth_polyorder"] = _SMOOTH[level]


SLOPE = Spec(
    "curve_smoothing_diff_integrate", CurveOpsOptions,
    [
        Heading("Watch the trace change", "Grey is the trace as typed; blue is what comes back."),
        Custom(lambda p: sketch(slope_scene, empty="With at least five points on the sheet, the trace is drawn here "
                                                   "before and after.")),
        Heading("How much smoothing?"),
        Choose("_smoothness", [Opt("off", "None"), Opt("light", "Light"), Opt("medium", "Medium"), Opt("strong", "Strong"),
                               Opt("own", "Set by hand")], style="segmented"),
        Sentence(["Through each point and", Int("smooth_neighbors", 1, 50, 50), "neighbours on either side, fit a polynomial "
                  "of degree", Int("smooth_polyorder", 1, 6, 44)], when=lambda s, c: s["_smoothness"] == "own"),
        Note("Savitzky-Golay smoothing. More neighbours smooth more; a higher degree keeps sharp peaks sharp.",
             tone="faint", when=lambda s, c: s["_smoothness"] != "off"),
        Heading("What do you want back?"),
        Choose("operation", [
            Opt("none", "The trace", "Smoothed, if you asked for that."),
            Opt("first_derivative", "Its slope", "How fast Y changes with X, at every point."),
            Opt("second_derivative", "Its curvature", "How fast the slope itself changes."),
            Opt("integrate", "Its running area", "The area built up from left to right."),
        ], columns=4),
        Sentence(["Start the running area at", Num("integral_start_y", decimals=6, width=80)],
                 when=lambda s, c: s["operation"] == "integrate"),
    ],
    virtual={"_smoothness": "medium"},
    finish=_slope_finish,
    methods=lambda s, c: ("The trace was " + ("smoothed (Savitzky-Golay) and " if s["_smoothness"] != "off" else "") +
                          {"none": "left as it is.", "first_derivative": "differentiated.",
                           "second_derivative": "differentiated twice.", "integrate": "integrated (trapezoid rule)."}[
                              s["operation"]]),
)

# ── a trace: trend ──────────────────────────────────────────────────────────


def _wiggle(x):
    return 0.5 + 0.35 * math.sin(6 * x) * (1 - x) + 0.3 * x


TREND = Spec(
    "spline_lowess_smoothing", SplineLowessOptions,
    [
        Heading("Draw the trend as"),
        Choose("method", [
            Opt("smoothing_spline", "A smoothing spline", "Near the points, not through them - a set number of bends.",
                glyph=sk.curve_glyph(lambda x: 0.5 + 0.3 * math.sin(3 * x) + 0.2 * x)),
            Opt("lowess", "LOWESS", "A local fit around every point - follows the data's own shape.",
                glyph=sk.curve_glyph(lambda x: 0.5 + 0.25 * math.sin(5 * x) + 0.2 * x)),
            Opt("cubic", "Cubic spline", "Through every point, smoothly - may swing past them.",
                glyph=sk.curve_glyph(_wiggle)),
            Opt("akima", "Akima spline", "Through every point, with less overshoot.",
                glyph=sk.curve_glyph(lambda x: _wiggle(x) * 0.9 + 0.05)),
            Opt("linear", "Straight segments", "From each point straight to the next; no smoothing.",
                glyph=sk.curve_glyph(lambda x: [0.2, 0.7, 0.4, 0.9, 0.6][min(4, int(x * 4.999))], points=5)),
        ], columns=2),
        Sentence(["Allow", Int("n_knots", 1, 50, 44), "bends (interior knots)"], when=lambda s, c: s["method"] == "smoothing_spline"),
        Row("Each local fit uses", Pick("lowess_preset", [(k, k) for k in LOWESS_PRESETS]),
            when=lambda s, c: s["method"] == "lowess"),
        Drawer("Fine-tune", [
            Sentence(["Draw the curve from", Int("n_segments", 2, 5000, 58), "straight pieces"]),
            Switch("report_residuals", "List each point's distance from the curve"),
        ]),
    ],
    methods=lambda s, c: {"smoothing_spline": f"A smoothing spline with {s['n_knots']} interior knots was fitted.",
                          "lowess": "The trend was drawn by LOWESS.", "cubic": "A cubic spline was drawn through the points.",
                          "akima": "An Akima spline was drawn through the points.",
                          "linear": "The points were joined by straight segments."}[s["method"]],
)

# ── data sets together ──────────────────────────────────────────────────────


def _own_parameters(panel):
    """A chip for each parameter of the chosen model: on = every data set has its own value of it."""
    from .models import _Rebuilding

    def signature(p):
        model = MODELS.get(p.state.get("model"))
        return tuple(model.param_names) if model is not None else ()

    def build(p, lay):
        names = signature(p)
        kept = [n for n in (p.state.get("separate") or []) if n in names]
        p.state["separate"] = kept
        group = sk.ChipGroup("")
        for name in names:
            chip = group.add(sk.Chip(name, name in kept))

            def toggle(on, n=name):
                current = [x for x in (p.state.get("separate") or []) if x != n]
                p.assign({"separate": current + ([n] if on else [])})

            chip.toggled.connect(toggle)
        lay.addWidget(group)

    return _Rebuilding(build, signature)


def _global_methods(s, ctx) -> str:
    model = MODELS.get(s["model"])
    name = model.label if model is not None else s["model"].replace("_", " ")
    names = list(model.param_names) if model is not None else []
    own = [n for n in names if n in (s.get("separate") or [])]
    shared = [n for n in names if n not in own]
    weight = "weighted by 1/Y²" if s["relative_weighting"] else "unweighted"
    text = f"The data sets were fitted together to the model \"{name}\" by nonlinear least squares ({weight})"
    if shared and own:
        text += f", with {', '.join(shared)} shared between them and {', '.join(own)} fitted separately"
    elif shared:
        text += ", with every parameter shared"
    text += "; shared and separate fits were compared with the extra sum-of-squares F test."
    return text


TOGETHER = Spec(
    "global_fit", GlobalFitOptions,
    [
        Heading("Which curve?", "The same model is fitted to every data set."),
        Custom(_picker),
        Heading("What may differ between the data sets?", "Parameters switched on get a value of their own in each data set. "
                                                         "The rest are one value, shared by all."),
        Custom(_own_parameters),
        Note("With nothing switched on, one curve is fitted to all the data sets - and the result says whether that is fair."),
        Switch("test_each", "Test each separate parameter", "Does it really differ between the data sets? The usual way to compare "
                                                            "an EC50, a Km or a rate."),
        Heading("How does the scatter behave?"),
        Choose("relative_weighting", [
            Opt(False, "The same everywhere", "Every point counts equally - ordinary least squares."),
            Opt(True, "It grows with Y", "Weight each point by 1/Y² - common in enzyme and binding assays."),
        ]),
        Drawer("Fine-tune", _LIMIT),
    ],
    virtual={"_limit": False},
    finish=_limit_finish,
    on_sheet=_x_limits,
    methods=_global_methods,
)

# ── two models ──────────────────────────────────────────────────────────────

_MODEL_LIST = [(key, f"{model.label}  ({len(model.param_names)} parameters)") for key, model in MODELS.items()]


def _is_special_case(s) -> bool:
    a, b = s["model_a"], s["model_b"]
    return (a, b) in NESTED or (b, a) in NESTED


def _compare_methods(s, ctx) -> str:
    a, b = MODELS.get(s["model_a"]), MODELS.get(s["model_b"])
    if a is None or b is None:
        return "Two models were compared."
    how = "the extra sum-of-squares F test and by AICc" if _is_special_case(s) else "AICc"
    return f"The fits of \"{a.label}\" and \"{b.label}\" were compared by {how}."


TWO_MODELS = Spec(
    "compare_models", CompareModelsOptions,
    [
        Heading("Which two models?"),
        Row("Compare", Pick("model_a", _MODEL_LIST)),
        Row("with", Pick("model_b", _MODEL_LIST)),
        Note("One of these is a special case of the other, so both the F test and AICc are given.",
             when=lambda s, c: _is_special_case(s)),
        Note("Neither is a special case of the other, so the comparison is by AICc alone.",
             when=lambda s, c: not _is_special_case(s) and s["model_a"] != s["model_b"]),
        Note("Choose two different models.", tone="warn", when=lambda s, c: s["model_a"] == s["model_b"]),
        Heading("How does the scatter behave?"),
        Choose("relative_weighting", [
            Opt(False, "The same everywhere", "Every point counts equally - ordinary least squares."),
            Opt(True, "It grows with Y", "Weight each point by 1/Y² - common in enzyme and binding assays."),
        ]),
        Drawer("Fine-tune", _LIMIT),
    ],
    virtual={"_limit": False},
    finish=_limit_finish,
    on_sheet=_x_limits,
    methods=_compare_methods,
)

# ── peaks ───────────────────────────────────────────────────────────────────


def _peak_methods(s, ctx) -> str:
    if not s["fit"]:
        return f"Peaks were found by their prominence (at least {s['min_height_percent']:g}% of the range of the trace)."
    base = {"none": "no baseline", "constant": "a constant baseline", "linear": "a linear baseline", "curved": "a quadratic baseline"}[s["baseline"]]
    shape = {"gaussian": "Gaussian", "lorentzian": "Lorentzian", "voigt": "Voigt"}[s["shape"]]
    return (f"Peaks were found by their prominence and fitted together as {shape} profiles with {base} by nonlinear least "
            "squares; position, height, full width at half height and area are reported for each.")


PEAKS = Spec(
    "peak_fitting", PeakOptions,
    [
        Note("X is the position - wavelength, time, angle. Each data set is one trace."),
        Heading("What shape are the peaks?"),
        Choose("shape", [
            Opt("gaussian", "Gaussian", "The bell shape: chromatography, instrument-broadened lines.",
                glyph=sk.curve_glyph(lambda x: math.exp(-((x - 0.5) / 0.16) ** 2))),
            Opt("lorentzian", "Lorentzian", "Sharper top, long tails: resonances, natural line shapes.",
                glyph=sk.curve_glyph(lambda x: 1 / (1 + ((x - 0.5) / 0.09) ** 2))),
            Opt("voigt", "Voigt", "Both at once. One parameter more per peak: needs clean peaks.",
                glyph=sk.curve_glyph(lambda x: 0.5 * math.exp(-((x - 0.5) / 0.14) ** 2) + 0.5 / (1 + ((x - 0.5) / 0.1) ** 2))),
        ], columns=3),
        Choose("direction", [Opt("up", "Peaks point up"), Opt("down", "Dips point down")], style="segmented"),
        Heading("What lies under them?"),
        Choose("baseline", [Opt("none", "Nothing"), Opt("constant", "A flat level"), Opt("linear", "A sloping line"), Opt("curved", "A gentle curve")],
               style="segmented"),
        Note("For a baseline that wanders more than a gentle curve, flatten it first with 'Baseline'.",
             when=lambda s, c: s["baseline"] == "curved"),
        Heading("Which peaks count?"),
        Sentence(["A peak must stand", Num("min_height_percent", 0.01, 100, 2, 58, " %"), "of the trace's range out of its surroundings"]),
        Sentence(["Keep at most", Int("max_peaks", 1, 40, 50), "peaks, the most prominent first"]),
        Switch("fit", "Fit the peaks", "Off: the peaks are only found and measured from the trace - quicker, and no shape is assumed."),
    ],
    methods=_peak_methods,
)

# ── baseline ────────────────────────────────────────────────────────────────


def _baseline_methods(s, ctx) -> str:
    return {"smooth": f"The baseline was estimated by asymmetric least squares (Eilers and Boelens 2005; log10 smoothness {s['stiffness']:g}, "
                      "asymmetry 0.01) and subtracted.",
            "polynomial": f"The baseline was estimated with an iteratively fitted polynomial of degree {s['degree']} (Lieber and "
                          "Mahadevan-Jansen 2003) and subtracted.",
            "ends": f"A straight baseline between the means of the first and last {s['end_points']} points was subtracted."}[s["method"]]


FLATTEN = Spec(
    "baseline_spectrum", BaselineOptions,
    [
        Note("X is the position; each data set is one trace. The flattened traces go on a new sheet."),
        Heading("How should the baseline be found?"),
        Choose("method", [
            Opt("smooth", "A smooth curve under the peaks", "Follows a drifting, bending baseline. The usual choice.",
                glyph=sk.curve_glyph(lambda x: 0.25 + 0.25 * x + 0.12 * math.sin(5 * x))),
            Opt("polynomial", "A polynomial", "For a baseline that is a simple bow or slope.",
                glyph=sk.curve_glyph(lambda x: 0.2 + 0.6 * (x - 0.5) ** 2 + 0.2 * x)),
            Opt("ends", "A line between the ends", "When the trace begins and ends on its baseline and the drift is straight.",
                glyph=sk.curve_glyph(lambda x: 0.25 + 0.4 * x)),
        ], columns=3),
        Choose("direction", [Opt("up", "Peaks point up"), Opt("down", "Dips point down")], style="segmented",
               when=lambda s, c: s["method"] != "ends"),
        Sentence(["Stiffness", Num("stiffness", 0, 12, 1, 54), "- lower follows every bend, higher is nearer a straight line"],
                 when=lambda s, c: s["method"] == "smooth"),
        Sentence(["A polynomial of degree", Int("degree", 0, 10, 44)], when=lambda s, c: s["method"] == "polynomial"),
        Sentence(["Average", Int("end_points", 1, 500, 50), "points at each end"], when=lambda s, c: s["method"] == "ends"),
        Switch("create_new_sheet", "Put the flattened traces on a new sheet", "The sheet you analyse is never changed."),
    ],
    methods=_baseline_methods,
)

# ── frequencies ─────────────────────────────────────────────────────────────

_BAND = ("bandpass", "bandstop")


def _fourier_methods(s, ctx) -> str:
    if s["task"] == "spectrum":
        window = {"hann": "a Hann window", "hamming": "a Hamming window", "blackman": "a Blackman window", "none": "no window"}[s["window"]]
        return f"Amplitude spectra were calculated by fast Fourier transform with {window}."
    kind = ("Frequencies outside the pass band were set to zero in the Fourier transform" if s["sharpness"] == "sharp" else
            "A fourth-order Butterworth filter was applied forward and backward (zero phase)")
    what = {"lowpass": f"low-pass, cut-off {s['cutoff']:g}", "highpass": f"high-pass, cut-off {s['cutoff']:g}",
            "bandpass": f"band-pass, {s['band_low']:g} to {s['band_high']:g}", "bandstop": f"band-stop, {s['band_low']:g} to {s['band_high']:g}"}[s["task"]]
    return f"{kind} ({what}, in cycles per unit of X)."


FREQUENCIES = Spec(
    "fourier_filter", FourierOptions,
    [
        Note("X is time or position, in equal steps; each data set is one trace. Frequency is in cycles per unit of X."),
        Heading("What do you want to do?"),
        Choose("task", [
            Opt("spectrum", "See the frequencies", "How strong each rhythm in the trace is."),
            Opt("lowpass", "Keep the slow part", "Smooth away fast noise."),
            Opt("highpass", "Keep the fast part", "Take out slow drift."),
            Opt("bandpass", "Keep one band", "Pick out one rhythm."),
            Opt("bandstop", "Take one band out", "Remove one interfering rhythm, such as mains hum."),
        ], columns=3),
        Sentence(["Turn at a frequency of", Num("cutoff", 0, 1e12, 6, 90), "(0 = chosen for you)"],
                 when=lambda s, c: s["task"] in ("lowpass", "highpass")),
        Sentence(["From a frequency of", Num("band_low", 0, 1e12, 6, 90), "to", Num("band_high", 0, 1e12, 6, 90), "(0 and 0 = chosen for you)"],
                 when=lambda s, c: s["task"] in _BAND),
        Choose("sharpness", [Opt("gentle", "A gentle turn", "No ripples in the trace - the usual choice."),
                             Opt("sharp", "A sharp cut", "Exact in frequency; the trace may ring near sudden steps.")],
               when=lambda s, c: s["task"] != "spectrum"),
        Switch("create_new_sheet", "Put the result on a new sheet", "The spectrum, or the filtered traces. The sheet you analyse is never changed."),
        Drawer("Fine-tune", [
            Row("Fade the ends of the trace with", Pick("window", [("hann", "a Hann window (usual)"), ("hamming", "a Hamming window"),
                                                                   ("blackman", "a Blackman window"), ("none", "nothing")])),
            Row("Before the transform, take off", Pick("detrend", [("mean", "the mean"), ("line", "a straight-line trend"), ("none", "nothing")])),
            Sentence(["List the", Int("peaks_listed", 1, 30, 44), "strongest frequencies"]),
        ]),
    ],
    methods=_fourier_methods,
)

SPECS = [CURVE, TOGETHER, TWO_MODELS, LINE, DEMING, YES_NO, STANDARD, AREA, SLOPE, TREND, PEAKS, FLATTEN, FREQUENCIES]
