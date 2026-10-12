"""Reshape and clean: transform values, rescale them, filter rows, and lay the
table out another way. Every one of these writes a new sheet and leaves the
original as it was - and shows the first rows before and after, as the
settings change, so nothing has to be imagined."""

from __future__ import annotations

import numpy as np

from ...backends.baseline_correction import CALCULATIONS, BaselineOptions
from ...backends.extract_rearrange import ExtractRearrangeOptions
from ...backends.normalize_values import NormalizeOptions
from ...backends.remove_rows import RemoveRowsOptions
from ...backends.select_transform import SelectTransformOptions
from ...backends.transform_data import TransformDataOptions
from ...backends.transform_x_concentration import LOG_BASES, TransformXOptions
from ...backends.transpose_xy import TransposeOptions
from ..spec import Chips, Choose, Custom, Drawer, Heading, Int, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch, Txt
from .common import repeats
from .editors import rule_list, variable_chips
from .shaping import before_after, new_variables, pairing_list, recipe_lane, scale_ruler

_GOES = Note("The result goes on a new sheet. This one stays exactly as it is.", tone="faint")


# ── transform values: a recipe for each axis ────────────────────────────────

def _transform_methods(s, c) -> str:
    from ... import recipes

    parts = [f"{axis}: {recipes.describe(s[field])}" for axis, field in (("Y", "y_steps"), ("X", "x_steps")) if s[field]]
    return "Values were transformed" + (f" ({'; '.join(parts)})" if parts else "") + "."


TRANSFORM = Spec(
    "transform_data", TransformDataOptions,
    [
        Heading("Build the change, one step at a time",
                "Every value passes through the steps from left to right. An axis with no step is left alone."),
        Custom(lambda p: recipe_lane("y_steps", "Y values", "each Y value")),
        Custom(lambda p: recipe_lane("x_steps", "X values", "each X value")),
        Row("Side-by-side repeats", Pick("replicate_handling", [("individual", "change every repeat"),
                                                                 ("average", "average them, then change the average")]),
            when=lambda s, c: repeats(c)),
        Custom(lambda p: before_after("transform_data")),
        _GOES,
    ],
    start={"y_steps": [{"op": "log10"}], "x_steps": []},
    methods=_transform_methods,
)


# ── doses onto a log scale: one sentence ────────────────────────────────────

_LOG_WORDS = {"log10 (common)": "its log₁₀ is taken", "ln (natural)": "its natural log is taken", "log2": "its log₂ is taken"}


def _doses_sheet(panel) -> None:
    x = getattr(panel.ctx.sheet, "x", None)
    if x is None:
        return
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    positive = x[x > 0]
    if positive.size:
        panel.suggest({"zero_replacement": float(f"{float(positive.min()) / 10:.3g}")})
    if np.any(x == 0):
        panel.suggest({"_zero": "stand"})
    panel.ctx.extra["zero_dose"] = bool(np.any(x == 0))


def _doses_finish(s, ctx) -> None:
    units, log, zero = s.pop("_units"), s.pop("_log"), s.pop("_zero")
    s["change_units"] = units != "none"
    if units != "none":
        s["units_mode"] = units
    s["log_transform"] = log != "none"
    if log != "none":
        s["log_base"] = log
    s["replace_zero"] = zero == "stand"
    s["zero_blank"] = zero == "blank"


DOSES = Spec(
    "transform_x_concentration", TransformXOptions,
    [
        Heading("Say what happens to each dose"),
        Sentence(["Each dose is", Pick("_units", [("none", "left in its units"), ("multiply", "multiplied by"),
                                                 ("divide", "divided by")]),
                  Num("units_constant", 1e-12, 1e12, 6, 80, live=lambda s, c: s["_units"] != "none"),
                  ", and then", Pick("_log", [(k, _LOG_WORDS[k]) for k in LOG_BASES] + [("none", "no log is taken")]), "."]),
        Sentence(["A dose of 0", Pick("_zero", [("stand", "is given a stand-in of"), ("blank", "is left blank")]),
                  Num("zero_replacement", 0, 1e12, 12, 110, live=lambda s, c: s["_zero"] == "stand"), "."]),
        Note("This sheet has a dose of 0. Zero has no logarithm, so an untreated control would drop off a log axis; "
             "a stand-in a tenth of the lowest real dose keeps it on the graph.", tone="warn",
             when=lambda s, c: c.extra.get("zero_dose", False)),
        Note("Zero has no logarithm. If a dose of 0 turns up, this is what is done with it.", tone="faint",
             when=lambda s, c: not c.extra.get("zero_dose", False)),
        Custom(lambda p: before_after("transform_x_concentration")),
        _GOES,
    ],
    virtual={"_units": "none", "_log": "log10 (common)", "_zero": "blank"},
    finish=_doses_finish,
    on_sheet=_doses_sheet,
    methods=lambda s, c: ("X values were log-transformed" if s["_log"] != "none" else "X values were rescaled")
    + (" after a stand-in replaced the zero doses" if s["_zero"] == "stand" and c.extra.get("zero_dose") else "") + ".",
)

# ── rescale: pin 0% and 100% ────────────────────────────────────────────────


def _pins(values, options):
    from ...backends.normalize_values import _hundred_ref, _zero_ref

    return float(_zero_ref(values, options)), float(_hundred_ref(values, options))


NORMALIZE = Spec(
    "normalize_values", NormalizeOptions,
    [
        Heading("Pin the two ends of the scale", "Pick the value that becomes 0 and the value that becomes 100."),
        Custom(lambda p: scale_ruler(_pins)),
        Heading("The bottom of the scale is"),
        Choose("zero_definition", [Opt("smallest_mean", "Its lowest value"),
                                   Opt("first_mean", "The lower of its two ends"),
                                   Opt("fixed", "A number I set")], style="segmented"),
        Sentence(["Zero sits at", Num("zero_fixed_value", decimals=6, width=90)],
                 when=lambda s, c: s["zero_definition"] == "fixed"),
        Heading("The top of the scale is"),
        Choose("hundred_definition", [Opt("largest_mean", "Its highest value"),
                                      Opt("last_mean", "The higher of its two ends"),
                                      Opt("sum", "Its total"), Opt("mean", "Its average"),
                                      Opt("fixed", "A number I set")], style="segmented"),
        Sentence(["The top sits at", Num("hundred_fixed_value", decimals=6, width=90)],
                 when=lambda s, c: s["hundred_definition"] == "fixed"),
        Heading("Write the result"),
        Choose("present_as", [Opt("percentages", "From 0 to 100"), Opt("fractions", "From 0 to 1")], style="segmented"),
        Row("Side-by-side repeats", Pick("subcolumn_mode", [("average", "average them first"),
                                                             ("separate", "rescale each repeat column")]),
            when=lambda s, c: repeats(c)),
        Custom(lambda p: before_after("normalize_values")),
        _GOES,
    ],
    methods=lambda s, c: "Each data set was rescaled between the chosen 0% and 100% references.",
)

# ── take a baseline away ────────────────────────────────────────────────────

_CALC = list(CALCULATIONS)   # in the backend's own order
_CALC_CARDS = [("Take it away", "value − baseline", "−"), ("Divide by it", "value ÷ baseline", "÷"),
               ("Percent of it", "100 × value ÷ baseline", "as a % of"),
               ("Percent change", "100 × (value − baseline) ÷ baseline", "% change from"),
               ("Fold change", "(value − baseline) ÷ baseline", "change relative to"),
               ("Add it", "value + baseline", "+"), ("Multiply by it", "value × baseline", "×")]
_ROW_WORDS = {"first": "its own first row", "last": "its own last row"}


def _baseline_lines(s, ctx):
    names = ctx.names or ["Set A", "Set B", "Set C", "Set D"]
    sign = _CALC_CARDS[_CALC.index(s["calculation"])][2] if s["calculation"] in _CALC else "−"
    if s["baseline_source"] == "rows":
        where = _ROW_WORDS.get(s["row_mode"], f"the mean of its first {s['mean_first_n']} and last {s['mean_last_n']} rows")
        return [(name, sign, where) for name in names]
    if s["column_mode"] == "first_column":
        return [(name, sign, names[0]) for name in names[1:]]
    return [(names[i], sign, names[i + 1]) for i in range(0, len(names) - 1, 2)]


BASELINE = Spec(
    "baseline_correction", BaselineOptions,
    [
        Heading("What should each value become?"),
        Choose("calculation", [Opt(key, title, formula) for key, (title, formula, _sign) in zip(_CALC, _CALC_CARDS)],
               columns=4),
        Heading("Which baseline belongs to which values?"),
        Choose("baseline_source", [Opt("columns", "A data set beside it", "A blank or a control measured alongside."),
                                   Opt("rows", "Its own rows", "The start of the trace, the end, or both.")]),
        Choose("column_mode", [Opt("every_other", "In pairs: each data set, then its baseline"),
                               Opt("first_column", "The first data set serves all the others")], style="segmented",
               when=lambda s, c: s["baseline_source"] == "columns"),
        Choose("row_mode", [Opt("first", "The first row"), Opt("last", "The last row"),
                            Opt("mean_ends", "An average of both ends")], style="segmented",
               when=lambda s, c: s["baseline_source"] == "rows"),
        Sentence(["Average the first", Int("mean_first_n", 1, 100000, 50), "rows and the last",
                  Int("mean_last_n", 0, 100000, 50)],
                 when=lambda s, c: s["baseline_source"] == "rows" and s["row_mode"] == "mean_ends"),
        Custom(lambda p: pairing_list(_baseline_lines)),
        Chips([("", [("assume_linear_with_x", "Steady the baseline with a straight line against X",
                      "Fits a line through the baseline values and uses the line, which evens out their noise."),
                     ("remove_baseline_from_results", "Leave the baseline data sets out of the new sheet")])],
              when=lambda s, c: s["baseline_source"] == "columns"),
        Row("Side-by-side repeats", Pick("subcolumn_mode", [("ignore", "average each row's repeats first"),
                                                             ("replicates", "take the baseline's average from each repeat"),
                                                             ("repeated_measures", "match repeat 1 to repeat 1, and so on")]),
            when=lambda s, c: repeats(c)),
        Custom(lambda p: before_after("baseline_correction")),
        _GOES,
    ],
    methods=lambda s, c: "A baseline was applied to each value ("
    + (_CALC_CARDS[_CALC.index(s["calculation"])][1] if s["calculation"] in _CALC else "value − baseline") + ").",
)

# ── thin the rows ───────────────────────────────────────────────────────────


def _rows_sheet(panel) -> None:
    rng = panel.ctx.x_range()
    if rng is not None:
        panel.suggest({"x_min": rng[0], "x_max": rng[1], "threshold": round((rng[0] + rng[1]) / 2, 6)})


ROWS = Spec(
    "remove_rows", RemoveRowsOptions,
    [
        Heading("How should the sheet get shorter?"),
        Choose("mode", [Opt("exclude_range", "Keep a stretch of X", "Rows outside it are dropped."),
                        Opt("average_k", "Merge neighbouring rows", "Thin a dense time course, or pool technical "
                                                                    "repeats entered as rows.")]),
        Sentence(["Keep X from", Num("x_min", decimals=6, width=80), "to", Num("x_max", decimals=6, width=80)],
                 when=lambda s, c: s["mode"] == "exclude_range"),
        Sentence(["Merge every", Int("k_rows", 2, 100000, 50), "rows (in order of X) into their average"],
                 when=lambda s, c: s["mode"] == "average_k"),
        Switch("only_above_threshold", "Only where X is above a value", "Rows below it are kept as they are.",
               when=lambda s, c: s["mode"] == "average_k"),
        Sentence(["X above", Num("threshold", decimals=6, width=80)],
                 when=lambda s, c: s["mode"] == "average_k" and s["only_above_threshold"]),
        Custom(lambda p: before_after("remove_rows")),
        _GOES,
    ],
    on_sheet=_rows_sheet,
    methods=lambda s, c: (f"Rows with X from {s['x_min']:g} to {s['x_max']:g} were kept." if s["mode"] == "exclude_range"
                          else f"Every {s['k_rows']} neighbouring rows were averaged."),
)

# ── turn the table on its side ──────────────────────────────────────────────

TRANSPOSE = Spec(
    "transpose_xy", TransposeOptions,
    [
        Heading("Turn the table on its side", "Each data set becomes a row; each row becomes a data set."),
        Custom(lambda p: before_after("transpose_xy")),
        Heading("Names on the new sheet"),
        Sentence(["Call each new row", Pick("row_titles", [("column_titles", "by the old data set's name"),
                                                           ("letters", "A, B, C…"),
                                                           ("none", "nothing")])]),
        Sentence(["and each new data set", Pick("dataset_titles", [("row_numbers", "by the old row's number"),
                                                                   ("x_values", "by the old row's X value")])]),
        _GOES,
    ],
    methods=lambda s, c: "The table was transposed.",
)


def _all_names(ctx):
    return [(n, n) for n in ["X"] + (ctx.names or ["Set A", "Set B"])]


def _optional_names(none_label: str):
    return lambda ctx: [("", none_label)] + _all_names(ctx)


def _stack_sheet(panel) -> None:
    names = ["X"] + panel.ctx.names
    panel.suggest({"x_variable": names[0], "y_variable": names[1] if len(names) > 1 else names[0]})


STACK = Spec(
    "extract_rearrange", ExtractRearrangeOptions,
    [
        Heading("What kind of table should come out?"),
        Choose("table_type", [Opt("xy", "Curves and regression", "A Bivariate table."),
                              Opt("column", "Comparing groups", "A Column table."),
                              Opt("grouped", "Two factors", "A Grouped table."),
                              Opt("contingency", "Counts", "A Cross-tab of two categories.")], columns=4),
        Heading("Say where each variable goes"),
        Sentence(["Take the values from", Pick("y_variable", _all_names), "and set them against", Pick("x_variable", _all_names)]),
        Sentence(["Start a new data set for every value of", Pick("grouping_variable", _optional_names("nothing - one data set"))]),
        Sentence(["and a new repeat column for every value of",
                  Pick("subject_variable", _optional_names("nothing - one value per cell"))]),
        Sentence(["Name the rows after", Pick("row_title_variable", _optional_names("nothing - number them"))],
                 when=lambda s, c: s["table_type"] in ("column", "grouped")),
        Chips([("", [("include_variable_prefix", "Start each new name with the variable's name"),
                     ("only_complete_rows", "Leave out rows missing any chosen variable")])]),
        Custom(lambda p: before_after("extract_rearrange"), when=lambda s, c: s["table_type"] != "contingency"),
    ],
    on_sheet=_stack_sheet,
    start={"x_variable": "X", "y_variable": ""},
    methods=lambda s, c: f"The data were rearranged into a new {s['table_type']} table.",
)

_OPS = [(">", "is above"), ("<", "is below"), (">=", "is at least"), ("<=", "is at most"), ("=", "equals"),
        ("!=", "is not")]
def _select_finish(s, ctx) -> None:
    s["use_variable_vs_value"] = bool(s["variable_vs_value_criteria"])
    s["use_variable_vs_variable"] = bool(s["variable_vs_variable_criteria"])
    s["row_selection_mode"] = "range" if s.pop("_row_range", False) else "all"
    s["custom_transforms"] = [r for r in s["custom_transforms"] if (r.get("expression") or "").strip()]


PICK = Spec(
    "select_transform", SelectTransformOptions,
    [
        Heading("Keep these variables"),
        Custom(lambda p: variable_chips("included_variables", "", with_x=True)),
        Heading("Keep rows where", "Every rule must hold. No rules keeps every row."),
        Custom(lambda p: rule_list("variable_vs_value_criteria",
                                   [("variable", "pick", _all_names), ("logic", "pick", _OPS), ("value", "num", None)],
                                   add_text="+ Add a rule comparing with a number",
                                   blank={"logic": ">", "value": 0.0})),
        Custom(lambda p: rule_list("variable_vs_variable_criteria",
                                   [("variable1", "pick", _all_names), ("logic", "pick", _OPS),
                                    ("variable2", "pick", _all_names)],
                                   add_text="+ Add a rule comparing two variables", blank={"logic": ">"})),
        Switch("_row_range", "Only rows in a range"),
        Sentence(["Rows", Int("row_from", 1, 10_000_000, 64), "to", Int("row_to", 1, 10_000_000, 64)],
                 when=lambda s, c: s["_row_range"]),
        Heading("Make new variables", "Each starts from a variable on the sheet and passes through the steps you add."),
        Custom(lambda p: new_variables("recipe_variables", _all_names)),
        Custom(lambda p: rule_list("custom_transforms",
                                   ["Make", ("title", "text", "a name"), "from the formula", ("expression", "text", "e.g. A+B or log(B/C)")],
                                   add_text="+ Make a new variable from a formula of several", blank={"expression": ""})),
        Custom(lambda p: before_after("select_transform")),
    ],
    virtual={"_row_range": False},
    finish=_select_finish,
    methods=lambda s, c: "Variables were selected and recoded into a new sheet.",
)

SPECS = [TRANSFORM, DOSES, NORMALIZE, BASELINE, ROWS, TRANSPOSE, STACK, PICK]
