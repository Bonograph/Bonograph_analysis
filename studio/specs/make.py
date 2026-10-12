"""Make data: simulate a curve, groups or a table of counts; draw an
equation; and test a design by running it many times."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...backends.function_library import EQUATIONS, param_help, suggested_x_range
from ...backends.monte_carlo_simulation import MonteCarloOptions
from ...backends.plot_function import PlotFunctionOptions
from ...backends.simulate_column_data import SimulateColumnOptions
from ...backends.simulate_contingency_table import SimulateContingencyOptions
from ...backends.simulate_xy_data import SimulateXYOptions
from .. import kit as sk
from ..spec import Choose, Chips, Custom, Drawer, Heading, Int, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch, Txt
from .models import _Rebuilding, _table_ss

# ── scatter, shared by the simulators ──────────────────────────────────────

_OFFSETS = [0.1, -0.35, 0.3, -0.05, 0.42, -0.25, 0.15, -0.45, 0.05, 0.33, -0.15, 0.22]


def _scatter_glyph(kind: str):
    """Dots about a rising line, scattered the way ``kind`` of noise scatters them."""

    def draw(p, r, colour):
        p.setPen(sk.pen(colour, 1.0))
        p.drawLine(QPointF(r.left(), r.bottom() - 4), QPointF(r.right(), r.top() + 4))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(colour))
        n = len(_OFFSETS)
        for i, off in enumerate(_OFFSETS):
            t = (i + 0.5) / n
            spread = {"none": 0.0, "gaussian_abs": 0.28, "gaussian_relative": 0.1 + 0.5 * t, "t_dist": 0.28,
                      "double_gaussian": 0.12, "poisson": 0.25, "cauchy": 0.2, "binomial": 0.3}[kind]
            dy = off * spread
            if kind == "double_gaussian" and i in (3, 8):
                dy = 0.55 if i == 3 else -0.5
            if kind == "t_dist" and i in (4,):
                dy = 0.45
            if kind == "cauchy" and i in (2, 9):
                dy = 0.95 if i == 2 else -0.9
            base = 1 - t
            y = base + dy
            if kind == "poisson":
                y = round(y * 4) / 4
            if kind == "binomial":
                y = min(1.0, max(0.0, y))
            x = r.left() + t * r.width()
            yy = r.top() + 4 + y * (r.height() - 8)
            p.drawEllipse(QPointF(x, max(r.top(), min(r.bottom(), yy))), 1.6, 1.6)

    return draw


_NOISE = [
    ("none", "None", "Exactly what the model gives."),
    ("gaussian_abs", "Even scatter", "Gaussian, the same SD everywhere - the usual assumption."),
    ("gaussian_relative", "Grows with the value", "Gaussian, SD a set share of each value - typical of assays."),
    ("double_gaussian", "Occasional strays", "Two Gaussians: most tight, a few much wider."),
    ("t_dist", "Heavy-tailed", "A t distribution - more extreme values than Gaussian."),
    ("cauchy", "Very heavy tails", "Cauchy - so heavy it has no SD: a stress test."),
    ("poisson", "Counts", "Each value a Poisson draw around the expected count."),
    ("binomial", "Proportions", "Successes out of N trials."),
]


_NOISE_WORDS = {"none": "none", "gaussian_abs": "Gaussian, constant SD", "gaussian_relative": "Gaussian, SD proportional to the value",
                "double_gaussian": "a mixture of two Gaussians", "t_dist": "t-distributed", "cauchy": "Cauchy",
                "poisson": "Poisson", "binomial": "binomial"}


def noise_controls(what: str = "points") -> list:
    return [
        Heading("Scatter"),
        Choose("error_type", [Opt(k, t, b, glyph=_scatter_glyph(k)) for k, t, b in _NOISE], columns=2),
        Sentence(["SD", Num("error_sd", 0, 1e12, 6, 74)],
                 when=lambda s, c: s["error_type"] in ("gaussian_abs", "t_dist", "cauchy", "double_gaussian")),
        Sentence(["SD of", Num("error_relative_pct", 0, 1000, 2, 58, " %"), "of each value"],
                 when=lambda s, c: s["error_type"] == "gaussian_relative"),
        Sentence([Num("double_gaussian_pct", 0, 100, 1, 54, " %"), f"of {what} use that SD; the rest use",
                  Num("error_sd2", 0, 1e12, 6, 74)], when=lambda s, c: s["error_type"] == "double_gaussian"),
        Sentence(["with", Int("t_df", 1, 10000, 50), "degrees of freedom (30 or more is close to Gaussian)"],
                 when=lambda s, c: s["error_type"] == "t_dist"),
        Sentence(["Out of", Int("binomial_n", 1, 10_000_000, 74), "trials, given as",
                  Pick("binomial_value_type", [("fractions", "fractions"), ("percentages", "percentages"),
                                               ("counts", "counts")])],
                 when=lambda s, c: s["error_type"] == "binomial"),
        Switch("add_outliers", f"Throw a few {what} off", "Shifted by a random amount between two limits."),
        Sentence(["Shift", Int("outlier_count", 0, 100000, 50), what, "by between",
                  Num("outlier_lower", decimals=4, width=70), "and", Num("outlier_upper", decimals=4, width=70)],
                 when=lambda s, c: s["add_outliers"]),
        Drawer("Fine-tune", [
            Row("Random numbers", Pick("seed_mode", [("auto", "new ones every run"), ("fixed", "repeatable, from a seed")])),
            Sentence(["Seed", Int("seed", 0, 2_000_000_000, 96)], when=lambda s, c: s["seed_mode"] == "fixed"),
            Sentence(["Round each value to", Int("significant_digits", 1, 15, 44), "significant digits"]),
        ]),
    ]


_NOISE_START = {"error_type": "gaussian_abs", "outlier_lower": -20.0, "outlier_upper": 20.0, "outlier_count": 2,
                "error_sd2": 5.0, "t_df": 3, "binomial_n": 20, "double_gaussian_pct": 90.0}

# ── equation and its parameters ────────────────────────────────────────────


def _equation_picker(field: str, on_pick=None):
    def build(panel):
        from ...equation_picker import EquationPicker, FunctionCatalog

        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        picker = EquationPicker(panel.state.get(field) or "straight_line", catalog=FunctionCatalog())
        picker.setMinimumHeight(330)
        lay.addWidget(picker)

        def chosen(key):
            if on_pick is not None:
                panel.suggest(on_pick(key))  # e.g. the X range that suits it - unless you have set your own
            panel.assign({field: key})

        picker.chosen.connect(chosen)
        box.refresh = lambda p: None
        return box

    return build


def parameter_grid(key_field: str, count):
    """Parameters down the side, one column per curve, each parameter's
    meaning beside it; the first row names each curve."""

    def signature(panel):
        return (panel.state.get(key_field), count(panel.state))

    def build(panel, lay):
        key, n = signature(panel)
        spec = EQUATIONS.get(key)
        if spec is None:
            return
        sets = list(panel.state.get("parameter_sets") or [])
        titles = list(panel.state.get("column_titles") or [])
        if panel.ctx.extra.get("grid_key") != key:  # a new equation starts from its own defaults
            sets, titles = [], []
            panel.ctx.extra["grid_key"] = key
        while len(sets) < n:
            sets.append({name: default for name, default in spec.params})
        while len(titles) < n:
            titles.append(spec.label if n == 1 else f"{spec.label} {len(titles) + 1}")
        sets, titles = sets[:n], titles[:n]
        panel.state["parameter_sets"], panel.state["column_titles"] = sets, titles

        table = QTableWidget(1 + len(spec.params), n + 1)
        table.setHorizontalHeaderLabels(["Meaning"] + [f"Curve {i + 1}" for i in range(n)])
        table.setVerticalHeaderLabels(["Name"] + [name for name, _d in spec.params])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(_table_ss())
        table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        meaning = QTableWidgetItem("what the curve is called")
        meaning.setFlags(Qt.ItemFlag.ItemIsEnabled)
        table.setItem(0, 0, meaning)
        for r, (name, _default) in enumerate(spec.params, start=1):
            item = QTableWidgetItem(param_help(name) or "")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            table.setItem(r, 0, item)
        for c in range(n):
            table.setItem(0, c + 1, QTableWidgetItem(titles[c]))
            for r, (name, default) in enumerate(spec.params, start=1):
                table.setItem(r, c + 1, QTableWidgetItem(f"{sets[c].get(name, default):g}"))
        table.setMinimumHeight(min(320, 40 + 30 * (1 + len(spec.params))))

        def read(*_args):
            new_sets, new_titles = [], []
            for c in range(n):
                title = table.item(0, c + 1)
                new_titles.append(title.text() if title else "")
                values = {}
                for r, (name, default) in enumerate(spec.params, start=1):
                    item = table.item(r, c + 1)
                    try:
                        values[name] = float(item.text())
                    except (ValueError, AttributeError):
                        values[name] = default
                new_sets.append(values)
            panel.assign({"parameter_sets": new_sets, "column_titles": new_titles})

        table.itemChanged.connect(read)
        lay.addWidget(table)

    return _Rebuilding(build, signature)


# ── a simulated curve ───────────────────────────────────────────────────────


def _x_series(s) -> tuple[list[float], str]:
    lo, hi, n = float(s["_x_from"]), float(s["_x_to"]), max(2, int(s["_x_points"]))
    if s["_x_spacing"] == "log":
        if lo <= 0 or hi <= 0:
            return [], "A log-spaced range needs X above 0 at both ends."
        return list(np.geomspace(lo, hi, n)), ""
    return list(np.linspace(lo, hi, n)), ""


def _x_preview(s, ctx) -> str:
    xs, problem = _x_series(s)
    if problem:
        return problem
    shown = ", ".join(f"{v:.4g}" for v in xs[:4])
    return f"X = {shown}, …, {xs[-1]:.4g}" if len(xs) > 5 else "X = " + ", ".join(f"{v:.4g}" for v in xs)


def _xy_finish(s, ctx) -> None:
    lo, hi, n = float(s.pop("_x_from")), float(s.pop("_x_to")), max(2, int(s.pop("_x_points")))
    spacing = s.pop("_x_spacing")
    s["x_start"] = lo
    s["x_count_mode"] = "count"
    s["x_count"] = n
    if spacing == "log" and lo > 0 and hi > 0:
        s["x_step_mode"], s["x_step_value"] = "times", (hi / lo) ** (1.0 / (n - 1))
    else:
        s["x_step_mode"], s["x_step_value"] = "plus", (hi - lo) / (n - 1)


def _x_range_for(key: str) -> dict:
    lo, hi = suggested_x_range(key)
    return {"_x_from": lo, "_x_to": hi}


CURVE = Spec(
    "simulate_xy_data", SimulateXYOptions,
    [
        Heading("The curve"),
        Custom(_equation_picker("equation_key", on_pick=_x_range_for)),
        Sentence(["Draw", Int("n_datasets", 1, 50, 44), "curve(s), each point repeated",
                  Int("replicates_per_dataset", 1, 100, 44), "time(s)"]),
        Custom(lambda p: parameter_grid("equation_key", lambda s: max(1, int(s["n_datasets"])))),
        Heading("Along X"),
        Choose("use_sheet_x", [Opt(False, "A new range"), Opt(True, "This sheet's X values")], style="segmented"),
        Sentence(["From", Num("_x_from", decimals=6, width=74), "to", Num("_x_to", decimals=6, width=74), "in",
                  Int("_x_points", 2, 100000, 58), "points, spaced",
                  Pick("_x_spacing", [("even", "evenly"), ("log", "evenly on a log scale")])],
                 when=lambda s, c: not s["use_sheet_x"]),
        Note(_x_preview, tone="faint", when=lambda s, c: not s["use_sheet_x"]),
        *noise_controls("points"),
        Drawer("Reuse this sheet", [Switch("use_sheet_y", "Keep this sheet's own Y values and only add scatter",
                                           "Nothing from the equation is used.")]),
    ],
    virtual={"_x_from": 0.0, "_x_to": 20.0, "_x_points": 21, "_x_spacing": "even"},
    finish=_xy_finish,
    own={"significant_digits"},
    start=dict(_NOISE_START, equation_key="michaelis_menten"),
    methods=lambda s, c: (f"Data were simulated from {EQUATIONS[s['equation_key']].equation if s['equation_key'] in EQUATIONS else s['equation_key']} "
                          f"({s['n_datasets']} curve{'s' if s['n_datasets'] != 1 else ''}; noise: {_NOISE_WORDS[s['error_type']]})."),
)

# ── simulated groups ────────────────────────────────────────────────────────


def group_table():
    def signature(panel):
        return (int(panel.state.get("n_datasets") or 1), panel.state.get("mean_mode"))

    def build(panel, lay):
        n, mode = signature(panel)
        rows = list(panel.state.get("dataset_rows") or [])
        means = list(panel.state.get("dataset_means") or [])
        titles = list(panel.state.get("dataset_titles") or [])
        while len(rows) < n:
            rows.append(rows[-1] if rows else 5)
        while len(means) < n:
            means.append(float(len(means)))
        while len(titles) < n:
            titles.append("")
        rows, means, titles = rows[:n], means[:n], titles[:n]
        panel.state.update(dataset_rows=rows, dataset_means=means, dataset_titles=titles)
        individual = mode == "individual"
        table = QTableWidget(n, 3 if individual else 2)
        table.setHorizontalHeaderLabels(["Name", "Values"] + (["True mean"] if individual else []))
        table.setVerticalHeaderLabels([chr(65 + i % 26) for i in range(n)])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(_table_ss())
        table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        for r in range(n):
            table.setItem(r, 0, QTableWidgetItem(titles[r]))
            table.setItem(r, 1, QTableWidgetItem(str(int(rows[r]))))
            if individual:
                table.setItem(r, 2, QTableWidgetItem(f"{means[r]:g}"))
        table.setMinimumHeight(min(280, 40 + 30 * n))

        def read(*_args):
            new_rows, new_means, new_titles = [], [], []
            for r in range(n):
                new_titles.append(table.item(r, 0).text().strip() if table.item(r, 0) else "")
                try:
                    new_rows.append(max(1, int(float(table.item(r, 1).text()))))
                except (ValueError, AttributeError):
                    new_rows.append(5)
                if individual:
                    try:
                        new_means.append(float(table.item(r, 2).text()))
                    except (ValueError, AttributeError):
                        new_means.append(0.0)
                else:
                    new_means.append(means[r])
            panel.assign({"dataset_rows": new_rows, "dataset_means": new_means, "dataset_titles": new_titles})

        table.itemChanged.connect(read)
        lay.addWidget(table)
        lay.addWidget(sk.note("Leave a name blank to use the group's letter. The gaps between the true means are "
                              "the effects a test should find.", "faint"))

    return _Rebuilding(build, signature)


GROUPS = Spec(
    "simulate_column_data", SimulateColumnOptions,
    [
        Heading("The groups"),
        Sentence(["Make", Int("n_datasets", 1, 52, 44), "groups"]),
        Choose("mean_mode", [Opt("individual", "I set each true mean", "Build in the effect you want to detect."),
                             Opt("random_gaussian", "Draw the means at random", "From one normal population.")]),
        Sentence(["Means drawn from a normal distribution with mean", Num("population_mean", decimals=6, width=70),
                  "and SD", Num("population_sd", 0, 1e12, 6, 70)], when=lambda s, c: s["mean_mode"] == "random_gaussian"),
        Custom(lambda p: group_table()),
        *noise_controls("values"),
    ],
    own={"significant_digits"},
    start=dict(_NOISE_START, n_datasets=3, dataset_rows=[10, 10, 10], dataset_means=[10.0, 12.0, 15.0],
               dataset_titles=["", "", ""]),
    methods=lambda s, c: f"{s['n_datasets']} groups were simulated (noise: {_NOISE_WORDS[s['error_type']]}).",
)

# ── simulated counts ────────────────────────────────────────────────────────


def _counts_finish(s, ctx) -> None:
    s["p_exposed"] = s.pop("_pct_exposed") / 100.0
    s["p_outcome_row1"] = s.pop("_pct_row1") / 100.0
    s["p_outcome_row2"] = s.pop("_pct_row2") / 100.0
    for key, fallback in (("row1_title", "Row 1"), ("row2_title", "Row 2"), ("col_a_title", "Column A"),
                          ("col_b_title", "Column B")):
        s[key] = (s.get(key) or "").strip() or fallback


COUNTS = Spec(
    "simulate_contingency_table", SimulateContingencyOptions,
    [
        Heading("The study", "Subjects are sampled at random, whatever their exposure or outcome."),
        Sentence(["Sample", Int("sample_size", 1, 100_000_000, 74), "subjects;", Num("_pct_exposed", 0, 100, 1, 54, " %"),
                  "are", Txt("row1_title", 110, "Exposed"), "and the rest", Txt("row2_title", 110, "Not exposed"), "."]),
        Sentence(["Among the first,", Num("_pct_row1", 0, 100, 1, 54, " %"), "have", Txt("col_a_title", 120, "Bad outcome"),
                  "; among the rest,", Num("_pct_row2", 0, 100, 1, 54, " %"), "do. Everyone else has",
                  Txt("col_b_title", 120, "Good outcome"), "."]),
        Drawer("Fine-tune", [
            Row("Random numbers", Pick("seed_mode", [("auto", "new ones every run"), ("fixed", "repeatable, from a seed")])),
            Sentence(["Seed", Int("seed", 0, 2_000_000_000, 96)], when=lambda s, c: s["seed_mode"] == "fixed"),
        ]),
    ],
    virtual={"_pct_exposed": 50.0, "_pct_row1": 30.0, "_pct_row2": 10.0},
    finish=_counts_finish,
    methods=lambda s, c: f"A 2×2 table of {s['sample_size']} subjects was simulated from the stated proportions.",
)

# ── draw an equation ────────────────────────────────────────────────────────


def _draw_finish(s, ctx) -> None:
    curves = max(1, int(s.pop("_curves", 1)))
    s["plot_family"] = curves > 1
    s["family_count"] = curves
    s["n_points"] = max(2, int(s.pop("_steps", 150)) + 1)


def _param_names(state):
    spec = EQUATIONS.get(state.get("equation_key"))
    return [(n, n) for n, _d in spec.params] if spec else [("", "—")]


def _range_for(key: str) -> dict:
    lo, hi = suggested_x_range(key)
    return {"x_min": lo, "x_max": hi}


DRAW = Spec(
    "plot_function", PlotFunctionOptions,
    [
        Heading("The equation"),
        Custom(_equation_picker("equation_key", on_pick=_range_for)),
        Sentence(["Draw", Int("_curves", 1, 50, 44), "curve(s) from X =", Num("x_min", decimals=6, width=74), "to",
                  Num("x_max", decimals=6, width=74)]),
        Custom(lambda p: parameter_grid("equation_key", lambda s: max(1, int(s["_curves"])))),
        Heading("Also draw"),
        Chips([("", [("plot_function", "The equation itself"), ("plot_first_derivative", "Its slope"),
                     ("plot_second_derivative", "Its curvature"), ("plot_integral", "The area built up under it")])]),
        Sentence(["Counting area from X =", Num("integral_x0", decimals=6, width=70), ", where it is",
                  Num("integral_y0", decimals=6, width=70)], when=lambda s, c: s["plot_integral"]),
        Drawer("Fine-tune", [
            Sentence(["Draw each curve from", Int("_steps", 1, 100000, 64), "straight pieces"]),
            Row("Name the curves", Pick("column_label_mode", [("manual", "as typed in the grid"),
                                                              ("parameter_value", "by their value of a parameter")])),
            Row("Parameter", Pick("column_label_parameter", lambda c: _param_names(c.state)),
                when=lambda s, c: s["column_label_mode"] == "parameter_value"),
        ]),
    ],
    virtual={"_curves": 1, "_steps": 150},
    finish=_draw_finish,
    own={"significant_digits"},
    start={"equation_key": "michaelis_menten", **_range_for("michaelis_menten")},
    methods=lambda s, c: f"The equation {EQUATIONS[s['equation_key']].equation if s['equation_key'] in EQUATIONS else ''} was drawn from X = {s['x_min']:g} to {s['x_max']:g}.",
)



# ── a design, run many times ────────────────────────────────────────────────

POWER = Spec(
    "monte_carlo_simulation", MonteCarloOptions,
    [
        Heading("The experiment"),
        Sentence(["Two groups of", Int("n_per_group", 2, 1_000_000, 64), "subjects, a true difference of",
                  Num("effect_size", decimals=6, width=70), "and an SD of", Num("sd", 0, 1e12, 6, 70), "in each."]),
        Heading("Run it"),
        Sentence(["Repeat it", Int("n_simulations", 10, 10_000_000, 80), "times and count how often the t test finds "
                  "the difference."]),
        Note("How often it does is the design's power. Significance uses the threshold set under Report style.", "faint"),
        Drawer("Fine-tune", [Sentence(["Seed", Int("seed", 0, 2_000_000_000, 96), "(0 = new numbers every run)"])]),
    ],
    methods=lambda s, c: (f"Power was estimated by simulating {s['n_simulations']} experiments with {s['n_per_group']} "
                          f"subjects per group, a true difference of {s['effect_size']:g} and SD {s['sd']:g}."),
)

SPECS = [CURVE, GROUPS, COUNTS, DRAW, POWER]
