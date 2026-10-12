"""Analysis settings, described instead of hand-built.

Each analysis is a ``Spec``: a list of controls (cards, sentences, chips,
switches, rows, a drawer), each bound to a field of that analysis's options
object. ``SettingsPanel`` draws any spec, keeps its state, reads the sheet to
suggest sensible choices, and hands back the options object the analysis
runs with - so the calculation code never needs to know how its settings
look on screen.

Binding rules
-------------
* A control's ``field`` names an options field; its value is read from and
  written to the panel's state dict.
* A card whose value is a dict sets several fields at once (and is shown
  as chosen when all of them match).
* Fields starting with "_" are the panel's own (a range written as
  from/to/points, say); ``Spec.finish`` turns them into options fields.
* ``when(state, ctx)`` hides a control, ``live(state, ctx)`` greys it out.

What a report holds is chosen by question (``Ask``), not by ticking named
statistics: "How well does X tell yes from no?" brings the area under the
ROC curve, its interval and the curve; the pieces can still be picked apart
inside the question. The grouping of statistics under questions is this
program's own.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget

from .. import ui_kit as kit
from . import kit as sk
from . import report_style


# ── what the sheet says ─────────────────────────────────────────────────────

class Context:
    """The sheet the settings are for: its data sets and what they suggest."""

    def __init__(self):
        self.names: list[str] = []
        self.sheet = None
        self.shape = None
        self.hints: dict[str, str] = {}
        self.extra: dict = {}  # anything a spec's on_sheet works out and its controls read
        self.state: dict = {}  # the panel's current settings (set by the panel)

    def set_sheet(self, sheet) -> None:
        self.sheet = sheet
        self.shape = None
        self.hints = {}
        if sheet is None:
            return
        try:
            from ..catalog import SheetShape
            from ..options_dialog_common import data_hints

            self.shape = SheetShape.read(sheet)
            self.names = list(sheet.group_labels())
            self.hints = data_hints(sheet.all_group_arrays())
        except Exception:  # noqa: BLE001 - an unreadable sheet just gives no suggestions
            self.shape = None

    def name(self, index: int) -> str:
        if 0 <= index < len(self.names):
            return self.names[index]
        return f"Set {chr(65 + index % 26)}"

    def column_options(self, extra=()) -> list[tuple[Any, str]]:
        names = self.names or ["Set A", "Set B", "Set C", "Set D"]
        return list(extra) + [(i, n) for i, n in enumerate(names)]

    def x_range(self) -> tuple[float, float] | None:
        x = getattr(self.sheet, "x", None)
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return (float(x.min()), float(x.max())) if x.size >= 2 else None

    def y_range(self) -> tuple[float, float] | None:
        if self.sheet is None:
            return None
        try:
            values = [v[np.isfinite(v)] for v in self.sheet.all_group_arrays().values()]
            values = np.concatenate([v for v in values if v.size]) if values else np.empty(0)
        except Exception:  # noqa: BLE001
            return None
        return (float(values.min()), float(values.max())) if values.size else None

    @property
    def groups(self) -> int:
        return self.shape.groups if self.shape is not None else len(self.names)


# ── controls ────────────────────────────────────────────────────────────────

Cond = Callable[[dict, Context], bool] | None


@dataclass
class Opt:
    value: Any
    title: str
    blurb: str = ""
    suggest: Callable[[Context], bool] | None = None  # marks the card "suits your data"
    phrase: str = ""                                  # how the methods sentence says it
    available: Callable[[Context], bool] | None = None  # greyed out when the sheet can't support it
    glyph: Callable | None = None                      # a small drawing on the card (see kit.curve_glyph)


@dataclass
class Control:
    when: Cond = field(default=None, kw_only=True)
    live: Cond = field(default=None, kw_only=True)

    def make(self, panel: "SettingsPanel") -> QWidget:
        raise NotImplementedError

    def refresh(self, panel: "SettingsPanel", widget: QWidget) -> None:
        pass


@dataclass
class Heading(Control):
    text: str
    note: str = ""

    def make(self, panel):
        return sk.heading(self.text, self.note)


@dataclass
class Note(Control):
    text: str | Callable[[dict, Context], str]
    tone: str = "dim"

    def make(self, panel):
        return sk.note("", self.tone)

    def refresh(self, panel, widget):
        widget.setText(self.text(panel.state, panel.ctx) if callable(self.text) else self.text)


def _matches(value, state: dict, field_name: str | None) -> bool:
    if isinstance(value, dict):
        return all(state.get(k) == v for k, v in value.items())
    return state.get(field_name) == value


def _assign(value, state: dict, field_name: str | None) -> dict:
    return dict(value) if isinstance(value, dict) else {field_name: value}


@dataclass
class Choose(Control):
    """The main decision, as cards (``style="cards"``) or a segmented row."""

    field: str | None
    options: list[Opt]
    style: str = "cards"
    columns: int = 0

    def make(self, panel):
        if self.style == "segmented":
            w = sk.Segmented([(i, o.title) for i, o in enumerate(self.options)])
        else:
            w = sk.Cards([(i, o.title, o.blurb, o.glyph) for i, o in enumerate(self.options)], self.columns)
        w.changed.connect(lambda i: panel.assign(_assign(self.options[i].value, panel.state, self.field)))
        return w

    def refresh(self, panel, widget):
        chosen = next((i for i, o in enumerate(self.options) if _matches(o.value, panel.state, self.field)), None)
        if chosen is not None:
            widget.set_value(chosen)
        if isinstance(widget, sk.Cards):
            widget.clear_badges()
            for i, o in enumerate(self.options):
                if o.available is not None:
                    try:
                        widget.set_option_enabled(i, bool(o.available(panel.ctx)))
                    except Exception:  # noqa: BLE001
                        pass
                if o.suggest is not None:
                    try:
                        if o.suggest(panel.ctx):
                            widget.set_badge(i, "suits your data")
                    except Exception:  # noqa: BLE001
                        pass

    def phrase(self, state) -> str:
        o = next((o for o in self.options if _matches(o.value, state, self.field)), None)
        return (o.phrase or o.title) if o else ""


@dataclass
class Switch(Control):
    field: str
    title: str
    blurb: str = ""

    def make(self, panel):
        w = sk.Switch(self.title, self.blurb)
        w.toggled.connect(lambda on: panel.assign({self.field: on}))
        return w

    def refresh(self, panel, widget):
        widget.toggle.blockSignals(True)
        widget.setChecked(bool(panel.state.get(self.field)))
        widget.toggle.blockSignals(False)
        widget.toggle.update()


@dataclass
class Chips(Control):
    """Extras to add, in titled groups: [(group title, [(field, label, tip)])]."""

    groups: list

    def make(self, panel):
        box = QWidget()
        box.chips = {}      # a plain attribute: the compiled program may not reach into a Qt widget's __dict__
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._chips = {}
        for title, items in self.groups:
            group = sk.ChipGroup(title)
            for item in items:
                fld, text = item[0], item[1]
                tip = item[2] if len(item) > 2 else ""
                chip = group.add(sk.Chip(text, bool(panel.state.get(fld)), tip))
                chip.toggled.connect(lambda on, f=fld: panel.assign({f: on}))
                box.chips[fld] = chip
            lay.addWidget(group)
        return box

    def refresh(self, panel, widget):
        for fld, chip in getattr(widget, "chips", {}).items():
            chip.blockSignals(True)
            chip.setChecked(bool(panel.state.get(fld)))
            chip._label(chip.isChecked())
            chip.blockSignals(False)


# inline parts of a Sentence
@dataclass
class Num:
    field: str
    lo: float = -1e12
    hi: float = 1e12
    decimals: int = 4
    width: int = 74
    suffix: str = ""
    live: Cond = None


@dataclass
class Int:
    field: str
    lo: int = 0
    hi: int = 1_000_000
    width: int = 58
    suffix: str = ""
    live: Cond = None


@dataclass
class Pick:
    field: str
    options: list | Callable[[Context], list]   # (value, label) pairs
    live: Cond = None


@dataclass
class Txt:
    field: str
    width: int = 140
    placeholder: str = ""
    live: Cond = None


def _inline(part, panel) -> QWidget:
    if isinstance(part, Num):
        w = sk.InlineNumber(float(panel.state.get(part.field) or 0.0), part.lo, part.hi, part.decimals, part.width,
                            part.suffix)
        w.valueChanged.connect(lambda v, f=part.field: panel.assign({f: float(v)}))
    elif isinstance(part, Int):
        w = sk.InlineInt(int(panel.state.get(part.field) or 0), part.lo, part.hi, part.width, part.suffix)
        w.valueChanged.connect(lambda v, f=part.field: panel.assign({f: int(v)}))
    elif isinstance(part, Pick):
        w = sk.InlineChoice(part.options(panel.ctx) if callable(part.options) else part.options)
        w.activated.connect(lambda _i, f=part.field, box=w: panel.assign({f: box.value()}))
    elif isinstance(part, Txt):
        w = sk.InlineText(str(panel.state.get(part.field) or ""), part.width, part.placeholder)
        w.textEdited.connect(lambda t, f=part.field: panel.assign({f: t}))
    else:
        raise TypeError(part)
    w._part = part
    return w


def _sync_inline(widget, panel) -> None:
    part = widget._part
    value = panel.state.get(part.field)
    widget.blockSignals(True)
    if isinstance(part, (Num, Int)):
        if value is not None and widget.value() != value:
            widget.setValue(value)
    elif isinstance(part, Pick):
        if callable(part.options):
            widget.set_options(part.options(panel.ctx))
        widget.set_value(value)
    elif isinstance(part, Txt):
        if widget.text() != str(value or ""):
            widget.setText(str(value or ""))
    widget.blockSignals(False)
    if part.live is not None:
        widget.setEnabled(bool(part.live(panel.state, panel.ctx)))


@dataclass
class Sentence(Control):
    """Words with inline inputs: ["X runs from", Num("x0"), "to", Num("x1")]."""

    parts: list

    def make(self, panel):
        w = sk.Sentence()
        w._inputs = []
        for part in self.parts:
            if isinstance(part, str):
                w.words(part)
            else:
                w._inputs.append(w.add(_inline(part, panel)))
        return w

    def refresh(self, panel, widget):
        for inp in widget._inputs:
            _sync_inline(inp, panel)


@dataclass
class Row(Control):
    """A labelled field for settings that are neither a decision nor an extra."""

    label: str
    part: Any       # a Num, Int, Pick or Txt
    hint: str = ""

    def make(self, panel):
        part = self.part
        if isinstance(part, Pick):
            inner = sk.boxed_combo(part.options(panel.ctx) if callable(part.options) else part.options)
            inner.activated.connect(lambda _i, f=part.field, box=inner: panel.assign({f: box.value()}))
            inner._part = part
        else:
            inner = _inline(part, panel)
        row = sk.Row(self.label, inner, self.hint)
        row._inner = inner
        return row

    def refresh(self, panel, widget):
        _sync_inline(widget._inner, panel)


@dataclass
class Checks(Control):
    """Every check an analysis can run, behind one switch. The individual
    checks are chips in a drawer: [(field, label)]."""

    fields: list
    title: str = "Check the assumptions for me"
    blurb: str = "Adds the checks that suit this analysis to the results, each with a plain verdict."

    def make(self, panel):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        box.switch = sk.Switch(self.title, self.blurb)
        box.switch.toggled.connect(lambda on: panel.assign({f: on for f, *_ in self.fields}))
        lay.addWidget(box.switch)
        drawer = sk.Drawer("Choose which checks")
        drawer.layout().setContentsMargins(44, 2, 0, 0)
        drawer.head.setStyleSheet(drawer.head.styleSheet().replace(f"border-top: 1px solid {kit.BORDER};", ""))
        box.chips = Chips([("", self.fields)]).make(panel)
        drawer.body.addWidget(box.chips)
        lay.addWidget(drawer)
        return box

    def refresh(self, panel, widget):
        on = any(panel.state.get(f) for f, *_ in self.fields)
        widget.switch.toggle.blockSignals(True)
        widget.switch.setChecked(on)
        widget.switch.toggle.blockSignals(False)
        widget.switch.toggle.update()
        Chips([("", self.fields)]).refresh(panel, widget.chips)


@dataclass
class Ask(Control):
    """A question the report can answer, in place of a list of statistics to tick.

    ``pieces`` are the parts of the answer - [(field, label, tip)], each an
    on/off field of the options. The question is on while any piece is on.
    Switching it on brings back the pieces it had last time (to begin with,
    those named in ``first``, or all of them); switching it off drops them
    all. ``inside`` holds the controls for the numbers the answer needs,
    shown only while the question is on. ``always`` is for the question
    every run answers: it has no switch."""

    title: str
    blurb: str = ""
    pieces: list = field(default_factory=list)
    inside: list = field(default_factory=list)
    first: list | None = None
    always: bool = False

    def _fields(self) -> list[str]:
        return [piece[0] for piece in self.pieces]

    def is_on(self, state: dict) -> bool:
        return self.always or any(bool(state.get(f)) for f in self._fields())

    def make(self, panel):
        card = sk.Question(self.title, self.blurb, self.always)
        memory = getattr(panel, "_asked", None)
        if memory is None:
            memory = panel._asked = {}

        def switch(on: bool) -> None:
            names = self._fields()
            if on:
                wanted = memory.get(id(self)) or self.first or names
                panel.assign({f: f in wanted for f in names})
            else:
                memory[id(self)] = [f for f in names if panel.state.get(f)]
                panel.assign({f: False for f in names})

        card.toggled.connect(switch)
        card.chips = None
        if len(self.pieces) > 1 or (self.always and self.pieces):
            card.chips = Chips([("", self.pieces)])
            card.chip_box = card.chips.make(panel)
            card.body.addWidget(card.chip_box)
        for control in self.inside:
            if not getattr(control, "_asked_by", None):   # shown only while its question is on
                own, control._asked_by = control.when, self
                control.when = (lambda s, c, own=own, ask=self: ask.is_on(s) and (own is None or bool(own(s, c))))
            card.body.addWidget(panel.place(control))
        return card

    def refresh(self, panel, widget):
        widget.set_on(self.is_on(panel.state))
        if widget.chips is not None:
            widget.chips.refresh(panel, widget.chip_box)


@dataclass
class Drawer(Control):
    title: str
    controls: list
    summary: str = ""

    def make(self, panel):
        w = sk.Drawer(self.title, self.summary)
        for control in self.controls:
            w.body.addWidget(panel.place(control))
        return w


@dataclass
class Custom(Control):
    """A control of its own: ``build(panel) -> QWidget``; the widget may
    define ``refresh(panel)`` to follow the state."""

    build: Callable

    def make(self, panel):
        return self.build(panel)

    def refresh(self, panel, widget):
        if hasattr(widget, "refresh"):
            widget.refresh(panel)


# ── the spec ────────────────────────────────────────────────────────────────

@dataclass
class Spec:
    key: str
    options: type                                    # the options dataclass
    controls: list
    virtual: dict = field(default_factory=dict)      # the panel's own fields, with their starting values
    finish: Callable[[dict, Context], None] | None = None   # panel fields -> options fields
    on_sheet: Callable[["SettingsPanel"], None] | None = None
    methods: Callable[[dict, Context], str] | None = None
    own: set = field(default_factory=set)            # fields the report style must leave alone
    preview: bool = True                             # safe to run as the settings change
    start: dict = field(default_factory=dict)        # starting values that differ from the options' defaults


class SettingsPanel(QDialog):
    """Draws a Spec and turns its state into the analysis's options.

    Keeps the calls the Analyze window and the rest of the app use on any
    settings panel: ``collect()``, ``options``, ``embed()``,
    ``set_variable_names()``, ``set_sheet_data()``.
    """

    changed = Signal()

    def __init__(self, spec: Spec, title: str = "", parent=None, embedded: bool = False):
        """``embedded`` builds the panel as a plain widget from the start, for
        the Analyze window: turning a finished dialog into a widget afterwards
        re-creates its window, which costs as much as a fifth of building it."""
        super().__init__(parent)
        self.spec = spec
        self.ctx = Context()
        self._embedded = bool(embedded)
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.setWindowTitle(title or "Settings")
        self.setObjectName("SettingsPanel")
        self.resize(640, 680)
        defaults = spec.options()
        self._fields = {f.name for f in dataclasses.fields(spec.options)}
        self.state: dict = {f: copy.deepcopy(getattr(defaults, f)) for f in self._fields}
        self.state.update(copy.deepcopy(spec.virtual))
        self.state.update(copy.deepcopy(spec.start))
        self.ctx.state = self.state  # option lists that depend on other settings read it here
        self.touched: set[str] = set()
        self.options = defaults
        self._placed: list[tuple[Control, QWidget]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(8)
        self._chrome = []
        if title:
            head = kit.label(title, size=kit.XL, weight=600)
            root.addWidget(head)
            self._chrome.append(head)

        self.content = QWidget()
        self.content.setObjectName("SettingsContent")
        self.content.setStyleSheet("#SettingsContent { background: transparent; }")
        body = QVBoxLayout(self.content)
        body.setContentsMargins(0, 0, 10, 12)
        body.setSpacing(4)
        for control in spec.controls:
            body.addWidget(self.place(control))
        body.addStretch()

        self.scroll = QScrollArea()
        self.scroll.setObjectName("SettingsScrollArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("#SettingsScrollArea { background: transparent; border: none; }" + kit.SCROLLBAR_SS)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        foot = QHBoxLayout()
        foot.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(kit.ghost_button_ss())
        cancel.clicked.connect(self.reject)
        run = QPushButton("Run")
        run.setStyleSheet(kit.accent_button_ss())
        run.setDefault(True)
        run.clicked.connect(self._run)
        foot.addWidget(cancel)
        foot.addWidget(run)
        self.ok_btn = run
        holder = QWidget()
        holder.setLayout(foot)
        foot.setContentsMargins(0, 0, 0, 0)
        root.addWidget(holder)
        self._chrome.append(holder)

        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(0)
        self._emit_timer.timeout.connect(self.changed.emit)
        # styled last: a child added under a styled parent is polished there and then
        self.setStyleSheet("#SettingsPanel { background: transparent; }" if embedded else f"#SettingsPanel {{ background: {kit.BG}; }}")
        if embedded:
            for widget in self._chrome:
                widget.hide()
            self.layout().setContentsMargins(0, 0, 0, 0)
        self.refresh()

    # ── building ────────────────────────────────────────────────────────────
    def place(self, control: Control) -> QWidget:
        widget = control.make(self)
        self._placed.append((control, widget))
        return widget

    # ── state ───────────────────────────────────────────────────────────────
    def assign(self, values: dict, user: bool = True) -> None:
        """Change fields (from a control, or from a spec's own logic) and redraw."""
        self.state.update(values)
        if user:
            self.touched.update(values)
        self.refresh()
        self._emit_timer.start()

    def suggest(self, values: dict) -> None:
        """Set fields the user hasn't set themselves - defaults read off the sheet."""
        fresh = {k: v for k, v in values.items() if k not in self.touched}
        if fresh:
            self.state.update(fresh)

    def refresh(self) -> None:
        for control, widget in self._placed:
            visible = control.when is None or bool(control.when(self.state, self.ctx))
            sk.set_shown(widget, visible)
            if not visible:
                continue
            if control.live is not None:
                widget.setEnabled(bool(control.live(self.state, self.ctx)))
            try:
                control.refresh(self, widget)
            except Exception:  # noqa: BLE001 - one control's refresh must not break the panel
                pass

    # ── the sheet ───────────────────────────────────────────────────────────
    def set_variable_names(self, names: list[str]) -> None:
        self.ctx.names = [n for n in names if n != "X"]
        self.refresh()

    def set_sheet_data(self, sheet) -> None:
        self.ctx.set_sheet(sheet)
        if self.spec.on_sheet is not None:
            try:
                self.spec.on_sheet(self)
            except Exception:  # noqa: BLE001 - suggestions are a nicety
                pass
        self.refresh()

    # ── out ─────────────────────────────────────────────────────────────────
    def collect(self):
        state = copy.deepcopy(self.state)
        if self.spec.finish is not None:
            self.spec.finish(state, self.ctx)
        report_style.apply(state, self._fields, self.spec.own)
        self.options = dataclasses.replace(self.spec.options(), **{k: v for k, v in state.items() if k in self._fields})
        return self.options

    def methods_text(self) -> str:
        if self.spec.methods is None:
            return ""
        try:
            return self.spec.methods(self.state, self.ctx)
        except Exception:  # noqa: BLE001
            return ""

    def embed(self) -> "SettingsPanel":
        """A plain panel for the Analyze window: no title, no buttons, no own scrolling frame."""
        if self._embedded:
            return self
        self._embedded = True
        self.setWindowFlags(Qt.WindowType.Widget)
        for widget in self._chrome:
            widget.hide()
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("#SettingsPanel { background: transparent; }")
        return self

    def _run(self) -> None:
        self.collect()
        self.accept()
