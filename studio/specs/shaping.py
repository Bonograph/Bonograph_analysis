"""Editors for the analyses that reshape a sheet.

* ``recipe_lane`` - a chain of steps for one axis, as a row of pills:
  "each Y value → log₁₀ → × 100 → + add a step".
* ``before_after`` - the first rows of the sheet as they are now and as the
  current settings would leave them, so every choice is seen at once.
* ``scale_ruler`` - where 0% and 100% land on a data set's own values.
* ``pairing_list`` - which data set is corrected by which baseline.
* ``new_variables`` - new columns, each made from one on the sheet by its own chain of steps.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from ... import recipes
from ... import ui_kit as kit
from ...backend_types import AnalysisError
from .. import kit as sk
from .models import _Rebuilding

PREVIEW_ROWS = 5
PREVIEW_COLUMNS = 3
PREVIEW_MOST_ROWS = 20_000   # a longer sheet is not run on every click


def _number(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "·"
    return f"{value:.4g}"


# ── a recipe, as a lane of pills ─────────────────────────────────────────────

_OPERATOR = {"times": "×", "divide": "÷", "plus": "+", "minus": "−", "power": "to the power"}


def _pill(name: str, dashed: bool = False) -> tuple[QFrame, QHBoxLayout]:
    frame = QFrame()
    frame.setObjectName(name)
    # the starting pill is quieter than a step: no fill, a faint outline
    frame.setStyleSheet(f"#{name} {{ background: {'transparent' if dashed else kit.SUNKEN}; border: 1px solid "
                        f"{kit.BORDER_STRONG}; border-radius: 13px; }}")
    lay = QHBoxLayout(frame)
    lay.setContentsMargins(11, 3, 6 if not dashed else 11, 3)
    lay.setSpacing(5)
    return frame, lay


def _lane(name: str, steps: list, store, head: QWidget, start: QWidget, tail: QWidget | None = None) -> QFrame:
    """A framed chain of step pills. ``steps`` is edited in place and ``store()``
    called after every change; ``head`` sits above the chain, ``start`` opens
    it, ``tail`` (a z-score chip, say) follows the last step."""
    lane = QFrame()
    lane.setObjectName(name)
    lane.setStyleSheet(f"#{name} {{ background: {sk.CARD_BG}; border: 1px solid {kit.BORDER}; border-radius: 8px; }}")
    box = QVBoxLayout(lane)
    box.setContentsMargins(12, 9, 12, 10)
    box.setSpacing(7)
    box.addWidget(head)
    holder = QWidget()
    flow = sk.FlowLayout(holder, 6, 7)
    box.addWidget(holder)
    flow.addWidget(start)
    if not steps:
        flow.addWidget(kit.label("stays as it is", size=kit.SM, colour=kit.TEXT_FAINT))
    for index, step in enumerate(steps):
        flow.addWidget(kit.label("→", size=kit.MD, colour=kit.TEXT_FAINT))
        pill, pill_lay = _pill(f"{name}_step{index}")
        op = step.get("op")
        if op == "equation":
            pill_lay.addWidget(kit.label("=", size=kit.MD, colour=kit.TEXT, weight=600))
            box_ = sk.InlineText(str(step.get("text") or "v"), 150, "v / x")
            box_.setToolTip(recipes.EQUATION_HINT)
            box_.textEdited.connect(lambda text, s=step: (s.__setitem__("text", text), store()))
            pill_lay.addWidget(box_)
        elif recipes.needs_number(op):
            pill_lay.addWidget(kit.label(_OPERATOR.get(op, ""), size=kit.MD, colour=kit.TEXT, weight=600))
            number = sk.InlineNumber(float(step.get("n", 0.0)), decimals=8, width=70)
            number.valueChanged.connect(lambda v, s=step: (s.__setitem__("n", float(v)), store()))
            pill_lay.addWidget(number)
        else:
            pill_lay.addWidget(kit.label(recipes.step_text(step), size=kit.MD, colour=kit.TEXT, weight=600))
        remove = QPushButton("×")
        remove.setFixedSize(18, 18)
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.setToolTip("Take this step out")
        remove.setAccessibleName(f"Remove step {index + 1}")
        remove.setStyleSheet(f"QPushButton {{ background: transparent; color: {kit.TEXT_FAINT}; border: none;"
                             f" font-size: 14px; }} QPushButton:hover {{ color: {kit.WARN}; }}")
        remove.clicked.connect(lambda _=False, i=index: (steps.pop(i), store()))
        pill_lay.addWidget(remove)
        flow.addWidget(pill)

    flow.addWidget(kit.label("→", size=kit.MD, colour=kit.TEXT_FAINT))
    add = QPushButton("+  Add a step")
    add.setCursor(Qt.CursorShape.PointingHandCursor)
    add.setStyleSheet(f"""
        QPushButton {{ background: transparent; color: {kit.ACCENT}; border: 1px dashed {kit.ACCENT};
                       border-radius: 13px; padding: 4px 12px; font-size: {kit.SM}px; font-weight: 600; }}
        QPushButton:hover {{ background: {kit.ACCENT_SOFT}; }}
        QPushButton::menu-indicator {{ image: none; width: 0; }}
    """)
    menu = QMenu(add)
    menu.setStyleSheet(f"""
        QMenu {{ background: {kit.RAISED}; color: {kit.TEXT}; border: 1px solid {kit.BORDER}; padding: 4px; }}
        QMenu::item {{ padding: 5px 22px 5px 14px; border-radius: 4px; }}
        QMenu::item:selected {{ background: {kit.ACCENT}; color: white; }}
        QMenu::item:disabled {{ color: {kit.TEXT_FAINT}; font-size: 10px; font-weight: 700; padding: 7px 14px 2px 10px; }}
        QMenu::separator {{ height: 1px; background: {kit.BORDER}; margin: 4px 6px; }}
    """)
    for heading, keys in recipes.MENU:
        menu.addAction(heading.upper()).setEnabled(False)
        for key in keys:
            menu.addAction(recipes.MENU_TEXT[key]).triggered.connect(
                lambda _=False, k=key: (steps.append(recipes.new_step(k)), store()))
    add.setMenu(menu)
    lane.add_button, lane.menu = add, menu
    flow.addWidget(add)
    if tail is not None:
        flow.addWidget(tail)
    return lane


def recipe_lane(field: str, title: str, start_text: str):
    """The recipe held in ``field`` (a list of step dicts), editable in place."""

    def signature(panel):
        return tuple(step.get("op") for step in (panel.state.get(field) or []))

    def build(panel, lay):
        steps = [dict(step) for step in (panel.state.get(field) or [])]
        name = f"Lane_{field}"
        start, start_lay = _pill(f"{name}_start", dashed=True)
        start_lay.addWidget(kit.label(start_text, size=kit.SM, colour=kit.TEXT_DIM))
        lay.addWidget(_lane(name, steps, lambda: panel.assign({field: [dict(step) for step in steps]}),
                            kit.label(title.upper(), size=10, colour=kit.TEXT_FAINT, weight=700), start))
        if any(step.get("op") == "equation" for step in steps):
            lay.addWidget(sk.note(recipes.EQUATION_HINT, "faint"))
        panel.state[field] = [dict(step) for step in steps]

    return _Rebuilding(build, signature)


def new_variables(field: str, names):
    """New variables, each made from one on the sheet by its own chain of
    steps: ``field`` holds [{"title", "variable", "steps", "standardise"}].
    ``names(ctx)`` lists the variables to start from as (value, label)."""

    def signature(panel):
        rules = panel.state.get(field) or []
        return (tuple(panel.ctx.names), tuple(tuple(s.get("op") for s in (r.get("steps") or [])) for r in rules))

    def build(panel, lay):
        rules = [dict(r, steps=[dict(s) for s in (r.get("steps") or [])]) for r in (panel.state.get(field) or [])]

        def store():
            panel.assign({field: [dict(r, steps=[dict(s) for s in r["steps"]]) for r in rules]})

        choices = names(panel.ctx)
        for index, rule in enumerate(rules):
            if rule.get("variable") is None and choices:
                rule["variable"] = choices[0][0]
            head = QWidget()
            head_lay = QHBoxLayout(head)
            head_lay.setContentsMargins(0, 0, 0, 0)
            head_lay.setSpacing(6)
            head_lay.addWidget(kit.label("Call it", size=kit.SM, colour=kit.TEXT_DIM))
            title = sk.InlineText(str(rule.get("title") or ""), 150, "a name")
            title.textEdited.connect(lambda text, r=rule: (r.__setitem__("title", text), store()))
            head_lay.addWidget(title)
            head_lay.addStretch()
            drop = QPushButton("Remove")
            drop.setCursor(Qt.CursorShape.PointingHandCursor)
            drop.setAccessibleName(f"Remove new variable {index + 1}")
            drop.setStyleSheet(kit.quiet_button_ss())
            drop.clicked.connect(lambda _=False, i=index: (rules.pop(i), store()))
            head_lay.addWidget(drop)
            source = sk.InlineChoice(choices)
            source.set_value(rule.get("variable"))
            source.activated.connect(lambda _i, r=rule, w=source: (r.__setitem__("variable", w.value()), store()))
            chip = sk.Chip("then as a z-score", bool(rule.get("standardise")),
                           "After the steps, take away the mean and divide by the standard deviation of the column.")
            chip.toggled.connect(lambda on, r=rule: (r.__setitem__("standardise", bool(on)), store()))
            lay.addWidget(_lane(f"NewVar_{field}_{index}", rule["steps"], store, head, source, chip))
        add = QPushButton("+ Make a new variable from one on the sheet")
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.setStyleSheet(kit.quiet_button_ss())
        add.clicked.connect(lambda: (rules.append({"title": "", "variable": None, "steps": [{"op": "log10"}],
                                                   "standardise": False}), store()))
        lay.addWidget(add)
        panel.state[field] = [dict(r, steps=[dict(s) for s in r["steps"]]) for r in rules]

    return _Rebuilding(build, signature)


# ── the sheet, before and after ──────────────────────────────────────────────

class _BeforeAfter(QWidget):
    def __init__(self, key: str):
        super().__init__()
        self._key, self._seen = key, None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 12, 0, 2)
        outer.setSpacing(6)
        outer.addWidget(kit.label("YOUR SHEET, BEFORE AND AFTER", size=10, colour=kit.TEXT_FAINT, weight=700))
        row = QHBoxLayout()
        row.setSpacing(10)
        self.before, self.after = self._pane("BeforePane"), self._pane("AfterPane")
        self.arrow = kit.label("→", size=20, colour=kit.ACCENT, weight=600)
        row.addWidget(self.before, 1)
        row.addWidget(self.arrow, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.after, 1)
        outer.addLayout(row)
        self.caption = kit.label("", size=kit.SM, colour=kit.TEXT_FAINT, wrap=True)
        outer.addWidget(self.caption)

    @staticmethod
    def _pane(name: str) -> QLabel:
        pane = QLabel()
        pane.setObjectName(name)
        pane.setTextFormat(Qt.TextFormat.RichText)
        pane.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        pane.setMinimumHeight(64)
        pane.setStyleSheet(f"#{name} {{ background: {kit.SUNKEN}; border: 1px solid {kit.BORDER}; border-radius: 8px;"
                           f" padding: 8px 10px; color: {kit.TEXT_DIM}; font-size: {kit.SM}px; }}")
        return pane

    @staticmethod
    def _table(headers, columns, total_rows: int, total_columns: int) -> str:
        shown = min(PREVIEW_ROWS, max((len(c) for c in columns), default=0))
        cell = "padding: 1px 12px 1px 0;"
        html = ["<table cellspacing='0'><tr>"]
        html += [f"<td style='{cell} color:{kit.TEXT_FAINT}; font-weight:600'>{_escape(h)[:14]}</td>" for h in headers]
        if total_columns > len(headers):
            html.append(f"<td style='{cell} color:{kit.TEXT_FAINT}'>+{total_columns - len(headers)} more</td>")
        html.append("</tr>")
        for r in range(shown):
            html.append("<tr>" + "".join(
                f"<td style='{cell} color:{kit.TEXT}'>{_number(c[r]) if r < len(c) else '·'}</td>" for c in columns) + "</tr>")
        if total_rows > shown:
            html.append(f"<tr><td colspan='{len(headers)}' style='color:{kit.TEXT_FAINT}'>… {total_rows - shown} more rows</td></tr>")
        return "".join(html) + "</table>"

    def refresh(self, panel) -> None:
        sheet = panel.ctx.sheet
        seen = (id(sheet), repr(sorted(panel.state.items(), key=lambda kv: kv[0])))
        if seen == self._seen:
            return
        self._seen = seen
        self.arrow.setVisible(sheet is not None)
        self.after.setVisible(sheet is not None)
        if sheet is None:
            self.before.setText("Once the sheet has data, its first rows show here - as they are, and as these "
                                "settings would leave them.")
            self.caption.setText("")
            return
        labels = list(sheet.group_labels())
        x = np.asarray(sheet.x[:sheet.n_rows], dtype=float)
        has_x = bool(np.any(np.isfinite(x)))
        headers = (["X"] if has_x else []) + labels[:PREVIEW_COLUMNS]
        columns = ([x] if has_x else []) + [sheet.group_rows(label, "average") for label in labels[:PREVIEW_COLUMNS]]
        self.before.setText(self._table(headers, columns, sheet.n_rows, len(labels) + int(has_x)))
        if sheet.n_rows > PREVIEW_MOST_ROWS:
            self.after.setText("This sheet is long, so the result is not worked out as you choose. Run it to see.")
            self.caption.setText("")
            return
        try:
            from ...registry import run_analysis

            options = panel.collect()
            if hasattr(options, "create_new_sheet"):
                options = dataclasses.replace(options, create_new_sheet=True)
            new = run_analysis(self._key, sheet, options).new_sheet
        except AnalysisError as exc:
            self.after.setText(f"<span style='color:{kit.WARN}'>{_escape(str(exc))}</span>")
            self.caption.setText("")
            return
        except Exception:  # noqa: BLE001 - a preview must never break the panel
            self.after.setText("No preview for these settings.")
            self.caption.setText("")
            return
        if new is None:
            self.after.setText("No new sheet with these settings.")
            self.caption.setText("")
            return
        names = list(new.columns)
        new_x = np.asarray(new.x_values, dtype=float) if _numeric(new.x_values) else list(new.x_values)
        show_x = not _numeric(new.x_values) or bool(np.any(np.isfinite(new_x)))
        after_headers = ([new.x_title or "X"] if show_x else []) + names[:PREVIEW_COLUMNS]
        after_columns = ([new_x] if show_x else []) + [new.columns[n] for n in names[:PREVIEW_COLUMNS]]
        rows_after = max([len(new.x_values)] + [len(new.columns[n]) for n in names]) if names else len(new.x_values)
        self.after.setText(self._table(after_headers, after_columns, rows_after, len(names) + int(show_x)))
        parts = []
        if rows_after != sheet.n_rows:
            parts.append(f"{sheet.n_rows} rows become {rows_after}")
        if len(names) != len(labels):
            parts.append(f"{len(labels)} data sets become {len(names)}")
        if rows_after == sheet.n_rows and len(names) == len(labels):
            lost = sum(int(np.sum(np.isnan(np.asarray(new.columns[n], dtype=float)))) for n in names) - sum(
                int(np.sum(np.isnan(sheet.group_rows(label, "average")))) for label in labels)
            if _numeric(new.x_values) and len(new.x_values) == x.size:   # an X with no answer counts as well
                lost += int(np.sum(np.isnan(new_x))) - int(np.sum(np.isnan(x)))
            if lost > 0:
                parts.append(f"{lost} value(s) have no answer and will be blank")
        self.caption.setText(".  ".join(parts) + ("." if parts else ""))


def _numeric(values) -> bool:
    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _escape(text) -> str:
    from html import escape

    return escape(str(text))


def before_after(key: str) -> QWidget:
    """The live preview for the analysis registered as ``key``."""
    return _BeforeAfter(key)


# ── where 0% and 100% land ───────────────────────────────────────────────────

class _Ruler(QWidget):
    """The first data set's values as marks on a line, with flags where the
    current settings put 0% and 100%."""

    def __init__(self, references):
        super().__init__()
        self._references = references
        self._values = np.empty(0)
        self._flags: tuple[float, float] | None = None
        self._name = ""
        self.setFixedHeight(96)
        self.setAccessibleName("Where 0% and 100% fall on the first data set")

    def refresh(self, panel) -> None:
        sheet = panel.ctx.sheet
        self._values, self._flags, self._name = np.empty(0), None, ""
        if sheet is not None and list(sheet.group_labels()):
            self._name = list(sheet.group_labels())[0]
            values = sheet.group_rows(self._name, "average")
            self._values = values[np.isfinite(values)]
            try:
                self._flags = self._references(values, panel.collect())
            except Exception:  # noqa: BLE001
                self._flags = None
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(kit.FONT)
        font.setPixelSize(11)
        p.setFont(font)
        w, mid = self.width(), 42.0
        if self._values.size == 0 or self._flags is None or not all(np.isfinite(self._flags)):
            p.setPen(QColor(kit.TEXT_FAINT))
            p.drawText(QRectF(0, 0, w, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       "With data on the sheet, this line shows where 0% and 100% fall.")
            return
        zero, hundred = self._flags
        lo, hi = min(float(self._values.min()), zero, hundred), max(float(self._values.max()), zero, hundred)
        span = (hi - lo) or 1.0
        left, right = 46.0, w - 46.0

        def at(value: float) -> float:
            return left + (value - lo) / span * (right - left)

        p.setPen(sk.pen(kit.BORDER_STRONG, 2))
        p.drawLine(QPointF(left, mid), QPointF(right, mid))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(160, 160, 173, 170))
        for value in self._values[:400]:
            p.drawEllipse(QPointF(at(float(value)), mid), 3.0, 3.0)
        for value, text in ((zero, "0%"), (hundred, "100%")):
            x = at(value)
            p.setPen(sk.pen(kit.ACCENT, 2))
            p.drawLine(QPointF(x, mid - 16), QPointF(x, mid + 10))
            p.setPen(QColor(kit.ACCENT))
            bold = QFont(font)
            bold.setBold(True)
            p.setFont(bold)
            p.drawText(QRectF(x - 45, mid - 34, 90, 16), Qt.AlignmentFlag.AlignCenter, text)
            p.setFont(font)
            p.setPen(QColor(kit.TEXT_DIM))
            p.drawText(QRectF(x - 45, mid + 12, 90, 16), Qt.AlignmentFlag.AlignCenter, _number(value))
        p.setPen(QColor(kit.TEXT_FAINT))
        p.drawText(QRectF(0, self.height() - 16, w, 16), Qt.AlignmentFlag.AlignLeft, f"Shown for {self._name}; every data "
                   "set is pinned the same way.")


def scale_ruler(references) -> QWidget:
    """``references(values, options) -> (value at 0%, value at 100%)``."""
    return _Ruler(references)


# ── which data set is corrected by which ─────────────────────────────────────

class _Pairs(QLabel):
    """A plain list of what will be worked out, with the sheet's own names."""

    def __init__(self, lines):
        super().__init__()
        self._lines = lines
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setObjectName("PairList")
        self.setStyleSheet(f"#PairList {{ background: {sk.CARD_BG}; border: 1px solid {kit.BORDER}; border-radius: 8px;"
                           f" padding: 9px 12px; color: {kit.TEXT_DIM}; font-size: {kit.SM}px; }}")

    def refresh(self, panel) -> None:
        try:
            lines = self._lines(panel.state, panel.ctx)
        except Exception:  # noqa: BLE001
            lines = []
        shown = [f"<span style='color:{kit.TEXT}'>{_escape(a)}</span> &nbsp;<span style='color:{kit.ACCENT}'>"
                 f"{_escape(sign)}</span>&nbsp; {_escape(b)}" for a, sign, b in lines[:6]]
        if len(lines) > 6:
            shown.append(f"… and {len(lines) - 6} more")
        self.setText("<br>".join(shown) or "Nothing to work out yet.")


def pairing_list(lines) -> QWidget:
    """``lines(state, ctx) -> [(value's name, sign, baseline's name)]``."""
    return _Pairs(lines)
