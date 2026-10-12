"""The controls analysis settings are built from.

Settings are decided, not filled in, so the controls say so:

* **Cards** - the main decision, each option with the consequence of
  choosing it, and a "suits your data" mark when the sheet argues for it.
* **Sentences** - numbers set inside the sentence they belong to
  ("X runs from 0 to 20 in 21 points, spaced evenly"), read as written.
* **Chips** - extras to add to the report, grouped by what they are.
* **Switches** - on/off choices with the reason beside them.
* **Rows** - a label and its control, for the few settings that are
  neither.
* **Drawer** - fine-tuning most runs never open.

All styles are scoped to object names (see ui_kit), so nothing cascades.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLayout, QLineEdit, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout,
                               QWidget, QWidgetItem)

from .. import ui_kit as kit

CARD_BG = "#26262d"
CARD_HOVER = "#2c2c34"
CARD_ON = "rgba(74, 144, 217, 40)"


def set_shown(widget: QWidget, on: bool) -> None:
    """setVisible, safe before the widget has a parent. Showing a parentless
    widget opens it as a window of its own for an instant - the flicker when
    a settings window was being built. A widget that will get a parent shows
    with it anyway, so "show" is simply skipped until then."""
    if on and widget.parentWidget() is None:
        return
    widget.setVisible(on)


def _uid(widget: QWidget, stem: str) -> str:
    name = f"{stem}{id(widget)}"
    widget.setObjectName(name)
    return name


# ── layout ─────────────────────────────────────────────────────────────────

class FlowLayout(QLayout):
    """Lays its items out left to right, wrapping like words in a paragraph."""

    def __init__(self, parent=None, h_spacing: int = 6, v_spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._h, self._v = h_spacing, v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, dry=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, dry: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        # items on one line share a baseline: centre them vertically
        rows, row = [], []
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and row:
                rows.append((row, line))
                x, y, line, row = area.x(), y + line + self._v, 0, []
            row.append((item, QPoint(x, y), hint))
            x += hint.width() + self._h
            line = max(line, hint.height())
        if row:
            rows.append((row, line))
        if not dry:
            for items, height in rows:
                for item, point, hint in items:
                    item.setGeometry(QRect(point + QPoint(0, (height - hint.height()) // 2), hint))
        return (y + line - rect.y() + m.bottom()) if rows else 0


# ── section heading ───────────────────────────────────────────────────────

def heading(text: str, note: str = "") -> QWidget:
    """A step's question, with an optional line of help under it."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 14, 0, 4)
    lay.setSpacing(1)
    lay.addWidget(kit.label(text, size=14, weight=600))
    if note:
        lay.addWidget(kit.label(note, size=kit.SM, colour=kit.TEXT_DIM, wrap=True))
    return box


def note(text: str, tone: str = "dim") -> QLabel:
    colour = {"dim": kit.TEXT_DIM, "faint": kit.TEXT_FAINT, "warn": kit.WARN, "accent": kit.ACCENT}[tone]
    return kit.label(text, size=kit.SM, colour=colour, wrap=True)


# ── cards: the main decision ──────────────────────────────────────────────

class Glyph(QWidget):
    """A small drawing on a card: ``draw(painter, rect, colour)``."""

    def __init__(self, draw, parent=None):
        super().__init__(parent)
        self._draw = draw
        self.on = False
        self.setFixedSize(46, 30)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = kit.ACCENT if self.on else kit.TEXT_DIM
        try:
            self._draw(p, QRectF(3, 3, self.width() - 6, self.height() - 6), colour)
        except Exception:  # noqa: BLE001 - a glyph is decoration
            pass
        p.end()


def curve_glyph(fn, x0: float = 0.0, x1: float = 1.0, points: int = 40):
    """A glyph drawing y = fn(x) over [x0, x1], scaled to fill its box."""
    import numpy as np

    xs = np.linspace(x0, x1, points)
    with np.errstate(all="ignore"):
        ys = np.array([fn(x) for x in xs], dtype=float)
    ok = np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]

    def draw(p: QPainter, r: QRectF, colour: str) -> None:
        if ys.size < 2:
            return
        lo, hi = float(ys.min()), float(ys.max())
        span = (hi - lo) or 1.0
        p.setPen(pen(colour, 1.8))
        prev = None
        for x, y in zip(xs, ys):
            pt = QPoint(int(r.left() + (x - x0) / (x1 - x0) * r.width()), int(r.bottom() - (y - lo) / span * r.height()))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

    return draw


class _Card(QFrame):
    clicked = Signal()

    def __init__(self, title: str, blurb: str, parent=None, glyph=None):
        super().__init__(parent)
        self._name = _uid(self, "Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.mark = QLabel("")
        self.mark.setFixedWidth(14)
        self.mark.setStyleSheet(f"color: {kit.ACCENT}; font-weight: 700; background: transparent; border: none;")
        top.addWidget(self.mark)
        self.glyph = Glyph(glyph) if glyph is not None else None
        if self.glyph is not None:
            top.addWidget(self.glyph)
        self.title = kit.label(title, size=kit.MD, weight=600, wrap=True)
        top.addWidget(self.title, 1)
        self.badge = QLabel("")
        self.badge.setStyleSheet(f"color: {kit.OK}; border: 1px solid {kit.OK}; border-radius: 8px; padding: 0 6px;"
                                 " font-size: 10px; font-weight: 600; background: transparent;")
        self.badge.hide()
        top.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        self.blurb = kit.label(blurb, size=kit.SM, colour=kit.TEXT_DIM, wrap=True)
        self.blurb.setContentsMargins(20, 0, 0, 0)
        lay.addWidget(self.blurb)
        if not blurb:  # hide only: showing a label that has no parent yet would flash it up as a window of its own
            self.blurb.hide()
        self._on = False
        self._hover = False
        self._restyle()

    def set_on(self, on: bool) -> None:
        self._on = on
        self.mark.setText("●" if on else "○")
        if self.glyph is not None:
            self.glyph.on = on
            self.glyph.update()
        self._restyle()

    def set_badge(self, text: str) -> None:
        self.badge.setText(text)
        set_shown(self.badge, bool(text))

    def _restyle(self) -> None:
        if self._on:
            rule = f"background: {CARD_ON}; border: 1.5px solid {kit.ACCENT};"
        elif not self.isEnabled():
            rule = f"background: transparent; border: 1px dashed {kit.BORDER};"
        else:
            rule = f"background: {CARD_HOVER if self._hover else CARD_BG}; border: 1px solid {kit.BORDER};"
        self.setStyleSheet(f"#{self._name} {{ {rule} border-radius: 8px; }}")
        self.mark.setStyleSheet(f"color: {kit.ACCENT if self._on else kit.TEXT_FAINT}; font-weight: 700;"
                                " background: transparent; border: none;")

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == event.Type.EnabledChange:
            self._restyle()

    def enterEvent(self, _event) -> None:  # noqa: N802
        self._hover = True
        self._restyle()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = False
        self._restyle()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.clicked.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class Cards(QWidget):
    """One choice from a few, each shown as a card saying what it means."""

    changed = Signal(object)

    def __init__(self, options, columns: int = 0, parent=None):
        """``options``: (value, title, blurb[, glyph]) tuples."""
        super().__init__(parent)
        self._values = [o[0] for o in options]
        self._cards: list[_Card] = []
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        cols = columns or (len(options) if len(options) <= 3 else 2)
        for i, (value, title, blurb, *rest) in enumerate(options):
            card = _Card(title, blurb, glyph=rest[0] if rest else None)
            card.clicked.connect(lambda v=value: self.set_value(v, emit=True))
            grid.addWidget(card, i // cols, i % cols)
            self._cards.append(card)
        for c in range(cols):
            grid.setColumnStretch(c, 1)
        self._value = self._values[0] if self._values else None
        self._paint()

    def value(self):
        return self._value

    def set_value(self, value, emit: bool = False) -> None:
        if value not in self._values:
            return
        changed = value != self._value
        self._value = value
        self._paint()
        if emit and changed:
            self.changed.emit(value)

    def set_badge(self, value, text: str) -> None:
        for v, card in zip(self._values, self._cards):
            if v == value:
                card.set_badge(text)

    def clear_badges(self) -> None:
        for card in self._cards:
            card.set_badge("")

    def set_option_enabled(self, value, on: bool) -> None:
        for v, card in zip(self._values, self._cards):
            if v == value:
                card.setEnabled(on)

    def card(self, value) -> _Card | None:
        return next((c for v, c in zip(self._values, self._cards) if v == value), None)

    def _paint(self) -> None:
        for v, card in zip(self._values, self._cards):
            card.set_on(v == self._value)


# ── segmented: a short choice in a row ─────────────────────────────────────

class Segmented(QWidget):
    """A few short options side by side, one lit - for modes and small choices."""

    changed = Signal(object)

    def __init__(self, options, parent=None):
        """``options``: (value, label) pairs."""
        super().__init__(parent)
        self._name = _uid(self, "Seg")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#{self._name} {{ background: {kit.SUNKEN}; border: 1px solid {kit.BORDER};"
                           f" border-radius: 8px; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._buttons = []
        for value, text in options:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton {{ background: transparent; color: {kit.TEXT_DIM}; border: none; border-radius: 6px;
                               padding: 5px 14px; font-size: {kit.SM}px; }}
                QPushButton:hover {{ color: {kit.TEXT}; }}
                QPushButton:checked {{ background: {kit.ACCENT}; color: white; font-weight: 600; }}
            """)
            bold = QFont(b.font())  # the lit option is bold: leave room, so its text is never clipped
            bold.setPixelSize(kit.SM)
            bold.setBold(True)
            b.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(text) + 30)
            b.clicked.connect(lambda _=False, v=value: self.set_value(v, emit=True))
            lay.addWidget(b)
            self._buttons.append((value, b))
        self._value = options[0][0] if options else None
        self._paint()
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def value(self):
        return self._value

    def set_value(self, value, emit: bool = False) -> None:
        if value not in [v for v, _b in self._buttons]:
            return
        changed = value != self._value
        self._value = value
        self._paint()
        if emit and changed:
            self.changed.emit(value)

    def _paint(self) -> None:
        for v, b in self._buttons:
            b.setChecked(v == self._value)


# ── switch ─────────────────────────────────────────────────────────────────

class Toggle(QAbstractButton):
    """A sliding on/off switch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 19)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(34, 19)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        enabled = self.isEnabled()
        track = QColor(kit.ACCENT if on else kit.BORDER_STRONG)
        if not enabled:
            track.setAlpha(90)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 1, 34, 17), 8.5, 8.5)
        knob = QColor("#ffffff" if enabled else "#8a8a94")
        p.setBrush(knob)
        x = 17 if on else 2
        p.drawEllipse(QRectF(x, 3, 13, 13))
        p.end()


class Switch(QWidget):
    """A switch with its title and, beside it, what turning it on does."""

    toggled = Signal(bool)

    def __init__(self, title: str, blurb: str = "", on: bool = False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(10)
        self.toggle = Toggle()
        self.toggle.setChecked(on)
        lay.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(0)
        self.title = kit.label(title, size=kit.MD, wrap=True)
        text.addWidget(self.title)
        self.blurb = kit.label(blurb, size=kit.SM, colour=kit.TEXT_DIM, wrap=True)
        set_shown(self.blurb, bool(blurb))
        text.addWidget(self.blurb)
        lay.addLayout(text, 1)
        self.toggle.toggled.connect(self.toggled.emit)
        for part in (self.title, self.blurb):
            part.mousePressEvent = lambda _e: self.toggle.toggle() if self.toggle.isEnabled() else None
            part.setCursor(Qt.CursorShape.PointingHandCursor)

    def isChecked(self) -> bool:  # noqa: N802
        return self.toggle.isChecked()

    def setChecked(self, on: bool) -> None:  # noqa: N802
        self.toggle.setChecked(on)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == event.Type.EnabledChange:
            colour = kit.TEXT if self.isEnabled() else kit.TEXT_FAINT
            self.title.setStyleSheet(f"color: {colour}; font-size: {kit.MD}px; background: transparent;"
                                     " border: none;")


# ── a question the report can answer ───────────────────────────────────────

class Question(QFrame):
    """A question as a card: a switch, the question, what answering it gives.
    Switched on, it opens to show the pieces of the answer (as chips) and any
    numbers the answer needs. ``always`` makes it a question every run answers."""

    toggled = Signal(bool)

    def __init__(self, title: str, blurb: str = "", always: bool = False, parent=None):
        super().__init__(parent)
        self._name = _uid(self, "Question")
        self._on, self._styled = always, False
        self.always = always
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 9, 12, 9)
        outer.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(11)
        self.toggle = Toggle()
        self.toggle.setChecked(always)
        self.toggle.setAccessibleName(title)
        if always:
            self.toggle.hide()
            mark = kit.label("ALWAYS", size=9, colour=kit.ACCENT, weight=700)
            mark.setToolTip("Every run answers this.")
            mark.setFixedWidth(46)
            top.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        else:
            top.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignTop)
        words = QVBoxLayout()
        words.setSpacing(1)
        self.title = kit.label(title, size=kit.MD, weight=600, wrap=True)
        self.blurb = kit.label(blurb, size=kit.SM, colour=kit.TEXT_DIM, wrap=True)
        words.addWidget(self.title)
        words.addWidget(self.blurb)
        set_shown(self.blurb, bool(blurb))
        top.addLayout(words, 1)
        outer.addLayout(top)
        self.inside = QWidget()
        self.body = QVBoxLayout(self.inside)
        self.body.setContentsMargins(57, 0, 0, 2)
        self.body.setSpacing(4)
        outer.addWidget(self.inside)
        self.toggle.toggled.connect(self.toggled.emit)
        if not always:
            for part in (self.title, self.blurb):
                part.mousePressEvent = lambda _e: self.toggle.toggle() if self.toggle.isEnabled() else None
                part.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_on(always)

    def set_on(self, on: bool) -> None:
        on = bool(on) or self.always
        if on == self._on and self._styled:      # every refresh of a panel comes through here: restyle only on a change
            return
        self._on, self._styled = on, True
        self.toggle.blockSignals(True)
        self.toggle.setChecked(self._on)
        self.toggle.blockSignals(False)
        self.toggle.update()
        set_shown(self.inside, self._on and self.body.count() > 0)
        edge = kit.ACCENT if self._on else kit.BORDER
        self.setStyleSheet(f"#{self._name} {{ background: {CARD_ON if self._on else CARD_BG}; border: 1px solid {edge};"
                           f" border-radius: 8px; }}")

    def is_on(self) -> bool:
        return self._on


# ── chips ──────────────────────────────────────────────────────────────────

class Chip(QPushButton):
    """An extra to include: tap to add it, tap again to leave it out."""

    def __init__(self, text: str, on: bool = False, tip: str = "", parent=None):
        super().__init__(text, parent)
        self._text = text
        self.setCheckable(True)
        self.setChecked(on)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            self.setToolTip(tip)
        self.setStyleSheet(f"""
            QPushButton {{ background: {CARD_BG}; color: {kit.TEXT_DIM}; border: 1px solid {kit.BORDER};
                           border-radius: 13px; padding: 4px 12px; font-size: {kit.SM}px; }}
            QPushButton:hover {{ color: {kit.TEXT}; border-color: {kit.BORDER_STRONG}; }}
            QPushButton:checked {{ background: {CARD_ON}; color: {kit.TEXT}; border: 1px solid {kit.ACCENT}; }}
            QPushButton:disabled {{ color: {kit.TEXT_FAINT}; border-style: dashed; }}
        """)
        self.toggled.connect(self._label)
        self._label(on)

    def _label(self, on: bool) -> None:
        text = ("✓  " if on else "+  ") + self._text
        if text != self.text():
            self.setText(text)


class ChipGroup(QWidget):
    """A few chips under a small title - one kind of extra."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 4)
        lay.setSpacing(5)
        if title:
            lay.addWidget(kit.label(title.upper(), size=10, colour=kit.TEXT_FAINT, weight=700))
        holder = QWidget()
        self.flow = FlowLayout(holder, 6, 6)
        lay.addWidget(holder)
        self.chips: list[Chip] = []

    def add(self, chip: Chip) -> Chip:
        self.flow.addWidget(chip)
        self.chips.append(chip)
        return chip


# ── inline inputs for sentences ────────────────────────────────────────────

def _inline_ss(kind: str, name: str) -> str:
    return f"""
        {kind}#{name} {{ background: {kit.SUNKEN}; color: {kit.ACCENT}; border: none;
            border-bottom: 1.5px solid {kit.ACCENT}; border-radius: 0; padding: 2px 4px; font-size: {kit.MD}px;
            font-weight: 600; selection-background-color: {kit.ACCENT}; }}
        {kind}#{name}:disabled {{ color: {kit.TEXT_FAINT}; border-bottom-color: {kit.BORDER}; }}
        {kind}#{name}:focus {{ background: #1b2330; }}
    """


class InlineNumber(QDoubleSpinBox):
    def __init__(self, value: float = 0.0, lo: float = -1e12, hi: float = 1e12, decimals: int = 4,
                 width: int = 74, suffix: str = "", parent=None):
        super().__init__(parent)
        name = _uid(self, "InNum")
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.setRange(lo, hi)
        self.setDecimals(decimals)
        self.setValue(value)
        self.setSuffix(suffix)
        self.setFixedWidth(width)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setKeyboardTracking(False)
        self.setStyleSheet(_inline_ss("QDoubleSpinBox", name))

    def textFromValue(self, value: float) -> str:  # noqa: N802 - shortest faithful form, no trailing zeros
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return text if text not in ("-0", "") else "0"


class InlineInt(QSpinBox):
    def __init__(self, value: int = 0, lo: int = 0, hi: int = 1_000_000, width: int = 58, suffix: str = "",
                 parent=None):
        super().__init__(parent)
        name = _uid(self, "InInt")
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.setRange(lo, hi)
        self.setValue(value)
        self.setSuffix(suffix)
        self.setFixedWidth(width)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setKeyboardTracking(False)
        self.setStyleSheet(_inline_ss("QSpinBox", name))


class InlineChoice(QComboBox):
    def __init__(self, options, parent=None):
        """``options``: (value, label) pairs."""
        super().__init__(parent)
        name = _uid(self, "InPick")
        self.set_options(options)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(_inline_ss("QComboBox", name) + f"""
            QComboBox#{name}::drop-down {{ border: none; width: 12px; }}
            QComboBox#{name}::down-arrow {{ image: none; border: none; width: 0; }}
            QComboBox#{name} QAbstractItemView {{ background: {kit.RAISED}; color: {kit.TEXT};
                selection-background-color: {kit.ACCENT}; border: 1px solid {kit.BORDER}; }}
        """)

    def set_options(self, options) -> None:
        current = self.value() if self.count() else None
        self.blockSignals(True)
        self.clear()
        for value, text in options:
            self.addItem(text, value)
        if current is not None:
            self.set_value(current)
        self.blockSignals(False)

    def value(self):
        return self.currentData()

    def set_value(self, value) -> None:
        for i in range(self.count()):
            if self.itemData(i) == value:
                self.setCurrentIndex(i)
                return


class InlineText(QLineEdit):
    def __init__(self, text: str = "", width: int = 140, placeholder: str = "", parent=None):
        super().__init__(text, parent)
        name = _uid(self, "InText")
        self.setFixedWidth(width)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet(_inline_ss("QLineEdit", name))


class Sentence(QWidget):
    """Words and inline inputs, wrapping like a paragraph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, 5, 7)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def words(self, text: str) -> None:
        for word in text.split():
            self.flow.addWidget(kit.label(word, size=kit.MD))

    def add(self, widget: QWidget) -> QWidget:
        self.flow.addWidget(widget)
        return widget


# ── rows and drawer ────────────────────────────────────────────────────────

class Row(QWidget):
    """A label on the left and its control on the right, with an optional hint."""

    LABEL_WIDTH = 190

    def __init__(self, label: str, control: QWidget, hint: str = "", parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(1)
        self.label = kit.label(label, size=kit.MD, colour=kit.TEXT_DIM, wrap=True)
        self.label.setFixedWidth(self.LABEL_WIDTH)
        lay.addWidget(self.label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(control, 0, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.setColumnStretch(1, 1)
        if hint:
            lay.addWidget(kit.label(hint, size=kit.XS, colour=kit.TEXT_FAINT, wrap=True), 1, 1)


def boxed_combo(options) -> InlineChoice:
    """A dropdown for a Row: same data API as InlineChoice, framed like a field."""
    box = InlineChoice(options)
    name = box.objectName()
    box.setStyleSheet(f"""
        QComboBox#{name} {{ background: {kit.SUNKEN}; color: {kit.TEXT}; border: 1px solid {kit.BORDER};
            border-radius: {kit.RADIUS}px; padding: 5px 10px; font-size: {kit.MD}px; min-width: 150px; }}
        QComboBox#{name}:focus {{ border-color: {kit.ACCENT}; }}
        QComboBox#{name}:disabled {{ color: {kit.TEXT_FAINT}; }}
        QComboBox#{name}::drop-down {{ border: none; width: 18px; }}
        QComboBox#{name} QAbstractItemView {{ background: {kit.RAISED}; color: {kit.TEXT};
            selection-background-color: {kit.ACCENT}; border: 1px solid {kit.BORDER}; }}
    """)
    return box


class Drawer(QWidget):
    """Settings folded away under a quiet heading, opened on demand."""

    def __init__(self, title: str = "Fine-tune", summary: str = "", parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 12, 0, 0)
        outer.setSpacing(4)
        self._title = title
        self.head = QPushButton()
        self.head.setCheckable(True)
        self.head.setCursor(Qt.CursorShape.PointingHandCursor)
        self.head.setStyleSheet(f"""
            QPushButton {{ color: {kit.TEXT_DIM}; background: transparent; border: none; border-top: 1px solid {kit.BORDER};
                           text-align: left; padding: 8px 0 4px 0; font-size: {kit.MD}px; font-weight: 600; }}
            QPushButton:hover {{ color: {kit.TEXT}; }}
        """)
        self.head.toggled.connect(self.set_open)
        outer.addWidget(self.head)
        self.summary = kit.label(summary, size=kit.SM, colour=kit.TEXT_FAINT, wrap=True)
        outer.addWidget(self.summary)
        set_shown(self.summary, bool(summary))
        self.panel = QWidget()
        self.body = QVBoxLayout(self.panel)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(4)
        outer.addWidget(self.panel)
        self.set_open(False)

    def set_open(self, open_: bool) -> None:
        self.head.blockSignals(True)
        self.head.setChecked(open_)
        self.head.blockSignals(False)
        self.head.setText(("▾  " if open_ else "▸  ") + self._title)
        set_shown(self.panel, open_)
        set_shown(self.summary, bool(self.summary.text()) and not open_)


def pen(colour: str, width: float = 1.0) -> QPen:
    p = QPen(QColor(colour))
    p.setWidthF(width)
    return p
