"""Shared look and boilerplate for every analysis's settings dialog
(analysis/*_options_dialog.py), so each individual dialog file only has
to describe its own fields, not rebuild the window chrome each time.

Styled to match graphs/g.py and its data-selection panels (e.g.
graphs/xy/data_selection.py, graphs/format_panel.py) - the app's actual
UI standard: dark backgrounds (#202020/#3a3a3a), #555 borders, plain
(unstyled) radio buttons/checkboxes that only set text color/size, and a
blue #4a90d9 accent for combos and the primary action button - rather
than a separately invented palette."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# QSpinBox/QDoubleSpinBox's ::up-arrow/::down-arrow stop being drawn by
# the native style the moment any stylesheet rule touches the
# surrounding ::up-button/::down-button (verified: with only
# background/border set on the buttons and no explicit arrow image, both
# arrows render as blank - and a plain CSS border-triangle on the arrow
# subcontrol doesn't get drawn either, nor does a data: URI in `image:
# url(...)`; only a real image file on disk renders). These small PNGs
# are pre-generated (see analysis/assets/_generate_spin_arrows.py) so a
# dialog can be styled without shipping a Qt resource file.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _arrow_url(name: str) -> str:
    return (_ASSETS_DIR / name).as_posix()


from . import ui_kit as _kit

# One set of colours for every analysis window (analysis/ui_kit.py). The
# names are kept because the 47 settings dialogs import them.
BG = _kit.PANEL
PANEL_BG = _kit.RAISED
INPUT_BG = _kit.SUNKEN
BORDER = _kit.BORDER_STRONG
ACCENT = _kit.ACCENT
TEXT = _kit.TEXT
TEXT_DIM = _kit.TEXT_DIM

# graphs/xy/data_selection.py never restyles the radio/checkbox indicator
# itself - only the label's color and font size - so we match that rather
# than drawing custom indicator boxes.
# Left padding is deliberate: with 0 the indicator sits at x=0 and its 1px
# border lands on the widget's own edge, which gets clipped - leaving every
# checkbox/radio visibly shaved off on the left.
CHECK_SS = f"QCheckBox {{ color: {TEXT}; font-size: 13px; spacing: 8px; padding: 2px 0 2px 3px; background: transparent; }}"
RADIO_SS = f"QRadioButton {{ color: {TEXT}; font-size: 13px; spacing: 8px; padding: 3px 0 3px 3px; background: transparent; }}"
SPIN_SS = f"""
    QSpinBox, QDoubleSpinBox {{
        background: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
        border-radius: 4px; padding: 3px 18px 3px 6px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 17px; height: 12px; border-left: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER}; background: #454545;
        border-top-right-radius: 4px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 17px; height: 12px; border-left: 1px solid {BORDER};
        background: #454545; border-bottom-right-radius: 4px;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: #505050;
    }}
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
        background: {ACCENT};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({_arrow_url("spin_up.png")}); width: 9px; height: 5px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({_arrow_url("spin_down.png")}); width: 9px; height: 5px;
    }}
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
        image: url({_arrow_url("spin_up_disabled.png")});
    }}
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
        image: url({_arrow_url("spin_down_disabled.png")});
    }}
"""
# Without an explicit right-side padding + explicit up/down button
# subcontrol geometry, Qt's styled QAbstractSpinBox lets the internal
# line-edit's hit-testable area extend UNDER the up-button (verified via
# QStyle.subControlRect: the plain "background/border/padding" stylesheet
# below made the edit field's rect overlap the up-button's rect but only
# just clip the bottom/down-button's rect) - so clicking "up" landed on
# the line edit (I-beam cursor, no increment) while "down" happened to
# still work. The padding-right + explicit button rects above give both
# buttons their own non-overlapping click area.
# Matches graphs/xy/data_selection.py's _combo_style() exactly.
COMBO_SS = f"""
    QComboBox {{
        background-color: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
        border-radius: 4px; padding: 5px; min-width: 150px;
    }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({_arrow_url("spin_down.png")}); width: 9px; height: 5px; }}
    QComboBox::down-arrow:disabled {{ image: url({_arrow_url("spin_down_disabled.png")}); }}
    QComboBox QAbstractItemView {{
        background-color: {INPUT_BG}; color: {TEXT}; selection-background-color: {ACCENT};
    }}
"""
# Styling ::drop-down (above) makes Qt stop drawing the native arrow, which
# left every combo looking like a plain text box - hence the explicit
# ::down-arrow image, same as SPIN_SS's buttons.


TAB_SS = _kit.tabs_ss() + _kit.SCROLLBAR_SS + f"""
    QTreeWidget, QListWidget {{ background: {_kit.SUNKEN}; color: {_kit.TEXT};
        border: 1px solid {_kit.BORDER}; border-radius: {_kit.RADIUS}px; outline: none;
        show-decoration-selected: 0; }}
    QTreeWidget::item, QListWidget::item {{ padding: 3px 2px; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{ background: {_kit.ACCENT_SOFT}; color: {_kit.TEXT}; }}
    QTableWidget, QTableView {{ background: {_kit.SUNKEN}; color: {_kit.TEXT}; gridline-color: {_kit.BORDER};
        border: 1px solid {_kit.BORDER}; border-radius: {_kit.RADIUS}px; outline: none;
        selection-background-color: {_kit.ACCENT_SOFT}; selection-color: {_kit.TEXT}; }}
    QHeaderView {{ background: {_kit.RAISED}; }}
    QHeaderView::section {{ background: {_kit.RAISED}; color: {_kit.TEXT_DIM}; border: none;
        border-right: 1px solid {_kit.BORDER}; border-bottom: 1px solid {_kit.BORDER}; padding: 4px 6px; }}
    QTableCornerButton::section {{ background: {_kit.RAISED}; border: none; }}
    QLineEdit {{ background: {_kit.SUNKEN}; color: {_kit.TEXT}; border: 1px solid {_kit.BORDER};
        border-radius: 4px; padding: 3px 6px; selection-background-color: {_kit.ACCENT}; }}
    QLineEdit:focus {{ border: 1px solid {_kit.ACCENT}; }}
    QLineEdit:disabled {{ color: {_kit.TEXT_FAINT}; }}
    QScrollArea#TabScroll, QWidget#TabScrollBody {{ background: transparent; border: none; }}
"""


def plan_box(text: str = "") -> tuple[QFrame, QLabel]:
    """The "here's what will actually run" panel some dialogs show.

    Tinted with the app's own accent rather than a warning colour: it is a
    summary of the choices above, not a caution.
    """
    box = QFrame()
    _kit.panel(box, f"planBox{id(box)}", _kit.ACCENT_SOFT, border=f"border: 1px solid {ACCENT};",
               radius=_kit.RADIUS)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(12, 9, 12, 9)
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {TEXT}; border: none; background: transparent; font-size: 13px;")
    layout.addWidget(label)
    return box, label


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: 600; padding-top: 10px;"
                      " background: transparent; border: none;")
    return lbl


def hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {_kit.TEXT_FAINT}; font-size: 12px; padding-left: 26px;"
                      " background: transparent; border: none;")
    return lbl


def check(text: str, checked: bool = False) -> QCheckBox:
    cb = QCheckBox(text)
    cb.setChecked(checked)
    cb.setStyleSheet(CHECK_SS)
    return cb


def radio(text: str, checked: bool = False) -> QRadioButton:
    rb = QRadioButton(text)
    rb.setChecked(checked)
    rb.setStyleSheet(RADIO_SS)
    return rb


def spin(minimum: int, maximum: int, value: int, width: int = 70) -> QSpinBox:
    s = QSpinBox()
    s.setRange(minimum, maximum)
    s.setValue(value)
    s.setStyleSheet(SPIN_SS)
    s.setFixedWidth(width)
    return s


def double_spin(minimum: float, maximum: float, value: float, step: float = 0.1, width: int = 80) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimum, maximum)
    s.setValue(value)
    s.setSingleStep(step)
    s.setStyleSheet(SPIN_SS)
    s.setFixedWidth(width)
    return s


def combo(items: list[str], current: str | None = None) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    if current is not None:
        c.setCurrentText(current)
    c.setStyleSheet(COMBO_SS)
    return c


def keep_distinct(first: QComboBox, second: QComboBox) -> None:
    """Two choices that must differ (method A and method B, control and
    patient values, the negative and the positive outcome): picking in one
    what the other shows swaps them, so the pair can never collide."""
    state = {"first": first.currentIndex(), "second": second.currentIndex()}

    def changed(which: str, box: QComboBox, other: QComboBox, other_key: str) -> None:
        new = box.currentIndex()
        if box.count() > 1 and new == other.currentIndex():
            other.blockSignals(True)
            old = state[which]
            # swap to where this one was - unless it was nowhere (a list being refilled)
            other.setCurrentIndex(old if 0 <= old < other.count() and old != new else (new + 1) % other.count())
            other.blockSignals(False)
            state[other_key] = other.currentIndex()
        state[which] = new

    first.currentIndexChanged.connect(lambda *_: changed("first", first, second, "second"))
    second.currentIndexChanged.connect(lambda *_: changed("second", second, first, "first"))
    # remembered positions follow a refill of the lists (set_variable_names)
    first.model().modelReset.connect(lambda: state.update(first=first.currentIndex()))
    second.model().modelReset.connect(lambda: state.update(second=second.currentIndex()))


# COMBO_SS's 150px min-width suits a form row, but inside a table cell it
# makes the combo wider than its column - it spills into the next cell and
# its own drop-down arrow ends up hidden underneath that neighbour.
CELL_COMBO_SS = COMBO_SS.replace("padding: 5px; min-width: 150px;", "padding: 2px 4px; min-width: 0px;")


def cell_combo(items: list[str], current: str | None = None) -> QComboBox:
    """A combo sized to fit inside a QTableWidget cell (see CELL_COMBO_SS)."""
    c = QComboBox()
    c.addItems(items)
    if current is not None:
        c.setCurrentText(current)
    c.setStyleSheet(CELL_COMBO_SS)
    c.setMinimumWidth(0)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(4)
    return c


def wrapping_label(text: str) -> QLabel:
    """A QLabel that wraps instead of forcing its row wider than the
    dialog - long descriptive labels (e.g. "Show this many significant
    digits (for everything except P values):") next to a spinbox were
    wide enough at our default dialog width to squeeze the spinbox down
    to where its up/down arrows visually overlapped the label text."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    return lbl


def labeled_row(label_text: str, widget) -> QHBoxLayout:
    """A "Label: [widget]" row - the common pattern for a plain settings
    dialog that isn't built from tabs/tables."""
    row = QHBoxLayout()
    label = QLabel(label_text)
    label.setStyleSheet(f"color: {TEXT_DIM};")
    row.addWidget(label)
    row.addWidget(widget)
    row.addStretch()
    return row


def p_value_style_row(default: str | None = None) -> tuple[QHBoxLayout, QComboBox]:
    """The "how should P values be written?" row.

    Every analysis that prints a P value offers the same list, in the same
    order, from analysis/p_values.py - so the choice means the same thing
    wherever it appears.
    """
    from .p_values import BY_KEY, DEFAULT, choices

    row = QHBoxLayout()
    label = QLabel("Write P values as:")
    label.setStyleSheet(f"color: {TEXT}; font-size: 14px;")
    row.addWidget(label)
    box = QComboBox()
    box.setStyleSheet(COMBO_SS)
    box.setMinimumWidth(320)
    for text, key in choices():
        box.addItem(text, key)
        box.setItemData(box.count() - 1, BY_KEY[key].note, Qt.ItemDataRole.ToolTipRole)
    index = box.findData(default or DEFAULT)
    box.setCurrentIndex(index if index >= 0 else 0)
    row.addWidget(box, 1)
    return row, box


def p_style_of(box: QComboBox) -> str:
    from .p_values import DEFAULT

    return box.currentData() or DEFAULT


def confidence_level_row(default: str = "95%") -> tuple[QHBoxLayout, QComboBox]:
    row = QHBoxLayout()
    label = QLabel("Confidence level:")
    label.setStyleSheet(f"color: {TEXT}; font-size: 14px;")
    row.addWidget(label)
    box = combo(["90%", "95%", "99%"], default)
    row.addWidget(box)
    row.addStretch()
    return row, box


def flat_btn_style() -> str:
    return f"""
        QPushButton {{
            background-color: {INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER};
            border-radius: 6px; padding: 6px 16px;
        }}
        QPushButton:hover {{ background-color: #454545; }}
    """


def accent_btn_style() -> str:
    return f"""
        QPushButton {{
            background-color: {ACCENT}; color: white; border: none;
            border-radius: 6px; padding: 6px 20px; font-weight: bold;
        }}
        QPushButton:hover {{ background-color: #5aa0e9; }}
        QPushButton:pressed {{ background-color: #3a80c9; }}
    """


def _gate_row_inputs(layout) -> None:
    """In a row that starts with a choice - "( ) Mean of the first [3] rows",
    "[x] Ignore peaks narrower than [2] points" - the boxes belong to that
    choice: grey them out while it is off, so it is always clear which
    numbers are in play. Each input follows the nearest radio button or
    checkbox to its left in the same row; rows led by a plain label are left
    alone, as are inputs something else already enables."""
    from PySide6.QtWidgets import QAbstractSpinBox, QBoxLayout, QCheckBox, QComboBox, QLineEdit, QRadioButton

    def walk(lay):
        if lay is None:
            return
        horizontal = isinstance(lay, QBoxLayout) and lay.direction() in (QBoxLayout.Direction.LeftToRight,
                                                                           QBoxLayout.Direction.RightToLeft)
        gate = None
        for i in range(lay.count()):
            item = lay.itemAt(i)
            child_layout = item.layout()
            if child_layout is not None:
                walk(child_layout)
                continue
            w = item.widget()
            if w is None:
                continue
            if hasattr(w, "count") and hasattr(w, "widget") and hasattr(w, "tabText"):  # a QTabWidget: every page
                for page_index in range(w.count()):
                    page = w.widget(page_index)
                    walk(page.layout() if page is not None else None)
            elif w.layout() is not None and not isinstance(w, (QAbstractSpinBox, QComboBox)):
                walk(w.layout())
            if not horizontal:
                continue
            if isinstance(w, (QRadioButton, QCheckBox)):
                gate = w
            elif isinstance(w, QLabel) and gate is None:
                return  # a row led by a label: nothing here is a choice's detail
            elif gate is not None and isinstance(w, (QAbstractSpinBox, QComboBox, QLineEdit))                     and not w.property("gated") and w.isEnabled() == True:
                w.setProperty("gated", True)
                w.setEnabled(gate.isChecked() and gate.isEnabled())
                gate.toggled.connect(lambda on, w=w, g=gate: w.setEnabled(on and g.isEnabled()))

    walk(layout)


class OptionsDialogBase(QDialog):
    """Base for an analysis's settings. Subclasses build their own fields
    into ``self.body`` (a QVBoxLayout) in ``_build_body()``, then implement
    ``_collect_options()`` returning the options dataclass instance to run
    the backend with. ``self.options`` holds the last accepted value; it
    starts out as ``default_options`` so a caller can read sensible defaults
    even without opening the dialog.

    Normally these settings live inside the Analyze window, not in a window
    of their own: ``embed()`` strips the window chrome (heading, buttons)
    and turns the dialog into a plain panel the Analyze window hosts. Used
    standalone, it is still a complete dialog.
    """

    BODY_FILLS = False  # True: the body grows with the window (no stretch below it)

    def __init__(self, title: str, default_options, parent=None, width: int = 520, height: int = 520,
                 subtitle: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"Set up: {title}")
        self.setObjectName("SettingsDialog")
        self.setStyleSheet(
            f"#SettingsDialog {{ background-color: {BG}; }}"
            f" QLabel {{ color: {TEXT}; font-size: 13px; background: transparent; }}" + TAB_SS)
        self.resize(width, height)
        self.options = default_options

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(10)
        self._root = root
        self._chrome: list = []  # the heading, rule and buttons - hidden when embedded

        # The analysis being set up, named in the dialog itself - a window
        # title is easy to miss, and these dialogs open on top of each other.
        heading = QLabel(title)
        heading.setWordWrap(True)
        heading.setStyleSheet(f"color: {TEXT}; font-size: 16px; font-weight: 600;")
        root.addWidget(heading)
        self._chrome.append(heading)
        if subtitle:
            note = QLabel(subtitle)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
            root.addWidget(note)
            self._chrome.append(note)
        rule = _kit.rule()
        root.addWidget(rule)
        self._chrome.append(rule)

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        # most settings sit at their natural height; a panel with a list to
        # browse (the equation picker) takes all the room there is instead
        root.addLayout(self.body)
        self._build_body()
        _gate_row_inputs(self._root)
        if self.BODY_FILLS:
            root.setStretchFactor(self.body, 1)
        else:
            root.addStretch()

        bottom = QHBoxLayout()
        self.help_btn = QPushButton("What do these mean?")
        self.help_btn.setStyleSheet(_kit.quiet_button_ss())
        self.help_btn.clicked.connect(self._open_guide)
        bottom.addWidget(self.help_btn)
        bottom.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(_kit.ghost_button_ss())
        self.cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(self.cancel_btn)
        self.ok_btn = QPushButton("Run")
        self.ok_btn.setStyleSheet(_kit.accent_button_ss())
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_accept)
        bottom.addWidget(self.ok_btn)
        self._button_row = QWidget()
        self._button_row.setLayout(bottom)
        bottom.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._button_row)
        self._chrome.append(self._button_row)

    def embed(self) -> "OptionsDialogBase":
        """Turn this dialog into a plain panel for the Analyze window: no
        window frame, no heading (the host shows its own) and no buttons
        (the host's Run collects the settings)."""
        self.setWindowFlags(Qt.WindowType.Widget)
        for widget in self._chrome:
            widget.hide()
        self._root.setContentsMargins(0, 4, 0, 0)
        self.setStyleSheet(
            f"#SettingsDialog {{ background: transparent; }}"
            f" QLabel {{ color: {TEXT}; font-size: 13px; background: transparent; }}" + TAB_SS)
        return self

    def collect(self):
        """The settings as they stand now - what the host runs with."""
        self.options = self._collect_options()
        return self.options

    @staticmethod
    def _open_guide() -> None:
        """Opens the written guide on choosing and reading an analysis."""
        try:
            import help_docs

            help_docs.open_topic("analyses")
        except Exception:  # noqa: BLE001 - help is never worth an error dialog
            pass

    def _build_body(self) -> None:
        raise NotImplementedError

    def _collect_options(self):
        raise NotImplementedError

    def _on_accept(self) -> None:
        self.options = self._collect_options()
        self.accept()

    @staticmethod
    def confidence_value(box: QComboBox) -> float:
        return int(box.currentText().rstrip("%")) / 100.0


# ---------------------------------------------------------------- shared building blocks
# Every analysis lays its tabs out in the same order and under the same
# names, so a setting is always where you last found it.
TAB_DESIGN = "Design"
TAB_FOLLOW_UP = "Follow-up tests"
TAB_CHECKS = "Checks"
TAB_OUTPUT = "Output"


class MoreSection(QWidget):
    """Options most people never need, folded away under "More options".
    Add widgets to ``self.body``."""

    def __init__(self, title: str = "More options", parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(4)
        self._title = title
        self.toggle = QPushButton(f"▸  {title}")
        self.toggle.setCheckable(True)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.setStyleSheet(
            f"QPushButton {{ color: {TEXT_DIM}; background: transparent; border: none; text-align: left;"
            f" font-size: 13px; padding: 4px 0; }} QPushButton:hover {{ color: {TEXT}; }}")
        self.toggle.toggled.connect(self._flip)
        outer.addWidget(self.toggle)
        self.panel = QWidget()
        self.body = QVBoxLayout(self.panel)
        self.body.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.panel)
        self.panel.hide()

    def _flip(self, open_: bool) -> None:
        self.toggle.setText(("▾  " if open_ else "▸  ") + self._title)
        self.panel.setVisible(open_)


class CorrectionChoice:
    """How follow-up comparisons are kept honest - the same words, in the
    same order, in every analysis that makes them. "Best for this design"
    picks Tukey, Dunnett, Games-Howell or Dunn from what was chosen
    elsewhere (see backends/_common.resolve_correction)."""

    TEXT = {
        "auto": ("Best for this design (recommended)",
                 "Tukey for every pair, Dunnett against a control, Games-Howell when the scatter differs, "
                 "Dunn after a rank test."),
        "holm": ("Holm", "Works after any test; a little conservative."),
        "tukey": ("Tukey", "Every pair, when the groups scatter about equally."),
        "dunnett": ("Dunnett", "Each group against one control."),
        "fdr": ("Benjamini-Hochberg", "Controls the false discovery rate - for screening many comparisons."),
        "none": ("No correction", "Each P read on its own; more false positives."),
    }

    def __init__(self, layout: QVBoxLayout, keys=("auto", "holm", "tukey", "dunnett", "fdr", "none"),
                 default: str = "auto", parent=None):
        from PySide6.QtWidgets import QButtonGroup

        layout.addWidget(section_title("Keeping many comparisons honest"))
        self._group = QButtonGroup(parent)
        self.buttons: dict[str, QRadioButton] = {}
        self.hints: dict[str, QLabel] = {}
        for key in keys:
            text, hint = self.TEXT[key]
            rb = radio(text, key == default)
            self._group.addButton(rb)
            self.buttons[key] = rb
            layout.addWidget(rb)
            self.hints[key] = hint_label(hint)
            layout.addWidget(self.hints[key])

    def value(self) -> str:
        return next((k for k, rb in self.buttons.items() if rb.isChecked()), "auto")

    def set_enabled(self, on: bool) -> None:
        for rb in self.buttons.values():
            rb.setEnabled(on)


class OutputSection:
    """The Output tab every analysis shares: confidence level, significant
    digits and how P values are written - always in this order."""

    def __init__(self, layout: QVBoxLayout, confidence: bool = True, digits: bool = True, p_style: bool = True,
                 digits_default: int = 4):
        layout.addWidget(section_title("How results are printed"))
        self.confidence_combo = self.digits_spin = self.p_style_combo = None
        if confidence:
            row, self.confidence_combo = confidence_level_row()
            layout.addLayout(row)
        if digits:
            row = QHBoxLayout()
            label = QLabel("Significant digits:")
            label.setStyleSheet(f"color: {TEXT}; font-size: 14px;")
            row.addWidget(label)
            self.digits_spin = spin(1, 10, digits_default, width=60)
            row.addWidget(self.digits_spin)
            row.addStretch()
            layout.addLayout(row)
        if p_style:
            row, self.p_style_combo = p_value_style_row()
            layout.addLayout(row)

    def confidence(self) -> float:
        return int(self.confidence_combo.currentText().rstrip("%")) / 100.0 if self.confidence_combo else 0.95

    def digits(self) -> int:
        return self.digits_spin.value() if self.digits_spin else 4

    def p_style(self) -> str:
        return p_style_of(self.p_style_combo) if self.p_style_combo else "number"


class SidednessChoice:
    """Two-sided unless a direction was fixed before looking at the data -
    and when one-sided, the direction is named, never implied."""

    def __init__(self, layout: QVBoxLayout, first: str = "A", second: str = "B"):
        layout.addWidget(section_title("Which differences count?"))
        self.combo = combo([
            "Either direction - two-sided (the usual choice)",
            f"Only {second} bigger than {first} - one-sided",
            f"Only {first} bigger than {second} - one-sided",
        ])
        layout.addWidget(self.combo)
        self.warning = hint_label("Use one-sided only if the direction was decided before the data were seen; "
                                  "a result the other way then can't be called significant.")
        self.warning.setStyleSheet(self.warning.styleSheet().replace(_kit.TEXT_FAINT, _kit.WARN))
        layout.addWidget(self.warning)
        self.combo.currentIndexChanged.connect(lambda i: self.warning.setVisible(i != 0))
        self.warning.setVisible(False)

    def value(self) -> str:
        """scipy's alternative for (first - second): 'less' means second is bigger."""
        return ("two-sided", "less", "greater")[self.combo.currentIndex()]


def suggestion_label() -> QLabel:
    """A line of advice drawn from the sheet itself ("these values look
    skewed ..."). Hidden until there is something to say."""
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {ACCENT}; font-size: 12px; padding: 2px 0 2px 26px; background: transparent;"
                      " border: none;")
    lbl.hide()
    return lbl


def data_hints(groups: dict, paired_possible: bool = True) -> dict[str, str]:
    """What the values themselves suggest about the right choices:
    'skewed' when every group is positive and clearly right-skewed, and
    'paired' when the rows of two equal-length groups move together."""
    import numpy as np
    from scipy import stats

    hints: dict[str, str] = {}
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[np.isfinite(a)] for a in arrays]
    usable = [a for a in arrays if a.size >= 5]
    if usable and len(usable) == len(arrays) and all((a > 0).all() for a in usable):
        skews = [float(stats.skew(a)) for a in usable]
        if min(skews) > 0.8 and max(a.max() / a.min() for a in usable) > 5:
            hints["skewed"] = ("These values are all positive and right-skewed (spanning over five-fold) - "
                               "'normal once logged' will probably describe them better.")
    if paired_possible and len(arrays) == 2 and arrays[0].size == arrays[1].size >= 4:
        r = float(np.corrcoef(arrays[0], arrays[1])[0, 1])
        if r > 0.7:
            hints["paired"] = (f"Rows move together (r = {r:.2f}) - if each row is one subject measured twice, "
                               "choose 'the same subjects measured twice'.")
    return hints
