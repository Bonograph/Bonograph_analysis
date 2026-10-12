"""The Analyze window: say what you want to find out, set it up, and see the
answer taking shape before you run it.

Three panes, read left to right:

    goals and their tools  |  the chosen tool's settings  |  a live preview

* **Left** - one list. Each goal ("Compare groups", "Fit a curve"...) opens
  to show its tools; "For this sheet" on top holds the ones that suit the
  table you are on. A tool the sheet can't feed says what is missing.
* **Middle** - the tool's settings (analysis/studio). Tools that come in
  several forms (two groups / three or more) switch form at the top; the
  form that suits the sheet is chosen for you.
* **Right** - the result on your own data, recalculated as you change the
  settings, and a methods sentence ready to paste into a paper.

How results are written - confidence level, digits, P format, the
significance threshold - is set once, in the footer, for every analysis.

After ``exec()`` returns Accepted the caller reads ``selected_analysis`` (the
registry key), ``selected_category`` (the goal's title), ``options`` and
``selected_data_sets``.
"""

from __future__ import annotations

import copy
import time

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from analysis import catalog, icons, registry

_INPUT_EVENTS = (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel)
from analysis import ui_kit as kit
from analysis.studio import kit as sk
from analysis.studio import report_style

SUGGESTED_ID = "suggested"
RECENT_ID = "recent"
_MAX_RECENTLY_USED = 6
# Session-lifetime history of (goal title, registry key), most recent first.
_recently_used: list[tuple[str, str]] = []
# Session-lifetime window size, so the window reopens the way it was left.
_last_size: list = [QSize(1320, 760)]
# Sheets with more values than this are previewed on request, not on every change.
PREVIEW_LIMIT = 40_000

_ROLE_KIND = Qt.ItemDataRole.UserRole       # "goal" or "tool"
_ROLE_ID = Qt.ItemDataRole.UserRole + 1     # the goal's or the tool's id
_ROLE_KEY = Qt.ItemDataRole.UserRole + 2    # a tool row: the registry key it would run
_ROLE_READY = Qt.ItemDataRole.UserRole + 3  # True / False / None (nothing to judge against)
_ROLE_DETAIL = Qt.ItemDataRole.UserRole + 4
_ROLE_GLYPH = Qt.ItemDataRole.UserRole + 5
_ROLE_OPEN = Qt.ItemDataRole.UserRole + 6
_ROLE_COUNT = Qt.ItemDataRole.UserRole + 7


def _record_recently_used(category: str, name: str) -> None:
    entry = (category, name)
    if entry in _recently_used:
        _recently_used.remove(entry)
    _recently_used.insert(0, entry)
    del _recently_used[_MAX_RECENTLY_USED:]


class _NavDelegate(QStyledItemDelegate):
    """Paints the one list: goal rows (glyph, title, count, open or shut) and,
    under an open goal, its tools (a status dot, the title, one line on what
    it answers - or what the sheet is missing)."""

    def sizeHint(self, option, index):
        return QSize(0, 36 if index.data(_ROLE_KIND) == "goal" else 50)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        goal = index.data(_ROLE_KIND) == "goal"
        rect = option.rect.adjusted(6, 2, -6, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected) and not goal
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(74, 144, 217, 46) if selected else QColor(kit.HOVER))
            painter.drawRoundedRect(QRectF(rect), 6, 6)
        font = QFont(option.font)
        if goal:
            is_open = bool(index.data(_ROLE_OPEN))
            tone = kit.ACCENT if is_open else (kit.TEXT if hovered else kit.TEXT_DIM)
            glyph = index.data(_ROLE_GLYPH)
            if glyph:
                painter.drawPixmap(QRect(rect.left() + 10, rect.center().y() - 9, 18, 18),
                                   icons.pixmap(glyph, tone, 18, max(2.0, painter.device().devicePixelRatioF())))
            font.setPixelSize(kit.MD)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.setPen(QColor(kit.TEXT if is_open else "#cfcfd8"))
            painter.drawText(QRect(rect.left() + 38, rect.top(), rect.width() - 76, rect.height()),
                             int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), index.data() or "")
            font.setPixelSize(kit.XS)
            font.setWeight(QFont.Weight.Normal)
            painter.setFont(font)
            painter.setPen(QColor(kit.TEXT_FAINT))
            painter.drawText(QRect(rect.right() - 44, rect.top(), 36, rect.height()),
                             int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                             f"{index.data(_ROLE_COUNT) or ''}  {'▾' if is_open else '▸'}")
            painter.restore()
            return

        if selected:
            painter.fillRect(QRectF(rect.left(), rect.top() + 9, 3, rect.height() - 18), QColor(kit.ACCENT))
        ready = index.data(_ROLE_READY)
        dot = QColor(kit.OK) if ready else QColor(kit.BORDER_STRONG) if ready is None else QColor(kit.WARN)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(QRectF(rect.left() + 22, rect.top() + 13, 6, 6))
        font.setPixelSize(kit.MD)
        font.setWeight(QFont.Weight.DemiBold if selected else QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor(kit.TEXT if ready is not False else kit.TEXT_DIM))
        text = QRect(rect.left() + 38, rect.top() + 5, rect.width() - 46, 18)
        painter.drawText(text, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         painter.fontMetrics().elidedText(index.data() or "", Qt.TextElideMode.ElideRight, text.width()))
        font.setPixelSize(kit.XS)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor(kit.WARN if ready is False else kit.TEXT_FAINT))
        sub = QRect(text.left(), text.bottom() + 1, text.width(), 16)
        painter.drawText(sub, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         painter.fontMetrics().elidedText(index.data(_ROLE_DETAIL) or "", Qt.TextElideMode.ElideRight,
                                                          sub.width()))
        painter.restore()


def _list_ss() -> str:
    return (f"QListWidget {{ background: transparent; border: none; outline: none; }}"
            f" QListWidget::item {{ border: none; }}" + kit.SCROLLBAR_SS)


class ReportStyleDialog(QDialog):
    """How every result is written - set once, used by every analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from analysis import p_values

        self.setWindowTitle("Report style")
        self.setObjectName("ReportStyle")
        self.setStyleSheet(f"#ReportStyle {{ background: {kit.BG}; }}")
        style = report_style.current()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(kit.label("Report style", size=kit.XL, weight=600))
        lay.addWidget(kit.label("Used by every analysis, so results read the same way throughout.", size=kit.SM,
                                colour=kit.TEXT_DIM, wrap=True))
        self.confidence = sk.boxed_combo([(c, f"{round(c * 100)}%") for c in report_style.CONFIDENCES])
        self.confidence.set_value(style.confidence)
        self.digits = QSpinBox()
        self.digits.setRange(2, 10)
        self.digits.setValue(style.digits)
        self.digits.setStyleSheet(f"QSpinBox {{ background: {kit.SUNKEN}; color: {kit.TEXT}; border: 1px solid {kit.BORDER};"
                                  f" border-radius: {kit.RADIUS}px; padding: 4px 8px; min-width: 60px; }}")
        self.p_style = sk.boxed_combo([(key, text) for text, key in p_values.choices()])
        self.p_style.set_value(style.p_style)
        self.alpha = sk.boxed_combo([(a, f"{a:g}") for a in report_style.ALPHAS])
        self.alpha.set_value(style.alpha)
        for text, widget, hint in (("Confidence intervals at", self.confidence, ""),
                                   ("Significant digits", self.digits, ""),
                                   ("Write P values as", self.p_style, ""),
                                   ("Call a result significant below P =", self.alpha,
                                    "Also used for stars and verdicts.")):
            lay.addWidget(sk.Row(text, widget, hint))
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(kit.ghost_button_ss())
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Use this style")
        ok.setStyleSheet(kit.accent_button_ss())
        ok.setDefault(True)
        ok.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        lay.addSpacing(8)
        lay.addLayout(buttons)

    def _save(self) -> None:
        report_style.save(report_style.Style(confidence=self.confidence.value() or 0.95, digits=self.digits.value(),
                                             p_style=self.p_style.value() or "number",
                                             alpha=self.alpha.value() or 0.05))
        self.accept()


class AnalyzeDataDialog(QDialog):
    analysisChosen = Signal(str, str)  # goal title, registry key

    def __init__(self, table_names: list[str] | None = None, sheet_data_lookup=None, parent=None,
                 table_kind: str | None = None, initial: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Analyze")
        self.setObjectName("AnalyzeWindow")
        self.setStyleSheet(f"#AnalyzeWindow {{ background: {kit.BG}; }}")
        self.resize(_last_size[0])
        self.setMinimumSize(1080, 580)

        self.selected_category: str | None = None
        self.selected_analysis: str | None = None
        self.options = None
        self._sheet_data_lookup = sheet_data_lookup
        self._table_kind = table_kind
        self._sheet = None
        self._shape: catalog.SheetShape | None = None
        self._settings: dict[str, object] = {}        # key -> embedded settings panel
        self._settings_sheet: dict[str, str] = {}     # key -> the sheet it was prepared for
        self._variant: dict[str, str] = {}            # tool id -> the form chosen for it
        self._tool: catalog.Tool | None = None
        self._open_goal: str = ""
        self._dataset_labels: list[str] = []
        self._dataset_checks: list[sk.Chip] = []
        self._current_settings = None
        self._preview_job = None
        self._preview_token = 0
        self._closing = False
        self._last_input = 0.0

        self._build_ui(table_names or ["Sheet 1"])
        self._refresh_sheet(self.table_combo.currentText())
        self._open_goal = SUGGESTED_ID if catalog.suggested_for(self._table_kind) else catalog.GROUPS[0].id
        self._fill_nav()
        if initial:
            self.select_analysis(initial)
        self.search_edit.setFocus()

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build_ui(self, table_names: list[str]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- top bar ----
        top = kit.panel(QFrame(), "AnalyzeTop", kit.RAISED, border=f"border-bottom: 1px solid {kit.BORDER};")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(16, 12, 16, 12)
        top_row.setSpacing(10)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("What do you want to find out?   Try “compare groups”, “EC50”, "
                                            "“is it normal” or a test's name")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(kit.input_ss())
        self.search_edit.textChanged.connect(self._on_search)
        self.search_edit.installEventFilter(self)
        top_row.addWidget(self.search_edit, 1)
        top_row.addSpacing(8)
        top_row.addWidget(kit.label("On", size=kit.SM, colour=kit.TEXT_DIM))
        self.table_combo = QComboBox()
        self.table_combo.addItems(table_names)
        self.table_combo.setMinimumWidth(150)
        self.table_combo.setStyleSheet(self._combo_ss())
        self.table_combo.currentTextChanged.connect(self._on_sheet_changed)
        top_row.addWidget(self.table_combo)
        self.shape_label = kit.label("", size=kit.SM, colour=kit.TEXT_FAINT)
        self.shape_label.setMinimumWidth(190)
        top_row.addWidget(self.shape_label)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        # ---- left: goals and tools ----
        left = kit.panel(QFrame(), "AnalyzeNav", kit.RAISED, border=f"border-right: 1px solid {kit.BORDER};")
        left.setFixedWidth(290)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 10, 4, 8)
        left_layout.setSpacing(4)
        self.list_heading = kit.label("", size=kit.XS, colour=kit.TEXT_FAINT, weight=600)
        self.list_heading.setContentsMargins(14, 0, 0, 2)
        self.list_heading.hide()
        left_layout.addWidget(self.list_heading)
        self.nav = QListWidget()
        self.nav.setItemDelegate(_NavDelegate(self.nav))
        self.nav.setStyleSheet(_list_ss())
        self.nav.setMouseTracking(True)
        self.nav.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.itemClicked.connect(self._on_nav_clicked)
        self.nav.currentItemChanged.connect(self._on_nav_current)
        self.nav.itemDoubleClicked.connect(self._on_nav_double)
        left_layout.addWidget(self.nav, 1)
        self.list_empty = kit.label("", size=kit.SM, colour=kit.TEXT_DIM, wrap=True)
        self.list_empty.setContentsMargins(14, 4, 14, 4)
        self.list_empty.hide()
        left_layout.addWidget(self.list_empty)
        body.addWidget(left)

        # ---- middle: the chosen tool ----
        settings = kit.panel(QFrame(), "AnalyzeSettings", kit.PANEL)
        settings.setMinimumWidth(500)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(22, 16, 16, 10)
        settings_layout.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(10)
        self.settings_title = kit.label("", size=kit.XL, weight=600)
        head.addWidget(self.settings_title)
        self.fit_pill = kit.pill("")
        self.fit_pill.hide()
        head.addWidget(self.fit_pill, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch()
        self.mode_holder = QHBoxLayout()
        self.mode_holder.setContentsMargins(0, 0, 0, 0)
        head.addLayout(self.mode_holder)
        self.mode_switch = None
        settings_layout.addLayout(head)
        self.settings_methods = kit.label("", size=kit.MD, colour=kit.TEXT_DIM, wrap=True)
        settings_layout.addWidget(self.settings_methods)
        self.settings_detail = kit.label("", size=kit.SM, colour=kit.ACCENT, wrap=True)  # the named tests
        settings_layout.addWidget(self.settings_detail)
        self.settings_note = kit.label("", size=kit.SM, colour=kit.WARN, wrap=True)
        self.settings_note.hide()
        settings_layout.addWidget(self.settings_note)
        settings_layout.addWidget(kit.rule())

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setObjectName("SettingsScroll")
        self.settings_scroll.setStyleSheet("#SettingsScroll { background: transparent; border: none; }"
                                           + kit.SCROLLBAR_SS)
        self.settings_scroll.viewport().setAutoFillBackground(False)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Every tool's panel lives in this one holder from the moment it is built, and
        # switching tools only hides one and shows another. (Moving a panel in and out
        # of the scroll area made Qt restyle all of its widgets on every switch.)
        self._settings_host = QWidget()
        self._settings_host.setObjectName("SettingsHost")
        self._settings_host.setStyleSheet("#SettingsHost { background: transparent; }")
        self._settings_host_layout = QVBoxLayout(self._settings_host)
        self._settings_host_layout.setContentsMargins(0, 0, 0, 0)
        self._settings_host_layout.setSpacing(0)
        self.settings_scroll.setWidget(self._settings_host)
        self._shown_panel = None
        settings_layout.addWidget(self.settings_scroll, 1)
        self.no_settings = kit.label("Nothing to set up — it runs as it is.", size=kit.MD, colour=kit.TEXT_DIM)
        self.no_settings.hide()
        settings_layout.addWidget(self.no_settings)

        # the data sets to include - only offered when there is a choice to make
        self.data_strip = kit.panel(QFrame(), "DataStrip", kit.PANEL, border=f"border-top: 1px solid {kit.BORDER};")
        strip_layout = QVBoxLayout(self.data_strip)
        strip_layout.setContentsMargins(0, 8, 0, 2)
        strip_layout.setSpacing(4)
        strip_head = QHBoxLayout()
        strip_head.addWidget(kit.label("DATA SETS TO INCLUDE", size=10, colour=kit.TEXT_FAINT, weight=700))
        strip_head.addStretch()
        self.select_all_btn = QPushButton("All")
        self.deselect_all_btn = QPushButton("None")
        for button in (self.select_all_btn, self.deselect_all_btn):
            button.setStyleSheet(kit.quiet_button_ss())
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            strip_head.addWidget(button)
        self.select_all_btn.clicked.connect(lambda: self._set_all_dataset_checks(True))
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_dataset_checks(False))
        strip_layout.addLayout(strip_head)
        self.checks_scroll = QScrollArea()
        self.checks_scroll.setWidgetResizable(True)
        self.checks_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.checks_scroll.setObjectName("ChecksScroll")
        self.checks_scroll.setStyleSheet("#ChecksScroll { background: transparent; border: none; }" + kit.SCROLLBAR_SS)
        holder = QWidget()
        holder.setObjectName("ChecksHolder")
        holder.setStyleSheet("#ChecksHolder { background: transparent; }")
        self.checks_layout = sk.FlowLayout(holder, 6, 6)
        self.checks_scroll.setWidget(holder)
        strip_layout.addWidget(self.checks_scroll)
        settings_layout.addWidget(self.data_strip)

        self.notice_panel = kit.panel(QFrame(), "EmptySheet", kit.PANEL, border=f"border-top: 1px solid {kit.BORDER};")
        notice_layout = QVBoxLayout(self.notice_panel)
        notice_layout.setContentsMargins(0, 10, 0, 2)
        self.notice_label = kit.label("", size=kit.SM, colour=kit.TEXT_DIM, wrap=True)
        notice_layout.addWidget(self.notice_label)
        settings_layout.addWidget(self.notice_panel)
        body.addWidget(settings, 1)

        # ---- right: the result, previewed ----
        preview = kit.panel(QFrame(), "AnalyzePreview", kit.RAISED, border=f"border-left: 1px solid {kit.BORDER};")
        preview.setFixedWidth(340)
        pv = QVBoxLayout(preview)
        pv.setContentsMargins(16, 14, 16, 12)
        pv.setSpacing(6)
        pv_head = QHBoxLayout()
        pv_head.addWidget(kit.label("PREVIEW", size=10, colour=kit.TEXT_FAINT, weight=700))
        self.preview_status = kit.label("", size=kit.XS, colour=kit.TEXT_FAINT)
        pv_head.addWidget(self.preview_status)
        pv_head.addStretch()
        self.preview_now_btn = QPushButton("Preview now")
        self.preview_now_btn.setStyleSheet(kit.quiet_button_ss())
        self.preview_now_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview_now_btn.clicked.connect(lambda: self._run_preview(force=True))
        self.preview_now_btn.hide()
        pv_head.addWidget(self.preview_now_btn)
        pv.addLayout(pv_head)
        self.preview_card = kit.panel(QFrame(), "PreviewCard", kit.SUNKEN, border=f"border: 1px solid {kit.BORDER};",
                                      radius=8)
        card = QVBoxLayout(self.preview_card)
        card.setContentsMargins(12, 10, 12, 10)
        self.preview_headline = kit.label("", size=kit.MD, wrap=True)
        self.preview_headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.addWidget(self.preview_headline)
        pv.addWidget(self.preview_card)
        self.preview_chart = QLabel()
        self.preview_chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_chart.hide()
        pv.addWidget(self.preview_chart)
        self.preview_table = kit.label("", size=kit.XS, colour=kit.TEXT_DIM, wrap=True)
        self.preview_table.setTextFormat(Qt.TextFormat.RichText)
        self.preview_table.setAlignment(Qt.AlignmentFlag.AlignTop)
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setFrameShape(QFrame.Shape.NoFrame)
        table_scroll.setObjectName("PreviewScroll")
        table_scroll.setStyleSheet("#PreviewScroll { background: transparent; border: none; }" + kit.SCROLLBAR_SS)
        table_scroll.viewport().setAutoFillBackground(False)
        table_scroll.setWidget(self.preview_table)
        pv.addWidget(table_scroll, 1)

        pv.addWidget(kit.rule())
        m_head = QHBoxLayout()
        m_head.addWidget(kit.label("METHODS, READY TO PASTE", size=10, colour=kit.TEXT_FAINT, weight=700))
        m_head.addStretch()
        self.copy_methods_btn = QPushButton("Copy")
        self.copy_methods_btn.setStyleSheet(kit.quiet_button_ss())
        self.copy_methods_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_methods_btn.clicked.connect(self._copy_methods)
        m_head.addWidget(self.copy_methods_btn)
        pv.addLayout(m_head)
        self.methods_label = kit.label("", size=kit.SM, colour=kit.TEXT_DIM, wrap=True, italic=True)
        self.methods_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.methods_label.setMinimumHeight(54)
        self.methods_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        pv.addWidget(self.methods_label)
        body.addWidget(preview)

        # ---- footer ----
        foot = kit.panel(QFrame(), "AnalyzeFoot", kit.RAISED, border=f"border-top: 1px solid {kit.BORDER};")
        foot_row = QHBoxLayout(foot)
        foot_row.setContentsMargins(16, 10, 16, 10)
        foot_row.setSpacing(8)
        self.style_btn = QPushButton("")
        self.style_btn.setStyleSheet(kit.quiet_button_ss())
        self.style_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.style_btn.setToolTip("How every result is written. Click to change.")
        self.style_btn.clicked.connect(self._edit_report_style)
        foot_row.addWidget(self.style_btn)
        foot_row.addStretch()
        self.help_btn = QPushButton("Which test do I need?")
        self.help_btn.setStyleSheet(kit.quiet_button_ss())
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.clicked.connect(self._open_guide)
        foot_row.addWidget(self.help_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(kit.ghost_button_ss())
        self.cancel_btn.clicked.connect(self.reject)
        foot_row.addWidget(self.cancel_btn)
        self.ok_btn = QPushButton("Run")
        self.ok_btn.setStyleSheet(kit.accent_button_ss())
        self.ok_btn.setMinimumWidth(96)
        self.ok_btn.setEnabled(False)
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_accept)
        foot_row.addWidget(self.ok_btn)
        root.addWidget(foot)

        QShortcut(QKeySequence.StandardKey.Find, self, activated=self._focus_search)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._run_preview)
        self._show_style()
        self._show_tool(None)

    @staticmethod
    def _combo_ss() -> str:
        from .options_dialog_common import _arrow_url

        return f"""
            QComboBox {{ background: {kit.SUNKEN}; color: {kit.TEXT}; border: 1px solid {kit.BORDER};
                         border-radius: {kit.RADIUS}px; padding: 6px 10px; font-size: {kit.MD}px; }}
            QComboBox:focus {{ border: 1px solid {kit.ACCENT}; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox::down-arrow {{ image: url({_arrow_url("spin_down.png")}); width: 9px; height: 5px; }}
            QComboBox QAbstractItemView {{ background: {kit.RAISED}; color: {kit.TEXT};
                                           selection-background-color: {kit.ACCENT}; border: 1px solid {kit.BORDER}; }}
        """

    # ------------------------------------------------------------------
    # the sheet
    # ------------------------------------------------------------------
    def _on_sheet_changed(self, table_name: str) -> None:
        self._refresh_sheet(table_name)
        self._fill_nav()

    def _refresh_sheet(self, table_name: str) -> None:
        """Read the sheet, describe it, and rebuild the data-set choices."""
        self._sheet, self._shape = None, None
        labels: list[str] = []
        if self._sheet_data_lookup is not None and table_name:
            try:
                self._sheet = self._sheet_data_lookup(table_name)
                labels = list(self._sheet.group_labels())
                self._shape = catalog.SheetShape.read(self._sheet)
            except Exception:  # noqa: BLE001 - an unreadable sheet just isn't checked against
                self._sheet, self._shape, labels = None, None, []

        shape = self._shape
        if shape is None or not shape.groups:
            self.shape_label.setText("Empty" if self._sheet_data_lookup else "")
        else:
            parts = [f"{shape.groups} data set{'s' if shape.groups != 1 else ''}",
                     f"{shape.rows} row{'s' if shape.rows != 1 else ''}"]
            if shape.replicates > 1:
                parts.append(f"{shape.replicates} repeats")
            self.shape_label.setText("  ·  ".join(parts))

        while self.checks_layout.count():
            item = self.checks_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
                item.widget().deleteLater()
        self._dataset_checks = []
        self._dataset_labels = labels
        for label in labels:
            chip = sk.Chip(label, True)
            chip.toggled.connect(self._on_datasets_changed)
            self.checks_layout.addWidget(chip)
            self._dataset_checks.append(chip)

        choose = len(labels) >= 2
        self.checks_scroll.setFixedHeight(34 if len(labels) <= 6 else 66)
        self._has_choice_of_sets = choose
        self.data_strip.setVisible(choose)
        self.select_all_btn.setEnabled(choose)
        self.deselect_all_btn.setEnabled(choose)
        empty = self._sheet_data_lookup is not None and not labels
        self.notice_panel.setVisible(empty)
        self.notice_label.setText(
            f"'{table_name}' doesn't have any data yet. Type or paste values into it, or use one of the "
            "tools under Make data." if table_name else "Choose a sheet to run on.")
        self._settings_sheet.clear()  # settings that read the sheet are prepared again for the new one
        self._update_ok_enabled()

    @property
    def _empty_sheet(self) -> bool:
        return self._sheet_data_lookup is not None and (self._shape is None or not self._shape.groups)

    def _fit(self, analysis) -> tuple[bool | None, str]:
        """(fits?, why) - None when there is nothing to judge against."""
        need = catalog.requirement(analysis)
        if need.no_data:
            return True, "Makes its own data"
        if self._shape is None or not self._shape.groups:
            return None, ""
        return catalog.verdict(analysis, self._shape)

    # ------------------------------------------------------------------
    # the list of goals and tools
    # ------------------------------------------------------------------
    def _key_for(self, tool: catalog.Tool, hint: str | None = None) -> str:
        """The form of ``tool`` to show: the one asked for, the one chosen
        before, or the one that suits the sheet (preferring a form that fits)."""
        if hint in tool.keys:
            return hint
        if tool.id in self._variant:
            return self._variant[tool.id]
        picked = tool.pick(self._shape)
        if self._fit(catalog.BY_KEY[picked])[0] is False:
            for key in tool.keys:
                if self._fit(catalog.BY_KEY[key])[0]:
                    return key
        return picked

    def _goals(self) -> list[tuple[str, str, str, str, list[tuple[catalog.Tool, str | None]]]]:
        """(id, title, tip, glyph, [(tool, the key to prefer)]) for every goal on show."""
        out = []
        suggested = self._tools_for([a.key for a in catalog.suggested_for(self._table_kind)])
        if suggested:
            out.append((SUGGESTED_ID, "For this sheet", "Good places to start on this kind of table", "suggested",
                        suggested))
        recent = self._tools_for([key for _cat, key in _recently_used if key in catalog.BY_KEY])
        if recent:
            out.append((RECENT_ID, "Recent", "What you ran last", "recent", recent))
        for group in catalog.GROUPS:
            out.append((group.id, group.title, group.question, group.glyph, [(t, None) for t in group.tools]))
        return out

    @staticmethod
    def _tools_for(keys: list[str]) -> list[tuple[catalog.Tool, str | None]]:
        seen, out = set(), []
        for key in keys:
            tool = catalog.TOOL_OF.get(key)
            if tool is not None and tool.id not in seen:
                seen.add(tool.id)
                out.append((tool, key))
        return out

    def _add_tool_row(self, tool: catalog.Tool, hint: str | None) -> None:
        key = self._key_for(tool, hint)
        ready, why = self._fit(catalog.BY_KEY[key])
        item = QListWidgetItem(tool.title)
        item.setData(_ROLE_KIND, "tool")
        item.setData(_ROLE_ID, tool.id)
        item.setData(_ROLE_KEY, key)
        item.setData(_ROLE_READY, ready)
        item.setData(_ROLE_DETAIL, why.split(";")[0] if ready is False else tool.question)
        item.setToolTip(f"{tool.question}\n{tool.methods}")
        self.nav.addItem(item)

    def _fill_nav(self) -> None:
        text = self.search_edit.text().strip()
        keep_tool = self._tool.id if self._tool is not None else None
        self.nav.blockSignals(True)
        self.nav.clear()
        if text:
            found = self._tools_for([a.key for a in catalog.search(text)])
            for tool, key in found:
                self._add_tool_row(tool, key)
            self.list_heading.setText(f"{len(found)} MATCH{'' if len(found) == 1 else 'ES'}" if found else "NO MATCHES")
            self.list_heading.show()
            self.list_empty.setVisible(not found)
            self.list_empty.setText("Nothing matches. Try a plainer word — “difference”, “curve”, “counts”.")
        else:
            self.list_heading.hide()
            self.list_empty.hide()
            for ident, title, tip, glyph, tools in self._goals():
                head = QListWidgetItem(title)
                head.setData(_ROLE_KIND, "goal")
                head.setData(_ROLE_ID, ident)
                head.setData(_ROLE_GLYPH, glyph)
                head.setData(_ROLE_OPEN, ident == self._open_goal)
                head.setData(_ROLE_COUNT, len(tools))
                head.setToolTip(tip)
                head.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.nav.addItem(head)
                if ident == self._open_goal:
                    # the tools this sheet can run first; the rest after, saying what is missing
                    ordered = sorted(tools, key=lambda t: self._fit(catalog.BY_KEY[self._key_for(*t)])[0] is False)
                    for tool, key in ordered:
                        self._add_tool_row(tool, key)
        self.nav.blockSignals(False)

        rows = [i for i in range(self.nav.count()) if self.nav.item(i).data(_ROLE_KIND) == "tool"]
        # build the panels on show in idle moments, so clicking one is instant
        self._prebuild_queue = [self.nav.item(i).data(_ROLE_KEY) for i in rows]
        self._prebuild_queue += [k for i in rows for k in catalog.TOOLS[self.nav.item(i).data(_ROLE_ID)].keys
                                 if k not in self._prebuild_queue]  # the other forms of those tools, after them
        self._prebuild_queue += [k for k in catalog.TOOL_OF if k not in self._prebuild_queue]  # then every other tool
        if not getattr(self, "_prebuild_running", False):
            self._prebuild_running = True
            QTimer.singleShot(60, self._prebuild_next)
        row = next((i for i in rows if self.nav.item(i).data(_ROLE_ID) == keep_tool), rows[0] if rows else -1)
        if row >= 0:
            self.nav.blockSignals(True)  # shown once, by the call below - not a second time by the list's signal
            self.nav.setCurrentRow(row)
            self.nav.blockSignals(False)
            self._on_nav_current(self.nav.item(row))
        else:
            self._show_tool(None)

    PREBUILD_QUIET = 0.35  # seconds without a key, click or mouse move before building the next panel

    def _prebuild_next(self) -> None:
        """Build one waiting settings panel, then yield to the event loop."""
        queue = getattr(self, "_prebuild_queue", [])
        while queue and queue[0] in self._settings:
            queue.pop(0)
        if not queue or not self.isVisible() or self._closing:
            self._prebuild_running = False
            return
        if self._user_busy():
            QTimer.singleShot(150, self._prebuild_next)  # the user is busy: building now would make the window stutter
            return
        try:
            self._settings_for(queue.pop(0))
        except Exception:  # noqa: BLE001 - it will be built (or explained) when clicked
            pass
        QTimer.singleShot(15, self._prebuild_next)

    def _touched(self, *_args) -> None:
        self._last_input = time.monotonic()

    def _user_busy(self) -> bool:
        """Has the user moved, clicked, typed or scrolled within the last moment?

        Asked when a panel is about to be built in the background. The pointer
        and buttons are looked at here, rather than every event of the whole
        program being passed through a filter: building a panel raises
        thousands of events, and filtering them cost more than it saved."""
        from PySide6.QtGui import QCursor

        where = QCursor.pos()
        if where != getattr(self, "_last_pointer", None) or QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
            self._last_pointer = where
            self._last_input = time.monotonic()
        return time.monotonic() - self._last_input < self.PREBUILD_QUIET

    def _on_search(self, _text: str) -> None:
        self._fill_nav()

    def _on_nav_clicked(self, item) -> None:
        if item.data(_ROLE_KIND) == "goal":
            self._open_goal = "" if self._open_goal == item.data(_ROLE_ID) else item.data(_ROLE_ID)
            self._fill_nav()

    def _on_nav_current(self, current, _previous=None) -> None:
        if current is None or current.data(_ROLE_KIND) != "tool":
            return
        self._show_tool(catalog.TOOLS.get(current.data(_ROLE_ID)), current.data(_ROLE_KEY))

    def _on_nav_double(self, item) -> None:
        if item.data(_ROLE_KIND) == "tool" and self.ok_btn.isEnabled():
            self._on_accept()

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------
    def _settings_for(self, key: str):
        """The embedded settings panel for ``key``, built once per window so
        choices survive moving between tools and back."""
        if key not in self._settings:
            factory = registry.get_options_dialog_factory(key)
            panel = None
            if factory is not None:
                try:
                    from . import studio

                    # built as a plain widget straight away when it is one of the described panels
                    panel = studio.panel_for(key, embedded=True) if studio.has_spec(key) else factory(parent=None)
                    if hasattr(panel, "embed"):
                        panel.embed()
                    if hasattr(panel, "changed"):
                        panel.changed.connect(self._on_settings_changed)
                except Exception as exc:  # noqa: BLE001 - a broken panel must not take the window down
                    panel = kit.label(f"These settings couldn't be shown: {exc}", colour=kit.WARN, wrap=True)
            if panel is not None:
                panel.hide()
                self._settings_host_layout.addWidget(panel)
            self._settings[key] = panel
        panel = self._settings[key]
        sheet_name = self.table_combo.currentText()
        if panel is not None and hasattr(panel, "collect") and self._settings_sheet.get(key) != sheet_name:
            try:
                registry.prepare_settings(panel, self._sheet)
            except Exception:  # noqa: BLE001 - defaults are better than no settings
                pass
            self._settings_sheet[key] = sheet_name
        return panel

    def _show_tool(self, tool: catalog.Tool | None, key: str | None = None) -> None:
        if self._shown_panel is not None:
            self._shown_panel.hide()
            self._shown_panel = None
        self._current_settings = None
        if self.mode_switch is not None:
            self.mode_switch.hide()
            self.mode_holder.removeWidget(self.mode_switch)
            self.mode_switch.deleteLater()
            self.mode_switch = None
        self._tool = tool

        if tool is None:
            self.selected_analysis = None
            self.selected_category = None
            self.settings_title.setText("Choose a tool")
            self.settings_methods.setText("Open a goal on the left and pick the tool that answers your question.")
            for widget in (self.settings_detail, self.fit_pill, self.settings_note, self.no_settings,
                           self.settings_scroll):
                widget.hide()
            self._show_preview_message("")
            self.methods_label.setText("")
            self._update_ok_enabled()
            return

        key = self._key_for(tool, key)
        self._variant[tool.id] = key
        analysis = catalog.BY_KEY[key]
        self.selected_analysis = key
        self.selected_category = catalog.GROUP_OF[key].title
        self.settings_title.setText(tool.title)
        self.settings_methods.setText(tool.question if len(tool.variants) == 1 else analysis.question)
        self.settings_detail.setText(analysis.methods)
        self.settings_detail.show()
        if len(tool.variants) > 1:
            self.mode_switch = sk.Segmented(list(tool.variants))
            self.mode_switch.set_value(key)
            self.mode_switch.changed.connect(lambda k, t=tool: self._switch_form(t, k))
            self.mode_holder.addWidget(self.mode_switch)

        ready, why = self._fit(analysis)
        if ready is None:
            self.fit_pill.setText("Needs data")
            self._tone(self.fit_pill, kit.TEXT_FAINT)
            self.settings_note.hide()
        elif ready:
            self.fit_pill.setText("Fits this sheet" if why != "Makes its own data" else "Makes its own data")
            self._tone(self.fit_pill, kit.OK)
            self.settings_note.hide()
        else:
            self.fit_pill.setText("Doesn't fit")
            self._tone(self.fit_pill, kit.WARN)
            self.settings_note.setText(why)
            self.settings_note.show()
        self.fit_pill.show()

        self.data_strip.setVisible(getattr(self, "_has_choice_of_sets", False)
                                   and not catalog.requirement(analysis).no_data)
        panel = self._settings_for(key)
        if panel is None:
            self.settings_scroll.hide()
            self.no_settings.show()
        else:
            self.no_settings.hide()
            self.settings_scroll.show()
            panel.show()
            self._shown_panel = panel
            self.settings_scroll.verticalScrollBar().setValue(0)
            self._current_settings = panel
        self._update_ok_enabled()
        self._on_settings_changed()

    def _switch_form(self, tool: catalog.Tool, key: str) -> None:
        self._variant[tool.id] = key
        item = self.nav.currentItem()
        if item is not None and item.data(_ROLE_ID) == tool.id:
            ready, why = self._fit(catalog.BY_KEY[key])
            item.setData(_ROLE_KEY, key)
            item.setData(_ROLE_READY, ready)
            item.setData(_ROLE_DETAIL, why.split(";")[0] if ready is False else tool.question)
        self._show_tool(tool, key)

    @staticmethod
    def _tone(pill: QLabel, colour: str) -> None:
        pill.setStyleSheet(
            f"color: {colour}; background: transparent; border: 1px solid {colour};"
            f" border-radius: 9px; padding: 1px 9px; font-size: {kit.XS}px; font-weight: 600;")

    # ------------------------------------------------------------------
    # data sets
    # ------------------------------------------------------------------
    def _set_all_dataset_checks(self, value: bool) -> None:
        for chip in self._dataset_checks:
            chip.setChecked(value)

    def _on_datasets_changed(self, *_args) -> None:
        self._update_ok_enabled()
        self._preview_timer.start()

    @property
    def selected_data_sets(self) -> list[str] | None:
        """Chosen data-set labels, or None when every one is chosen (no
        filtering needed) or the table's data sets aren't known at all."""
        if not self._dataset_checks:
            return None
        checked = [label for label, chip in zip(self._dataset_labels, self._dataset_checks) if chip.isChecked()]
        return None if len(checked) == len(self._dataset_labels) else checked

    def _update_ok_enabled(self, *_args) -> None:
        analysis = catalog.BY_KEY.get(self.selected_analysis or "")
        if analysis is None:
            self.ok_btn.setEnabled(False)
            return
        # An empty sheet can only feed the tools that make their own data.
        if self._empty_sheet and not catalog.requirement(analysis).no_data:
            self.ok_btn.setEnabled(False)
            return
        any_checked = not self._dataset_checks or any(chip.isChecked() for chip in self._dataset_checks)
        self.ok_btn.setEnabled(any_checked)

    # ------------------------------------------------------------------
    # preview and methods
    # ------------------------------------------------------------------
    def _on_settings_changed(self) -> None:
        self._last_input = time.monotonic()
        panel = self._current_settings
        text = panel.methods_text() if panel is not None and hasattr(panel, "methods_text") else ""
        self.methods_label.setText(text or "A methods sentence appears here for tools that run a test or a fit.")
        self.copy_methods_btn.setEnabled(bool(text))
        self._preview_timer.start()

    def _copy_methods(self) -> None:
        panel = self._current_settings
        text = panel.methods_text() if panel is not None and hasattr(panel, "methods_text") else ""
        if text:
            QGuiApplication.clipboard().setText(text)
            self.copy_methods_btn.setText("Copied")
            QTimer.singleShot(1200, lambda: self.copy_methods_btn.setText("Copy"))

    def _show_preview_message(self, text: str, status: str = "", tone: str = kit.TEXT_DIM) -> None:
        self.preview_status.setText(status)
        self.preview_headline.setText(text)
        self.preview_headline.setStyleSheet(f"color: {tone}; font-size: {kit.MD}px; background: transparent;"
                                            " border: none;")
        self.preview_card.setVisible(bool(text))
        self.preview_table.setText("")
        self.preview_chart.hide()

    def _preview_data(self):
        """A copy of the sheet, narrowed to the chosen data sets."""
        if self._sheet is None:
            return None
        data = copy.copy(self._sheet)
        chosen = self.selected_data_sets
        if chosen:
            data.columns = [c for c in self._sheet.columns if c.group_label in chosen]
        return data

    def _run_preview(self, force: bool = False) -> None:
        if self._closing or not self.isVisible():  # nothing is calculated for a window nobody is looking at
            return
        if self._preview_job is not None:
            self._preview_job.cancel()
            self._preview_job = None
        self._preview_token += 1
        self.preview_now_btn.hide()
        key = self.selected_analysis
        analysis = catalog.BY_KEY.get(key or "")
        if analysis is None:
            return
        makes_data = catalog.requirement(analysis).no_data
        if not makes_data and self._empty_sheet:
            self._show_preview_message("Put data on the sheet and the result appears here as you set things up.")
            return
        if not self.ok_btn.isEnabled():
            self._show_preview_message("Choose at least one data set.")
            return
        data = self._preview_data()
        if data is None:
            from analysis.data_source import SheetData
            import numpy as np

            data = SheetData(x=np.array([]), columns=[], n_rows=0)
        size = sum(getattr(c.values, "size", 0) for c in getattr(data, "columns", []))
        if size > PREVIEW_LIMIT and not force:
            self._show_preview_message("This sheet is large, so the preview waits for you.", "paused")
            self.preview_now_btn.show()
            return
        panel = self._current_settings
        try:
            options = panel.collect() if panel is not None and hasattr(panel, "collect") else None
        except Exception as exc:  # noqa: BLE001
            self._show_preview_message(f"These settings aren't complete yet: {exc}", tone=kit.WARN)
            return

        from analysis import runner
        from analysis.registry import run_analysis

        token = self._preview_token
        self.preview_status.setText("updating…")
        self._preview_job = runner.submit(lambda: run_analysis(key, data, options),
                                          lambda result, error: self._preview_done(token, result, error),
                                          label="preview")

    def _preview_done(self, token: int, result, error) -> None:
        if self._closing or token != self._preview_token:
            return
        self._preview_job = None
        try:
            if error is not None:
                from analysis.backend_types import AnalysisError

                exc = error[0]
                text = str(exc) if isinstance(exc, AnalysisError) else f"This can't be previewed: {exc}"
                self._show_preview_message(text, "can't run yet", kit.WARN)
                return
            sheet = self.table_combo.currentText()
            self._show_preview_message(result.summary or "Done.", f"on {sheet}" if sheet else "", kit.TEXT)
            self.preview_table.setText(self._table_html(result))
            self._show_chart(result)
        except RuntimeError:  # the window went away while the preview was running
            pass

    @staticmethod
    def _table_html(result) -> str:
        if not result.tables:
            return ""
        table = result.tables[0]
        cols = min(4, len(table.headers))
        cell = f"padding: 2px 8px 2px 0; color: {kit.TEXT_DIM};"
        head = "".join(f"<td style='{cell} color: {kit.TEXT_FAINT};'><b>{_esc(h)}</b></td>" for h in table.headers[:cols])
        rows = "".join("<tr>" + "".join(f"<td style='{cell}'>{_esc(str(v))}</td>" for v in row[:cols]) + "</tr>"
                       for row in table.rows[:8])
        more = len(table.rows) - 8
        rest = [f"{more} more row{'s' if more != 1 else ''}"] if more > 0 else []
        if len(result.tables) > 1:
            rest.append(f"{len(result.tables) - 1} more table{'s' if len(result.tables) != 2 else ''}")
        extra = (f"<p style='color: {kit.TEXT_FAINT};'>… and {' and '.join(rest)} when you run it.</p>" if rest else "")
        return (f"<p style='color: {kit.TEXT}; margin-bottom: 4px;'><b>{_esc(table.title)}</b></p>"
                f"<table cellspacing='0'><tr>{head}</tr>{rows}</table>{extra}")

    def _show_chart(self, result) -> None:
        """A thumbnail of the first graph the analysis draws."""
        if not result.charts:
            self.preview_chart.hide()
            return
        try:
            import graphs  # noqa: F401 - sys.path bootstrap for the graphs package
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            from graphs.engine.charts import render_chart

            fig = Figure(figsize=(3.08, 2.0), dpi=100)
            canvas = FigureCanvasAgg(fig)
            render_chart(fig, result.charts[0])
            canvas.draw()
            width, height = canvas.get_width_height()
            image = QImage(bytes(canvas.buffer_rgba()), width, height, QImage.Format.Format_RGBA8888).copy()
            self.preview_chart.setPixmap(QPixmap.fromImage(image))
            self.preview_chart.show()
        except Exception:  # noqa: BLE001 - the thumbnail is a nicety
            self.preview_chart.hide()

    # ------------------------------------------------------------------
    # report style
    # ------------------------------------------------------------------
    def _show_style(self) -> None:
        self.style_btn.setText("Report style:  " + report_style.current().summary() + "   ·  change")

    def _edit_report_style(self) -> None:
        if ReportStyleDialog(self).exec():
            self._show_style()
            self._on_settings_changed()

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _on_accept(self) -> None:
        if self.selected_analysis is None or not self.ok_btn.isEnabled():
            return
        panel = self._current_settings
        self.options = panel.collect() if panel is not None and hasattr(panel, "collect") else None
        _record_recently_used(self.selected_category or "", self.selected_analysis)
        self.analysisChosen.emit(self.selected_category or "", self.selected_analysis)
        self.accept()

    def done(self, result: int) -> None:
        _last_size[0] = self.size()
        self._closing = True
        self._preview_timer.stop()
        if self._preview_job is not None:
            self._preview_job.cancel()
        super().done(result)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._preview_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_scroll_watched", False):   # scrolling counts as the user being busy (see _prebuild_next)
            self._scroll_watched = True
            for bar in (self.nav.verticalScrollBar(), self.settings_scroll.verticalScrollBar()):
                bar.valueChanged.connect(self._touched)
        self._preview_timer.start()

    def _open_guide(self) -> None:
        try:
            import help_docs

            help_docs.open_topic("analyses")
        except Exception:  # noqa: BLE001 - help is never worth an error dialog
            pass

    def _focus_search(self) -> None:
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    # ------------------------------------------------------------------
    # keyboard
    # ------------------------------------------------------------------
    def _step(self, delta: int) -> None:
        rows = [i for i in range(self.nav.count()) if self.nav.item(i).data(_ROLE_KIND) == "tool"]
        if not rows:
            return
        current = self.nav.currentRow()
        position = rows.index(current) if current in rows else -1
        row = rows[max(0, min(position + delta, len(rows) - 1))]
        self.nav.setCurrentRow(row)
        self.nav.scrollToItem(self.nav.currentItem())

    def eventFilter(self, watched, event) -> bool:
        if event.type() in _INPUT_EVENTS:  # typing in the search box (see _prebuild_next)
            self._last_input = time.monotonic()
        # Up/Down from the search box walk the tools, so typing then arrowing works.
        if watched is self.search_edit and event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._step(1 if event.key() == Qt.Key.Key_Down else -1)
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_accept()
            return
        if key == Qt.Key.Key_Escape and self.search_edit.text():
            self.search_edit.clear()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # for callers and tests
    # ------------------------------------------------------------------
    def select_analysis(self, key: str) -> bool:
        """Show and select an analysis by its registry key: its tool, in that form."""
        tool = catalog.TOOL_OF.get(key)
        if tool is None:
            return False
        self.search_edit.blockSignals(True)
        self.search_edit.clear()
        self.search_edit.blockSignals(False)
        self._variant[tool.id] = key
        self._open_goal = catalog.GROUP_OF[key].id
        self._tool = tool
        self._fill_nav()
        return self.selected_analysis == key

    def select_tool(self, tool_id: str) -> bool:
        """Show a tool in whichever form suits the sheet."""
        tool = catalog.TOOLS.get(tool_id)
        if tool is None:
            return False
        self._variant.pop(tool_id, None)
        return self.select_analysis(self._key_for(tool))

    def visible_entries(self) -> list[tuple[str, bool | None, str]]:
        """(registry key, fits?, the line under the title) for every tool row shown."""
        return [(item.data(_ROLE_KEY), item.data(_ROLE_READY), item.data(_ROLE_DETAIL))
                for item in (self.nav.item(i) for i in range(self.nav.count())) if item.data(_ROLE_KIND) == "tool"]

    def rail_ids(self) -> list[str]:
        return [item.data(_ROLE_ID) for item in (self.nav.item(i) for i in range(self.nav.count()))
                if item.data(_ROLE_KIND) == "goal"]


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
