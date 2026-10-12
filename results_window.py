"""The window that shows one analysis result.

The answer comes first, in a headline card: the one line most people open
the window for. Under it, the tables and any graphs, each on its own tab
only when there is more than one kind of thing to show. From here the
analysis can be exported, or reopened in the Analyze window with its
settings to change and run again.

Non-modal by convention (callers use ``show()``), so several results can sit
open beside the sheet at once.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import ui_kit as kit
from .backend_types import AnalysisResult, ResultTable

# kept for modules that still read these names
BG, PANEL_BG, INPUT_BG, BORDER = kit.BG, kit.PANEL, kit.SUNKEN, kit.BORDER
ACCENT, TEXT, TEXT_DIM = kit.ACCENT, kit.TEXT, kit.TEXT_DIM
_TABS_SS = kit.tabs_ss()

_MAX_TABLE_ROWS = 12  # taller tables scroll inside themselves
_NUMBER = re.compile(r"^[<>≤≥±\s]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*(%|\([^)]*\))?$")


def refresh_theme() -> None:
    """Kept for callers; the colours come from analysis/ui_kit.py."""


def _looks_numeric(text: str) -> bool:
    return bool(_NUMBER.match(str(text).strip())) or str(text).strip() in ("n/a", "ns", "*", "**", "***", "****")


class ResultsWindow(QDialog):
    DRAWN_AT_ONCE = 3  # graphs drawn when the window opens; the rest when scrolled to

    def __init__(self, result: AnalysisResult, subtitle: str | None = None, parent=None, on_rerun=None):
        super().__init__(parent)
        self.result = result
        self.subtitle = subtitle
        self._on_rerun = on_rerun
        self.chart_figures: list = []
        self.tabs = None
        self.setObjectName("ResultsWindow")
        self.setStyleSheet(f"#ResultsWindow {{ background: {kit.BG}; }}")
        self.setWindowTitle(self._title())
        self.resize(820, 700 if result.charts else 600)
        self._build_ui()

    def _title(self) -> str:
        from . import catalog

        entry = catalog.BY_KEY.get(getattr(self.result, "analysis_key", "") or "")
        return entry.title if entry else self.result.title

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        result = self.result
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- header: what was run, on what, and the answer ----
        header = kit.panel(QFrame(), "ResultsHead", kit.RAISED, border=f"border-bottom: 1px solid {kit.BORDER};")
        head = QVBoxLayout(header)
        head.setContentsMargins(22, 16, 22, 16)
        head.setSpacing(4)
        head.addWidget(kit.label(self._title(), size=kit.XL, weight=600))
        meta = " · ".join(p for p in (result.title if result.title != self._title() else "", self.subtitle or "") if p)
        if meta:
            head.addWidget(kit.label(meta, size=kit.SM, colour=kit.TEXT_DIM))
        if result.summary:
            head.addSpacing(8)
            card = kit.panel(QFrame(), "Headline", kit.ACCENT_SOFT,
                             border=f"border: 1px solid {kit.ACCENT}; border-left: 3px solid {kit.ACCENT};",
                             radius=kit.RADIUS)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            self.headline = kit.label(result.summary, size=kit.LG, weight=600, wrap=True)
            self.headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card_layout.addWidget(self.headline)
            head.addWidget(card)
        root.addWidget(header)

        # ---- body ----
        pages = []
        if result.charts:
            pages.append((f"Graphs ({len(result.charts)})", self._build_charts(result.charts)))
        if result.tables:
            pages.append((f"Tables ({len(result.tables)})" if result.charts else "Tables", self._build_tables()))
        if result.notes:
            pages.append((f"Notes ({len(result.notes)})", self._build_notes()))
        body = QWidget()
        body.setObjectName("ResultsBody")
        body.setStyleSheet(f"#ResultsBody {{ background: {kit.BG}; }}")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(22, 10, 22, 6)
        if len(pages) > 1:
            self.tabs = QTabWidget()
            self.tabs.setStyleSheet(kit.tabs_ss())
            self.tabs.tabBar().setDrawBase(False)  # the pane's own top rule is the only line
            for name, page in pages:
                self.tabs.addTab(page, name)
            body_layout.addWidget(self.tabs, 1)
        elif pages:
            body_layout.addWidget(pages[0][1], 1)
        root.addWidget(body, 1)

        # ---- footer ----
        foot = kit.panel(QFrame(), "ResultsFoot", kit.RAISED, border=f"border-top: 1px solid {kit.BORDER};")
        row = QHBoxLayout(foot)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(8)
        if self._on_rerun is not None:
            rerun = QPushButton("Change settings and run again…")
            rerun.setStyleSheet(kit.quiet_button_ss())
            rerun.setCursor(Qt.CursorShape.PointingHandCursor)
            rerun.clicked.connect(self._rerun)
            row.addWidget(rerun)
            self.rerun_btn = rerun
        row.addStretch()
        pdf = QPushButton("Export PDF…")
        pdf.setStyleSheet(kit.ghost_button_ss())
        pdf.clicked.connect(self._export_pdf)
        row.addWidget(pdf)
        close = QPushButton("Close")
        close.setStyleSheet(kit.accent_button_ss())
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addWidget(foot)

    # ------------------------------------------------------------------
    def _scroll(self, name: str) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName(f"{name}Scroll")
        scroll.setStyleSheet(f"#{name}Scroll {{ background: {kit.BG}; border: none; }}" + kit.SCROLLBAR_SS)
        content = QWidget()
        content.setObjectName(f"{name}Content")
        content.setStyleSheet(f"#{name}Content {{ background: {kit.BG}; }}")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 6, 12)
        layout.setSpacing(18)
        scroll.setWidget(content)
        return scroll, layout

    def _build_tables(self) -> QWidget:
        scroll, layout = self._scroll("Tables")
        for table in self.result.tables:
            layout.addWidget(self._build_table(table))
        layout.addStretch()
        return scroll

    def _build_notes(self) -> QWidget:
        scroll, layout = self._scroll("Notes")
        layout.setSpacing(8)
        for note in self.result.notes:
            layout.addWidget(kit.label(f"•  {note}", size=kit.MD, colour=kit.TEXT_DIM, wrap=True))
        layout.addStretch()
        return scroll

    @staticmethod
    def _build_table(table: ResultTable) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.addWidget(kit.label(table.title, size=kit.MD, weight=600, wrap=True), 1)
        copy = QPushButton("Copy")
        copy.setStyleSheet(kit.quiet_button_ss())
        copy.setCursor(Qt.CursorShape.PointingHandCursor)
        copy.setToolTip("Copy this table, ready to paste into a spreadsheet")
        copy.clicked.connect(lambda _=False, t=table: QGuiApplication.clipboard().setText(
            "\n".join(["\t".join(t.headers)] + ["\t".join(str(v) for v in r) for r in t.rows])))
        head.addWidget(copy, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)

        grid = QTableWidget(len(table.rows), len(table.headers))
        grid.setObjectName("ResultTable")
        grid.setHorizontalHeaderLabels(table.headers)
        grid.verticalHeader().setVisible(False)
        grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        grid.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        grid.setAlternatingRowColors(True)
        grid.setShowGrid(False)
        grid.setWordWrap(False)
        grid.setStyleSheet(f"""
            #ResultTable {{ background: {kit.SUNKEN}; alternate-background-color: #1a1a1f; color: {kit.TEXT};
                            border: 1px solid {kit.BORDER}; border-radius: {kit.RADIUS}px; font-size: {kit.MD}px;
                            selection-background-color: {kit.ACCENT_SOFT}; selection-color: {kit.TEXT}; }}
            #ResultTable QHeaderView::section {{ background: {kit.RAISED}; color: {kit.TEXT_DIM}; border: none;
                            border-bottom: 1px solid {kit.BORDER}; padding: 6px 10px; font-weight: 600;
                            font-size: {kit.SM}px; }}
            #ResultTable::item {{ padding: 4px 10px; }}
        """ + kit.SCROLLBAR_SS)
        # A column of numbers lines up on its right edge, heading included, so
        # digits sit under one another; words and ranges read from the left.
        numeric = [c > 0 and bool(table.rows) and all(_looks_numeric(row[c]) for row in table.rows if c < len(row))
                   for c in range(len(table.headers))]
        for c, is_number in enumerate(numeric):
            align = Qt.AlignmentFlag.AlignRight if is_number else Qt.AlignmentFlag.AlignLeft
            grid.horizontalHeaderItem(c).setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
        for r, values in enumerate(table.rows):
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                align = Qt.AlignmentFlag.AlignRight if c < len(numeric) and numeric[c] else Qt.AlignmentFlag.AlignLeft
                item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
                grid.setItem(r, c, item)
        header = grid.horizontalHeader()
        header.setHighlightSections(False)
        grid.resizeColumnsToContents()
        # the spare width goes to the last text column, so a number never sits
        # stranded at the far edge away from its neighbours
        text_cols = [c for c, is_number in enumerate(numeric) if not is_number]
        for c, is_number in enumerate(numeric):
            if is_number:
                grid.setColumnWidth(c, grid.columnWidth(c) + 18)
        header.setSectionResizeMode(text_cols[-1] if text_cols else len(table.headers) - 1,
                                    QHeaderView.ResizeMode.Stretch)
        row_h = grid.verticalHeader().defaultSectionSize()
        shown = min(len(table.rows), _MAX_TABLE_ROWS)
        grid.setFixedHeight(header.sizeHint().height() + row_h * shown + 4)
        grid.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(grid)
        return frame

    def _build_charts(self, charts) -> QWidget:
        # matplotlib is only imported for results that actually have graphs
        import graphs  # noqa: F401 - sys.path bootstrap for the graphs package
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from graphs.engine.charts import render_chart

        class ChartCanvas(FigureCanvasQTAgg):
            """A graph in the scrolling list.

            The wheel belongs to the list: matplotlib's canvas takes every
            wheel event for itself, so the list would stop scrolling whenever
            the pointer was over a graph - and the graphs fill the list.

            A graph far down a long list (an MD report has dozens) is drawn
            when it first comes into view, not when the window opens."""

            def __init__(self, figure, chart):
                super().__init__(figure)
                self._chart = chart

            def show_chart(self, chart) -> None:
                """Draw another graph in this place (the choice above the graph changed)."""
                self._chart = chart
                self.ensure_drawn()
                self.draw_idle()

            def ensure_drawn(self) -> None:
                chart, self._chart = self._chart, None
                if chart is None:
                    return
                try:
                    render_chart(self.figure, chart)
                except Exception as exc:  # noqa: BLE001 - a bad chart must not hide the tables
                    self.figure.clear()
                    self.figure.text(0.5, 0.5, f"This graph couldn't be drawn:\n{exc}", ha="center", va="center",
                                     color="#555555")

            def paintEvent(self, event):
                if self._chart is not None:
                    self.ensure_drawn()
                    self.draw()
                super().paintEvent(event)

            def wheelEvent(self, event):
                QApplication.sendEvent(scroll.verticalScrollBar(), event)   # handed straight to the list

        scroll, layout = self._scroll("Charts")
        # graphs that share a group stand in one place, with a choice between them
        places: list[list] = []
        for chart in charts:
            if getattr(chart, "group", "") and places and getattr(places[-1][0], "group", "") == chart.group:
                places[-1].append(chart)
            else:
                places.append([chart])
        self.chart_choices = []
        for number, members in enumerate(places):
            chart = members[0]
            card = kit.panel(QFrame(), f"Chart{id(chart)}", "#ffffff", radius=kit.RADIUS)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 4)
            fig = Figure(figsize=(6.4, 4.2))
            canvas = ChartCanvas(fig, chart)
            choice = None
            if len(members) > 1:
                choice = QComboBox()
                choice.addItems([m.option or m.title for m in members])
                choice.setCursor(Qt.CursorShape.PointingHandCursor)
                choice.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                choice.wheelEvent = lambda event: QApplication.sendEvent(scroll.verticalScrollBar(), event)  # the wheel scrolls the list
                choice.setStyleSheet(
                    "QComboBox { background: #f3f5f8; color: #1f2733; border: 1px solid #c9d0da; border-radius: 5px;"
                    f" padding: 4px 10px; font-size: {kit.MD}px; min-width: 190px; }}"
                    "QComboBox:hover { border-color: #4a90d9; }"
                    "QComboBox QAbstractItemView { background: white; color: #1f2733; selection-background-color: #4a90d9;"
                    " selection-color: white; }")
                choice.currentIndexChanged.connect(lambda i, c=canvas, m=members: c.show_chart(m[i]))
                top = QHBoxLayout()
                top.setContentsMargins(6, 2, 6, 0)
                label = QLabel("Show")
                label.setStyleSheet(f"color: #4a5568; font-size: {kit.MD}px; background: transparent;")
                top.addWidget(label)
                top.addWidget(choice)
                top.addStretch()
                card_layout.addLayout(top)
                self.chart_choices.append(choice)
            canvas.setMinimumHeight(360)
            if number < self.DRAWN_AT_ONCE:
                canvas.ensure_drawn()
            self.chart_figures.append(fig)
            card_layout.addWidget(canvas)
            actions = QHBoxLayout()
            actions.addStretch()
            export = QPushButton("Export graph…")
            export.setStyleSheet(f"""
                QPushButton {{ background: transparent; color: #4a5568; border: none; padding: 4px 6px;
                               font-size: {kit.SM}px; }}
                QPushButton:hover {{ color: {kit.ACCENT}; }}""")
            export.setCursor(Qt.CursorShape.PointingHandCursor)
            export.clicked.connect(lambda _=False, c=canvas, m=members, ch=choice: (
                c.ensure_drawn(), self._export_chart(c.figure, m[ch.currentIndex() if ch is not None else 0].title)))
            actions.addWidget(export)
            card_layout.addLayout(actions)
            layout.addWidget(card)
        layout.addStretch()
        return scroll

    # ------------------------------------------------------------------
    def _rerun(self) -> None:
        if self._on_rerun is not None:
            self._on_rerun()

    def _export_pdf(self) -> None:
        from .report_pdf import export_result_pdf

        default = re.sub(r'[\\/:*?"<>|]+', "_", self._title() or "results") + ".pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export results as PDF", default, "PDF document (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            pages = export_result_pdf(self.result, path, subtitle=self.subtitle)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Exported", f"Saved a {pages}-page report to:\n{path}")

    def _export_chart(self, fig, title: str) -> None:
        default = re.sub(r'[\\/:*?"<>|]+', "_", title or "graph") + ".png"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export graph", default, "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf);;TIFF image (*.tiff)")
        if not path:
            return
        ext = chosen.split("*.")[-1].rstrip(")")
        if "." not in path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]:
            path += f".{ext}"
        try:
            fig.savefig(path, dpi=300, facecolor="white")
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            QMessageBox.warning(self, "Export failed", str(exc))
