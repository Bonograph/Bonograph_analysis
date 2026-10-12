"""The look shared by every analysis window: the Analyze window, the settings
embedded in it, and the results window.

One set of tokens and a handful of building blocks, so the three windows
cannot drift apart. Two rules keep it glitch-free:

  * every frame style is scoped to its own objectName (``#name``) - an
    unscoped ``border`` on a QFrame cascades onto every label inside it,
    which is what drew stray underlines under text;
  * every scroll area gets SCROLLBAR_SS, which styles the track pages as
    well as the handle - leave them out and Windows paints its hatched
    default pattern into the track.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

import ui_theme

_ACCENT = ui_theme.colors()["accent"]

# ── tokens ───────────────────────────────────────────────────────────────────
BG = "#18181c"          # window
RAISED = "#1f1f24"      # rail, top bar, footer
SUNKEN = "#141417"      # inputs, table cells
PANEL = "#222228"       # settings pane
HOVER = "#2a2a31"
BORDER = "#2e2e36"
BORDER_STRONG = "#3d3d47"
TEXT = "#ececf1"
TEXT_DIM = "#a0a0ad"
TEXT_FAINT = "#6d6d7a"
ACCENT = _ACCENT
ACCENT_SOFT = "rgba(74, 144, 217, 38)"
ACCENT_HOVER = "#5c9fe3"
ACCENT_PRESSED = "#3b7fc7"
OK = "#4ac97e"
WARN = "#e0a84a"

RADIUS = 6
FONT = "Segoe UI"
# type scale
XS, SM, MD, LG, XL = 11, 12, 13, 15, 18

SCROLLBAR_SS = f"""
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px 1px; border: none; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 1px 2px; border: none; }}
    QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 3px; min-height: 28px; }}
    QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 3px; min-width: 28px; }}
    QScrollBar::handle:hover {{ background: #55555f; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: transparent; }}
"""


def panel(widget: QWidget, name: str, background: str, *, border: str = "", radius: int = 0) -> QWidget:
    """Give ``widget`` a background (and optional border sides) that apply to
    it alone, never to the labels and checkboxes inside it."""
    widget.setObjectName(name)
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    rule = f"background: {background};"
    if border:
        rule += f" {border}"
    if radius:
        rule += f" border-radius: {radius}px;"
    widget.setStyleSheet(f"#{name} {{ {rule} }}")
    return widget


def label(text: str = "", *, size: int = MD, colour: str = TEXT, weight: int = 400,
          wrap: bool = False, italic: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(wrap)
    style = "italic" if italic else "normal"
    lbl.setStyleSheet(f"color: {colour}; font-size: {size}px; font-weight: {weight}; font-style: {style};"
                      " background: transparent; border: none;")
    return lbl


def pill(text: str, tone: str = "ok") -> QLabel:
    """A small rounded status chip - "Fits this sheet", "Needs repeats"."""
    colour = {"ok": OK, "warn": WARN, "info": ACCENT}.get(tone, TEXT_DIM)
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {colour}; background: transparent; border: 1px solid {colour};"
        f" border-radius: 9px; padding: 1px 9px; font-size: {XS}px; font-weight: 600;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


def rule() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    panel(line, f"rule{id(line)}", BORDER)
    return line


def accent_button_ss() -> str:
    return f"""
        QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: {RADIUS}px;
                       padding: 7px 22px; font-size: {MD}px; font-weight: 600; }}
        QPushButton:hover {{ background: {ACCENT_HOVER}; }}
        QPushButton:pressed {{ background: {ACCENT_PRESSED}; }}
        QPushButton:disabled {{ background: {BORDER_STRONG}; color: {TEXT_FAINT}; }}
    """


def ghost_button_ss() -> str:
    return f"""
        QPushButton {{ background: transparent; color: {TEXT}; border: 1px solid {BORDER_STRONG};
                       border-radius: {RADIUS}px; padding: 7px 16px; font-size: {MD}px; }}
        QPushButton:hover {{ background: {HOVER}; }}
        QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}
    """


def quiet_button_ss() -> str:
    """A link-like button for secondary actions ("Which test do I need?")."""
    return f"""
        QPushButton {{ background: transparent; color: {TEXT_DIM}; border: none; padding: 7px 4px;
                       font-size: {SM}px; text-align: left; }}
        QPushButton:hover {{ color: {ACCENT}; }}
    """


def input_ss() -> str:
    return f"""
        QLineEdit {{ background: {SUNKEN}; color: {TEXT}; border: 1px solid {BORDER};
                     border-radius: {RADIUS}px; padding: 7px 10px; font-size: {MD}px;
                     selection-background-color: {ACCENT}; }}
        QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
    """


def tabs_ss() -> str:
    return f"""
        QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; top: -1px; background: transparent; }}
        QTabBar {{ background: transparent; }}
        QTabBar::tab {{ background: transparent; color: {TEXT_DIM}; padding: 8px 14px 7px 14px;
                        margin-right: 4px; border: none; border-bottom: 2px solid transparent;
                        font-size: {MD}px; }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
        QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; font-weight: 600; }}
    """


def row(*widgets, spacing: int = 8, stretch_after: bool = True) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    if stretch_after:
        layout.addStretch()
    return layout
