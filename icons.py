"""The Analyze icon set: one small line drawing per question.

Every icon is drawn on the same 24-unit grid with the same 1.6-unit round
stroke and no fills except data points, so the set reads as one family at
16-20 px. Each one shows the *kind of result* the question produces - a
significance bracket over two groups, a histogram under its curve, a
saturating fit, survival steps, an ROC bow - rather than a generic
shape. Colour is supplied at draw time, so the same drawing serves the idle,
hover and selected states and any theme.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_STROKE = 'fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'
_DOT = 'fill="{c}" stroke="none"'

# body of each <svg viewBox="0 0 24 24">; {S} = stroke attributes, {D} = dot attributes
_BODIES = {
    # a spark: where to start
    "suggested": '<path {S} d="M11 3.5 12.6 9.4 18.5 11 12.6 12.6 11 18.5 9.4 12.6 3.5 11 9.4 9.4Z"/>'
                 '<path {S} d="M18.5 16v4M16.5 18h4"/>',
    # a clock with the arrow running back
    "recent": '<path {S} d="M4.6 9.2A8 8 0 1 1 4 12"/><path {S} d="M4.2 4.8v4.6h4.6"/>'
              '<path {S} d="M12 8v4.2l2.8 1.8"/>',
    # two groups side by side, and the bracket that asks "is the difference real?"
    "compare": '<path {S} d="M6.5 6V4h11v2"/>'
               '<rect {S} x="4" y="12" width="5" height="8.5" rx="1.2"/>'
               '<rect {S} x="15" y="8.5" width="5" height="12" rx="1.2"/>',
    # a distribution: a broad bell over its baseline, the spread marked
    "describe": '<path {S} d="M3 19.5h18"/>'
                '<path {S} d="M3 17.5C6.5 17.5 8 7 12 7s5.5 10.5 9 10.5"/>'
                '<path {S} d="M8.5 13.5h7" stroke-dasharray="0.1 2.4"/>',
    # axes, points and a saturating model through them
    "fit": '<path {S} d="M3.5 3.5v17h17"/>'
           '<path {S} d="M4 19.5C7 10 11 7.5 20.5 7"/>'
           '<circle {D} cx="7.2" cy="13.2" r="1.3"/><circle {D} cx="11" cy="10.4" r="1.3"/>'
           '<circle {D} cx="16" cy="8.8" r="1.3"/>',
    # a scatter with its trend line
    "relate": '<path {S} d="M4 19.5 20 5"/>'
              '<circle {D} cx="6.5" cy="14.5" r="1.3"/><circle {D} cx="10" cy="15" r="1.3"/>'
              '<circle {D} cx="11.5" cy="9.8" r="1.3"/><circle {D} cx="15.5" cy="11" r="1.3"/>'
              '<circle {D} cx="17" cy="6" r="1.3"/>',
    # a 2 x 2 table with counts in the cells
    "counts": '<rect {S} x="3.5" y="3.5" width="17" height="17" rx="2.5"/><path {S} d="M12 3.5v17M3.5 12h17"/>'
              '<circle {D} cx="7.75" cy="7.75" r="1.5"/><circle {D} cx="16.25" cy="16.25" r="1.5"/>',
    # survival steps with a censoring tick
    "time": '<path {S} d="M3.5 3.5v17h17"/>'
            '<path {S} d="M4 5.5h4v3.5h4v4h3.5v3.5h4.5"/><path {S} d="M10 7.5v3"/>',
    # an ROC curve bowing above the chance line
    "agreement": '<rect {S} x="3.5" y="3.5" width="17" height="17" rx="2.5"/>'
                 '<path {S} d="M5 19 19 5" stroke-dasharray="0.1 2.8"/>'
                 '<path {S} d="M5 19C5.5 9 9 5.5 19 5"/>',
    # two clusters, each ringed
    "patterns": '<circle {S} cx="8" cy="8.5" r="4.8"/><circle {S} cx="16.5" cy="15.5" r="4.3"/>'
                '<circle {D} cx="6.8" cy="7.5" r="1.2"/><circle {D} cx="9.4" cy="9.6" r="1.2"/>'
                '<circle {D} cx="15.4" cy="16.5" r="1.2"/><circle {D} cx="17.8" cy="14.6" r="1.2"/>',
    # a table with its columns being swapped
    "reshape": '<rect {S} x="3.5" y="3.5" width="11" height="17" rx="2"/><path {S} d="M3.5 8.5h11M9 3.5v17"/>'
               '<path {S} d="M17.5 8h3.5l-1.8-1.8M21 16h-3.5l1.8 1.8"/>',
    # a die: values drawn at random
    "make": '<rect {S} x="3.5" y="3.5" width="17" height="17" rx="3.5"/>'
            '<circle {D} cx="8.2" cy="8.2" r="1.4"/><circle {D} cx="12" cy="12" r="1.4"/>'
            '<circle {D} cx="15.8" cy="15.8" r="1.4"/>',
}

KINDS = tuple(_BODIES)

# the catalog's older glyph names
_ALIASES = {"star": "suggested", "clock": "recent"}

_pixmaps: dict[tuple[str, str, int, float], QPixmap] = {}
_icons: dict[tuple[str, str, int, float], QIcon] = {}


def svg(kind: str, color: str) -> str:
    kind = _ALIASES.get(kind, kind)
    body = _BODIES.get(kind, _BODIES["make"])
    body = body.replace("{S}", _STROKE).replace("{D}", _DOT).replace("{c}", color)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'


def pixmap(kind: str, color: str, size: int = 18, ratio: float = 2.0) -> QPixmap:
    """Rendered at ``ratio`` x the logical size so it stays crisp on high-DPI screens."""
    key = (kind, color, size, ratio)
    if key in _pixmaps:
        return _pixmaps[key]
    px = QPixmap(round(size * ratio), round(size * ratio))
    px.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg(kind, color).encode()))
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, px.width(), px.height()))
    painter.end()
    px.setDevicePixelRatio(ratio)
    _pixmaps[key] = px
    return px


def icon(kind: str, color: str, size: int = 18, ratio: float = 2.0) -> QIcon:
    key = (kind, color, size, ratio)
    if key not in _icons:
        _icons[key] = QIcon(pixmap(kind, color, size, ratio))
    return _icons[key]
