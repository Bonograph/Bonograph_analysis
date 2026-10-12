"""Small live drawings for settings panels: the sheet's own trace or points,
with what the current settings would do to them drawn on top - the level an
area is measured from, a smoothed curve over the raw one, the limits two
methods agree within. A choice is seen on the data instead of being imagined.

``sketch(scene)`` wraps a function ``scene(panel) -> dict | None`` giving:

* ``lines``  - [{"x", "y", "colour", "width", "own"}]; ``own`` draws the line
  to its own vertical scale (a slope has other units than the trace it is of).
* ``fills``  - [{"x", "y", "base"}]: the area between a line and a level.
* ``levels`` - [{"y", "colour", "dash", "text", "under"}]: horizontal lines, labelled above (or under).
* ``dots``   - [{"x", "y", "colour"}].
* ``caption`` - a line of words under the drawing.

Also the little grid drawings for cards that choose rows, columns or the
whole table.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPolygonF
from PySide6.QtWidgets import QWidget

from ... import ui_kit as kit
from .. import kit as sk

MOST_POINTS = 600   # a longer trace is thinned for drawing only


class _Sketch(QWidget):
    def __init__(self, scene, height: int, empty: str):
        super().__init__()
        self._make, self._scene, self._empty, self._seen = scene, None, empty, None
        self.setFixedHeight(height)
        self.setAccessibleName("A drawing of the first data set under these settings")

    def refresh(self, panel) -> None:
        seen = (id(panel.ctx.sheet), repr(sorted(panel.state.items(), key=lambda kv: kv[0])))
        if seen == self._seen:
            return
        self._seen = seen
        try:
            self._scene = self._make(panel)
        except Exception:  # noqa: BLE001 - a drawing must never break the panel
            self._scene = None
        self.setAccessibleDescription((self._scene or {}).get("caption", ""))
        self.update()

    @property
    def caption(self) -> str:
        return (self._scene or {}).get("caption", "")

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(frame, 8, 8)
        p.fillPath(path, QColor(kit.SUNKEN))
        p.setPen(sk.pen(kit.BORDER, 1))
        p.drawPath(path)
        font = QFont(kit.FONT)
        font.setPixelSize(11)
        p.setFont(font)
        scene = self._scene
        if not scene:
            p.setPen(QColor(kit.TEXT_FAINT))
            p.drawText(frame.adjusted(14, 0, -14, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                       | Qt.TextFlag.TextWordWrap, self._empty)
            return
        caption = scene.get("caption", "")
        area = frame.adjusted(14, 14, -14, -(42 if caption else 16))
        shared = [ln for ln in scene.get("lines", []) if not ln.get("own")]
        xs = [np.asarray(ln["x"], dtype=float) for ln in scene.get("lines", [])] + [
            np.asarray([d["x"]], dtype=float) for d in scene.get("dots", [])]
        ys = [np.asarray(ln["y"], dtype=float) for ln in shared] + [np.asarray([lv["y"]], dtype=float) for lv in scene.get("levels", [])] + [
            np.asarray([d["y"]], dtype=float) for d in scene.get("dots", [])]
        xs = np.concatenate(xs) if xs else np.empty(0)
        ys = np.concatenate(ys) if ys else np.empty(0)
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if xs.size < 2 or ys.size < 1:
            return
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        pad = (y1 - y0) * 0.08 or 1.0
        y0, y1 = y0 - pad, y1 + pad
        xspan = (x1 - x0) or 1.0

        def px(x: float) -> float:
            return area.left() + (x - x0) / xspan * area.width()

        def py(y: float, lo: float = y0, hi: float = y1) -> float:
            return area.bottom() - (y - lo) / ((hi - lo) or 1.0) * area.height()

        fill = QColor(kit.ACCENT)
        fill.setAlpha(60)
        for f in scene.get("fills", []):
            fx, fy = _thin(f["x"], f["y"])
            if fx.size < 2:
                continue
            poly = QPolygonF([QPointF(px(fx[0]), py(f["base"]))] + [QPointF(px(a), py(b)) for a, b in zip(fx, fy)]
                             + [QPointF(px(fx[-1]), py(f["base"]))])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawPolygon(poly)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for lv in scene.get("levels", []):
            pen = sk.pen(lv.get("colour", kit.ACCENT), 1.2)
            if lv.get("dash"):
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            y = py(lv["y"])
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            if lv.get("text"):
                p.setPen(QColor(lv.get("colour", kit.ACCENT)))
                p.drawText(QRectF(area.left(), y + 1 if lv.get("under") else y - 15, area.width(), 14),
                           Qt.AlignmentFlag.AlignRight, lv["text"])
        for ln in scene.get("lines", []):
            lx, ly = _thin(ln["x"], ln["y"])
            if lx.size < 2:
                continue
            lo, hi = (y0, y1)
            if ln.get("own"):
                good = ly[np.isfinite(ly)]
                if good.size < 2:
                    continue
                gap = (float(good.max()) - float(good.min())) * 0.08 or 1.0
                lo, hi = float(good.min()) - gap, float(good.max()) + gap
            p.setPen(sk.pen(ln.get("colour", kit.TEXT_DIM), ln.get("width", 1.6)))
            line = QPainterPath()
            started = False
            for a, b in zip(lx, ly):
                if not np.isfinite(b):
                    started = False
                    continue
                point = QPointF(px(a), py(b, lo, hi))
                line.lineTo(point) if started else line.moveTo(point)
                started = True
            p.drawPath(line)
        p.setPen(Qt.PenStyle.NoPen)
        for d in scene.get("dots", []):
            p.setBrush(QColor(d.get("colour", kit.ACCENT)))
            p.drawEllipse(QPointF(px(d["x"]), py(d["y"])), 3.2, 3.2)
        if caption:
            p.setPen(QColor(kit.TEXT_DIM))
            p.drawText(QRectF(frame.left() + 14, frame.bottom() - 22, frame.width() - 28, 16),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideRight, int(frame.width() - 28)))


def _thin(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size > MOST_POINTS:
        keep = np.linspace(0, x.size - 1, MOST_POINTS).astype(int)
        x, y = x[keep], y[keep]
    return x, y


def sketch(scene, height: int = 170, empty: str = "With data on the sheet, the first data set is drawn here as these "
                                                  "settings would treat it.") -> QWidget:
    return _Sketch(scene, height, empty)


def _g(value: float) -> str:
    return f"{value:.4g}"


# ── the scenes ───────────────────────────────────────────────────────────────

def _first_trace(panel):
    sheet = panel.ctx.sheet
    labels = list(sheet.group_labels()) if sheet is not None else []
    if not labels:
        return None
    x, y = sheet.xy_points(labels[0])
    order = np.argsort(x)
    return labels[0], x[order], y[order]


def area_scene(panel):
    """The trace, the level its area is measured from, the height a bump must reach, and the bumps that count."""
    from ...backends.area_under_curve import outline

    got = _first_trace(panel)
    if got is None or got[1].size < 2:
        return None
    name, x, y = got
    opt = panel.collect()
    baseline, height, peaks, _signal, _working = outline(x, y, opt)
    shaded = y if opt.allow_below_baseline else np.maximum(y, baseline)
    levels = [{"y": baseline, "colour": kit.ACCENT, "text": f"measured from {_g(baseline)}", "under": True}]
    if height > 0:
        levels.append({"y": baseline + height, "colour": kit.WARN, "dash": True, "text": "a bump must rise above this"})
    count = len(peaks)
    return {
        "lines": [{"x": x, "y": y, "colour": kit.TEXT, "width": 1.6}],
        "fills": [{"x": x, "y": shaded, "base": baseline}],
        "levels": levels,
        "dots": [{"x": float(x[i]), "y": float(y[i]), "colour": kit.OK} for i in peaks],
        "caption": f"{name}: {count} bump{'' if count == 1 else 's'} counted (green dots); the shaded part is the area.",
    }


def slope_scene(panel):
    """The trace as typed, and what the settings turn it into."""
    from ...backends.curve_smoothing_diff_integrate import compute

    got = _first_trace(panel)
    if got is None or got[1].size < 5:
        return None
    name, x, y = got
    opt = panel.collect()
    new, _steps, label = compute(x, y, opt)
    own = opt.operation != "none"
    what = {"none": "smoothed" if opt.smooth else "unchanged", "first_derivative": "its slope", "second_derivative":
            "how its slope changes", "integrate": "the area built up"}[opt.operation]
    return {
        "lines": [{"x": x, "y": y, "colour": kit.TEXT_FAINT, "width": 1.4},
                  {"x": x, "y": new, "colour": kit.ACCENT, "width": 2.0, "own": own}],
        "caption": f"{name}: grey as typed, blue {what}" + (f" ({label}, drawn to its own scale)." if own else "."),
    }


def agreement_scene(panel):
    """Each pair's difference against its mean, with the bias and the limits of agreement."""
    from ...backends.bland_altman import _compute

    sheet = panel.ctx.sheet
    labels = list(sheet.group_labels()) if sheet is not None else []
    opt = panel.collect()
    a_i, b_i = opt.method_a_index, opt.method_b_index
    if len(labels) < 2 or a_i == b_i or max(a_i, b_i) >= len(labels):
        return None
    rows = sheet.row_matrix([labels[a_i], labels[b_i]])
    both = ~np.any(np.isnan(rows), axis=1)
    a, b = rows[both, 0], rows[both, 1]
    if a.size < 3:
        return None
    with np.errstate(all="ignore"):
        values, _label = _compute(a, b, opt.calculate)
    good = np.isfinite(values)
    values, means = values[good], ((a + b) / 2)[good]
    if values.size < 3:
        return None
    bias, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    return {
        "lines": [{"x": [float(means.min()), float(means.max())], "y": [bias, bias], "colour": kit.ACCENT, "width": 1.6}],
        "levels": [{"y": bias + 1.96 * sd, "colour": kit.WARN, "dash": True, "text": f"upper limit {_g(bias + 1.96 * sd)}"},
                   {"y": bias - 1.96 * sd, "colour": kit.WARN, "dash": True, "text": f"lower limit {_g(bias - 1.96 * sd)}"}],
        "dots": [{"x": float(m), "y": float(v), "colour": kit.TEXT_DIM} for m, v in zip(means[:400], values[:400])],
        "caption": f"{values.size} pairs. Blue: the average gap, {_g(bias)}. Dashed: where 95% of gaps fall.",
    }


# ── grid drawings for cards ──────────────────────────────────────────────────

def grid_glyph(lit: str):
    """A 3 × 3 table with part of it lit: "column", "row", "all" (the whole
    table), "each" (every cell its own), or "three" (a column, a row and the
    table at once)."""

    def draw(p: QPainter, r: QRectF, colour: str) -> None:
        side = min(r.width(), r.height())
        cell = side / 3
        left, top = r.left() + (r.width() - side) / 2, r.top() + (r.height() - side) / 2
        soft = QColor(colour)
        soft.setAlpha(70)
        for row in range(3):
            for col in range(3):
                on = {"column": col == 1, "row": row == 1, "all": True, "each": (row + col) % 2 == 0,
                      "three": col == 1 or row == 1}.get(lit, False)
                box = QRectF(left + col * cell + 1, top + row * cell + 1, cell - 2, cell - 2)
                p.setPen(sk.pen(colour, 1))
                p.setBrush(soft if on else Qt.BrushStyle.NoBrush)
                p.drawRect(box)

    return draw
