"""One-off generator for the spin_up*.png / spin_down*.png triangles
used by options_dialog_common.SPIN_SS. Re-run this only if the arrow
size/color needs to change - the PNGs are committed, not built on the
fly, since building them needs a live QApplication."""

import sys
from pathlib import Path

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygon
from PySide6.QtWidgets import QApplication


def make_triangle(path: Path, direction: str, color: str, w: int = 9, h: int = 5, scale: int = 2) -> None:
    pix = QPixmap(w * scale, h * scale)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    qcolor = QColor(color)
    painter.setBrush(qcolor)
    painter.setPen(qcolor)
    ww, hh = w * scale, h * scale
    if direction == "up":
        points = [QPoint(0, hh), QPoint(ww, hh), QPoint(ww // 2, 0)]
    else:
        points = [QPoint(0, 0), QPoint(ww, 0), QPoint(ww // 2, hh)]
    painter.drawPolygon(QPolygon(points))
    painter.end()
    pix.save(str(path), "PNG")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    here = Path(__file__).resolve().parent
    make_triangle(here / "spin_up.png", "up", "#cccccc")
    make_triangle(here / "spin_down.png", "down", "#cccccc")
    make_triangle(here / "spin_up_disabled.png", "up", "#888888")
    make_triangle(here / "spin_down_disabled.png", "down", "#888888")
    print("wrote spin arrow assets to", here)
