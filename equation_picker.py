"""Choosing an equation: shared by "Draw an equation" and "Simulated XY data".

The old picker was one long tree of collapsed categories, half of them
"Dose-response - Special, X is ..." variants and one ("Classic equations")
repeating equations listed elsewhere. This one works like the new-table
panel on the start screen:

  * a search box that matches names, formulas, parameter names (Km, EC50…)
    and the old names of retired duplicates;
  * ten families down the left, each with a glyph and a count;
  * the equations as two-line rows: the name, with a tag such as "rising"
    or "inhibition", and the formula under it;
  * a card for the chosen equation: its formula written out, what every
    parameter means, and a sketch of the curve's shape at the defaults.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from . import ui_kit as K
from .backends import function_library as F

KEY_ROLE = Qt.ItemDataRole.UserRole
DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1
TAG_ROLE = Qt.ItemDataRole.UserRole + 2
COUNT_ROLE = Qt.ItemDataRole.UserRole + 3
ALL = "All equations"
MONO = "Cascadia Mono"


class FunctionCatalog:
    """The equation library (Draw an equation, Simulated XY data)."""

    noun = "equations"

    def families(self):
        return F.FAMILIES

    def listed(self, family=None):
        return F.listed_equations(family)

    def has(self, key):
        return key in F.EQUATIONS

    def resolve(self, key):
        return F.resolve(key)

    def family_of(self, key):
        return F.family_of(key)

    def search(self, query):
        return F.search(query)

    def label(self, key):
        return F.EQUATIONS[key].label

    def formula(self, key):
        return F.EQUATIONS[key].equation

    def x_label(self, key):
        return F.EQUATIONS[key].x_label

    def params(self, key):
        """[(name, meaning, default or None)]"""
        return [(n, F.param_help(n), d) for n, d in F.EQUATIONS[key].params]

    def curve(self, key):
        """(x, y) sketching the shape, or None."""
        spec = F.EQUATIONS[key]
        lo, hi = F.suggested_x_range(key)
        x = np.linspace(lo, hi, 160)
        with np.errstate(all="ignore"):
            y = np.asarray(spec.fn(x, *[d for _n, d in spec.params]), dtype=float)
        return x, np.broadcast_to(y, x.shape).astype(float)


class FitModelCatalog:
    """The models Fit a curve can fit (backends.nonlinear_regression.MODELS),
    shown with the same families, meanings and sketches as the equation
    library."""

    noun = "models"
    FAMILIES = [
        ("Enzyme kinetics", "⊗", "Michaelis–Menten, allosteric, substrate inhibition"),
        ("Binding", "◔", "Saturation binding"),
        ("Dose-response", "∫", "Sigmoid curves, four and five parameters"),
        ("Lines & polynomials", "╱", "Straight lines and polynomials"),
        ("Exponential", "↘", "Growth and decay"),
        ("Peaks", "∩", "Gaussian peaks"),
    ]
    _FAMILY = {"Lines": "Lines & polynomials", "Polynomials": "Lines & polynomials"}
    _TAG = {"michaelis_menten": "velocity", "allosteric_sigmoidal": "cooperative", "substrate_inhibition": "inhibition",
            "hyperbola": "one site", "sigmoid_4pl": "symmetric", "sigmoid_5pl": "asymmetric",
            "exp_growth": "growth", "exp_plateau": "decay to a plateau"}
    # the library equation drawn as the sketch
    _SHAPE = {"michaelis_menten": "michaelis_menten", "allosteric_sigmoidal": "allosteric_sigmoidal",
              "substrate_inhibition": "substrate_inhibition", "hyperbola": "one_site_specific",
              "sigmoid_4pl": "dr_stim_4pl_log", "sigmoid_5pl": "dr_asymmetric_5pl_log", "linear": "straight_line",
              "quadratic": "poly_2", "cubic": "poly_3", "exp_growth": "exp_malthusian",
              "exp_plateau": "one_phase_decay", "gaussian": "gaussian"}
    _HELP = {
        "Vmax": "maximum velocity", "Km": "[substrate] giving half of Vmax", "Ki": "substrate inhibition constant",
        "K′ (half-saturation)": "[substrate] giving half of Vmax", "h (Hill)": "Hill coefficient (cooperativity)",
        "Bmax": "maximum specific binding", "Kd": "concentration giving half-maximal binding",
        "bottom": "plateau at low dose", "top": "plateau at high dose", "EC50": "dose giving a half response",
        "Hill slope": "steepness (1 = standard; negative for falling curves)",
        "S (asymmetry)": "asymmetry (1 = symmetric, as the 4PL)",
        "Amplitude": "peak height", "Mean": "centre of the peak", "SD": "spread (standard deviation)",
    }
    _POLY_HELP = {"linear": {"a": "slope", "b": "intercept"},
                  "quadratic": {"a": "coefficient of X²", "b": "coefficient of X", "c": "constant term"},
                  "cubic": {"a": "coefficient of X³", "b": "coefficient of X²", "c": "coefficient of X",
                            "d": "constant term"},
                  "exp_growth": {"a": "Y at X = 0", "b": "rate constant (doubling time = ln 2 / b)"},
                  "exp_plateau": {"a": "span of the decay", "b": "rate constant (half-life = ln 2 / b)",
                                  "c": "plateau it decays to"}}

    def __init__(self):
        from .backends.nonlinear_regression import MODELS

        self.models = MODELS

    def families(self):
        return self.FAMILIES

    def listed(self, family=None):
        return [k for k in self.models if family is None or self.family_of(k)[0] == family]

    def has(self, key):
        return key in self.models

    def resolve(self, key):
        return key

    def family_of(self, key):
        m = self.models[key]
        return self._FAMILY.get(m.category, m.category), self._TAG.get(key, "")

    def search(self, query):
        words = [w for w in query.lower().replace("-", " ").split() if w]
        found = []
        for key in self.models:
            text = " ".join([self.label(key), self.formula(key), *self.family_of(key), key.replace("_", " "),
                             " ".join(n for n, _h, _d in self.params(key))]).lower().replace("-", " ")
            if all(w in text for w in words):
                found.append(key)
        return found

    def label(self, key):
        return self.models[key].label

    def formula(self, key):
        return self.models[key].equation

    def x_label(self, key):
        m = self.models[key]
        return m.x_label + (" (log axis)" if getattr(m, "log_x", False) else "")

    def params(self, key):
        names = self.models[key].param_names
        extra = self._POLY_HELP.get(key, {})
        return [(n, extra.get(n) or self._HELP.get(n, ""), None) for n in names]

    def curve(self, key):
        shape = self._SHAPE.get(key)
        return FunctionCatalog().curve(shape) if shape in F.EQUATIONS else None


def pretty(formula: str) -> str:
    """Y = Vmax*X/(Km+X) -> Y = Vmax·X/(Km + X): the formula as it is written on paper."""
    out = formula.replace("*", "·").replace("+", " + ").replace("  ", " ")
    return out.replace("( ", "(").replace(" )", ")")


def _list_ss() -> str:
    return f"""
        QListWidget {{ background: {K.SUNKEN}; border: 1px solid {K.BORDER}; border-radius: {K.RADIUS}px;
                       outline: none; padding: 3px; color: {K.TEXT}; }}
        QListWidget::item {{ border-radius: 4px; }}
        QListWidget::item:hover {{ background: {K.HOVER}; }}
        QListWidget::item:selected {{ background: {K.ACCENT_SOFT}; }}
    """ + K.SCROLLBAR_SS


class _FamilyDelegate(QStyledItemDelegate):
    """Glyph, family name, and a count at the right."""

    def sizeHint(self, option, index):
        return QSize(0, 30)

    def paint(self, painter, option, index):
        widget = option.widget
        widget.style().drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, option, painter, widget)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        r = option.rect.adjusted(8, 0, -8, 0)
        painter.save()
        glyph = index.data(DETAIL_ROLE) or ""
        font = QFont(K.FONT)
        font.setPixelSize(K.MD)
        painter.setFont(font)
        painter.setPen(QColor(K.ACCENT if selected else K.TEXT_FAINT))
        painter.drawText(QRect(r.x(), r.y(), 18, r.height()), int(Qt.AlignmentFlag.AlignCenter), glyph)
        count = str(index.data(COUNT_ROLE) or "")
        small = QFont(font)
        small.setPixelSize(K.XS)
        count_w = QFontMetrics(small).horizontalAdvance(count) + 4
        if selected:
            font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(K.TEXT))
        name_rect = QRect(r.x() + 24, r.y(), r.width() - 24 - count_w, r.height())
        painter.drawText(name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         QFontMetrics(font).elidedText(str(index.data() or ""), Qt.TextElideMode.ElideRight,
                                                       name_rect.width()))
        painter.setFont(small)
        painter.setPen(QColor(K.TEXT_FAINT))
        painter.drawText(QRect(r.right() - count_w, r.y(), count_w, r.height()),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), count)
        painter.restore()


class _EquationDelegate(QStyledItemDelegate):
    """Name (with a tag at the right), and the formula underneath in a
    monospaced face."""

    def sizeHint(self, option, index):
        return QSize(0, 44)

    def paint(self, painter, option, index):
        widget = option.widget
        widget.style().drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, option, painter, widget)
        r = option.rect.adjusted(10, 5, -10, -5)
        painter.save()
        name_font = QFont(K.FONT)
        name_font.setPixelSize(K.MD)
        tag_font = QFont(K.FONT)
        tag_font.setPixelSize(K.XS)
        mono = QFont(MONO)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPixelSize(K.XS)
        tag = str(index.data(TAG_ROLE) or "")
        tag_w = QFontMetrics(tag_font).horizontalAdvance(tag) + 14 if tag else 0
        half = r.height() // 2
        name_rect = QRect(r.x(), r.y(), r.width() - tag_w - 6, half)
        painter.setFont(name_font)
        painter.setPen(QColor(K.TEXT))
        painter.drawText(name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         QFontMetrics(name_font).elidedText(str(index.data() or ""), Qt.TextElideMode.ElideRight,
                                                            name_rect.width()))
        if tag:
            painter.setFont(tag_font)
            pill = QRectF(r.right() - tag_w, r.y() + 2, tag_w, half - 4)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(K.BORDER_STRONG)))
            painter.drawRoundedRect(pill, 7, 7)
            painter.setPen(QColor(K.TEXT_DIM))
            painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), tag)
        painter.setFont(mono)
        painter.setPen(QColor(K.TEXT_FAINT))
        formula_rect = QRect(r.x(), r.y() + half, r.width(), half)
        painter.drawText(formula_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         QFontMetrics(mono).elidedText(str(index.data(DETAIL_ROLE) or ""), Qt.TextElideMode.ElideRight,
                                                       formula_rect.width()))
        painter.restore()


class ShapePreview(QWidget):
    """A sketch of the equation at its default parameters over its suggested
    X range - enough to tell a hyperbola from a sigmoid at a glance."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(190, 112)
        self._xy: tuple[np.ndarray, np.ndarray] | None = None
        self._x_label = "X"

    def show_curve(self, xy, x_label: str) -> None:
        self._xy = xy
        self._x_label = x_label
        self.update()

    def show_equation(self, key: str) -> None:
        """The library equation ``key`` (kept for callers that draw one directly)."""
        self.show_curve(FunctionCatalog().curve(key), F.EQUATIONS[key].x_label)

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(K.BORDER)))
        p.setBrush(QColor(K.SUNKEN))
        p.drawRoundedRect(rect, K.RADIUS, K.RADIUS)
        plot = rect.adjusted(12, 10, -10, -18)
        p.setPen(QPen(QColor(K.BORDER_STRONG), 1))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())
        p.drawLine(plot.bottomLeft(), plot.topLeft())
        font = QFont(K.FONT)
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QColor(K.TEXT_FAINT))
        p.drawText(QRectF(plot.left(), plot.bottom() + 2, plot.width(), 14),
                   int(Qt.AlignmentFlag.AlignCenter), self._x_label)
        if self._xy is None:
            return
        x, y = self._xy
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return
        y_lo, y_hi = float(np.min(y[ok])), float(np.max(y[ok]))
        if y_hi - y_lo < 1e-12 * max(1.0, abs(y_hi)):
            y_lo, y_hi = y_lo - 1, y_hi + 1  # a flat line sits in the middle
        pad = 0.08 * (y_hi - y_lo)
        y_lo, y_hi = y_lo - pad, y_hi + pad
        x_lo, x_hi = float(x[0]), float(x[-1])
        path = QPainterPath()
        started = False
        for xi, yi, good in zip(x, y, ok):
            if not good:
                started = False
                continue
            pt = QPointF(plot.left() + (xi - x_lo) / (x_hi - x_lo) * plot.width(),
                         plot.bottom() - (yi - y_lo) / (y_hi - y_lo) * plot.height())
            if started:
                path.lineTo(pt)
            else:
                path.moveTo(pt)
                started = True
        p.setPen(QPen(QColor(K.ACCENT), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()


class EquationPicker(QWidget):
    """Search + families + equations + a card for the chosen one."""

    chosen = Signal(str)  # equation key

    def __init__(self, current: str = "straight_line", parent=None, catalog=None):
        super().__init__(parent)
        self.catalog = cat = catalog or FunctionCatalog()
        self._family_names = [f[0] for f in cat.families()]
        first = cat.listed()[0]
        self._current = cat.resolve(current) if cat.has(current) else first
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.search = QLineEdit()
        self.search.setPlaceholderText(f"Search {cat.noun}: a name, a parameter (Km, EC50) or a shape (sigmoid, peak)")
        self.search.setClearButtonEnabled(True)
        self.search.setStyleSheet(K.input_ss())
        self.search.textChanged.connect(self._fill_equations)
        self.search.returnPressed.connect(self._choose_first)
        lay.addWidget(self.search)

        lists = QHBoxLayout()
        lists.setSpacing(8)
        self.families = QListWidget()
        self.families.setFixedWidth(196)
        self.families.setStyleSheet(_list_ss())
        self.families.setItemDelegate(_FamilyDelegate(self.families))
        self.families.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for name, glyph, blurb in [(ALL, "☰", f"Every one of the {cat.noun}")] + list(cat.families()):
            item = QListWidgetItem(f"All {cat.noun}" if name == ALL else name)
            item.setData(KEY_ROLE, name)
            item.setData(DETAIL_ROLE, glyph)
            item.setData(COUNT_ROLE, len(cat.listed(None if name == ALL else name)))
            item.setToolTip(blurb)
            self.families.addItem(item)
        self.families.currentRowChanged.connect(lambda _row: self._fill_equations())
        lists.addWidget(self.families)

        right = QVBoxLayout()
        right.setSpacing(0)
        self.equations = QListWidget()
        self.equations.setStyleSheet(_list_ss())
        self.equations.setItemDelegate(_EquationDelegate(self.equations))
        self.equations.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.equations.setUniformItemSizes(True)
        self.equations.currentItemChanged.connect(self._on_current_changed)
        right.addWidget(self.equations)
        self.empty = K.label("", size=K.SM, colour=K.TEXT_DIM, wrap=True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.hide()
        right.addWidget(self.empty)
        lists.addLayout(right, 1)
        lay.addLayout(lists, 1)

        # the chosen equation
        card = QFrame()
        K.panel(card, "EquationCard", K.RAISED, border=f"border: 1px solid {K.BORDER};", radius=K.RADIUS)
        card_lay = QHBoxLayout(card)
        card_lay.setContentsMargins(14, 10, 12, 10)
        card_lay.setSpacing(14)
        text = QVBoxLayout()
        text.setSpacing(3)
        self.name_label = K.label("", size=K.LG, weight=600, wrap=True)
        self.family_label = K.label("", size=K.XS, colour=K.TEXT_FAINT)
        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)
        self.formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.formula_label.setStyleSheet(
            f"color: {K.TEXT}; font-family: '{MONO}', Consolas, monospace; font-size: {K.SM}px;"
            " background: transparent; padding: 2px 0 4px 0;")
        self.params_grid = QGridLayout()
        self.params_grid.setHorizontalSpacing(10)
        self.params_grid.setVerticalSpacing(1)
        text.addWidget(self.name_label)
        text.addWidget(self.family_label)
        text.addWidget(self.formula_label)
        text.addLayout(self.params_grid)
        text.addStretch()
        card_lay.addLayout(text, 1)
        self.preview = ShapePreview()
        card_lay.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignTop)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        lay.addWidget(card)

        family = cat.family_of(self._current)[0]
        self.families.setCurrentRow(1 + self._family_names.index(family) if family in self._family_names else 0)
        self._show(self._current)

    # ---------------------------------------------------------------- API
    def current_key(self) -> str:
        return self._current

    def set_current(self, key: str) -> None:
        key = self.catalog.resolve(key)
        if not self.catalog.has(key):
            return
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        family = self.catalog.family_of(key)[0]
        self._current = key
        self.families.setCurrentRow(1 + self._family_names.index(family) if family in self._family_names else 0)
        self._fill_equations()
        self._show(key)

    # ---------------------------------------------------------------- internals
    def _fill_equations(self, *_args) -> None:
        query = self.search.text().strip()
        if query:
            keys = self.catalog.search(query)
        else:
            item = self.families.currentItem()
            family = item.data(KEY_ROLE) if item is not None else ALL
            keys = self.catalog.listed(None if family == ALL else family)
        self.equations.blockSignals(True)
        self.equations.clear()
        for key in keys:
            family, tag = self.catalog.family_of(key)
            label, formula = self.catalog.label(key), pretty(self.catalog.formula(key))
            row = QListWidgetItem(label)
            row.setData(KEY_ROLE, key)
            row.setData(DETAIL_ROLE, formula)
            row.setData(TAG_ROLE, (family if query else tag) or "")
            row.setToolTip(f"{label}\n{formula}")
            self.equations.addItem(row)
            if key == self._current:
                self.equations.setCurrentItem(row)
        self.equations.blockSignals(False)
        self.empty.setVisible(not keys)
        self.equations.setVisible(bool(keys))
        if not keys:
            self.empty.setText(f"Nothing matches “{query}”. Try a parameter such as Km or EC50, "
                               "or a shape such as sigmoid or decay.")
        if self.equations.currentItem() is not None:
            self.equations.scrollToItem(self.equations.currentItem())

    def _choose_first(self) -> None:
        if self.equations.count():
            self.equations.setCurrentRow(0)
            self.equations.setFocus()

    def _on_current_changed(self, item, _previous) -> None:
        if item is None:
            return
        key = item.data(KEY_ROLE)
        if key and key != self._current:
            self._current = key
            self._show(key)
            self.chosen.emit(key)

    def _show(self, key: str) -> None:
        cat = self.catalog
        family, tag = cat.family_of(key)
        x_label = cat.x_label(key)
        self.name_label.setText(cat.label(key))
        self.family_label.setText(" · ".join(part for part in (family, tag, f"X is {x_label}"
                                                                     if x_label != "X" else "") if part))
        self.formula_label.setText(pretty(cat.formula(key)))
        while self.params_grid.count():
            widget = self.params_grid.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for row, (name, meaning, default) in enumerate(cat.params(key)):
            self.params_grid.addWidget(K.label(name, size=K.SM, weight=600), row, 0)
            text = meaning or ""
            if default is not None:
                text = f"{text}  (default {default:g})" if text else f"default {default:g}"
            self.params_grid.addWidget(K.label(text, size=K.SM, colour=K.TEXT_DIM), row, 1)
        self.params_grid.setColumnStretch(1, 1)
        try:
            xy = cat.curve(key)
        except Exception:  # noqa: BLE001 - a sketch is optional
            xy = None
        self.preview.show_curve(xy, x_label)
