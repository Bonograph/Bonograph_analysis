"""Editors shared by the regression models (multiple linear, multiple
logistic, Cox): the variables in play, categories and their reference
levels, which terms a simpler model drops, and tables of values to predict
for. Each is a ``Custom`` control that rebuilds itself when the set of
predictors changes."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ... import ui_kit as kit
from .. import kit as sk
from ..spec import Context


def all_variables(ctx: Context) -> list[str]:
    """X and every data set - what the multiple regressions choose from."""
    if ctx.sheet is not None:
        try:
            return list(ctx.sheet.full_variable_matrix()[0])
        except Exception:  # noqa: BLE001
            pass
    return ["X"] + (ctx.names or ["Set A", "Set B", "Set C"])


def levels_of(ctx: Context, include_x: bool = True) -> dict[str, list[float]]:
    """Variables with two to six distinct values, read as categories."""
    out: dict[str, list[float]] = {}
    if ctx.sheet is None:
        return out
    try:
        if include_x:
            labels, matrix = ctx.sheet.full_variable_matrix()
            columns = {label: matrix[:, i] for i, label in enumerate(labels)}
        else:
            columns = {label: ctx.sheet.group_values(label) for label in ctx.sheet.group_labels()}
    except Exception:  # noqa: BLE001
        return out
    for label, values in columns.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        levels = sorted(set(finite.tolist())) if finite.size else []
        if 2 <= len(levels) <= 6:
            out[label] = levels
    return out


def outcome_of(state: dict, variables: list[str], field: str = "outcome_index") -> str:
    i = state.get(field, -1)
    if not variables:
        return ""
    i = len(variables) - 1 if i is None or i < 0 or i >= len(variables) else i
    return variables[i]


class _Rebuilding(QWidget):
    """A custom editor that redraws itself when its inputs change."""

    def __init__(self, build, signature):
        super().__init__()
        self._build, self._signature, self._last = build, signature, object()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(4)

    def refresh(self, panel) -> None:
        sig = self._signature(panel)
        if sig == self._last:
            return
        self._last = sig
        while self.lay.count():
            item = self.lay.takeAt(0)
            old = item.widget()
            if old is not None:
                # taken out of sight at once: a widget only marked for deletion stays drawn
                # (under the new ones) until the event loop it was made in comes round again
                old.hide()
                old.setParent(None)
                old.deleteLater()
        self._build(panel, self.lay)


# ── categories ──────────────────────────────────────────────────────────────

def reference_editor(predictors, include_x: bool = True):
    """One row per categorical predictor: compare its levels with the lowest
    (automatic) or with a level you choose."""

    def signature(panel):
        preds = predictors(panel)
        lv = levels_of(panel.ctx, include_x)
        return tuple((p, tuple(lv[p])) for p in preds if p in lv)

    def build(panel, lay):
        cats = [(p, tuple(levels)) for p, levels in signature(panel)]
        if not cats:
            lay.addWidget(sk.note("No predictor has only a few distinct values, so every one is treated as a "
                                  "number.", "faint"))
            panel.state["reference_levels"], panel.state["auto_reference"] = {}, {}
            return
        lay.addWidget(sk.note("Predictors with six or fewer distinct values are read as categories; each level is "
                              "compared with a reference level."))
        auto = dict(panel.state.get("auto_reference") or {})
        chosen = dict(panel.state.get("reference_levels") or {})
        for label, levels in cats:
            options = [("", "the lowest level (automatic)")] + [(f"{v:g}", f"level {v:g}") for v in levels]
            box = sk.boxed_combo(options)
            box.set_value("" if auto.get(label, True) else chosen.get(label, ""))

            def pick(_i, lbl=label, b=box):
                a = dict(panel.state.get("auto_reference") or {})
                r = dict(panel.state.get("reference_levels") or {})
                a[lbl] = not b.value()
                if b.value():
                    r[lbl] = b.value()
                else:
                    r.pop(lbl, None)
                panel.assign({"auto_reference": a, "reference_levels": r})

            box.activated.connect(pick)
            lay.addWidget(sk.Row(f"{label}: compare with", box))
            auto.setdefault(label, True)
        panel.state["auto_reference"] = auto

    return _Rebuilding(build, signature)


# ── a simpler model ─────────────────────────────────────────────────────────

def drop_editor(predictors, field: str):
    def signature(panel):
        return tuple(predictors(panel))

    def build(panel, lay):
        group = sk.ChipGroup("Leave out of the simpler model")
        kept = [p for p in (panel.state.get(field) or []) if p in signature(panel)]
        panel.state[field] = kept
        for label in signature(panel):
            chip = group.add(sk.Chip(label, label in kept))

            def toggle(on, lbl=label):
                current = [p for p in (panel.state.get(field) or []) if p != lbl]
                panel.assign({field: current + ([lbl] if on else [])})

            chip.toggled.connect(toggle)
        lay.addWidget(group)

    return _Rebuilding(build, signature)


def subset_editor(items, field: str, title: str):
    """Chips for choosing some of a list. An empty choice means all of them,
    so unticking the last chip ticks every one again."""

    def signature(panel):
        return tuple(items(panel))

    def build(panel, lay):
        names = list(signature(panel))
        chosen = [n for n in (panel.state.get(field) or []) if n in names]
        panel.state[field] = chosen
        group = sk.ChipGroup(title)
        chips = {name: group.add(sk.Chip(name, not chosen or name in chosen)) for name in names}

        def changed(_on=False):
            keep = [n for n in names if chips[n].isChecked()]
            if not keep:
                for chip in chips.values():
                    chip.blockSignals(True)
                    chip.setChecked(True)
                    chip.blockSignals(False)
                    chip._label(True)
            panel.assign({field: [] if len(keep) in (0, len(names)) else keep})

        for chip in chips.values():
            chip.toggled.connect(changed)
        lay.addWidget(group)

    return _Rebuilding(build, signature)


# ── tables of values ────────────────────────────────────────────────────────

def _table_ss() -> str:
    return f"""
        QTableWidget {{ background: {kit.SUNKEN}; color: {kit.TEXT}; gridline-color: {kit.BORDER};
            border: 1px solid {kit.BORDER}; border-radius: {kit.RADIUS}px; }}
        QHeaderView::section {{ background: {kit.RAISED}; color: {kit.TEXT_DIM}; border: none;
            border-right: 1px solid {kit.BORDER}; border-bottom: 1px solid {kit.BORDER}; padding: 4px 6px; }}
        QTableCornerButton::section {{ background: {kit.RAISED}; border: none; }}
        QTableWidget QLineEdit {{ background: {kit.SUNKEN}; color: {kit.TEXT}; border: 1px solid {kit.ACCENT}; }}
    """


def _small_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(kit.quiet_button_ss())
    return b


def scenario_editor(predictors, field: str, *, named: bool = False, baseline: bool = False, noun: str = "scenario"):
    """A table: one row per case to predict for, one column per predictor.
    ``named`` adds a name column and ``baseline`` a draw-the-baseline tick (Cox curves)."""

    def signature(panel):
        return tuple(predictors(panel))

    def build(panel, lay):
        preds = list(signature(panel))
        rows = list(panel.state.get(field) or [])
        extra = (["Name"] if named else []) + (["With baseline"] if baseline else [])
        table = QTableWidget(0, len(extra) + len(preds))
        table.setHorizontalHeaderLabels(extra + preds)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(_table_ss())
        table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)

        def row_values(r):
            if named or baseline:
                values = rows[r].get("covariate_values", {}) if isinstance(rows[r], dict) else {}
            else:
                values = rows[r]
            return values

        def write_back(*_args):
            out = []
            for r in range(table.rowCount()):
                values = {}
                for c, label in enumerate(preds, start=len(extra)):
                    item = table.item(r, c)
                    try:
                        values[label] = float(item.text()) if item and item.text().strip() else 0.0
                    except ValueError:
                        values[label] = 0.0
                if named or baseline:
                    title_item = table.item(r, 0) if named else None
                    check = table.cellWidget(r, 1 if named else 0) if baseline else None
                    out.append({"title": title_item.text() if title_item else f"{noun.capitalize()} {r + 1}",
                                "include_baseline": bool(check.isChecked()) if check else False,
                                "covariate_values": values})
                else:
                    out.append(values)
            panel.assign({field: out})

        def add_row(values=None, title=None, with_base=False):
            r = table.rowCount()
            table.insertRow(r)
            if named:
                table.setItem(r, 0, QTableWidgetItem(title or f"{noun.capitalize()} {r + 1}"))
            if baseline:
                check = QCheckBox()
                check.setChecked(with_base)
                check.toggled.connect(write_back)
                table.setCellWidget(r, 1 if named else 0, check)
            for c, label in enumerate(preds, start=len(extra)):
                v = (values or {}).get(label, 0.0)
                table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

        table.blockSignals(True)
        for r in range(len(rows)):
            meta = rows[r] if isinstance(rows[r], dict) else {}
            add_row(row_values(r), meta.get("title"), meta.get("include_baseline", False))
        table.blockSignals(False)
        table.itemChanged.connect(write_back)
        table.setMinimumHeight(110)
        lay.addWidget(table)

        buttons = QHBoxLayout()
        plus = _small_button(f"+ Add a {noun}")
        minus = _small_button("Remove the last")
        plus.clicked.connect(lambda: (table.blockSignals(True), add_row(), table.blockSignals(False), write_back()))
        minus.clicked.connect(lambda: (table.removeRow(table.rowCount() - 1) if table.rowCount() else None,
                                       write_back()))
        buttons.addWidget(plus)
        buttons.addWidget(minus)
        buttons.addStretch()
        holder = QWidget()
        holder.setLayout(buttons)
        buttons.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(holder)

    return _Rebuilding(build, signature)
