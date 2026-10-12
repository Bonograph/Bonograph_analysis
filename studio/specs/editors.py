"""Small custom editors used by a few specs: a pick-the-variables chip set,
and the expected-counts table of the goodness-of-fit test."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QPushButton, QTableWidget, QTableWidgetItem

from ... import ui_kit as kit
from .. import kit as sk
from .models import _Rebuilding, _table_ss


def variable_chips(field: str = "selected_variables", title: str = "Use these variables", with_x: bool = False):
    """Every data set (and X, if asked) as a chip, all on to start. An empty list means all."""

    def signature(panel):
        return (("X",) if with_x else ()) + tuple(panel.ctx.names)

    def build(panel, lay):
        names = list(signature(panel)) or ["Set A", "Set B", "Set C"]
        chosen = [n for n in (panel.state.get(field) or []) if n in names]
        panel.state[field] = chosen
        group = sk.ChipGroup(title)
        for name in names:
            chip = group.add(sk.Chip(name, not chosen or name in chosen))

            def toggle(on, nm=name):
                current = [n for n in names if (n != nm and (not panel.state.get(field) or n in panel.state[field]))]
                if on:
                    current.append(nm)
                keep = [n for n in names if n in current]
                panel.assign({field: [] if len(keep) == len(names) else keep})

            chip.toggled.connect(toggle)
        lay.addWidget(group)
        lay.addWidget(sk.note("Each must hold numbers; one row per subject.", "faint"))

    return _Rebuilding(build, signature)


def observed_counts(ctx, index: int) -> list[float]:
    if ctx.sheet is None:
        return []
    labels = ctx.sheet.group_labels()
    if not 0 <= index < len(labels):
        return []
    return [float(v) for v in ctx.sheet.group_values(labels[index])]


def observed_rows(ctx, index: int) -> list[int]:
    """The sheet row (counting from 1) each observed count sits on: a row left blank is not a category."""
    if ctx.sheet is None:
        return []
    labels = ctx.sheet.group_labels()
    if not 0 <= index < len(labels):
        return []
    values = ctx.sheet.group_rows(labels[index], "average")
    return [int(i) + 1 for i in range(values.size) if values[i] == values[i]]


def expected_table(field: str = "expected_values"):
    """Observed counts beside the expected share typed for each row, with the
    total and a one-click 'all equal'."""

    def signature(panel):
        return (tuple(observed_counts(panel.ctx, int(panel.state.get("dataset_index") or 0))),
                bool(panel.state.get("expected_as_percent")))

    def build(panel, lay):
        observed, as_percent = signature(panel)
        if not observed:
            lay.addWidget(sk.note("The table of expected values appears once the sheet has counts in this data set.",
                                  "faint"))
            return
        current = list(panel.state.get(field) or [])
        previous = panel.ctx.extra.get("expected_as_percent")
        panel.ctx.extra["expected_as_percent"] = as_percent
        if as_percent and previous is False and len(current) == len(observed) and sum(current) > 0:
            # relative amounts become the percentages they stand for, so nothing typed is lost
            total_before = sum(current)
            current = [round(v / total_before * 100.0, 4) for v in current]
            panel.state[field] = current
        if len(current) != len(observed):
            current = [round(100.0 / len(observed), 4) if as_percent else 1.0] * len(observed)
            panel.state[field] = current
        table = QTableWidget(len(observed), 3)
        table.setHorizontalHeaderLabels(["Row", "Observed", "Expected, as " + ("%" if as_percent else "a share")])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(_table_ss())
        table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        on_rows = observed_rows(panel.ctx, int(panel.state.get("dataset_index") or 0))
        for r, obs in enumerate(observed):
            for c, text in ((0, f"Row {on_rows[r] if r < len(on_rows) else r + 1}"), (1, f"{obs:g}")):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table.setItem(r, c, item)
            table.setItem(r, 2, QTableWidgetItem(f"{current[r]:g}"))
        table.setMinimumHeight(min(260, 34 + 28 * len(observed)))
        lay.addWidget(table)
        total = sk.note("", "dim")
        lay.addWidget(total)

        def read(*_args):
            values = []
            for r in range(table.rowCount()):
                try:
                    values.append(float(table.item(r, 2).text()))
                except (ValueError, AttributeError):
                    values.append(0.0)
            s = sum(values)
            total.setText(f"Total {s:g}" + (" - should be 100%" if as_percent and abs(s - 100) > 1e-6 else ""))
            panel.assign({field: values})

        def equal():
            table.blockSignals(True)
            for r in range(table.rowCount()):
                table.item(r, 2).setText(f"{(100.0 / table.rowCount()) if as_percent else 1.0:g}")
            table.blockSignals(False)
            read()

        table.itemChanged.connect(read)
        button = QPushButton("Make them all equal")
        button.setStyleSheet(kit.quiet_button_ss())
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(equal)
        lay.addWidget(button)
        s = sum(current)
        total.setText(f"Total {s:g}" + (" - should be 100%" if as_percent and abs(s - 100) > 1e-6 else ""))

    return _Rebuilding(build, signature)


def is_2x2(ctx) -> bool:
    if ctx.sheet is None:
        return False
    try:
        _labels, matrix = ctx.sheet.first_replicate_matrix()
        rows = ~np.any(np.isnan(matrix), axis=1)
        return matrix[rows].shape == (2, 2)
    except Exception:  # noqa: BLE001
        return False


# ── a list of rules, each a short sentence ───────────────────────────────────

def rule_list(field: str, parts, *, add_text: str, blank: dict, options_for=None):
    """Rows of one sentence each - "keep rows where [Age] [is above] [40]" -
    stored as a list of dicts in ``field``. ``parts``: words (str) and
    (key, kind, choices) tuples with kind "pick", "num" or "text"; for "pick",
    ``choices`` is a list of (value, label) or a callable(ctx)."""
    from PySide6.QtWidgets import QHBoxLayout, QWidget

    def signature(panel):
        return (tuple(panel.ctx.names), len(panel.state.get(field) or []))

    def build(panel, lay):
        rules = [dict(r) for r in (panel.state.get(field) or [])]

        def store():
            panel.assign({field: [dict(r) for r in rules]})

        for index, rule in enumerate(rules):
            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            sentence = sk.Sentence()
            for part in parts:
                if isinstance(part, str):
                    sentence.words(part)
                    continue
                key, kind, choices = part
                if kind == "pick":
                    opts = choices(panel.ctx) if callable(choices) else choices
                    w = sk.InlineChoice(opts)
                    w.set_value(rule.get(key))
                    if rule.get(key) is None and opts:
                        rule[key] = opts[0][0]
                    w.activated.connect(lambda _i, k=key, b=w, r=rule: (r.__setitem__(k, b.value()), store()))
                elif kind == "num":
                    w = sk.InlineNumber(float(rule.get(key) or 0.0), decimals=6, width=74)
                    w.valueChanged.connect(lambda v, k=key, r=rule: (r.__setitem__(k, float(v)), store()))
                else:
                    w = sk.InlineText(str(rule.get(key) or ""), 150, choices or "")
                    w.textEdited.connect(lambda t, k=key, r=rule: (r.__setitem__(k, t), store()))
                sentence.add(w)
            row_lay.addWidget(sentence, 1)
            remove = QPushButton("×")
            remove.setFixedWidth(26)
            remove.setToolTip("Remove this rule")
            remove.setCursor(Qt.CursorShape.PointingHandCursor)
            remove.setStyleSheet(kit.quiet_button_ss())
            remove.clicked.connect(lambda _=False, i=index: (rules.pop(i), store()))
            row_lay.addWidget(remove, 0, Qt.AlignmentFlag.AlignTop)
            lay.addWidget(row)
        add = QPushButton(add_text)
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.setStyleSheet(kit.quiet_button_ss())
        add.clicked.connect(lambda: (rules.append(dict(blank)), store()))
        lay.addWidget(add)
        panel.state[field] = [dict(r) for r in rules]

    return _Rebuilding(build, signature)
