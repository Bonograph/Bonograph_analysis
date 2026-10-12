"""
Pull usable numeric data straight out of a *live* InfiniteSpreadsheetGrid.

This reads whatever the user actually typed into the grid's sparse
CellStore - never the full (dynamically-growing, mostly-empty) Qt model
dimensions - so extraction cost is proportional to cells the user
actually filled in. That makes this work the same whether a sheet has
ten rows or a hundred thousand, and it always looks at whichever grid
object is passed in at call time - nothing here is cached at import or
app-startup time - so a sheet added a second ago via the + beside Sheets
works exactly like every other one, with no special-casing needed.

spreadsheet/Infinite_spreadsheet.py is intentionally left untouched (see
main_sp.py), so this module only ever reads its public attributes/methods
plus the CellStore's own ``_data`` dict - CellStore has no public
enumeration API, and adding one would mean touching that file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_float(text: str) -> float:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class DataColumn:
    """One Y column - one replicate of one group - with its raw values
    (NaN wherever a cell was blank or not a number)."""

    group_index: int
    group_label: str
    replicate: int
    values: np.ndarray


@dataclass
class SheetData:
    """Everything extracted from one sheet: the shared X column plus
    every populated Y column, grouped by which group they belong to."""

    x: np.ndarray
    columns: list[DataColumn]
    n_rows: int
    # What was typed in the X column, row by row (e.g. "Under 35") - lets
    # graphs name rows by category text, which the numeric ``x`` can't hold.
    row_names: list[str] | None = None

    def __post_init__(self) -> None:
        # an infinity typed into a cell ("inf", or a number too large to hold) is not a measurement:
        # it is read as a blank here, once, so that no analysis has to defend itself against it
        x = np.asarray(self.x, dtype=float)
        if x.size and not np.all(np.isfinite(x) | np.isnan(x)):
            self.x = np.where(np.isinf(x), np.nan, x)
        for column in self.columns:
            values = np.asarray(column.values, dtype=float)
            if values.size and np.any(np.isinf(values)):
                column.values = np.where(np.isinf(values), np.nan, values)

    def group_labels(self) -> list[str]:
        """Group labels in the order their columns first appear."""
        seen: list[str] = []
        for col in self.columns:
            if col.group_label not in seen:
                seen.append(col.group_label)
        return seen

    def _group_columns(self, label: str) -> list["DataColumn"]:
        return sorted(
            (c for c in self.columns if c.group_label == label),
            key=lambda c: c.replicate,
        )

    def group_replicate_matrix(self, label: str) -> tuple[list[int], np.ndarray]:
        """Replicate numbers and their row-aligned values (n_rows x
        n_replicates, NaN where a cell was blank) for one group - the
        shape every subcolumn-aware calculation starts from."""
        cols = self._group_columns(label)
        if not cols:
            return [], np.empty((0, 0))
        reps = [c.replicate for c in cols]
        matrix = np.column_stack([c.values for c in cols])
        return reps, matrix

    def group_replicate_arrays(self, label: str) -> dict[int, np.ndarray]:
        """Each replicate (sub)column of one group as its own,
        NaN-stripped 1-D array - for "treat every subcolumn separately"."""
        cols = self._group_columns(label)
        return {c.replicate: c.values[~np.isnan(c.values)] for c in cols}

    def group_values(self, label: str, mode: str = "average") -> np.ndarray:
        """One group's values as a single 1-D array, combining replicate
        (sub)columns, the way a "Subcolumns" option would:

        - "average" (default): average the replicate columns *within
          each row* first, then treat each row's average as one data
          point. This is the correct default for duplicate/triplicate
          data - a group with 5 rows x 2 replicate columns yields 5
          values here, not 10, so it isn't silently double- or
          triple-counted.
        - "pooled": every raw value from every replicate column,
          flattened together into one long list, ignoring row
          structure entirely - only correct when replicates really are
          independent observations rather than repeats of the same row.
        - "separate": not handled here - see ``group_replicate_arrays``,
          since that mode needs one array *per* replicate column, not one
          combined array.
        """
        _reps, matrix = self.group_replicate_matrix(label)
        if matrix.size == 0:
            return np.array([])

        if mode == "pooled" or matrix.shape[1] == 1:
            flat = matrix.flatten()
            return flat[~np.isnan(flat)]

        if mode == "average":
            row_has_data = ~np.all(np.isnan(matrix), axis=1)
            with np.errstate(invalid="ignore"):
                row_means = np.nanmean(matrix[row_has_data], axis=1)
            return row_means

        raise ValueError(f"Unknown subcolumn mode: {mode!r}")

    def group_rows(self, label: str, mode: str = "first") -> np.ndarray:
        """One value per sheet row for a group, with a blank cell kept as NaN.

        ``group_values`` drops blanks, so after a blank every later value
        sits one place higher than the row it was typed on - harmless for a
        column that is a sample on its own, wrong for anything that pairs
        values row by row with X or with another data set (a time with its
        event code and covariates, a before with its after). This keeps each
        value on its own row.

        - "first" (default): the first replicate column - for codes and
          anything else that must not be averaged.
        - "average": the mean of the row's replicates (NaN when all are blank).
        """
        _reps, matrix = self.group_replicate_matrix(label)
        if matrix.size == 0:
            return np.full(self.n_rows, np.nan)
        if mode == "average" and matrix.shape[1] > 1:
            filled = ~np.all(np.isnan(matrix), axis=1)
            out = np.full(matrix.shape[0], np.nan)
            out[filled] = np.nanmean(matrix[filled], axis=1)
            return out
        if mode not in ("first", "average"):
            raise ValueError(f"Unknown row mode: {mode!r}")
        return matrix[:, 0].astype(float, copy=True)

    def xy_points(self, label: str, mode: str = "average") -> tuple[np.ndarray, np.ndarray]:
        """X paired with one group's Y, each Y with the X of its own row;
        points with a blank X or Y are dropped only after the pairing.

        - "average" (default): one point per row, the mean of its replicates.
        - "each": one point per filled replicate cell, at its row's X.
        """
        _reps, matrix = self.group_replicate_matrix(label)
        if matrix.size == 0:
            return np.array([]), np.array([])
        n = min(self.x.size, matrix.shape[0])
        x = np.asarray(self.x[:n], dtype=float)
        if mode == "each" and matrix.shape[1] > 1:
            xs, ys = np.repeat(x, matrix.shape[1]), matrix[:n].ravel()
        else:
            xs, ys = x, self.group_rows(label, "average")[:n]
        keep = ~np.isnan(xs) & ~np.isnan(ys)
        return xs[keep], ys[keep]

    def row_matrix(self, labels: list[str], mode: str = "average") -> np.ndarray:
        """Rows x data sets: each group's value on every sheet row, blanks
        kept as NaN - what an analysis of matched rows (the same subject
        under several conditions, the same sample by two methods) starts from."""
        columns = [self.group_rows(label, mode) for label in labels]
        length = max([c.size for c in columns] + [0])
        out = np.full((length, len(columns)), np.nan)
        for j, column in enumerate(columns):
            out[: column.size, j] = column
        return out

    def all_group_arrays(self, mode: str = "average") -> dict[str, np.ndarray]:
        return {label: self.group_values(label, mode) for label in self.group_labels()}

    def first_replicate_matrix(self) -> tuple[list[str], np.ndarray]:
        """The first replicate column of each group, row-aligned (no NaN
        stripping) - what a real matrix-shaped analysis (contingency
        tables, PCA, clustering, multiple regression) needs, since those
        require the columns to line up row-by-row rather than being
        pooled into one long list per group."""
        first_reps = [c for c in self.columns if c.replicate == 1]
        labels = [c.group_label for c in first_reps]
        if not first_reps:
            return [], np.empty((0, 0))
        matrix = np.column_stack([c.values for c in first_reps])
        return labels, matrix

    def full_variable_matrix(self) -> tuple[list[str], np.ndarray]:
        """X plus every group's first replicate column, row-aligned (no
        NaN stripping) - the "one row per observation, one column per
        variable" shape multivariate analyses (correlation matrix,
        multiple regression) need, with X included as a variable."""
        labels, group_matrix = self.first_replicate_matrix()
        n = min(self.x.size, group_matrix.shape[0]) if group_matrix.size else self.x.size
        if group_matrix.size == 0:
            matrix = self.x[:n].reshape(-1, 1) if n else np.empty((0, 1))
        else:
            matrix = np.column_stack([self.x[:n], group_matrix[:n]])
        return ["X"] + labels, matrix

    def paired_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """X paired with the first data column's Y, row-aligned and
        stripped of rows where either side is missing - what simple XY
        analyses (regression, correlation) want."""
        if not self.columns:
            return np.array([]), np.array([])
        y = self.columns[0].values
        n = min(len(self.x), len(y))
        x, y = self.x[:n], y[:n]
        mask = ~np.isnan(x) & ~np.isnan(y)
        return x[mask], y[mask]


def extract_sheet_data(grid) -> SheetData:
    """``grid`` is an InfiniteSpreadsheetGrid instance (see
    spreadsheet/Infinite_spreadsheet.py). Column 0 is the shared X
    column; columns 1.. belong to groups of ``grid.replicates`` columns
    each, exactly as that module lays them out."""
    store = grid.store
    cells = store._data  # sparse {(row, col): CellData} - see module docstring

    # size the sheet by cells that hold something - displaying or clearing
    # a cell can leave an empty entry behind, which isn't data
    filled = [key for key, cell in cells.items() if cell.code]
    if not filled:
        return SheetData(x=np.array([]), columns=[], n_rows=0)

    max_row = max(r for r, _ in filled) + 1
    max_col = max(c for _, c in filled) + 1

    dense = np.full((max_row, max_col), np.nan, dtype=float)
    for (r, c), cell in cells.items():
        if cell.code:
            dense[r, c] = _to_float(cell.code)

    x = dense[:, 0].copy() if max_col > 0 else np.array([])

    columns: list[DataColumn] = []
    for col in range(1, max_col):
        values = dense[:, col]
        if np.all(np.isnan(values)):
            continue
        group_idx, letter, rep_num, _anchor = grid._col_group_info(col)
        title = grid.get_group_title(group_idx).strip()
        # The title cell isn't blank by default - it starts pre-filled
        # with the literal placeholder text ("Name") for the user to
        # type over - so that placeholder means "not renamed", not "the
        # group is actually called Name".
        placeholder = getattr(grid, "group_subheader", "Name")
        if title and title != placeholder:
            label = title
        elif hasattr(grid, "group_header_text"):  # "Variable A" / "A" / "Set A", as the sheet's header shows
            label = grid.group_header_text(letter)
        else:
            label = f"Set {letter}"
        columns.append(DataColumn(group_idx, label, rep_num, values))

    row_names = [cells[(r, 0)].code.strip() if (r, 0) in cells else "" for r in range(max_row)]
    return SheetData(x=x, columns=columns, n_rows=max_row, row_names=row_names)
