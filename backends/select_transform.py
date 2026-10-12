"""Pick the variables to carry forward and convert them.
Produces a genuinely new sheet: the selected original variables
(filtered by row criteria) plus any standard/custom transformed
columns - not just a preview."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register

_STANDARD_TRANSFORMS = {
    "log10": ("Log10(Y)", lambda v, k: np.log10(v), lambda v: v > 0, "values must be > 0"),
    "ln": ("Ln(Y)", lambda v, k: np.log(v), lambda v: v > 0, "values must be > 0"),
    "sqrt": ("Square root of Y", lambda v, k: np.sqrt(v), lambda v: v >= 0, "values must be >= 0"),
    "reciprocal": ("1 / Y", lambda v, k: 1.0 / v, lambda v: v != 0, "values must be nonzero"),
    "zscore": ("Z-score", None, lambda v: True, ""),  # handled specially - needs the whole array
    "power": ("Y ^ K", lambda v, k: np.power(v, k), lambda v: True, ""),
    "scale": ("K * Y", lambda v, k: v * k, lambda v: True, ""),
    "offset": ("Y + K", lambda v, k: v + k, lambda v: True, ""),
}
_USES_K = {"power", "scale", "offset"}

_LOGIC_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_ALLOWED_FUNCS = {
    "log": np.log, "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt, "exp": np.exp,
    "abs": np.abs, "sin": np.sin, "cos": np.cos, "tan": np.tan, "min": np.minimum, "max": np.maximum,
}
_SAFE_EXPR_RE = re.compile(r"^[A-Za-z0-9_\s\.\+\-\*/\(\),]+$")


@dataclass
class SelectTransformOptions:
    # new variables
    standard_transforms: list[dict] = field(default_factory=list)  # {"variable", "transform", "k", "title"}
    custom_transforms: list[dict] = field(default_factory=list)  # {"expression", "title"}
    recipe_variables: list[dict] = field(default_factory=list)  # {"variable", "steps", "standardise", "title"}
    # rows to keep
    row_selection_mode: str = "all"  # "all" | "range"
    row_from: int = 1
    row_to: int = 20
    use_variable_vs_value: bool = False
    variable_vs_value_criteria: list[dict] = field(default_factory=list)  # {"variable", "logic", "value"}
    use_variable_vs_variable: bool = False
    variable_vs_variable_criteria: list[dict] = field(default_factory=list)  # {"variable1", "logic", "variable2"}
    # variables to keep
    included_variables: list[str] = field(default_factory=list)  # empty = all


def _variable_array(name: str, all_labels: list[str], all_matrix: np.ndarray) -> np.ndarray:
    if name in all_labels:
        return all_matrix[:, all_labels.index(name)]
    raise AnalysisError(f"'{name}' isn't a variable on this sheet.")


def _evaluate_custom(expression: str, all_labels: list[str], all_matrix: np.ndarray) -> np.ndarray:
    if not _SAFE_EXPR_RE.match(expression):
        raise AnalysisError(f"Custom transform \"{expression}\" contains characters that aren't allowed (only variable names, numbers, and + - * / ( ) , . are permitted).")
    namespace = dict(_ALLOWED_FUNCS)
    for i, label in enumerate(all_labels):
        if label.isidentifier():
            namespace[label] = all_matrix[:, i]
    try:
        result = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307 - sandboxed: no builtins, whitelisted names only, and _SAFE_EXPR_RE already rejected anything but names/numbers/arithmetic
    except Exception as exc:  # noqa: BLE001
        raise AnalysisError(f"Couldn't evaluate custom transform \"{expression}\": {exc}") from None
    return np.asarray(result, dtype=float)


@register("select_transform")
def run(data: SheetData, options: SelectTransformOptions | None = None) -> AnalysisResult:
    opt = options or SelectTransformOptions()
    all_labels, all_matrix = data.full_variable_matrix()
    if not all_labels:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    n = all_matrix.shape[0]

    # ---- Select rows ----
    mask = np.ones(n, dtype=bool)
    if opt.row_selection_mode == "range":
        idx = np.arange(1, n + 1)
        mask &= (idx >= opt.row_from) & (idx <= opt.row_to)
    if opt.use_variable_vs_value:
        for crit in opt.variable_vs_value_criteria:
            var, logic, value = crit.get("variable"), crit.get("logic", "="), crit.get("value")
            if not var or value is None:
                continue
            arr = _variable_array(var, all_labels, all_matrix)
            mask &= _LOGIC_OPS.get(logic, _LOGIC_OPS["="])(arr, float(value))
    if opt.use_variable_vs_variable:
        for crit in opt.variable_vs_variable_criteria:
            var1, logic, var2 = crit.get("variable1"), crit.get("logic", "="), crit.get("variable2")
            if not var1 or not var2:
                continue
            arr1 = _variable_array(var1, all_labels, all_matrix)
            arr2 = _variable_array(var2, all_labels, all_matrix)
            mask &= _LOGIC_OPS.get(logic, _LOGIC_OPS["="])(arr1, arr2)

    if not np.any(mask):
        raise AnalysisError("No rows match the selected row criteria.")

    filtered_matrix = all_matrix[mask]

    # ---- Select columns ---- ("X" is excluded here since it already becomes the new sheet's x-axis, not a Y column)
    if opt.included_variables:
        keep_idx = [i for i, lbl in enumerate(all_labels) if lbl in opt.included_variables and lbl != "X"]
        if not keep_idx:
            raise AnalysisError("Keep at least one variable.")
        out_labels = [all_labels[i] for i in keep_idx]
        out_columns = {lbl: filtered_matrix[:, i].tolist() for lbl, i in zip(out_labels, keep_idx)}
    else:
        out_columns = {lbl: filtered_matrix[:, i].tolist() for i, lbl in enumerate(all_labels) if lbl != "X"}

    # ---- Standard transforms ----
    notes = []
    for t in opt.standard_transforms:
        var, kind, k, title = t.get("variable"), t.get("transform"), float(t.get("k", 1.0)), t.get("title") or ""
        if not var or kind not in _STANDARD_TRANSFORMS:
            continue
        raw = _variable_array(var, all_labels, all_matrix)[mask]
        display_name, fn, valid_mask, requirement = _STANDARD_TRANSFORMS[kind]
        col_title = title or f"{display_name.replace('Y', var)}"
        if kind == "zscore":
            sd = np.std(raw, ddof=1) if raw.size > 1 else 0.0
            if sd == 0:
                notes.append(f"Skipped Z-score of '{var}': its SD is zero.")
                continue
            transformed = (raw - np.mean(raw)) / sd
        else:
            ok = valid_mask(raw)
            if not np.all(ok):
                notes.append(f"'{var}' had values that don't satisfy {requirement} for {display_name}; those rows became blank in '{col_title}'.")
            transformed = np.full(raw.shape, np.nan)
            transformed[ok] = fn(raw[ok], k)
        out_columns[col_title] = transformed.tolist()

    # ---- New variables, each a chain of steps (and, if asked, a z-score at the end) ----
    from .. import recipes

    for position, t in enumerate(opt.recipe_variables, 1):
        var, steps = t.get("variable"), list(t.get("steps") or [])
        if not var:
            continue
        wrong = recipes.check(steps)
        if wrong:
            raise AnalysisError(f"New variable {position}: {wrong}")
        raw = _variable_array(var, all_labels, all_matrix)[mask]
        made = recipes.apply_many(steps, raw, x=raw, y=raw)
        lost = int(np.sum(np.isnan(made) & ~np.isnan(raw)))
        col_title = (t.get("title") or "").strip() or f"{var}: {recipes.describe(steps)}"
        if lost:
            notes.append(f"{lost} value(s) of '{var}' have no answer under {recipes.describe(steps)}; they are blank in '{col_title}'.")
        if t.get("standardise"):
            good = made[~np.isnan(made)]
            sd = float(np.std(good, ddof=1)) if good.size > 1 else 0.0
            if sd == 0:
                notes.append(f"'{col_title}' was not turned into z-scores: its values do not vary.")
            else:
                made = (made - float(np.mean(good))) / sd
        out_columns[col_title] = made.tolist()

    # ---- Custom transforms ----
    for t in opt.custom_transforms:
        expr, title = t.get("expression", "").strip(), t.get("title") or ""
        if not expr:
            continue
        full_result = np.asarray(_evaluate_custom(expr, all_labels, all_matrix), dtype=float)
        if full_result.ndim == 0:  # a constant ("7", "pi"): the same value on every row
            full_result = np.full(n, float(full_result))
        if full_result.shape[0] != n:
            raise AnalysisError(f"Custom transform \"{expr}\" didn't produce one value per row.")
        out_columns[title or expr] = full_result[mask].tolist()

    if not out_columns:
        raise AnalysisError("The new sheet would be empty: keep a variable, or make a new one.")

    x_title = "X"
    x_values = data.x[mask][: filtered_matrix.shape[0]].tolist() if data.x.size >= n else list(range(1, filtered_matrix.shape[0] + 1))

    new_sheet = NewSheetData(suggested_name="Selected variables", x_title=x_title, x_values=[float(v) for v in x_values], columns=out_columns)
    preview_rows = [[f"{x_values[i]:.4g}" if i < len(x_values) else ""] + [f"{out_columns[c][i]:.4g}" if i < len(out_columns[c]) and out_columns[c][i] == out_columns[c][i] else "" for c in out_columns] for i in range(min(filtered_matrix.shape[0], 50))]
    preview_table = ResultTable(title="Preview", headers=[x_title] + list(out_columns.keys()), rows=preview_rows)

    summary = f"{filtered_matrix.shape[0]} of {n} row(s) kept; {len(out_columns)} output column(s) ({len(opt.standard_transforms) + len(opt.recipe_variables) + len(opt.custom_transforms)} new)."

    return AnalysisResult(title="Pick and convert variables", summary=summary, tables=[preview_table], notes=notes, new_sheet=new_sheet)
