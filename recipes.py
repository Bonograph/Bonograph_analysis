"""A recipe: a short chain of steps applied to a value, one after another.

"log, then times 100" is two steps. Each step is a dict - ``{"op": "log10"}``,
``{"op": "times", "n": 100}`` or ``{"op": "equation", "text": "v / x"}`` -
so a recipe is plain data: it is drawn as a row of pills on the settings
panel, saved with a session, and run here.

A step that cannot be worked out for a value (the log of zero, a division by
zero) gives a blank for that value, never an error.
"""

from __future__ import annotations

import math

import numpy as np

from .backend_types import AnalysisError
from .safe_eval import evaluate

# key -> (how the pill reads, the default number or None, what it does)
_STEPS = {
    "log10": ("log₁₀", None, lambda v, n: math.log10(v)),
    "ln": ("ln", None, lambda v, n: math.log(v)),
    "log2": ("log₂", None, lambda v, n: math.log2(v)),
    "times": ("× {n}", 100.0, lambda v, n: v * n),
    "divide": ("÷ {n}", 10.0, lambda v, n: v / n),
    "plus": ("+ {n}", 1.0, lambda v, n: v + n),
    "minus": ("− {n}", 1.0, lambda v, n: v - n),
    "power": ("to the power {n}", 2.0, lambda v, n: v ** n),
    "root": ("square root", None, lambda v, n: math.sqrt(v)),
    "reciprocal": ("1 ÷ value", None, lambda v, n: 1.0 / v),
    "ten_to": ("10 to the value", None, lambda v, n: 10.0 ** v),
    "e_to": ("e to the value", None, lambda v, n: math.exp(v)),
    "absolute": ("drop the sign", None, lambda v, n: abs(v)),
    "negate": ("flip the sign", None, lambda v, n: -v),
    "equation": ("= {text}", None, None),
}

# the menu a step is added from: (heading, [keys])
MENU = [
    ("Logs", ["log10", "ln", "log2"]),
    ("Arithmetic", ["times", "divide", "plus", "minus"]),
    ("Powers", ["power", "root", "reciprocal", "ten_to", "e_to"]),
    ("Sign", ["absolute", "negate"]),
    ("Anything else", ["equation"]),
]
MENU_TEXT = {
    "log10": "Log, base 10", "ln": "Natural log", "log2": "Log, base 2",
    "times": "Multiply by a number", "divide": "Divide by a number", "plus": "Add a number", "minus": "Subtract a number",
    "power": "Raise to a power", "root": "Square root", "reciprocal": "Reciprocal (1 ÷ value)",
    "ten_to": "10 to the value (undo log₁₀)", "e_to": "e to the value (undo ln)",
    "absolute": "Drop the sign", "negate": "Flip the sign", "equation": "An equation of my own…",
}
EQUATION_HINT = ("Write it with v for the value so far, and x or y for the row's own X or Y - for example v / x. "
                 "Functions: log10, log (natural), exp, sqrt, abs, sin, cos, tan.")


def new_step(op: str) -> dict:
    """A step of this kind with its starting number (or an equation that changes nothing)."""
    if op not in _STEPS:
        raise KeyError(op)
    if op == "equation":
        return {"op": op, "text": "v"}
    number = _STEPS[op][1]
    return {"op": op} if number is None else {"op": op, "n": number}


def needs_number(op: str) -> bool:
    return op in _STEPS and _STEPS[op][1] is not None


def _number_text(n: float) -> str:
    return f"{n:.10g}"


def step_text(step: dict) -> str:
    """How one step reads: "× 100", "log₁₀", "= v / x"."""
    template = _STEPS.get(step.get("op"), ("?", None, None))[0]
    return template.format(n=_number_text(float(step.get("n", 0.0))), text=(step.get("text") or "").strip() or "v")


def describe(steps) -> str:
    """The whole recipe on one line, or "unchanged" when it has no step."""
    return " → ".join(step_text(s) for s in (steps or [])) or "unchanged"


def apply(steps, value: float, x: float = float("nan"), y: float = float("nan")) -> float:
    """``value`` after every step in turn. ``x`` and ``y`` are the row's own
    values as they stand on the sheet, for an equation step to use. A blank
    in gives a blank out; so does a step that has no answer for this value."""
    v = float(value)
    if math.isnan(v):
        return v
    for step in steps or []:
        op = step.get("op")
        try:
            if op == "equation":
                v = float(evaluate((step.get("text") or "v").strip() or "v", v=v, x=float(x), y=float(y)))
            else:
                v = float(_STEPS[op][2](v, float(step.get("n", 0.0))))
        except (AnalysisError, KeyError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            return float("nan")
        if isinstance(v, complex) or math.isnan(v) or math.isinf(v):
            return float("nan")
    return v


_WHOLE_COLUMN = {
    "log10": lambda v, n: np.log10(v), "ln": lambda v, n: np.log(v), "log2": lambda v, n: np.log2(v),
    "times": lambda v, n: v * n, "divide": lambda v, n: v / n, "plus": lambda v, n: v + n, "minus": lambda v, n: v - n,
    "power": lambda v, n: np.power(v, n), "root": lambda v, n: np.sqrt(v), "reciprocal": lambda v, n: 1.0 / v,
    "ten_to": lambda v, n: np.power(10.0, v), "e_to": lambda v, n: np.exp(v), "absolute": lambda v, n: np.abs(v),
    "negate": lambda v, n: -v,
}


def apply_many(steps, values, x=None, y=None) -> np.ndarray:
    """``apply`` for a whole column at once: the same answers, worked out on
    arrays. Only an equation step is still taken value by value, since it
    is the user's own text. ``x`` and ``y`` are the rows' own values."""
    v = np.array(values, dtype=float)
    if not steps:
        return v
    x = np.full(v.shape, np.nan) if x is None else np.asarray(x, dtype=float)
    y = np.full(v.shape, np.nan) if y is None else np.asarray(y, dtype=float)
    with np.errstate(all="ignore"):
        for step in steps:
            op = step.get("op")
            if op == "equation" or op not in _WHOLE_COLUMN:
                return np.array([apply(steps, values[i], x[i], y[i]) for i in range(v.size)], dtype=float)                     if op != "equation" else _with_equations(steps, np.asarray(values, dtype=float), x, y)
            v = np.asarray(_WHOLE_COLUMN[op](v, float(step.get("n", 0.0))), dtype=float)
            v[~np.isfinite(v)] = np.nan       # no answer for this value: blank from here on, as apply() gives
    return v


def _with_equations(steps, values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A recipe holding an equation: the plain steps before the first equation
    go by column, the rest value by value."""
    first = next(i for i, step in enumerate(steps) if step.get("op") == "equation")
    head = apply_many(steps[:first], values, x, y)
    return np.array([apply(steps[first:], head[i], x[i], y[i]) for i in range(head.size)], dtype=float)


def check(steps) -> str:
    """"" when every step is well formed, else what is wrong with the first one that is not."""
    for position, step in enumerate(steps or [], 1):
        op = step.get("op")
        if op not in _STEPS:
            return f"Step {position} is not a step this version knows."
        if op == "equation":
            text = (step.get("text") or "").strip()
            if not text:
                return f"Step {position}: the equation is empty."
            try:
                evaluate(text, v=1.5, x=1.5, y=1.5)
            except AnalysisError as exc:
                message = str(exc)
                if "division by zero" in message or "math domain" in message:
                    continue  # the equation is sound; it only has no answer at the trial value
                return f"Step {position}: the equation '{text}' cannot be read."
        elif op == "divide" and float(step.get("n", 0.0)) == 0.0:
            return f"Step {position} divides by zero."
    return ""
