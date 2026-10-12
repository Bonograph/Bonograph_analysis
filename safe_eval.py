"""A small, restricted expression evaluator shared by the analyses that
let a user type a formula (Transform Data, Plot a Mathematical Function).

This is not a security sandbox against untrusted input from other users
or the network - it's a local, single-user desktop app, and the "attacker"
would just be the user running arbitrary Python on their own machine
anyway. The restricted namespace exists to catch typos with a clear
error message and to keep formulas limited to plain math, not to defend
a trust boundary.
"""

from __future__ import annotations

import math

from .backend_types import AnalysisError

SAFE_NAMES = {
    name: getattr(math, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "log10", "log2", "sqrt", "pi", "e", "floor", "ceil",
    )
}
SAFE_NAMES["abs"] = abs
SAFE_NAMES["pow"] = pow


def evaluate(expression: str, **variables: float) -> float:
    namespace = dict(SAFE_NAMES)
    namespace.update(variables)
    namespace["__builtins__"] = {}
    # people write powers as y^2; in Python ^ is bitwise XOR, which fails on decimals
    expression = expression.replace("^", "**")
    try:
        return float(eval(expression, namespace))  # noqa: S307 - see module docstring
    except Exception as exc:
        given = ", ".join(f"{k}={v:g}" for k, v in variables.items())
        raise AnalysisError(f"Couldn't evaluate '{expression}' at {given}: {exc}") from None
