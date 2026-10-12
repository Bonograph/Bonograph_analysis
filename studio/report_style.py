"""How every result is written, set once for all analyses.

Confidence level, significant digits, how P values are written and the
threshold for calling a result significant are properties of the report,
not of any one test - so they live here, are chosen in the Analyze
window's footer, and are applied to whichever analysis runs. An analysis
that uses one of these words for something else (a simulator's rounding,
say) lists it in its spec's ``own`` fields and keeps its own value.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULTS = {"confidence": 0.95, "digits": 4, "p_style": "number", "alpha": 0.05}
CONFIDENCES = (0.90, 0.95, 0.99)
ALPHAS = (0.10, 0.05, 0.01, 0.001)

# options field -> the report-style value it takes
FIELDS = {
    "confidence_level": "confidence",
    "ci_level": "confidence",
    "confidence_band_level": "confidence",
    "significant_digits": "digits",
    "p_style": "p_style",
    "alpha": "alpha",
}

_override: dict | None = None  # tests and scripts can pin a style without touching saved settings


@dataclass
class Style:
    confidence: float = 0.95
    digits: int = 4
    p_style: str = "number"
    alpha: float = 0.05

    def summary(self) -> str:
        from ..p_values import format_p

        return (f"{round(self.confidence * 100)}% intervals  ·  {self.digits} digits  ·  "
                f"P as {format_p(0.02137, self.p_style, digits=self.digits)}  ·  significant below {self.alpha:g}")


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings("Bonograph", "Analyze")


def current() -> Style:
    if _override is not None:
        return Style(**_override)
    values = dict(DEFAULTS)
    try:
        s = _settings()
        values["confidence"] = float(s.value("report/confidence", DEFAULTS["confidence"]))
        values["digits"] = int(s.value("report/digits", DEFAULTS["digits"]))
        values["p_style"] = str(s.value("report/p_style", DEFAULTS["p_style"]))
        values["alpha"] = float(s.value("report/alpha", DEFAULTS["alpha"]))
    except Exception:  # noqa: BLE001 - a broken settings store falls back to the defaults
        values = dict(DEFAULTS)
    from ..p_values import BY_KEY

    if values["p_style"] not in BY_KEY:
        values["p_style"] = DEFAULTS["p_style"]
    return Style(**values)


def save(style: Style) -> None:
    if _override is not None:
        _override.update(style.__dict__)
        return
    try:
        s = _settings()
        s.setValue("report/confidence", style.confidence)
        s.setValue("report/digits", style.digits)
        s.setValue("report/p_style", style.p_style)
        s.setValue("report/alpha", style.alpha)
    except Exception:  # noqa: BLE001
        pass


def pin(**values) -> None:
    """Use this style (defaults for anything not given) instead of the saved one."""
    global _override
    _override = dict(DEFAULTS, **values)


def apply(state: dict, fields: set[str], own: set[str] = frozenset(), style: Style | None = None) -> None:
    """Write the report style into ``state`` for every field the options have."""
    style = style or current()
    for field, attr in FIELDS.items():
        if field in fields and field not in own:
            state[field] = getattr(style, attr)
