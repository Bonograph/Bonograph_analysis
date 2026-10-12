"""How a P value is written out.

Different fields write P values differently, and the difference is not
cosmetic: a paper that prints ".03" and one that prints "0.0214 (*)" are
following different house rules. This is the one place that decides, so every
analysis prints them the same way and a single setting changes all of them.

Ordered from the plainest to the most interpreted: the number on its own, the
number with a verdict attached, then the two journal conventions, then the
forms that drop the number altogether.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT = "number"
DEFAULT_DIGITS = 4
DEFAULT_ALPHA = 0.05

# The thresholds behind the symbols, loosest first. Widely used, and the
# same ones the "stars" styles below print.
_STAR_STEPS = ((0.05, "*"), (0.01, "**"), (0.001, "***"), (0.0001, "****"))
NOT_SIGNIFICANT = "ns"


@dataclass(frozen=True)
class PStyle:
    key: str
    label: str      # what the dropdown shows
    example: str    # the same P value (0.02137) written this way
    note: str       # one line on when to use it


STYLES: tuple[PStyle, ...] = (
    PStyle("number", "Just the number",
           "0.0214", "The P value alone, to as many digits as you set below."),
    PStyle("number_stars", "Number, with stars",
           "0.0214 (*)", "The number, with the usual * ** *** marks beside it."),
    PStyle("number_verdict", "Number, with a verdict",
           "0.0214 (significant)", "The number, said plainly against your threshold."),
    PStyle("apa", "APA style, no leading zero",
           ".021", "Three decimals and no leading zero, as psychology journals ask."),
    PStyle("medical", "Medical journal style",
           "0.021", "Three decimals with the leading zero, as medical journals ask."),
    PStyle("scientific", "Scientific notation",
           "2.137e-02", "Keeps small P values readable instead of collapsing to <0.0001."),
    PStyle("stars", "Stars only",
           "*", "Just the marks - for a figure where the number would not fit."),
    PStyle("verdict", "Significant or not",
           "Significant", "No number at all: only whether it passed your threshold."),
)

BY_KEY = {style.key: style for style in STYLES}
KEYS = tuple(style.key for style in STYLES)


def stars(p: float, alpha: float = DEFAULT_ALPHA) -> str:
    """The * ** *** mark for a P value, or "ns" above the threshold."""
    if not _is_number(p):
        return ""
    if p > alpha:
        return NOT_SIGNIFICANT
    mark = NOT_SIGNIFICANT
    for threshold, symbol in _STAR_STEPS:
        if p <= threshold:
            mark = symbol
    return mark


def is_significant(p: float, alpha: float = DEFAULT_ALPHA) -> bool:
    return _is_number(p) and p <= alpha


def format_p(p: float, style: str = DEFAULT, *, digits: int = DEFAULT_DIGITS,
             alpha: float = DEFAULT_ALPHA) -> str:
    """One P value, written in the chosen style.

    A value that isn't a number at all (a test that couldn't run) always comes
    out as "n/a", whichever style is chosen - never as a misleading "0".
    """
    if not _is_number(p):
        return "n/a"
    p = min(max(float(p), 0.0), 1.0)
    style = style if style in BY_KEY else DEFAULT

    if style == "stars":
        return stars(p, alpha)
    if style == "verdict":
        return "Significant" if is_significant(p, alpha) else "Not significant"
    if style == "apa":
        return _fixed(p, 3, leading_zero=False)
    if style == "medical":
        return _fixed(p, 3, leading_zero=True)
    if style == "scientific":
        return f"{p:.{max(digits - 1, 1)}e}"

    number = _fixed(p, digits, leading_zero=True)
    if style == "number_stars":
        return f"{number} ({stars(p, alpha)})"
    if style == "number_verdict":
        return f"{number} ({'significant' if is_significant(p, alpha) else 'not significant'})"
    return number


def _fixed(p: float, digits: int, *, leading_zero: bool) -> str:
    """A P value to ``digits`` decimals, with the extremes called out.

    Rounding 0.00004 to "0.0000" would read as an impossible P of zero, so
    anything below what the chosen precision can show is written as "less
    than" the smallest value it can - and the same at the top end.
    """
    digits = max(int(digits), 1)
    smallest = 10.0 ** -digits
    if p < smallest:
        text = f"<{smallest:.{digits}f}"
    elif p > 1 - smallest:
        text = f">{1 - smallest:.{digits}f}"
    else:
        text = f"{p:.{digits}f}"
    return text if leading_zero else text.replace("0.", ".", 1)


def _is_number(value) -> bool:
    try:
        return value is not None and not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def choices() -> list[tuple[str, str]]:
    """(label, key) pairs for a dropdown, in the order above. The label
    carries its example so the list can be read without opening the help."""
    return [(f"{style.label}   —   {style.example}", style.key) for style in STYLES]
