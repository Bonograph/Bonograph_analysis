"""Pieces several analyses share: the shape-of-the-values cards, the ways of
keeping many comparisons honest, column lists, and methods-text helpers."""

from __future__ import annotations

from ..spec import Context, Opt, Pick
from .. import report_style


def skewed(ctx: Context) -> bool:
    return "skewed" in ctx.hints


def paired(ctx: Context) -> bool:
    return "paired" in ctx.hints


def repeats(ctx: Context) -> bool:
    return ctx.shape is not None and ctx.shape.replicates >= 2


def shape_cards(field: str = "distribution", ratio_word: str = "geometric means") -> list[Opt]:
    return [
        Opt("gaussian", "Roughly bell-shaped", "Compares means. The usual choice for measurements.",
            phrase="means"),
        Opt("lognormal", "Skewed, bell-shaped once logged",
            f"Compares {ratio_word} - typical of concentrations, titres and expression levels.",
            suggest=skewed, phrase=ratio_word),
        Opt("nonparametric", "No assumption about shape",
            "Compares ranks: robust to odd values, a little less able to spot a real effect.",
            phrase="ranks"),
    ]


CORRECTIONS = {
    "auto": "the best method for this design",
    "holm": "Holm's step-down method",
    "tukey": "Tukey's method (every pair, equal scatter)",
    "dunnett": "Dunnett's method (each against a control)",
    "fdr": "a false-discovery-rate limit (Benjamini-Hochberg)",
    "none": "no adjustment - each P on its own",
}


def correction_pick(field: str = "correction", keys=("auto", "holm", "tukey", "dunnett", "fdr", "none")) -> Pick:
    return Pick(field, [(k, CORRECTIONS[k]) for k in keys])


def columns(ctx: Context, extra=()) -> list:
    return ctx.column_options(extra)


def a_b(ctx: Context) -> tuple[str, str]:
    return ctx.name(0), ctx.name(1)


def confidence() -> str:
    return f"{round(report_style.current().confidence * 100)}%"


def join(parts) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]
