"""
Shared result types every analysis backend returns, and the error type a
backend raises when the selected sheet's data doesn't fit what it needs
(too few groups, too few rows, non-numeric data, ...). Kept separate from
individual backend files since every backend imports these same types.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class AnalysisError(Exception):
    """Raised by a backend when the data it was given can't be analyzed
    this way (e.g. a t-test asked to compare only one group). The message
    is shown to the user as-is, so keep it plain and specific."""


@dataclass
class ResultTable:
    title: str
    headers: list[str]
    rows: list[list[str]]


@dataclass
class NewSheetData:
    """A brand-new sheet a backend wants created (data-transform analyses
    - Normalize, Transform, Transpose, ... - produce a new table rather
    than a report, so the source table is left as it was
    instead of overwriting the source data). ``columns`` maps each new
    group's title to its values, row-aligned with ``x_values``; a column
    shorter than ``x_values`` just leaves the remaining rows blank."""

    suggested_name: str
    x_title: str
    x_values: list[float]
    columns: dict[str, list[float]]


@dataclass
class ChartSeries:
    """One set of points on an analysis chart. ``kind``: scatter | line |
    step | bar | band (a shaded region between ``y`` and ``y_upper``). ``hollow_x/hollow_y`` are extra open symbols (e.g. censored
    subjects on a survival curve)."""

    label: str
    x: list[float]
    y: list[float]
    kind: str = "scatter"
    y_err: list[float] | None = None
    color: str | None = None
    hollow_x: list[float] | None = None
    hollow_y: list[float] | None = None
    y_upper: list[float] | None = None  # kind "band": shaded from y (lower edge) up to y_upper


@dataclass
class ChartSpec:
    """A graph an analysis wants shown next to its tables - plain numbers
    only, so backends never import matplotlib (graphs/engine/charts.py
    draws it). ``kind`` is "xy" (uses ``series``) or "heatmap" (uses
    ``matrix``). ``reference_lines`` entries: {"kind": "h"|"v", "value": v}
    or {"kind": "line", "points": [[x0, y0], [x1, y1]]}."""

    title: str
    kind: str = "xy"
    series: list[ChartSeries] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    reference_lines: list[dict] = field(default_factory=list)
    x_categories: list[str] | None = None
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    matrix: list[list[float]] | None = None
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    cmap: str = "viridis"
    color_limits: tuple[float, float] | None = None  # heat map colour scale (e.g. -1..1 for correlations)
    annotations: list[dict] = field(default_factory=list)  # {"x": .., "y": .., "text": ..}
    message: str = ""  # kind "message": why this requested graph couldn't be drawn, and what data it needs
    x_log: bool = False  # log-scaled X, for concentrations spanning several decades
    y_log: bool = False  # log-scaled Y, for ratios (odds, hazards) where 0.5 and 2 are equally far from 1
    # Graphs with the same ``group`` are one graph with a choice: the results window shows one of them
    # and a selector of their ``option`` names ("PC1 against PC2", "PC1 against PC3", ...).
    group: str = ""
    option: str = ""
    # kind "angles" (a map of two angles, such as a Ramachandran plot): {"density": rows of counts,
    # "regions": rows of 0 | 1 | 2 (outside, marginal, allowed), "inside": text for the legend}
    extra: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """What a backend hands back, and what ResultsWindow renders.

    ``summary`` is the one- or two-line headline (the key statistic and
    its p-value, typically) - the thing a user glances at first.
    ``tables`` are supporting detail (per-group stats, a fitted-parameter
    table, ...). ``notes`` are caveats: assumptions made, what a "not
    implemented yet" analysis is missing, warnings about small n, etc.
    ``new_sheet``, if set, tells the caller (main_sp.py) to also create a
    new sheet populated with this data - for analyses that transform the
    data itself rather than just report on it. ``charts`` are graphs shown
    among the graphs of the results window.
    """

    title: str
    summary: str
    tables: list[ResultTable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    new_sheet: NewSheetData | None = None
    charts: list[ChartSpec] = field(default_factory=list)


def _known(cls, data: dict) -> dict:
    from dataclasses import fields

    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def result_to_dict(result: AnalysisResult) -> dict:
    """JSON-ready copy of a result, for saving in the project file."""
    from dataclasses import asdict

    data = asdict(result)
    data["new_sheet"] = None  # already turned into a sheet when the analysis ran
    return data


def result_from_dict(data: dict) -> AnalysisResult:
    """Rebuild a saved result (unknown keys from other versions are ignored)."""
    charts = []
    for raw in data.get("charts") or []:
        chart = _known(ChartSpec, dict(raw))
        chart["series"] = [ChartSeries(**_known(ChartSeries, s)) for s in chart.get("series") or []]
        for key in ("x_limits", "y_limits", "color_limits"):
            if chart.get(key) is not None:
                chart[key] = tuple(chart[key])
        charts.append(ChartSpec(**chart))
    return AnalysisResult(
        title=data["title"],
        summary=data.get("summary", ""),
        tables=[ResultTable(**_known(ResultTable, t)) for t in data.get("tables") or []],
        notes=list(data.get("notes") or []),
        charts=charts,
    )
