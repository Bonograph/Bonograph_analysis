"""How the values are spread: counts of values in each range, as plain,
relative or running totals, with the ranges chosen for you or set by hand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class FrequencyDistributionOptions:
    create_cumulative: bool = False  # False = plain frequency distribution
    tabulate: str = "count"  # "count" | "fraction" | "percent"
    bin_width_mode: str = "auto"  # "auto" | "fixed" | "exact"
    bin_width: float = 1.0
    first_bin_center_auto: bool = True
    first_bin_center: float = 1.0
    last_bin_center_auto: bool = True
    last_bin_center: float = 5.0
    replicate_mode: str = "means"  # "each" | "means" - only meaningful with subcolumns
    graph_results: bool = False
    graph_kind: str = "bars"  # "bars" (histogram) | "line" (frequency polygon)
    new_sheet: bool = False  # also put the table on a new sheet


def _build_bins(values: np.ndarray, opt: FrequencyDistributionOptions) -> np.ndarray:
    if opt.bin_width_mode == "exact":
        uniques = np.unique(values)
        # exact-frequency "bins" are just each distinct value; represent as
        # edges straddling each unique value by half the smallest gap.
        if uniques.size < 2:
            half = 0.5
        else:
            half = float(np.min(np.diff(uniques))) / 2
        edges = np.empty(uniques.size + 1)
        edges[:-1] = uniques - half
        edges[-1] = uniques[-1] + half
        return edges

    if opt.bin_width_mode == "fixed":
        width = opt.bin_width if opt.bin_width > 0 else 1.0
        start = opt.first_bin_center - width / 2 if not opt.first_bin_center_auto else np.min(values) - width / 2
        end = opt.last_bin_center + width / 2 if not opt.last_bin_center_auto else np.max(values) + width
        n_bins = max(1, int(np.ceil((end - start) / width)))
        return start + width * np.arange(n_bins + 1)

    return np.histogram_bin_edges(values, bins="auto")


@register("frequency_distribution")
def run(data: SheetData, options: FrequencyDistributionOptions | None = None) -> AnalysisResult:
    """Bins every data set on the same bin edges (worked out from all the
    values together), so their distributions can be compared bin by bin."""
    opt = options or FrequencyDistributionOptions()
    groups = {k: v[np.isfinite(v)] for k, v in
              data.all_group_arrays(mode="average" if opt.replicate_mode == "means" else "pooled").items()}
    groups = {k: v for k, v in groups.items() if v.size}
    if not groups:
        raise AnalysisError("This sheet has no numeric data in its Y columns yet.")
    pooled = np.concatenate(list(groups.values()))
    if pooled.size < 2:
        raise AnalysisError("Binning needs at least 2 values.")

    edges = _build_bins(pooled, opt)
    centers = (edges[:-1] + edges[1:]) / 2
    kind = "Cumulative frequency distribution" if opt.create_cumulative else "Frequency distribution"
    if opt.tabulate == "fraction":
        what = "Relative frequency (fraction)"
    elif opt.tabulate == "percent":
        what = "Relative frequency (%)"
    else:
        what = "Number of values"
    if opt.create_cumulative:
        what = "Cumulative " + what[0].lower() + what[1:]

    columns: dict[str, np.ndarray] = {}
    for label, values in groups.items():
        counts, _ = np.histogram(values, bins=edges)
        vals = counts.astype(float)
        if opt.tabulate == "fraction":
            vals = vals / values.size
        elif opt.tabulate == "percent":
            vals = 100 * vals / values.size
        columns[label] = np.cumsum(vals) if opt.create_cumulative else vals

    headers = ["Bin center", "Bin range"] + [f"{label} (n = {groups[label].size})" for label in columns]
    rows = [[f"{c:.4g}", f"{edges[i]:.4g} to {edges[i + 1]:.4g}"] + [f"{columns[k][i]:.4g}" for k in columns]
            for i, c in enumerate(centers)]
    table = ResultTable(title=f"{kind}: {what.lower()} per bin", headers=headers, rows=rows)
    width = float(edges[1] - edges[0]) if edges.size > 1 else float("nan")
    summary = (f"{len(columns)} data set(s) binned into {len(centers)} bins"
               + (f" of width {width:.4g}" if opt.bin_width_mode != "exact" else " (one per distinct value)")
               + f", {pooled.size} values in all.")

    new_sheet = None
    if opt.new_sheet or opt.graph_results:
        new_sheet = NewSheetData(suggested_name=kind, x_title="Bin center", x_values=[float(c) for c in centers],
                                 columns={k: [float(v) for v in vals] for k, vals in columns.items()})

    charts = []
    if opt.graph_results:
        from ..backend_types import ChartSeries, ChartSpec

        line = opt.create_cumulative or opt.graph_kind == "line"
        series = []
        n_sets = len(columns)
        for i, (label, vals) in enumerate(columns.items()):
            if line:
                series.append(ChartSeries(label, centers.tolist(), vals.tolist(), kind="line_markers"))
            else:  # side-by-side bars within each bin, so several data sets stay readable
                step = width * 0.8 / n_sets if np.isfinite(width) and width > 0 else 0.0
                offset = (i - (n_sets - 1) / 2) * step
                series.append(ChartSeries(label, (centers + offset).tolist(), vals.tolist(), kind="bar"))
        charts.append(ChartSpec(title=kind, series=series, x_label="Bin center", y_label=what))
    notes = []
    if len(columns) > 1:
        notes.append("Every data set uses the same bins, so their columns can be compared bin by bin.")
    return AnalysisResult(title="How values are spread", summary=summary, tables=[table], notes=notes,
                          new_sheet=new_sheet, charts=charts)
