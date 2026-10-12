"""A tree of which rows resemble which (hierarchical clustering).
Clusters rows and/or columns using every selected group's first
replicate column as a variable."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

_LINKAGE_MAP = {
    "complete": "complete",
    "single": "single",
    "simple_average": "weighted",  # WPGMA - doesn't re-weight by prior cluster size
    "centroid": "centroid",
    "median": "median",
    "group_average": "average",  # UPGMA - weighted by cluster size
    "ward": "ward",
}
_DISTANCE_MAP = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
    "correlation": "correlation",
    "cosine": "cosine",
}


@dataclass
class HierarchicalClusteringOptions:
    # the data
    selected_variables: list[str] = field(default_factory=list)  # empty = all
    clustering_direction: str = "rows"  # "rows" | "columns" | "both"
    # what to report
    column_scaling: str = "standardize"  # "standardize" | "center" | "none"
    row_scaling: str = "none"  # "standardize" | "center" | "none"
    distance_method: str = "euclidean"
    linkage_method: str = "complete"
    define_rows: bool = False
    row_cluster_mode: str = "count"  # "count" | "height"
    row_n_clusters: int = 2
    row_height: float = 1.0
    define_columns: bool = False
    col_cluster_mode: str = "count"
    col_n_clusters: int = 2
    col_height: float = 1.0
    # tables to report
    report_scaled_data: bool = False
    report_distance_matrix: bool = False
    report_merge_map: bool = False
    report_cophenetic_distance: bool = False
    report_cophenetic_correlation: bool = False
    report_full_matrices: bool = True
    significant_digits: int = 4
    # graphs
    graph_dendrogram: bool = True
    graph_heatmap_data: bool = False
    graph_heatmap_distances: bool = False
    graph_elbow: bool = False
    graph_silhouette: bool = False
    graph_gap: bool = False


_MAX_ROWS = 5000  # the distance matrix grows with rows squared


def _scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix
    mean = matrix.mean(axis=0)
    if mode == "center":
        return matrix - mean
    sd = matrix.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (matrix - mean) / sd


def _cluster_one_axis(matrix: np.ndarray, opt: HierarchicalClusteringOptions, defining: bool, mode: str, n_clusters: int, height: float):
    dist = pdist(matrix, metric=_DISTANCE_MAP[opt.distance_method])
    z = linkage(dist, method=_LINKAGE_MAP[opt.linkage_method])
    assignments = None
    if defining:
        if mode == "count":
            k = max(1, min(n_clusters, matrix.shape[0]))
            assignments = fcluster(z, t=k, criterion="maxclust")
        else:
            assignments = fcluster(z, t=height, criterion="distance")
    return z, dist, assignments


def _format_matrix(mat: np.ndarray, labels: list[str], sig: int, full: bool) -> ResultTable:
    rows = []
    n = mat.shape[0]
    for i in range(n):
        row = [labels[i]]
        for j in range(n):
            if not full and j > i:
                row.append("")
            else:
                row.append(f"{mat[i, j]:.{sig}g}")
        rows.append(row)
    return ResultTable(title="", headers=["", *labels], rows=rows)


@register("hierarchical_clustering")
def run(data: SheetData, options: HierarchicalClusteringOptions | None = None) -> AnalysisResult:
    from ..ml_utils import silhouette_score

    opt = options or HierarchicalClusteringOptions()
    all_labels, all_matrix = data.first_replicate_matrix()
    if len(all_labels) < 1:
        raise AnalysisError("Need at least 1 variable (group column) with data.")

    if opt.selected_variables:
        keep = [i for i, lbl in enumerate(all_labels) if lbl in opt.selected_variables]
        labels, matrix = [all_labels[i] for i in keep], all_matrix[:, keep]
    else:
        labels, matrix = all_labels, all_matrix

    complete_mask = ~np.any(np.isnan(matrix), axis=1)
    complete = matrix[complete_mask]
    if complete.shape[0] < 3:
        raise AnalysisError(f"Need at least 3 complete rows to cluster; found {complete.shape[0]}.")
    if complete.shape[0] > _MAX_ROWS:
        gb = complete.shape[0] ** 2 * 4 / 1e9
        raise AnalysisError(
            f"Hierarchical clustering compares every row with every other: {complete.shape[0]:,} rows would need "
            f"about {gb:.0f} GB and a very long wait. It works up to {_MAX_ROWS:,} rows - use 'Split into k "
            "groups' (K-means) for more.")

    sig = opt.significant_digits
    col_scaled = _scale(complete, opt.column_scaling)
    row_scaled = _scale(col_scaled.T, opt.row_scaling).T if opt.row_scaling != "none" else col_scaled
    row_labels = [str(i + 1) for i in range(row_scaled.shape[0])]

    tables = []
    charts = []
    from ..chart_helpers import dendrogram_chart, heatmap_chart, xy_chart
    notes = []
    summary_bits = []

    def one_axis(matrix, axis_labels, axis_name, defining, mode, n_clusters, height):
        """Every output for one direction - rows (subjects) or columns (variables) - so both get the same."""
        z, dist, assign = _cluster_one_axis(matrix, opt, defining, mode, n_clusters, height)
        what = "rows" if axis_name == "rows" else "variables"
        if assign is not None:
            sizes = np.bincount(assign)[1:]
            if axis_name == "rows":
                tables.append(ResultTable(title="Row cluster sizes", headers=["Cluster", "N rows"],
                                          rows=[[f"Cluster {i+1}", str(s)] for i, s in enumerate(sizes)]))
            else:
                tables.append(ResultTable(title="Column (variable) clusters", headers=["Cluster", "Variables"],
                                          rows=[[f"Cluster {i+1}", ", ".join(axis_labels[j] for j in range(len(axis_labels)) if assign[j] == i + 1)]
                                                for i in range(len(sizes))]))
            summary_bits.append(f"{len(sizes)} {'row' if axis_name == 'rows' else 'column'} cluster(s)")
        if opt.report_merge_map:
            tables.append(ResultTable(title=f"The order clusters joined ({axis_name})", headers=["Step", "Cluster A", "Cluster B", "Distance", "N members"],
                                      rows=[[str(i+1), f"{a:g}", f"{b:g}", f"{d:.{sig}g}", f"{int(m)}"] for i, (a, b, d, m) in enumerate(z)][:200]))
        if opt.report_cophenetic_distance or opt.report_cophenetic_correlation:
            coph_corr, coph_dists = cophenet(z, dist)
            if opt.report_cophenetic_correlation:
                tables.append(ResultTable(title=f"Cophenetic correlation ({axis_name})", headers=["Cophenetic correlation"],
                                          rows=[[f"{coph_corr:.{sig}g}"]]))
                notes.append("The cophenetic correlation says how faithfully the tree keeps the original distances "
                             "(above about 0.75 is usually taken as good).")
            if opt.report_cophenetic_distance:
                t = _format_matrix(squareform(coph_dists), axis_labels, sig, opt.report_full_matrices)
                t.title = f"Cophenetic distance matrix ({axis_name})"
                tables.append(t)
        if opt.report_distance_matrix or opt.graph_heatmap_distances:
            mat = squareform(dist)
            if opt.report_distance_matrix:
                t = _format_matrix(mat, axis_labels, sig, opt.report_full_matrices)
                t.title = f"Distance matrix ({axis_name})"
                tables.append(t)
            if opt.graph_heatmap_distances:
                charts.append(heatmap_chart(f"Heat map of distances ({axis_name})", mat, axis_labels, axis_labels, cmap="magma"))
        if opt.graph_dendrogram:
            charts.append(dendrogram_chart(f"Dendrogram ({axis_name})", z, axis_labels))
            tables.append(ResultTable(title=f"The tree, join by join ({axis_name} linkage)", headers=["Step", "Cluster A", "Cluster B", "Distance"],
                                      rows=[[str(i+1), f"{a:g}", f"{b:g}", f"{d:.{sig}g}"] for i, (a, b, d, _m) in enumerate(z)][:200]))
        if (opt.graph_elbow or opt.graph_silhouette or opt.graph_gap) and matrix.shape[0] >= 3:
            max_k = min(10, matrix.shape[0] - 1)
            elbow_rows, sil_rows, gap_rows = [], [], []
            rng = np.random.default_rng(0)
            mins, maxs = matrix.min(axis=0), matrix.max(axis=0)

            def wcss(m, assignment):
                return sum(float(np.sum((m[assignment == c] - m[assignment == c].mean(axis=0)) ** 2)) for c in np.unique(assignment))

            for k in range(2, max_k + 1):
                assign_k = fcluster(z, t=k, criterion="maxclust")
                if opt.graph_elbow:
                    elbow_rows.append([str(k), f"{wcss(matrix, assign_k):.{sig}g}"])
                if opt.graph_silhouette and 2 <= len(np.unique(assign_k)) < matrix.shape[0]:
                    try:
                        sil_rows.append([str(k), f"{silhouette_score(matrix, assign_k):.{sig}g}"])
                    except ValueError:
                        pass
                if opt.graph_gap:
                    real_disp = np.log(max(wcss(matrix, assign_k), 1e-12))
                    ref_disps = []
                    for _ in range(10):
                        ref_data = rng.uniform(mins, maxs, size=matrix.shape)
                        ref_z = linkage(pdist(ref_data), method=_LINKAGE_MAP[opt.linkage_method])
                        ref_disps.append(np.log(max(wcss(ref_data, fcluster(ref_z, t=k, criterion="maxclust")), 1e-12)))
                    gap_rows.append([str(k), f"{float(np.mean(ref_disps) - real_disp):.{sig}g}"])
            for rows_k, chart_title, y_label in ((elbow_rows, "Elbow plot", "Within-cluster sum of squares"),
                                                 (sil_rows, "Silhouette plot", "Silhouette score"),
                                                 (gap_rows, "Gap statistic plot", "Gap statistic")):
                if rows_k:
                    charts.append(xy_chart(f"{chart_title} ({axis_name})", [float(r[0]) for r in rows_k],
                                           {"": [float(r[1]) for r in rows_k]}, x_label="Number of clusters", y_label=y_label))
                    tables.append(ResultTable(title=f"{chart_title} data ({axis_name})", headers=["Number of clusters", y_label], rows=rows_k))
        return what

    if opt.clustering_direction in ("rows", "both"):
        one_axis(row_scaled, row_labels, "rows", opt.define_rows, opt.row_cluster_mode, opt.row_n_clusters, opt.row_height)
    if opt.clustering_direction in ("columns", "both"):
        if row_scaled.shape[1] < 2:
            raise AnalysisError("Clustering the columns needs at least 2 variables.")
        one_axis(row_scaled.T, labels, "columns", opt.define_columns, opt.col_cluster_mode, opt.col_n_clusters, opt.col_height)

    if opt.report_scaled_data:
        kind = "Standardized" if opt.column_scaling == "standardize" else "Centered" if opt.column_scaling == "center" else "Raw"
        tables.append(ResultTable(title=f"{kind} data", headers=["Row"] + labels, rows=[[row_labels[i]] + [f"{v:.{sig}g}" for v in row] for i, row in enumerate(row_scaled)][:200]))
    if opt.graph_heatmap_data:
        charts.append(heatmap_chart("Heat map of data", row_scaled, row_labels, labels))
        tables.append(ResultTable(title="The data, in the order of the tree", headers=["Row"] + labels, rows=[[row_labels[i]] + [f"{v:.{sig}g}" for v in row] for i, row in enumerate(row_scaled)][:200]))

    if not tables and not charts:
        raise AnalysisError("Nothing was asked for: switch on at least one question.")

    summary = f"{complete.shape[0]} rows, {len(labels)} variable(s); " + ("; ".join(summary_bits) if summary_bits else "clustering computed") + f" ({opt.linkage_method.replace('_', ' ')} linkage, {opt.distance_method} distance)."
    notes.append(f"Columns were scaled ({opt.column_scaling}) before rows ({opt.row_scaling}).")
    if opt.linkage_method in ("ward", "centroid", "median") and opt.distance_method != "euclidean":
        notes.append(f"Ward's/centroid/median linkage are defined for Euclidean distance; using them with {opt.distance_method} distance is computed but no longer has the usual minimum-variance interpretation.")

    return AnalysisResult(title="Group by similarity", summary=summary, tables=tables, notes=notes, charts=charts)
