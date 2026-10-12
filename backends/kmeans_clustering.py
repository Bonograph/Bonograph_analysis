"""K-Means Clustering. Clusters rows using
every selected group's first replicate column as a variable."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register

_METRIC_MAP = {"euclidean": "euclidean", "manhattan": "cityblock", "correlation": "correlation", "cosine": "cosine"}


@dataclass
class KMeansClusteringOptions:
    # the data
    selected_variables: list[str] = field(default_factory=list)  # empty = all
    # what to report
    scaling: str = "standardize"  # "none" | "standardize" | "center"
    distance_method: str = "euclidean"  # "euclidean" | "correlation" | "manhattan" | "cosine"
    algorithm: str = "hartigan_wong"  # "hartigan_wong" | "forgy_lloyd" | "macqueen"
    max_iterations: int = 100
    min_clusters: int = 1
    max_clusters: int = 3
    determine_optimal: bool = False
    random_seed_mode: str = "auto"  # "auto" | "fixed"
    random_seed: int = 0
    # tables to report
    report_cluster_details: bool = False
    report_distance_to_centroid: bool = False
    report_initial_centroids: bool = False
    report_scaled_data: bool = False
    significant_digits: int = 4
    # graphs
    graph_elbow: bool = True
    graph_silhouette: bool = True
    graph_gap: bool = False


def _scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix
    mean = matrix.mean(axis=0)
    if mode == "center":
        return matrix - mean
    sd = matrix.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (matrix - mean) / sd


def _init_centroids(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Forgy initialization - k randomly chosen observations - the
    initialization "Forgy-Lloyd" refers to."""
    idx = rng.choice(data.shape[0], size=k, replace=False)
    return data[idx].copy()


def _assign(data: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    dists = cdist(data, centroids, metric=_METRIC_MAP[metric])
    return np.argmin(dists, axis=1)


def _lloyd(data: np.ndarray, centroids: np.ndarray, metric: str, max_iter: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Forgy-Lloyd: alternate "assign every point to nearest centroid"
    and "recompute centroids as cluster means" until stable."""
    k = centroids.shape[0]
    assignments = _assign(data, centroids, metric)
    for it in range(max_iter):
        new_centroids = np.array([data[assignments == c].mean(axis=0) if np.any(assignments == c) else centroids[c] for c in range(k)])
        new_assignments = _assign(data, new_centroids, metric)
        if np.array_equal(new_assignments, assignments) and np.allclose(new_centroids, centroids):
            return new_centroids, new_assignments, it + 1
        centroids, assignments = new_centroids, new_assignments
    return centroids, assignments, max_iter


def _macqueen(data: np.ndarray, centroids: np.ndarray, metric: str, max_iter: int) -> tuple[np.ndarray, np.ndarray, int]:
    """MacQueen's algorithm: process points one at a time, immediately
    updating the centroid of whichever cluster a point is assigned to
    (rather than waiting for a full pass, as Lloyd's does)."""
    k = centroids.shape[0]
    n = data.shape[0]
    counts = np.ones(k)
    assignments = _assign(data, centroids, metric)
    for c in range(k):
        counts[c] = max(1, int(np.sum(assignments == c)))
    for it in range(max_iter):
        changed = False
        for i in range(n):
            dists = cdist(data[i : i + 1], centroids, metric=_METRIC_MAP[metric])[0]
            new_c = int(np.argmin(dists))
            old_c = assignments[i]
            if new_c != old_c:
                changed = True
                if counts[old_c] > 1:
                    counts[old_c] -= 1
                    centroids[old_c] = (centroids[old_c] * (counts[old_c] + 1) - data[i]) / counts[old_c]
                counts[new_c] += 1
                centroids[new_c] = centroids[new_c] + (data[i] - centroids[new_c]) / counts[new_c]
                assignments[i] = new_c
        if not changed:
            return centroids, assignments, it + 1
    return centroids, assignments, max_iter


def _hartigan_wong(data: np.ndarray, centroids: np.ndarray, metric: str, max_iter: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Hartigan-Wong: start from a Lloyd's solution, then repeatedly try
    moving each point to whichever cluster would most reduce total
    within-cluster sum-of-squared-distance, updating that pair of
    centroids immediately - the extra single-point-move refinement that
    distinguishes it from plain Lloyd's."""
    centroids, assignments, used_iter = _lloyd(data, centroids, metric, max_iter)
    k = centroids.shape[0]
    counts = np.array([max(1, int(np.sum(assignments == c))) for c in range(k)])
    for it in range(max_iter):
        moved = False
        dists = cdist(data, centroids, metric=_METRIC_MAP[metric])
        for i in range(data.shape[0]):
            cur = assignments[i]
            if counts[cur] <= 1:
                continue
            best_c, best_gain = cur, 0.0
            for c in range(k):
                if c == cur:
                    continue
                gain = dists[i, cur] - dists[i, c]
                if gain > best_gain:
                    best_gain, best_c = gain, c
            if best_c != cur:
                centroids[cur] = (centroids[cur] * counts[cur] - data[i]) / (counts[cur] - 1)
                counts[cur] -= 1
                centroids[best_c] = (centroids[best_c] * counts[best_c] + data[i]) / (counts[best_c] + 1)
                counts[best_c] += 1
                assignments[i] = best_c
                moved = True
        if not moved:
            return centroids, assignments, used_iter + it + 1
        dists = cdist(data, centroids, metric=_METRIC_MAP[metric])
    return centroids, assignments, used_iter + max_iter


_ALGORITHMS = {"forgy_lloyd": _lloyd, "macqueen": _macqueen, "hartigan_wong": _hartigan_wong}


def _fit_once(data: np.ndarray, k: int, opt: KMeansClusteringOptions, rng: np.random.Generator):
    initial = _init_centroids(data, k, rng)
    fn = _ALGORITHMS.get(opt.algorithm, _lloyd)
    centroids, assignments, n_iter = fn(data, initial.copy(), opt.distance_method, opt.max_iterations)
    inertia = float(np.sum([np.sum((data[assignments == c] - centroids[c]) ** 2) for c in range(k) if np.any(assignments == c)]))
    return centroids, assignments, n_iter, initial, inertia


_SIL_SAMPLE = 5000  # silhouettes compare every row with every other; above this, a fixed sample


def _sil_score(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette - on a fixed random sample of rows for large sheets,
    where comparing every pair of rows would take minutes."""
    from ..ml_utils import silhouette_score

    if not 2 <= len(np.unique(labels)) < x.shape[0]:
        return float("nan")  # undefined: every row its own cluster, or only one cluster
    if x.shape[0] <= _SIL_SAMPLE:
        return float(silhouette_score(x, labels))
    idx = np.random.default_rng(0).choice(x.shape[0], _SIL_SAMPLE, replace=False)
    if not 2 <= len(np.unique(labels[idx])) < idx.size:
        return float("nan")
    return float(silhouette_score(x[idx], labels[idx]))


def _sil_rows(x: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Silhouette of the first ``count`` rows (the ones the table shows),
    each measured against every row."""
    from ..ml_utils import silhouette_samples

    if not 2 <= len(np.unique(labels)) < x.shape[0]:
        return np.zeros(count)       # one cluster in all (rows that do not differ): no silhouette to speak of
    if x.shape[0] <= _SIL_SAMPLE:
        return silhouette_samples(x, labels)[:count]
    out = np.zeros(count)
    clusters = np.unique(labels)
    d = cdist(x[:count], x)
    for i in range(count):
        own = labels == labels[i]
        n_own = own.sum() - 1
        a_i = d[i, own].sum() / n_own if n_own > 0 else 0.0
        b_i = min((d[i, labels == c].mean() for c in clusters if c != labels[i]), default=0.0)
        out[i] = 0.0 if n_own == 0 else (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
    return out


@register("kmeans_clustering")
def run(data: SheetData, options: KMeansClusteringOptions | None = None) -> AnalysisResult:
    from ..ml_utils import silhouette_samples, silhouette_score

    opt = options or KMeansClusteringOptions()
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

    sig = opt.significant_digits
    scaled = _scale(complete, opt.scaling)
    seed = None if opt.random_seed_mode == "auto" else opt.random_seed
    rng = np.random.default_rng(seed)

    lo_k = max(1, min(opt.min_clusters, complete.shape[0] - 1))
    hi_k = max(lo_k, min(opt.max_clusters, complete.shape[0] - 1))

    if opt.determine_optimal and hi_k > max(lo_k, 1):
        best_k, best_score = max(lo_k, 2), -1.0
        for k in range(max(lo_k, 2), hi_k + 1):
            _c, assign_k, _n, _init, _inertia = _fit_once(scaled, k, opt, rng)
            if len(np.unique(assign_k)) < 2:
                continue
            score = _sil_score(scaled, assign_k)
            if score > best_score:
                best_k, best_score = k, score
        k = best_k
        k_note = f"k = {k} chosen automatically (highest average silhouette score = {best_score:.3f} across k = {max(lo_k,2)}..{hi_k})."
    else:
        k = hi_k
        k_note = f"k = {k}."

    centroids, assignments, n_iter, initial_centroids, inertia = _fit_once(scaled, k, opt, rng)

    algo_names = {"hartigan_wong": "Hartigan-Wong", "forgy_lloyd": "Forgy-Lloyd", "macqueen": "MacQueen"}
    sizes = np.bincount(assignments, minlength=k)
    tables = [ResultTable(title="Cluster sizes", headers=["Cluster", "N rows"], rows=[[f"Cluster {i+1}", str(sizes[i])] for i in range(k)])]

    mean, sd = complete.mean(axis=0), complete.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    if opt.scaling == "standardize":
        centers_original = centroids * sd + mean
    elif opt.scaling == "center":
        centers_original = centroids + mean
    else:
        centers_original = centroids
    tables.append(ResultTable(title="Centre of each cluster, in the units typed", headers=["Cluster"] + labels, rows=[[f"Cluster {i+1}"] + [f"{v:.{sig}g}" for v in centers_original[i]] for i in range(k)]))

    row_labels = [str(i + 1) for i in range(complete.shape[0])]
    if opt.report_cluster_details:
        shown = min(200, complete.shape[0])
        sil = _sil_rows(scaled, assignments, shown) if 2 <= k < complete.shape[0] else None
        headers = ["Row", "Cluster"] + (["Silhouette"] if sil is not None else [])
        rows = [[row_labels[i], f"Cluster {c+1}"] + ([f"{sil[i]:.{sig}g}"] if sil is not None else [])
                for i, c in enumerate(assignments[:shown])]
        tables.append(ResultTable(title="Each row's cluster", headers=headers, rows=rows[:200]))
    if opt.report_distance_to_centroid:
        dists = cdist(scaled, centroids, metric=_METRIC_MAP[opt.distance_method])
        row_dists = dists[np.arange(scaled.shape[0]), assignments]
        tables.append(ResultTable(title="Distance of each row to its cluster center", headers=["Row", "Cluster", "Distance"], rows=[[row_labels[i], f"Cluster {assignments[i]+1}", f"{d:.{sig}g}"] for i, d in enumerate(row_dists)][:200]))
    if opt.report_initial_centroids:
        init_original = initial_centroids * sd + mean if opt.scaling == "standardize" else (initial_centroids + mean if opt.scaling == "center" else initial_centroids)
        tables.append(ResultTable(title="The centres the search started from", headers=["Cluster"] + labels, rows=[[f"Cluster {i+1}"] + [f"{v:.{sig}g}" for v in init_original[i]] for i in range(k)]))
    if opt.report_scaled_data:
        kind = "Standardized" if opt.scaling == "standardize" else "Centered" if opt.scaling == "center" else "Raw"
        tables.append(ResultTable(title=f"{kind} data", headers=["Row"] + labels, rows=[[row_labels[i]] + [f"{v:.{sig}g}" for v in row] for i, row in enumerate(scaled)][:200]))

    charts = []
    if opt.graph_elbow or opt.graph_silhouette or opt.graph_gap:
        from ..chart_helpers import xy_chart

        eval_hi = max(hi_k, max(lo_k, 2) + 1)
        elbow_rows, sil_rows, gap_rows = [], [], []
        ref_rng = np.random.default_rng(0)
        mins, maxs = scaled.min(axis=0), scaled.max(axis=0)
        for kk in range(max(lo_k, 2), min(eval_hi, 10) + 1):
            _c, assign_kk, _n, _init, inertia_kk = _fit_once(scaled, kk, opt, rng)
            if opt.graph_elbow:
                elbow_rows.append([str(kk), f"{inertia_kk:.{sig}g}"])
            if opt.graph_silhouette and len(np.unique(assign_kk)) >= 2:
                sil_rows.append([str(kk), f"{_sil_score(scaled, assign_kk):.{sig}g}"])
            if opt.graph_gap:
                real_disp = np.log(max(inertia_kk, 1e-12))
                ref_disps = []
                for _ in range(10):
                    ref_data = ref_rng.uniform(mins, maxs, size=scaled.shape)
                    _rc, _ra, _rn, _ri, ref_inertia = _fit_once(ref_data, kk, opt, ref_rng)
                    ref_disps.append(np.log(max(ref_inertia, 1e-12)))
                gap_rows.append([str(kk), f"{float(np.mean(ref_disps) - real_disp):.{sig}g}"])
        for rows_k, title, y_label in (
            (elbow_rows, "Elbow plot", "Inertia (WCSS)"),
            (sil_rows, "Silhouette plot", "Silhouette score"),
            (gap_rows, "Gap statistic plot", "Gap statistic"),
        ):
            if rows_k:
                charts.append(xy_chart(title, [float(r[0]) for r in rows_k], {"": [float(r[1]) for r in rows_k]},
                                       x_label="Number of clusters", y_label=y_label))
        if elbow_rows:
            tables.append(ResultTable(title="Within-cluster scatter for each number of clusters", headers=["Number of clusters", "Inertia (WCSS)"], rows=elbow_rows))
        if sil_rows:
            tables.append(ResultTable(title="Silhouette width for each number of clusters", headers=["Number of clusters", "Silhouette score"], rows=sil_rows))
        if gap_rows:
            tables.append(ResultTable(title="Gap statistic for each number of clusters", headers=["Number of clusters", "Gap statistic"], rows=gap_rows))

    summary = f"{complete.shape[0]} rows grouped into {k} clusters across {len(labels)} variable(s) ({algo_names.get(opt.algorithm, opt.algorithm)}, {n_iter} iteration(s)). {k_note}"
    notes = [f"Data were scaled ({opt.scaling}) before clustering, using {opt.distance_method} distance."]
    if opt.algorithm == "macqueen":
        notes.append("MacQueen's algorithm updates a centroid immediately after each point is (re)assigned, rather than waiting for a full pass like Lloyd's.")
    elif opt.algorithm == "hartigan_wong":
        notes.append("Hartigan-Wong here is a Lloyd's fit refined by single-point moves that reduce total within-cluster sum of squares - the idea of the method, written afresh, and not a line-by-line copy of the original Fortran routine.")

    return AnalysisResult(title="Split into k groups", summary=summary, tables=tables, notes=notes, charts=charts)
