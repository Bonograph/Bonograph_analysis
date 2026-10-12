"""Find the main axes of many variables (principal components). Runs across every selected group's first replicate
column (one variable each, rows as observations)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register


@dataclass
class PCAOptions:
    # the data
    selected_variables: list[str] = field(default_factory=list)  # empty = all
    perform_pcr: bool = False
    pcr_outcome: str = ""
    # what to report
    method: str = "standardize"  # "standardize" | "center"
    pc_selection: str = "parallel_analysis"  # "parallel_analysis" | "all"
    percentile_level: float = 0.95
    num_simulations: int = 1000
    random_seed_mode: str = "auto"  # "auto" | "fixed"
    random_seed: int = 0
    # tables to report
    report_standardized_data: bool = False
    report_eigenvectors: bool = False
    report_variable_contribution: bool = False
    report_variable_pc_correlation: bool = False
    report_case_contribution: bool = False
    report_corr_cov_matrix: bool = False
    significant_digits: int = 4
    # graphs (each adds a chart plus its data table)
    graph_score_plot: bool = True
    graph_loading_plot: bool = True
    graph_biplot: bool = True
    graph_scree_plot: bool = False
    graph_proportion_variance: bool = True


def _parallel_analysis(n_obs: int, n_vars: int, eigenvalues: np.ndarray, percentile: float, n_sims: int, seed: int | None) -> np.ndarray:
    """Horn's parallel analysis: eigenvalues of PCA run on random data of
    the same shape, compared to the real data's eigenvalues at the given
    percentile - a PC is "real" if its eigenvalue beats the simulated
    threshold for that component."""
    rng = np.random.default_rng(seed)
    sim_eigs = np.zeros((n_sims, n_vars))
    for i in range(n_sims):
        random_data = rng.standard_normal((n_obs, n_vars))
        corr = np.corrcoef(random_data, rowvar=False)
        eigs = np.linalg.eigvalsh(corr)[::-1]
        sim_eigs[i] = eigs
    thresholds = np.percentile(sim_eigs, percentile * 100, axis=0)
    return thresholds


@register("pca")
def run(data: SheetData, options: PCAOptions | None = None) -> AnalysisResult:
    from ..ml_utils import PCA as _SKPCA

    opt = options or PCAOptions()
    all_labels, all_matrix = data.first_replicate_matrix()
    if len(all_labels) < 2:
        raise AnalysisError(f"PCA needs at least 2 variables (group columns); found {len(all_labels)}.")

    if opt.selected_variables:
        keep = [i for i, lbl in enumerate(all_labels) if lbl in opt.selected_variables]
        if len(keep) < 2:
            raise AnalysisError("Select at least 2 variables for PCA.")
        labels = [all_labels[i] for i in keep]
        matrix = all_matrix[:, keep]
    else:
        labels, matrix = all_labels, all_matrix

    complete_mask = ~np.any(np.isnan(matrix), axis=1)
    complete = matrix[complete_mask]
    if complete.shape[0] < 3:
        raise AnalysisError(f"Need at least 3 complete rows across all {matrix.shape[1]} variables; found {complete.shape[0]}.")

    n_obs, n_vars = complete.shape
    sig = opt.significant_digits
    means = complete.mean(axis=0)
    if opt.method == "standardize":
        sds = complete.std(axis=0, ddof=1)
        sds = np.where(sds == 0, 1.0, sds)
        scaled = (complete - means) / sds
        basis_name = "correlation matrix"
    else:
        scaled = complete - means
        sds = np.ones(n_vars)
        basis_name = "covariance matrix"

    n_components_max = min(n_obs, n_vars)
    pca = _SKPCA(n_components=n_components_max)
    scores = pca.fit_transform(scaled)
    eigenvalues = pca.explained_variance_

    n_keep = n_components_max
    parallel_note = None
    if opt.pc_selection == "parallel_analysis":
        seed = None if opt.random_seed_mode == "auto" else opt.random_seed
        corr_eigs = np.linalg.eigvalsh(np.corrcoef(scaled, rowvar=False))[::-1] if opt.method == "standardize" else eigenvalues
        thresholds = _parallel_analysis(n_obs, n_vars, corr_eigs, opt.percentile_level, opt.num_simulations, seed)
        n_keep = int(np.sum(corr_eigs[:len(thresholds)] > thresholds))
        n_keep = max(1, n_keep)
        parallel_note = f"Parallel analysis ({opt.num_simulations} simulations, {int(opt.percentile_level*100)}th percentile) retained {n_keep} of {n_components_max} components."

    tables = []
    charts = []
    variance_rows = [
        [f"PC{i + 1}", f"{eigenvalues[i]:.{sig}g}", f"{ratio * 100:.2f}%", f"{np.cumsum(pca.explained_variance_ratio_)[i] * 100:.2f}%", "Yes" if i < n_keep else "No"]
        for i, ratio in enumerate(pca.explained_variance_ratio_)
    ]
    tables.append(ResultTable(title=f"Explained variance (based on the {basis_name})", headers=["Component", "Eigenvalue", "% variance", "Cumulative %", "Retained"], rows=variance_rows))

    loading_rows = [[labels[j]] + [f"{pca.components_[i, j]:.{sig}g}" for i in range(n_components_max)] for j in range(len(labels))]
    tables.append(ResultTable(title="Component loadings", headers=["Variable"] + [f"PC{i + 1}" for i in range(n_components_max)], rows=loading_rows))

    score_headers = ["Row"] + [f"PC{i + 1}" for i in range(n_components_max)]
    obs_indices = np.where(complete_mask)[0]
    score_rows = [[str(int(r) + 1)] + [f"{v:.{sig}g}" for v in scores[i]] for i, r in enumerate(obs_indices)]
    tables.append(ResultTable(title="Each row's score on the components", headers=score_headers, rows=score_rows[:200]))

    if opt.report_standardized_data:
        tables.append(ResultTable(title=f"{'Standardized' if opt.method == 'standardize' else 'Centered'} data", headers=["Row"] + labels, rows=[[str(int(r) + 1)] + [f"{v:.{sig}g}" for v in row] for r, row in zip(obs_indices, scaled)][:200]))
    if opt.report_eigenvectors:
        tables.append(ResultTable(title="Eigenvectors", headers=["Variable"] + [f"PC{i+1}" for i in range(n_components_max)], rows=loading_rows))
    if opt.report_variable_contribution:
        contrib = (pca.components_.T ** 2) * 100
        tables.append(ResultTable(title="Each variable's share of each component (%)", headers=["Variable"] + [f"PC{i+1}" for i in range(n_components_max)], rows=[[labels[j]] + [f"{contrib[j, i]:.{sig}g}" for i in range(n_components_max)] for j in range(len(labels))]))
    if opt.report_variable_pc_correlation:
        var_pc_corr = pca.components_.T * np.sqrt(np.clip(eigenvalues, 0, None))[None, :]
        if opt.method == "standardize":
            var_pc_corr = var_pc_corr  # already correlations when input is standardized
        tables.append(ResultTable(title="Correlation of each variable with each component", headers=["Variable"] + [f"PC{i+1}" for i in range(n_components_max)], rows=[[labels[j]] + [f"{var_pc_corr[j, i]:.{sig}g}" for i in range(n_components_max)] for j in range(len(labels))]))
    if opt.report_case_contribution:
        case_contrib = (scores ** 2) / np.where(eigenvalues * (n_obs - 1) == 0, 1, eigenvalues * (n_obs - 1))[None, :] * 100
        tables.append(ResultTable(title="Each row's share of each component (%)", headers=score_headers, rows=[[str(int(r) + 1)] + [f"{v:.{sig}g}" for v in case_contrib[i]] for i, r in enumerate(obs_indices)][:200]))
    if opt.report_corr_cov_matrix:
        mat = np.corrcoef(complete, rowvar=False) if opt.method == "standardize" else np.cov(complete, rowvar=False)
        title = "Correlation matrix between variables" if opt.method == "standardize" else "Covariance matrix between variables"
        tables.append(ResultTable(title=title, headers=["Variable"] + labels, rows=[[labels[i]] + [f"{mat[i,j]:.{sig}g}" for j in range(len(labels))] for i in range(len(labels))]))

    from ..backend_types import ChartSeries, ChartSpec
    from ..chart_helpers import xy_chart

    pc_numbers = np.arange(1, len(eigenvalues) + 1)
    pc_axes = [f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
               f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)" if n_components_max >= 2 else "PC2"]
    origin_lines = [{"kind": "h", "value": 0.0}, {"kind": "v", "value": 0.0}]
    if n_components_max < 2:
        from ..chart_helpers import unavailable_chart

        for wanted, name in ((opt.graph_score_plot, "Scores"), (opt.graph_loading_plot, "Loadings"), (opt.graph_biplot, "Biplot")):
            if wanted:
                charts.append(unavailable_chart(name, f"The {name.lower()} shows PC1 against PC2, but only 1 component "
                                                      "could be computed. It needs at least 2 variables and 2 complete rows."))
    if opt.graph_scree_plot:
        charts.append(xy_chart("Scree plot", pc_numbers, {"": eigenvalues}, x_label="Principal component", y_label="Eigenvalue",
                               reference_lines=[{"kind": "h", "value": 1.0}] if opt.method == "standardize" else None))
        tables.append(ResultTable(title="Eigenvalue of each component", headers=["PC", "Eigenvalue"], rows=[[f"PC{i+1}", f"{e:.{sig}g}"] for i, e in enumerate(eigenvalues)]))
    if opt.graph_proportion_variance:
        charts.append(xy_chart("Variance carried by each component", pc_numbers, {"% of variance": pca.explained_variance_ratio_ * 100,
                                                                      "Cumulative %": np.cumsum(pca.explained_variance_ratio_) * 100},
                               kind="line_markers", x_label="Principal component", y_label="% of variance"))
        charts[-1].series[0].kind = "bar"
        tables.append(ResultTable(title="Share of the variance carried by each component", headers=["PC", "% Variance"], rows=[[f"PC{i+1}", f"{r*100:.{sig}g}%"] for i, r in enumerate(pca.explained_variance_ratio_)]))
    if opt.graph_score_plot and n_components_max >= 2:
        notes_rows = [{"x": float(scores[i, 0]), "y": float(scores[i, 1]), "text": str(int(r) + 1)} for i, r in enumerate(obs_indices)]
        charts.append(ChartSpec(title="Scores", series=[ChartSeries("", [float(v) for v in scores[:, 0]], [float(v) for v in scores[:, 1]])],
                                x_label=pc_axes[0], y_label=pc_axes[1], reference_lines=origin_lines,
                                annotations=notes_rows if len(notes_rows) <= 40 else []))
        tables.append(ResultTable(title="Scores on the first two components", headers=["Row", "PC1", "PC2"], rows=[[str(int(r) + 1), f"{scores[i,0]:.{sig}g}", f"{scores[i,1]:.{sig}g}"] for i, r in enumerate(obs_indices)][:200]))
    if opt.graph_loading_plot and n_components_max >= 2:
        charts.append(ChartSpec(title="Loadings", series=[ChartSeries("", [float(v) for v in pca.components_[0]], [float(v) for v in pca.components_[1]])],
                                x_label="PC1 loading", y_label="PC2 loading", reference_lines=origin_lines,
                                annotations=[{"x": float(pca.components_[0, j]), "y": float(pca.components_[1, j]), "text": labels[j]} for j in range(len(labels))]))
        tables.append(ResultTable(title="Loadings on the first two components", headers=["Variable", "PC1 loading", "PC2 loading"], rows=[[labels[j], f"{pca.components_[0,j]:.{sig}g}", f"{pca.components_[1,j]:.{sig}g}"] for j in range(len(labels))]))
    if opt.graph_biplot and n_components_max >= 2:
        # loadings stretched to the score cloud so both are readable on one set of axes
        reach = float(np.max(np.abs(scores[:, :2]))) or 1.0
        stretch = reach / (float(np.max(np.abs(pca.components_[:2]))) or 1.0) * 0.9
        arrows = [ChartSeries("", [0.0, float(pca.components_[0, j] * stretch)], [0.0, float(pca.components_[1, j] * stretch)], kind="line", color="#d62728")
                  for j in range(len(labels))]
        charts.append(ChartSpec(title="Biplot", series=[ChartSeries("", [float(v) for v in scores[:, 0]], [float(v) for v in scores[:, 1]])] + arrows,
                                x_label=pc_axes[0], y_label=pc_axes[1], reference_lines=origin_lines,
                                annotations=[{"x": float(pca.components_[0, j] * stretch), "y": float(pca.components_[1, j] * stretch), "text": labels[j]} for j in range(len(labels))]))
        tables.append(ResultTable(
            title="Biplot data (scores + loadings on PC1/PC2)",
            headers=["Kind", "Name", "PC1", "PC2"],
            rows=[["Case", str(int(r) + 1), f"{scores[i,0]:.{sig}g}", f"{scores[i,1]:.{sig}g}"] for i, r in enumerate(obs_indices)][:100]
            + [["Variable", labels[j], f"{pca.components_[0,j]:.{sig}g}", f"{pca.components_[1,j]:.{sig}g}"] for j in range(len(labels))],
        ))

    notes = [f"{'Standardized' if opt.method=='standardize' else 'Mean-centered'} the {len(labels)} selected variable(s) before PCA (based on their {basis_name})."]
    if parallel_note:
        notes.append(parallel_note)

    summary = f"{n_obs} observations, {len(labels)} variables; PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance."

    if opt.perform_pcr and opt.pcr_outcome:
        outcome_values = None
        for i, lbl in enumerate(all_labels):
            if lbl == opt.pcr_outcome:
                outcome_values = all_matrix[complete_mask, i]
        if outcome_values is None or np.any(np.isnan(outcome_values)):
            notes.append(f"Principal Component Regression outcome '{opt.pcr_outcome}' wasn't found, or has missing values among the complete rows used for PCA.")
        else:
            y = outcome_values
            k = n_keep
            design = np.column_stack([np.ones(n_obs), scores[:, :k]])
            coeffs, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coeffs
            resid = y - fitted
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
            pcr_rows = [["Intercept", f"{coeffs[0]:.{sig}g}"]] + [[f"PC{i+1}", f"{coeffs[i+1]:.{sig}g}"] for i in range(k)]
            tables.append(ResultTable(title=f"Principal Component Regression predicting '{opt.pcr_outcome}' (R² = {r2:.4f}, using {k} retained PC(s))", headers=["Term", "Coefficient"], rows=pcr_rows))
            summary += f" PCR: R² = {r2:.4f} predicting '{opt.pcr_outcome}' from {k} retained PC(s)."

    return AnalysisResult(title="Principal Component Analysis", summary=summary, tables=tables, notes=notes, charts=charts)
