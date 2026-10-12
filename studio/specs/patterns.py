"""Patterns across variables: principal components, and clusters (a tree, or
a set number)."""

from __future__ import annotations

from ...backends.discriminant_analysis import DiscriminantOptions
from ...backends.factor_analysis import FactorAnalysisOptions
from ...backends.hierarchical_clustering import HierarchicalClusteringOptions
from ...backends.kmeans_clustering import KMeansClusteringOptions
from ...backends.pca import PCAOptions
from ..spec import Ask, Chips, Choose, Custom, Drawer, Heading, Int, Note, Num, Opt, Pick, Row, Sentence, Spec, Switch
from .editors import variable_chips
from .models import subset_editor

_SEED = [
    Row("Random numbers", Pick("random_seed_mode", [("auto", "new ones every run"), ("fixed", "repeatable, from a seed")])),
    Sentence(["Seed", Int("random_seed", 0, 2_000_000_000, 96)], when=lambda s, c: s["random_seed_mode"] == "fixed"),
]
_SCALES = [("standardize", "to mean 0 and SD 1"), ("center", "to mean 0 only"), ("none", "not at all")]
_DISTANCES = [("euclidean", "straight-line (Euclidean)"), ("manhattan", "city-block (Manhattan)"),
              ("correlation", "1 − Pearson correlation"), ("cosine", "cosine")]

PCA = Spec(
    "pca", PCAOptions,
    [
        Custom(lambda p: variable_chips()),
        Heading("Put the variables on one scale first?"),
        Choose("method", [
            Opt("standardize", "Yes, the same scale", "For different units or very different spreads - the usual "
                                                     "choice (correlation matrix)."),
            Opt("center", "No, keep their spreads", "Same units, where a bigger spread should count for more "
                                                    "(covariance matrix)."),
        ]),
        Heading("How many components to keep?"),
        Choose("pc_selection", [
            Opt("parallel_analysis", "Those that beat chance", "Kept only if they explain more than the same "
                                                               "component in random tables of the same size."),
            Opt("all", "All of them"),
        ]),
        Sentence(["Beat random data at the", Pick("percentile_level", [(0.95, "95th"), (0.99, "99th"), (0.5, "50th")]),
                  "percentile of", Int("num_simulations", 10, 100000, 64), "random tables"],
                 when=lambda s, c: s["pc_selection"] == "parallel_analysis"),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("How many dimensions matter?",
            "How much of the variation each component carries, and where the amounts level off.",
            pieces=[("graph_scree_plot", "Scree plot"), ("graph_proportion_variance", "Graph of variance explained")]),
        Ask("What does each component stand for?",
            "Which variables pull along it, and how strongly.",
            pieces=[("graph_loading_plot", "Graph of the loadings"), ("report_eigenvectors", "Loadings (eigenvectors)"),
                    ("report_variable_pc_correlation", "Correlation of each variable with each component"),
                    ("report_variable_contribution", "Each variable's share of a component")],
            first=["graph_loading_plot", "report_eigenvectors"]),
        Ask("Where does each row fall?",
            "The rows placed on the components: groups, gradients and stragglers show up here.",
            pieces=[("graph_score_plot", "Graph of the scores"), ("graph_biplot", "Biplot", "Rows and variables on one graph."),
                    ("report_case_contribution", "Each row's share of a component")],
            first=["graph_score_plot"]),
        Ask("What went into the analysis?",
            "The matrix the components were found from, and the data as scaled.",
            pieces=[("report_corr_cov_matrix", "Correlations between the variables"),
                    ("report_standardized_data", "The scaled data")]),
        Drawer("Fine-tune", [
            Switch("perform_pcr", "Also regress an outcome on the components", "Principal component regression."),
            Row("Outcome", Pick("pcr_outcome", lambda c: [(n, n) for n in c.names] or [("", "—")]),
                when=lambda s, c: s["perform_pcr"]),
            *_SEED,
        ]),
    ],
    on_sheet=lambda p: p.suggest({"pcr_outcome": p.ctx.names[-1] if p.ctx.names else ""}),
    methods=lambda s, c: ("Principal components were extracted from the " +
                          ("correlation" if s["method"] == "standardize" else "covariance") + " matrix" +
                          ("; components were kept by parallel analysis." if s["pc_selection"] == "parallel_analysis" else ".")),
)

TREE = Spec(
    "hierarchical_clustering", HierarchicalClusteringOptions,
    [
        Custom(lambda p: variable_chips()),
        Heading("Cluster"),
        Choose("clustering_direction", [Opt("rows", "The rows", "Subjects that resemble each other."),
                                        Opt("columns", "The variables", "Variables that move together."),
                                        Opt("both", "Both")], columns=3),
        Heading("Distance and joining"),
        Row("Scale each variable", Pick("column_scaling", _SCALES)),
        Row("Scale each row", Pick("row_scaling", [("none", "not at all"), ("standardize", "to mean 0 and SD 1"),
                                                  ("center", "to mean 0 only")])),
        Row("Distance", Pick("distance_method", _DISTANCES)),
        Row("Join clusters by", Pick("linkage_method", [
            ("complete", "their farthest members"), ("single", "their nearest members"),
            ("group_average", "the average over all pairs (UPGMA)"), ("simple_average", "the average of the two (WPGMA)"),
            ("centroid", "their centroids"), ("median", "their medians"), ("ward", "least added variance (Ward)")])),
        Heading("Cut the tree into clusters?"),
        Switch("define_rows", "Cut the rows' tree", when=lambda s, c: s["clustering_direction"] != "columns"),
        Sentence(["Cut", Pick("row_cluster_mode", [("count", "into"), ("height", "at height")]),
                  Int("row_n_clusters", 2, 1000, 50, live=lambda s, c: s["row_cluster_mode"] == "count"),
                  "clusters, or height", Num("row_height", 0, 1e12, 4, 70, live=lambda s, c: s["row_cluster_mode"] == "height")],
                 when=lambda s, c: s["define_rows"] and s["clustering_direction"] != "columns"),
        Switch("define_columns", "Cut the variables' tree", when=lambda s, c: s["clustering_direction"] != "rows"),
        Sentence(["Cut", Pick("col_cluster_mode", [("count", "into"), ("height", "at height")]),
                  Int("col_n_clusters", 2, 1000, 50, live=lambda s, c: s["col_cluster_mode"] == "count"),
                  "clusters, or height", Num("col_height", 0, 1e12, 4, 70, live=lambda s, c: s["col_cluster_mode"] == "height")],
                 when=lambda s, c: s["define_columns"] and s["clustering_direction"] != "rows"),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("What joins to what?",
            "The tree, read from the leaves up, and the order in which clusters were joined.",
            pieces=[("graph_dendrogram", "Dendrogram"), ("report_merge_map", "The order clusters joined")],
            first=["graph_dendrogram"]),
        Ask("How many clusters are there?",
            "Three ways to judge it. They need not agree; a number is convincing when they do.",
            pieces=[("graph_elbow", "Elbow plot"), ("graph_silhouette", "Silhouette plot"), ("graph_gap", "Gap statistic")]),
        Ask("Does the tree keep the distances?",
            "How closely the heights in the tree follow the distances between the rows themselves.",
            pieces=[("report_cophenetic_correlation", "Cophenetic correlation"),
                    ("report_cophenetic_distance", "Cophenetic distances")],
            first=["report_cophenetic_correlation"]),
        Ask("What do the data and the distances look like?",
            "Heat maps in the order of the tree, and the numbers behind them.",
            pieces=[("graph_heatmap_data", "Heat map of the data"), ("graph_heatmap_distances", "Heat map of the distances"),
                    ("report_distance_matrix", "Table of distances"), ("report_scaled_data", "The scaled data"),
                    ("report_full_matrices", "Both halves of each matrix")],
            first=["graph_heatmap_data"]),
    ],
    methods=lambda s, c: (f"Rows were clustered hierarchically ({dict(_DISTANCES)[s['distance_method']]} distance, "
                          f"{s['linkage_method'].replace('_', ' ')} linkage)."),
)

KMEANS = Spec(
    "kmeans_clustering", KMeansClusteringOptions,
    [
        Custom(lambda p: variable_chips()),
        Heading("How many clusters?"),
        Sentence(["Try", Int("min_clusters", 1, 100, 44), "to", Int("max_clusters", 1, 100, 44), "clusters"]),
        Switch("determine_optimal", "Suggest the best number in that range", "By silhouette score."),
        Heading("Distance"),
        Row("Scale each variable", Pick("scaling", _SCALES)),
        Row("Distance", Pick("distance_method", _DISTANCES)),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("What is each cluster like?",
            "Its size and its centre, and how far each row lies from the centre of its own cluster.",
            pieces=[("report_cluster_details", "Each cluster's size and centre"),
                    ("report_distance_to_centroid", "Each row's distance to its centre")],
            first=["report_cluster_details"]),
        Ask("How many clusters are there?",
            "Three ways to judge it. They need not agree; a number is convincing when they do.",
            pieces=[("graph_elbow", "Elbow plot"), ("graph_silhouette", "Silhouette plot"), ("graph_gap", "Gap statistic")]),
        Ask("What went into the analysis?",
            "The centres the search started from, and the data as scaled.",
            pieces=[("report_initial_centroids", "Starting centres"), ("report_scaled_data", "The scaled data")]),
        Drawer("Fine-tune", [
            Row("Algorithm", Pick("algorithm", [("hartigan_wong", "Hartigan-Wong (usual)"), ("forgy_lloyd", "Lloyd"),
                                               ("macqueen", "MacQueen")])),
            Sentence(["Stop after", Int("max_iterations", 1, 100000, 64), "rounds"]),
            *_SEED,
        ]),
    ],
    methods=lambda s, c: f"Rows were divided into clusters by k-means ({s['min_clusters']} to {s['max_clusters']} tried).",
)

# ── hidden factors ──────────────────────────────────────────────────────────

_ROTATION_WORDS = {"none": "no rotation", "varimax": "varimax rotation", "quartimax": "quartimax rotation",
                   "promax": "promax rotation", "oblimin": "direct oblimin rotation"}


def _factor_methods(s, ctx) -> str:
    how = "principal axis factoring" if s["extraction"] == "principal_axis" else "maximum likelihood"
    many = {"parallel": "the number of factors was chosen by parallel analysis",
            "eigenvalue": "factors with eigenvalues above 1 were kept",
            "fixed": f"{s['n_factors']} factor{'s were' if s['n_factors'] != 1 else ' was'} extracted"}[s["how_many"]]
    return f"Exploratory factor analysis by {how} with {_ROTATION_WORDS[s['rotation']]}; {many}."


FACTORS = Spec(
    "factor_analysis", FactorAnalysisOptions,
    [
        Custom(lambda p: variable_chips()),
        Heading("How many factors?"),
        Choose("how_many", [
            Opt("parallel", "Those that beat chance", "Kept while they explain more than the same factor does in random "
                                                      "tables of this size."),
            Opt("eigenvalue", "Eigenvalues above 1", "The classical rule; it tends to keep too many."),
            Opt("fixed", "A number I choose", "When theory says how many there should be."),
        ], columns=3),
        Sentence(["Extract", Int("n_factors", 1, 50, 50), "factors"], when=lambda s, c: s["how_many"] == "fixed"),
        Heading("How to find them"),
        Choose("extraction", [
            Opt("principal_axis", "Principal axis", "Assumes nothing about the shape of the data - the safe choice."),
            Opt("maximum_likelihood", "Maximum likelihood", "For roughly bell-shaped variables; adds a test of whether "
                                                            "the factors are enough."),
        ]),
        Heading("How to turn them"),
        Choose("rotation", [
            Opt("varimax", "Varimax", "Factors stay unrelated; each variable leans on as few as possible."),
            Opt("promax", "Promax", "Factors may be related - usual for traits and abilities."),
            Opt("oblimin", "Oblimin", "Factors may be related; slower and more exact."),
            Opt("quartimax", "Quartimax", "Factors stay unrelated; favours one general factor."),
            Opt("none", "Not at all", "The factors as first found."),
        ], columns=3),
        Heading("Show the loadings"),
        Switch("sort_loadings", "Grouped by the factor each variable follows most"),
        Sentence(["Leave blank any loading smaller than", Num("hide_below", 0, 1, 2, 58), "(0 shows them all)"]),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("Are these data suited to factor analysis?",
            "Whether the variables share enough for factors to be worth looking for.",
            pieces=[("show_suitability", "Sampling adequacy and Bartlett's test"),
                    ("show_each_variable", "Each variable's adequacy and shared variance")],
            first=["show_suitability"]),
        Ask("How many factors are there?", "The eigenvalues in order, against what random data would give.",
            pieces=[("graph_scree", "Scree plot")]),
        Ask("What does each factor stand for?",
            "Which variables follow it, seen as a picture.",
            pieces=[("graph_heatmap", "Loadings at a glance"), ("graph_loadings", "Loadings, factor against factor"),
                    ("show_structure", "Correlations of variables with factors", "For turns that let the factors relate.")],
            first=["graph_heatmap"]),
        Ask("How well do the factors reproduce the correlations?",
            "What is left over once the factors have accounted for what they can.",
            pieces=[("show_fit", "Reproduced correlations and what is left")]),
        Ask("What is each row's score on the factors?", "The scores on a new sheet, ready for further analysis.",
            pieces=[("make_scores", "Factor scores, on a new sheet"), ("show_score_weights", "The weights the scores are made with")],
            first=["make_scores"]),
        Drawer("Fine-tune", [
            Sentence(["Promax power", Num("promax_power", 2, 8, 1, 58)], when=lambda s, c: s["rotation"] == "promax"),
            Sentence(["Judge chance from", Int("num_simulations", 50, 5000, 64), "random tables, seed",
                      Int("random_seed", 0, 2_000_000_000, 96)]),
        ]),
    ],
    methods=_factor_methods,
)

# ── known groups ────────────────────────────────────────────────────────────


def _group_sheet(panel) -> None:
    """The group column is most likely the one holding a few whole-number codes: the last such data set."""
    import numpy as np

    names = panel.ctx.names
    guess = len(names) - 1
    if panel.ctx.sheet is not None:
        try:
            labels, matrix = panel.ctx.sheet.first_replicate_matrix()
            for i in range(len(labels) - 1, -1, -1):
                values = matrix[:, i]
                values = values[np.isfinite(values)]
                if values.size and np.all(values == np.round(values)) and 2 <= np.unique(values).size <= 12:
                    guess = i
                    break
        except Exception:  # noqa: BLE001
            pass
    current = panel.state.get("group_index", -1)
    if current < 0 or current >= max(len(names), 1):
        panel.state["group_index"] = max(guess, 0)
    else:
        panel.suggest({"group_index": max(guess, 0)})


def _measurements(panel):
    names = panel.ctx.names or ["Set A", "Set B", "Set C"]
    return [n for i, n in enumerate(names) if i != panel.state.get("group_index", -1)]


def _group_finish(s, ctx) -> None:
    if s["group_index"] < 0 or s["group_index"] >= max(len(ctx.names), 1):
        s["group_index"] = max(len(ctx.names) - 1, 0)


GROUPS_APART = Spec(
    "discriminant_analysis", DiscriminantOptions,
    [
        Heading("Which data set holds the groups?"),
        Row("The groups are in", Pick("group_index", lambda c: c.column_options())),
        Note("Each subject's group is a code - 1, 2, 3 ... Every other data set ticked below is a measurement.", tone="faint"),
        Custom(lambda p: subset_editor(_measurements, "selected_variables", "Measurements to use")),
        Heading("Before looking at a subject, how likely is each group?"),
        Choose("priors", [
            Opt("equal", "All equally likely", "The neutral choice."),
            Opt("sizes", "As common as in this sample", "When the sample's group sizes reflect the real world."),
        ]),
        Heading("What do you want to know?", "Switch on the questions you want answered. Each brings its own tables and graphs."),
        Ask("Can the groups be told apart?",
            "Tests of the functions that separate them, and of each measurement taken alone.",
            pieces=[("show_functions", "The discriminant functions and their tests"),
                    ("show_each_variable", "Each measurement on its own")]),
        Ask("What tells them apart?",
            "The weight each measurement carries in the separation.",
            pieces=[("show_coefficients", "Weights and correlations of the measurements"),
                    ("graph_weights", "Graph of what Function 1 reflects"),
                    ("show_raw_coefficients", "Weights for the values as typed"),
                    ("show_group_means", "Group sizes and means")],
            first=["show_coefficients", "graph_weights"]),
        Ask("How well would the rule place a new subject?",
            "Each subject classified by a rule built without it - the honest estimate of how well the rule works.",
            pieces=[("show_cross_validation", "Leave-one-out classification"),
                    ("show_classification_functions", "Classification functions")],
            first=["show_cross_validation"]),
        Ask("Where does each subject fall?",
            "Subjects plotted on the functions, and their scores and predicted groups on a new sheet.",
            pieces=[("graph_scores", "Graph of subjects on the functions"),
                    ("make_scores", "Scores and predicted groups, on a new sheet")],
            first=["graph_scores"]),
        Ask("Do the groups vary alike?",
            "The method assumes they do. Box's M test checks it; it is very sensitive in large samples.",
            pieces=[("show_box_m", "Box's M test")]),
    ],
    finish=_group_finish,
    on_sheet=_group_sheet,
    methods=lambda s, c: ("Groups were separated by linear discriminant analysis ("
                          + ("equal" if s["priors"] == "equal" else "sample-proportional") + " prior probabilities); the overall "
                          "difference was tested with Wilks' lambda"
                          + (" and classification was checked by leave-one-out cross-validation." if s["show_cross_validation"] else ".")),
)

SPECS = [PCA, FACTORS, TREE, KMEANS, GROUPS_APART]
