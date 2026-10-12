"""Discriminant analysis: which measurements tell known groups apart, and how
well a new subject could be placed in one of them.

Each row is a subject. One data set holds the group as a code (1, 2, 3 ...);
the others are the measurements. The analysis finds the combinations of
measurements that separate the groups most (the discriminant functions),
tests whether the groups differ at all, and classifies every subject - both
with all the data and, more honestly, with that subject left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _regress as rg

MOST_GROUPS = 12


@dataclass
class DiscriminantOptions:
    group_index: int = -1               # which data set holds the group codes; -1 = the last
    selected_variables: list[str] = field(default_factory=list)   # measurements to use; empty = every other data set
    priors: str = "equal"               # chance of each group before looking: "equal" | "sizes"
    show_group_means: bool = True
    show_each_variable: bool = True     # each measurement on its own: F test and Wilks' lambda
    show_functions: bool = True         # eigenvalues, canonical correlations and tests of the functions
    show_coefficients: bool = True      # standardised weights and correlations with the functions
    show_raw_coefficients: bool = False
    show_classification_functions: bool = False
    show_box_m: bool = True             # are the groups' covariance matrices alike?
    show_cross_validation: bool = True  # leave-one-out classification
    make_scores: bool = False           # each subject's scores, predicted group and chances on a new sheet
    graph_scores: bool = True
    graph_weights: bool = True
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def scatter_matrices(X: np.ndarray, codes: np.ndarray, g: int):
    """(group sizes, group means, within-groups and between-groups sums of squares and products)."""
    n, p = X.shape
    sizes = np.bincount(codes, minlength=g).astype(float)
    means = np.vstack([X[codes == j].mean(axis=0) for j in range(g)])
    centred = X - means[codes]
    W = centred.T @ centred
    gap = means - X.mean(axis=0)
    B = (gap * sizes[:, None]).T @ gap
    return sizes, means, W, B


def canonical(W: np.ndarray, B: np.ndarray, n: int, g: int):
    """Discriminant functions: (eigenvalues, raw coefficients scaled so the
    pooled within-group variance of every function is 1)."""
    p = W.shape[0]
    values, vectors = linalg.eigh(B, W)
    order = np.argsort(values)[::-1][: min(p, g - 1)]
    values = np.clip(values[order], 0, None)
    vectors = vectors[:, order] * np.sqrt(n - g)          # eigh gives v'Wv = 1; make it v'(W/(n-g))v = 1
    return values, vectors


def wilks_tests(values: np.ndarray, n: int, p: int, g: int) -> list[tuple[int, float, float, int, float]]:
    """Functions k..last together: (first function, Wilks' lambda, chi-square, df, P) by Bartlett's approximation."""
    out = []
    for k in range(values.size):
        lam = float(np.prod(1 / (1 + values[k:])))
        chi2 = -(n - 1 - (p + g) / 2) * np.log(lam) if lam > 0 else float("inf")
        df = (p - k) * (g - 1 - k)
        out.append((k + 1, lam, float(chi2), int(df), float(stats.chi2.sf(chi2, df))))
    return out


def rao_f(lam: float, n: int, p: int, g: int) -> tuple[float, float, float, float]:
    """Rao's F approximation to Wilks' lambda for all functions: (F, df1, df2, P). Exact for two or three groups."""
    q = g - 1
    df1 = p * q
    w = n - 1 - (p + q + 1) / 2
    t = np.sqrt((p**2 * q**2 - 4) / (p**2 + q**2 - 5)) if p**2 + q**2 - 5 > 0 else 1.0
    df2 = w * t - (p * q - 2) / 2
    root = lam ** (1 / t)
    F = (1 - root) / root * df2 / df1 if root > 0 else float("inf")
    return float(F), float(df1), float(df2), float(stats.f.sf(F, df1, df2))


def box_m(X: np.ndarray, codes: np.ndarray, g: int, W: np.ndarray):
    """Box's M test that the groups share one covariance matrix:
    (M, chi-square, df, P), or None when a group is too small for it."""
    n, p = X.shape
    sizes = np.bincount(codes, minlength=g)
    if np.any(sizes <= p):
        return None
    pooled_sign, pooled_log = np.linalg.slogdet(W / (n - g))
    if pooled_sign <= 0:
        return None
    M = (n - g) * pooled_log
    for j in range(g):
        sign, logdet = np.linalg.slogdet(np.cov(X[codes == j], rowvar=False).reshape(p, p))
        if sign <= 0:
            return None
        M -= (sizes[j] - 1) * logdet
    c = (np.sum(1 / (sizes - 1)) - 1 / (n - g)) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    df = p * (p + 1) * (g - 1) // 2
    chi2 = float(M * (1 - c))
    return float(M), chi2, int(df), float(stats.chi2.sf(chi2, df))


def posterior(D2: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Chance of each group from squared distances to the group centres (rows x groups)."""
    score = -0.5 * D2 + np.log(prior)
    score -= score.max(axis=1, keepdims=True)
    e = np.exp(score)
    return e / e.sum(axis=1, keepdims=True)


def distances(X: np.ndarray, means: np.ndarray, S_inv: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every row to every group centre."""
    out = np.empty((X.shape[0], means.shape[0]))
    for j in range(means.shape[0]):
        gap = X - means[j]
        out[:, j] = np.einsum("ij,jk,ik->i", gap, S_inv, gap)
    return out


def leave_one_out(X: np.ndarray, codes: np.ndarray, sizes: np.ndarray, means: np.ndarray, W: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """The group each subject is placed in when the rule is rebuilt without it.

    Removing subject i from its group a moves that group's centre and takes
    c·d·d' off the within-groups matrix (d = the subject's gap from its group
    centre, c = n_a / (n_a - 1)); the inverse follows from the
    Sherman-Morrison formula, so nothing is refitted row by row."""
    n, p = X.shape
    g = means.shape[0]
    W_inv = np.linalg.inv(W)
    na = sizes[codes]
    c = na / (na - 1)
    d = X - means[codes]
    u = d @ W_inv                                   # rows: W^-1 d_i
    dWd = np.einsum("ij,ij->i", u, d)
    denom = 1 - c * dWd
    D2 = np.empty((n, g))
    for j in range(g):
        gap = X - means[j]
        own = codes == j
        gap = np.where(own[:, None], gap * c[:, None], gap)       # the subject's own centre moves away from it
        gWg = np.einsum("ij,jk,ik->i", gap, W_inv, gap)
        gWd = np.einsum("ij,ij->i", gap @ W_inv, d)
        D2[:, j] = (n - g - 1) * (gWg + c * gWd**2 / denom)
    return posterior(D2, prior).argmax(axis=1)


@register("discriminant_analysis")
def run(data: SheetData, options: DiscriminantOptions | None = None) -> AnalysisResult:
    opt = options or DiscriminantOptions()
    names, matrix = data.first_replicate_matrix()
    if len(names) < 2:
        raise AnalysisError("Discriminant analysis needs one data set holding each subject's group (a code such as 1, 2, 3) and at "
                            "least one data set of measurements, one subject on each row.")
    at = opt.group_index if 0 <= opt.group_index < len(names) else len(names) - 1
    group_name = names[at]
    use = [i for i in range(len(names)) if i != at and (not opt.selected_variables or names[i] in opt.selected_variables)
           and not np.all(np.isnan(matrix[:, i]))]
    if not use:
        raise AnalysisError("Choose at least one measurement besides the data set that holds the groups.")
    filled = ~np.any(np.isnan(matrix[:, [at, *use]]), axis=1)
    rows = np.where(filled)[0]
    labels_raw = matrix[filled, at]
    X = matrix[filled][:, use]
    variables = [names[i] for i in use]
    levels, codes = np.unique(labels_raw, return_inverse=True)
    codes = codes.astype(int)
    g = levels.size
    if g > MOST_GROUPS:
        raise AnalysisError(f"'{group_name}' has {g} distinct values, which does not look like a set of groups. Choose the data set "
                            f"that holds the group codes (up to {MOST_GROUPS} groups).")
    if g < 2:
        raise AnalysisError(f"'{group_name}' holds only one group on the rows that are filled in; two or more are needed.")
    n, p = X.shape
    sizes = np.bincount(codes, minlength=g).astype(float)
    if sizes.min() < 2:
        raise AnalysisError("Every group needs at least 2 subjects.")
    if n - g < p + 1:
        raise AnalysisError(f"There are {n} subjects in {g} groups for {p} measurements: too few. Discriminant analysis needs more "
                            "subjects than measurements plus groups.")
    group_names = [f"{v:g}" for v in levels]
    sig = opt.significant_digits

    def num(v) -> str:
        return rg.clean_zero(f"{v:.{sig}g}") if np.isfinite(v) else "-"

    def three(v) -> str:
        return rg.clean_zero(f"{v:.3f}")

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    sizes, means, W, B = scatter_matrices(X, codes, g)
    S = W / (n - g)
    spread = np.sqrt(np.diag(S))
    if np.any(spread == 0):
        flat = [variables[i] for i in range(p) if spread[i] == 0]
        raise AnalysisError("A measurement that does not vary within the groups cannot be used: " + ", ".join(flat[:6]) + ".")
    if np.linalg.matrix_rank(S) < p:
        raise AnalysisError("Some measurements can be worked out exactly from the others (for example a total alongside its "
                            "parts). Leave one of them out.")
    values, raw = canonical(W, B, n, g)
    m = values.size
    # a function and its mirror image separate the groups equally well: each is turned so that the
    # measurement it reflects most counts positively
    within_corr = S / np.outer(spread, spread)
    reflects = within_corr @ (raw * spread[:, None])
    for j in range(m):
        if reflects[int(np.argmax(np.abs(reflects[:, j]))), j] < 0:
            raw[:, j] = -raw[:, j]
    functions = [f"Function {j + 1}" for j in range(m)]
    tests = wilks_tests(values, n, p, g)
    lam = tests[0][1]
    F, df1, df2, p_all = rao_f(lam, n, p, g)
    grand = X.mean(axis=0)
    scores = (X - grand) @ raw
    centroids = (means - grand) @ raw
    constants = -grand @ raw
    notes: list[str] = []

    tables = []
    tables.append(ResultTable("Do the groups differ on these measurements taken together?", ["Test", "Value", "F", "df", "P value"],
                              [["Wilks' lambda (0 = groups fully apart, 1 = no difference)", f"{lam:.4f}", f"{F:.3f}",
                                f"{df1:g}, {df2:.4g}", p_text(p_all)]]))

    # ---- classification
    prior = sizes / n if opt.priors == "sizes" else np.full(g, 1 / g)
    S_inv = np.linalg.inv(S)
    post = posterior(distances(X, means, S_inv), prior)
    picked = post.argmax(axis=1)
    table, right = rg.classification(codes, picked, group_names, "group")
    table.title = "Subjects by their group and the group the rule puts them in (all subjects used to build the rule)"
    tables.append(table)
    right_loo = None
    if opt.show_cross_validation:
        picked_loo = leave_one_out(X, codes, sizes, means, W, prior)
        table_loo, right_loo = rg.classification(codes, picked_loo, group_names, "group")
        table_loo.title = "The same, each subject classified by a rule built without it (leave-one-out)"
        tables.append(table_loo)
    by_chance = float(np.max(prior)) if opt.priors == "sizes" else 1 / g

    if opt.show_functions:
        total = values.sum()
        tables.append(ResultTable("The discriminant functions", ["Function", "Eigenvalue", "Share of the separation", "Running total",
                                                               "Canonical correlation"],
                                  [[functions[j], num(values[j]), f"{values[j] / total:.1%}" if total > 0 else "-",
                                    f"{values[: j + 1].sum() / total:.1%}" if total > 0 else "-", f"{np.sqrt(values[j] / (1 + values[j])):.4f}"]
                                   for j in range(m)]))
        tables.append(ResultTable("Which functions are needed?", ["Functions tested", "Wilks' lambda", "Chi-square", "df", "P value"],
                                  [[(f"{k} to {m}" if k < m else f"{k}") if m > 1 else "1", f"{lm:.4f}", f"{c2:.3f}", str(df), p_text(pv)]
                                   for k, lm, c2, df, pv in tests]))
        tables.append(ResultTable("Where each group sits on the functions (group centres)", ["Group", *functions],
                                  [[group_names[j], *[three(v) for v in centroids[j]]] for j in range(g)]))

    standard = raw * spread[:, None]
    structure = within_corr @ standard                   # pooled within-group correlation of each measurement with each function
    if opt.show_coefficients:
        order = sorted(range(p), key=lambda i: -abs(structure[i, 0]))
        tables.append(ResultTable("Standardised weights: each measurement's own contribution to a function", ["Measurement", *functions],
                                  [[variables[i], *[three(v) for v in standard[i]]] for i in range(p)]))
        tables.append(ResultTable("Correlation of each measurement with each function (within the groups)", ["Measurement", *functions],
                                  [[variables[i], *[three(v) for v in structure[i]]] for i in order]))
    if opt.show_raw_coefficients:
        tables.append(ResultTable("Weights for the measurements as typed (score = constant + sum of weight × value)", ["Term", *functions],
                                  [[variables[i], *[num(v) for v in raw[i]]] for i in range(p)] + [["Constant", *[num(v) for v in constants]]]))

    if opt.show_each_variable:
        each = []
        total_ss = np.diag(W + B)
        for i in range(p):
            lam_i = W[i, i] / total_ss[i] if total_ss[i] > 0 else float("nan")
            F_i = (B[i, i] / (g - 1)) / (W[i, i] / (n - g))
            each.append([variables[i], f"{lam_i:.4f}", f"{F_i:.3f}", f"{g - 1}, {n - g}", p_text(float(stats.f.sf(F_i, g - 1, n - g)))])
        tables.append(ResultTable("Each measurement on its own", ["Measurement", "Wilks' lambda", "F", "df", "P value"], each))

    if opt.show_group_means:
        mean_rows = [[group_names[j], str(int(sizes[j])), *[num(v) for v in means[j]]] for j in range(g)]
        mean_rows.append(["All", str(n), *[num(v) for v in grand]])
        tables.append(ResultTable("Group sizes and means", ["Group", "Subjects", *variables], mean_rows))

    if opt.show_classification_functions:
        weights = means @ S_inv                             # one row per group
        consts = -0.5 * np.einsum("ij,ij->i", weights, means) + np.log(prior)
        tables.append(ResultTable("Classification functions: a subject goes to the group with the highest score",
                                  ["Term", *[f"Group {nm}" for nm in group_names]],
                                  [[variables[i], *[num(v) for v in weights[:, i]]] for i in range(p)] + [["Constant", *[num(v) for v in consts]]]))

    box_text = ""
    if opt.show_box_m:
        box = box_m(X, codes, g, W)
        if box is None:
            notes.append("Box's M test of equal covariance matrices could not be run: a group has no more subjects than there are "
                         "measurements.")
        else:
            M, chi2, df, pv = box
            tables.append(ResultTable("Do the groups vary alike? (Box's M test of equal covariance matrices)",
                                      ["Box's M", "Chi-square", "df", "P value"], [[f"{M:.3f}", f"{chi2:.3f}", str(df), p_text(pv)]]))
            if pv < 0.001:
                box_text = " The groups do not vary alike (Box's M), so read the classification with some care."
                notes.append("Box's M is significant at 0.001: the groups have different spreads or correlations, which this "
                             "(linear) rule assumes they do not. With groups of similar size the classification is usually still "
                             "sound; with very unequal groups, subjects drift towards the more variable group. The test is "
                             "over-sensitive in large samples and to non-normal data, which is why 0.001 is the usual threshold.")

    # ---- graphs
    charts = []
    if opt.graph_scores:
        if m >= 2:
            pairs = [(a, b) for a in range(min(m, 3)) for b in range(a + 1, min(m, 3))]
            for a, b in pairs:
                series = []
                for j in range(g):
                    mask = codes == j
                    sx, sy = rg.thin(scores[mask, a], scores[mask, b], most=1500)
                    series.append(ChartSeries(f"Group {group_names[j]}", sx, sy, color=rg.SHADES[j % len(rg.SHADES)]))
                charts.append(ChartSpec(f"Subjects' scores: {functions[a]} against {functions[b]}",
                                        x_label=f"{functions[a]} ({values[a] / values.sum():.0%} of the separation)",
                                        y_label=f"{functions[b]} ({values[b] / values.sum():.0%})", series=series,
                                        annotations=[{"x": float(centroids[j, a]), "y": float(centroids[j, b]), "text": f"centre of {group_names[j]}",
                                                      "size": 9, "color": "#000000"} for j in range(g)],
                                        group="discriminant_scores", option=f"{functions[a]} against {functions[b]}"))
        else:
            rng = np.random.default_rng(0)
            series = []
            for j in range(g):
                mask = codes == j
                jitter = j + rng.uniform(-0.18, 0.18, int(mask.sum()))
                sx, sy = rg.thin(jitter, scores[mask, 0], most=1500)
                series.append(ChartSeries(f"Group {group_names[j]}", sx, sy, color=rg.SHADES[j % len(rg.SHADES)]))
            cut_lines = [{"kind": "h", "value": float(c)} for c in centroids[:, 0]]
            charts.append(ChartSpec("Subjects' scores on the discriminant function", x_label=group_name, y_label="Discriminant score",
                                    series=series, x_categories=group_names, reference_lines=cut_lines))
    if opt.graph_weights:
        xs = [float(i) for i in range(p)]
        charts.append(ChartSpec("How strongly each measurement goes with Function 1", x_label="", y_label="Correlation with Function 1",
                                series=[ChartSeries("", xs, structure[:, 0].tolist(), kind="bar", color="#1f5aa6")],
                                x_categories=variables, reference_lines=[{"kind": "h", "value": 0.0}], y_limits=(-1.05, 1.05)))

    # ---- headline
    lead = variables[int(np.argmax(np.abs(structure[:, 0])))]
    summary = (f"{g} groups, {p} measurement{'s' if p != 1 else ''}, {n} subjects. The groups "
               f"{'differ' if p_all < opt.alpha else 'do not clearly differ'}: Wilks' lambda = {lam:.4f}, F({df1:g}, {df2:.4g}) = {F:.3f}, "
               f"P = {p_all:.4g}. The rule places {right:.1%} of subjects in their own group"
               + (f", {right_loo:.1%} when each is left out in turn" if right_loo is not None else "")
               + f" (chance: {by_chance:.0%}). {lead} separates the groups most." + box_text)

    # ---- notes
    dropped = int((~np.all(np.isnan(matrix[:, [at, *use]]), axis=1)).sum() - n)
    if dropped:
        notes.insert(0, f"{dropped} row{'s' if dropped != 1 else ''} with a blank in the group or a measurement "
                        f"{'were' if dropped != 1 else 'was'} left out.")
    notes.append("The share classified correctly with every subject used to build the rule flatters it. The leave-one-out share - "
                 "each subject classified by a rule that never saw it - is the one to quote.")
    notes.append("A discriminant function is a weighted sum of the measurements, chosen so that the groups are as far apart as "
                 "possible compared with the spread inside them; the first separates most. The standardised weights show each "
                 "measurement's own contribution with the others present; the correlations show which measurements a function "
                 "reflects, and are steadier when measurements overlap.")
    notes.append(("Every group was given the same chance beforehand." if opt.priors != "sizes" else
                  "Groups were given chances in proportion to their sizes, so a subject on the border goes to the larger group.")
                 + " A subject is placed in the group it most probably belongs to, from its distance to each group centre.")
    if sizes.min() < 5 * p / max(g - 1, 1) and sizes.min() < 20:
        notes.append(f"The smallest group has {int(sizes.min())} subjects for {p} measurements. With few subjects per measurement "
                     "the functions fit this sample's quirks; expect a clear drop from the first classification table to the "
                     "leave-one-out one.")
    notes.append("Linear discriminant analysis (Fisher 1936): assumes roughly normal measurements with the same covariance matrix "
                 "in every group. Wilks' lambda is tested with Rao's F (exact for two or three groups) and, function by function, "
                 "with Bartlett's chi-square.")

    result = AnalysisResult(title="Discriminant analysis", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.make_scores:
        columns = {functions[j]: scores[:, j].tolist() for j in range(m)}
        columns["Group (as typed)"] = labels_raw.tolist()
        columns["Group the rule picks"] = levels[picked].tolist()
        for j in range(g):
            columns[f"Chance of group {group_names[j]}"] = post[:, j].tolist()
        result.new_sheet = NewSheetData("Discriminant scores", "Row", (rows + 1).astype(float).tolist(), columns)
        notes.append("Each subject's scores, the group the rule picks and the chance of each group are on a new sheet, "
                     "'Discriminant scores'.")
    return result
