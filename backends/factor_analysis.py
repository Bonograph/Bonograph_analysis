"""Factor analysis: a few hidden factors behind many correlated variables.

Each data set is a variable and each row a subject. Where principal
components re-describe all the variation, factor analysis models only what
the variables share: every variable is a mix of a few common factors plus a
part of its own. The loadings say how strongly each variable follows each
factor; rotating them makes the pattern easier to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..chart_helpers import heatmap_chart
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register

EXTRACTIONS = {"principal_axis": "principal axis factoring", "maximum_likelihood": "maximum likelihood"}
ROTATIONS = {"none": "no rotation", "varimax": "varimax", "quartimax": "quartimax", "promax": "promax", "oblimin": "direct oblimin"}
OBLIQUE = ("promax", "oblimin")


@dataclass
class FactorAnalysisOptions:
    selected_variables: list[str] = field(default_factory=list)   # empty = every data set
    extraction: str = "principal_axis"     # "principal_axis" | "maximum_likelihood"
    how_many: str = "parallel"             # "parallel" | "eigenvalue" | "fixed"
    n_factors: int = 2                     # how_many = "fixed"
    rotation: str = "varimax"              # "none" | "varimax" | "quartimax" | "promax" | "oblimin"
    promax_power: float = 4.0
    sort_loadings: bool = True             # variables grouped under the factor they follow most
    hide_below: float = 0.0                # loadings smaller than this are left blank (0 = show all)
    show_suitability: bool = True          # sampling adequacy and Bartlett's test
    show_each_variable: bool = True        # each variable's adequacy and shared variance
    show_fit: bool = True
    show_structure: bool = False           # correlated factors: the correlations of variables with factors
    show_score_weights: bool = False
    make_scores: bool = False              # each subject's factor scores on a new sheet
    graph_scree: bool = True
    graph_loadings: bool = True
    graph_heatmap: bool = True
    num_simulations: int = 500             # random tables for the "beat chance" rule
    random_seed: int = 0
    significant_digits: int = 4
    alpha: float = 0.05
    p_style: str = "number"


# ── suitability ─────────────────────────────────────────────────────────────

def kmo(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Kaiser-Meyer-Olkin sampling adequacy: overall, and for each variable.
    Compares the correlations with the partial correlations (each pair with
    every other variable held fixed): near 1 when the variables share common
    factors, near 0.5 or below when they pair off or stand alone."""
    inverse = np.linalg.inv(R)
    scale = np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    partial = -inverse / scale
    off = ~np.eye(R.shape[0], dtype=bool)
    r2, q2 = np.where(off, R**2, 0.0), np.where(off, partial**2, 0.0)
    each = r2.sum(axis=0) / (r2.sum(axis=0) + q2.sum(axis=0))
    return float(r2.sum() / (r2.sum() + q2.sum())), each


def bartlett_sphericity(R: np.ndarray, n: int) -> tuple[float, int, float]:
    """Bartlett's test that the variables are uncorrelated: (chi-square, df, P)."""
    p = R.shape[0]
    _sign, logdet = np.linalg.slogdet(R)
    chi2 = float(-(n - 1 - (2 * p + 5) / 6) * logdet)
    df = p * (p - 1) // 2
    return chi2, df, float(stats.chi2.sf(chi2, df))


def parallel_thresholds(n: int, p: int, simulations: int, seed: int, percentile: float = 95.0) -> np.ndarray:
    """Horn's parallel analysis: the eigenvalues that unrelated variables
    would give in a table of this size, at the given percentile. The random
    correlation matrices are drawn directly (a Wishart matrix scaled to unit
    diagonal), which is the same as simulating whole tables and far quicker."""
    draws = stats.wishart.rvs(df=n - 1, scale=np.eye(p), size=simulations, random_state=np.random.default_rng(seed))
    draws = np.asarray(draws).reshape(simulations, p, p)
    d = 1 / np.sqrt(np.einsum("ijj->ij", draws))
    correlations = draws * d[:, :, None] * d[:, None, :]
    eigenvalues = np.linalg.eigvalsh(correlations)[:, ::-1]
    return np.percentile(eigenvalues, percentile, axis=0)


# ── extraction ──────────────────────────────────────────────────────────────

def principal_axis(R: np.ndarray, k: int, rounds: int = 2000, tolerance: float = 1e-8):
    """Iterated principal axis factoring. Returns (loadings, communalities,
    rounds used, settled?, whether a communality had to be held at 1)."""
    p = R.shape[0]
    shared = 1 - 1 / np.diag(np.linalg.inv(R))          # squared multiple correlations to start
    held, settled, used = False, False, 0
    loadings = np.zeros((p, k))
    for used in range(1, rounds + 1):
        reduced = R.copy()
        np.fill_diagonal(reduced, shared)
        values, vectors = np.linalg.eigh(reduced)
        top = np.argsort(values)[::-1][:k]
        loadings = vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))
        new = (loadings**2).sum(axis=1)
        if np.any(new > 1):
            held = True
            new = np.minimum(new, 1.0)
        change = float(np.max(np.abs(new - shared)))
        shared = new
        if change < tolerance:
            settled = True
            break
    return loadings, shared, used, settled, held


def maximum_likelihood(R: np.ndarray, k: int, floor: float = 0.005):
    """Maximum-likelihood factor analysis (Joreskog 1967): the unique
    variances are searched for; for any set of them the best loadings follow
    from an eigen-decomposition. Returns (loadings, unique variances, the
    minimised discrepancy, settled?, whether a unique variance hit its floor)."""
    p = R.shape[0]

    def pieces(psi):
        scale = 1 / np.sqrt(psi)
        values, vectors = np.linalg.eigh(R * np.outer(scale, scale))
        order = np.argsort(values)[::-1]
        return values[order], vectors[:, order]

    def loadings_of(psi):
        values, vectors = pieces(psi)
        return np.sqrt(psi)[:, None] * vectors[:, :k] * np.sqrt(np.clip(values[:k] - 1, 0, None))

    def discrepancy(psi):
        rest = pieces(psi)[0][k:]
        return float(np.sum(rest - np.log(rest) - 1))

    def slope(psi):
        L = loadings_of(psi)
        return np.diag(L @ L.T + np.diag(psi) - R) / psi**2

    start = (1 - 0.5 * k / p) / np.diag(np.linalg.inv(R))
    best = optimize.minimize(discrepancy, np.clip(start, floor, 1.0), jac=slope, method="L-BFGS-B", bounds=[(floor, 1.0)] * p,
                             options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
    psi = best.x
    return loadings_of(psi), psi, float(best.fun), bool(best.success), bool(np.any(psi <= floor * 1.0001))


# ── rotation ────────────────────────────────────────────────────────────────

def _orthomax(A: np.ndarray, gamma: float, rounds: int = 1000, tolerance: float = 1e-10) -> np.ndarray:
    """Rotation matrix for the orthomax family: gamma 1 is varimax, 0 quartimax."""
    p, k = A.shape
    T = np.eye(k)
    last = 0.0
    for _ in range(rounds):
        L = A @ T
        u, s, vt = np.linalg.svd(A.T @ (L**3 - (gamma / p) * L @ np.diag((L**2).sum(axis=0))))
        T = u @ vt
        now = float(s.sum())
        if now < last * (1 + tolerance):
            break
        last = now
    return T


def _oblimin(A: np.ndarray, gamma: float = 0.0, rounds: int = 2000, tolerance: float = 1e-8):
    """Direct oblimin by gradient projection (Bernaards and Jennrich 2005):
    (pattern loadings, factor correlations)."""
    p, k = A.shape
    N = np.ones((k, k)) - np.eye(k)
    C = np.eye(p) - gamma / p * np.ones((p, p))

    def criterion(L):
        squared = L**2
        mixed = C @ squared @ N
        return float(np.sum(squared * mixed) / 4), L * mixed

    T = np.eye(k)
    L = A @ np.linalg.inv(T).T
    f, Gq = criterion(L)
    G = -(L.T @ Gq @ np.linalg.inv(T)).T
    step = 1.0
    for _ in range(rounds):
        Gp = G - T @ np.diag(np.sum(T * G, axis=0))
        size = float(np.sqrt(np.sum(Gp**2)))
        if size < tolerance:
            break
        step *= 2
        for _try in range(12):
            X = T - step * Gp
            Tt = X / np.sqrt(np.sum(X**2, axis=0))
            L = A @ np.linalg.inv(Tt).T
            ft, Gq = criterion(L)
            if ft < f - 0.5 * size**2 * step:
                break
            step /= 2
        T, f = Tt, ft
        G = -(L.T @ Gq @ np.linalg.inv(T)).T
    return A @ np.linalg.inv(T).T, T.T @ T


def rotate(loadings: np.ndarray, method: str, power: float = 4.0):
    """(pattern loadings, correlations between the factors). The rows are put
    on a common length first (Kaiser's normalisation) so that variables with
    little shared variance do not steer the rotation."""
    p, k = loadings.shape
    if method == "none" or k < 2:
        return loadings.copy(), np.eye(k)
    length = np.sqrt((loadings**2).sum(axis=1))
    length = np.where(length > 1e-12, length, 1.0)
    A = loadings / length[:, None]
    if method in ("varimax", "quartimax"):
        return (A @ _orthomax(A, 1.0 if method == "varimax" else 0.0)) * length[:, None], np.eye(k)
    if method == "promax":
        # Hendrickson and White 1964: a varimax solution, its loadings raised to a power to make a
        # sharper target, and the oblique turn that brings the varimax loadings closest to that target
        V = (A @ _orthomax(A, 1.0)) * length[:, None]
        target = V * np.abs(V) ** (power - 1)
        U = np.linalg.lstsq(V, target, rcond=None)[0]
        U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))
        inverse = np.linalg.inv(U)
        return V @ U, inverse @ inverse.T
    pattern, phi = _oblimin(A)
    return pattern * length[:, None], phi


def tidy(pattern: np.ndarray, phi: np.ndarray):
    """Largest factor first, and each factor signed so its loadings sum to a positive number."""
    order = np.argsort(-(pattern**2).sum(axis=0), kind="stable")
    pattern, phi = pattern[:, order], phi[np.ix_(order, order)]
    signs = np.where(pattern.sum(axis=0) < 0, -1.0, 1.0)
    return pattern * signs, phi * np.outer(signs, signs)


def _verdict(value: float) -> str:
    for cut, word in ((0.9, "excellent"), (0.8, "good"), (0.7, "fair"), (0.6, "modest"), (0.5, "weak")):
        if value >= cut:
            return word
    return "too low - not suited"


@register("factor_analysis")
def run(data: SheetData, options: FactorAnalysisOptions | None = None) -> AnalysisResult:
    opt = options or FactorAnalysisOptions()
    extraction = opt.extraction if opt.extraction in EXTRACTIONS else "principal_axis"
    rotation = opt.rotation if opt.rotation in ROTATIONS else "varimax"
    all_names, all_matrix = data.first_replicate_matrix()
    keep = [i for i, nm in enumerate(all_names) if not opt.selected_variables or nm in opt.selected_variables]
    if len(keep) < 3:
        raise AnalysisError("Factor analysis needs at least 3 variables (data sets), with one subject on each row.")
    names = [all_names[i] for i in keep]
    matrix = all_matrix[:, keep]
    filled = ~np.any(np.isnan(matrix), axis=1)
    values = matrix[filled]
    n, p = values.shape
    if n < p + 2:
        raise AnalysisError(f"There are {n} complete rows for {p} variables. Factor analysis needs more rows than variables - "
                            "and many more to be trustworthy.")
    spread = values.std(axis=0, ddof=1)
    if np.any(spread == 0) or not np.all(np.isfinite(spread)):
        flat = [names[i] for i in range(p) if not spread[i] > 0]
        raise AnalysisError("A variable that never changes cannot be analysed: " + ", ".join(flat[:6]) + ".")
    Zs = (values - values.mean(axis=0)) / spread
    R = (Zs.T @ Zs) / (n - 1)
    eigen = np.sort(np.linalg.eigvalsh(R))[::-1]
    if not np.all(np.isfinite(eigen)):
        raise AnalysisError("The correlations between these variables could not be computed - check for extreme values.")
    if eigen[-1] < 1e-8:
        raise AnalysisError("Some variables can be worked out exactly from the others (for example a total alongside its parts), "
                            "so their correlations cannot be untangled. Leave one of them out.")
    sig = opt.significant_digits

    def num(v) -> str:
        return f"{v:.{sig}g}"

    def three(v) -> str:
        text = f"{v:.3f}"
        return text[1:] if text.startswith("-") and float(text) == 0 else text

    def p_text(v) -> str:
        return format_p(v, opt.p_style, digits=sig, alpha=opt.alpha)

    notes: list[str] = []
    # ---- how many factors
    most = p - 1
    if extraction == "maximum_likelihood":           # the test needs degrees of freedom to spare
        while most > 1 and ((p - most) ** 2 - (p + most)) < 0:
            most -= 1
    thresholds = parallel_thresholds(n, p, int(np.clip(opt.num_simulations, 50, 5000)), int(opt.random_seed))
    if opt.how_many == "fixed":
        k = int(opt.n_factors)
        rule = "as asked"
    elif opt.how_many == "eigenvalue":
        k = int(np.sum(eigen > 1.0))
        rule = "eigenvalues above 1"
    else:
        beats = eigen > thresholds
        k = int(np.argmin(beats)) if not beats.all() else p       # the leading run of eigenvalues that beat chance
        rule = "eigenvalues larger than random tables of this size give (parallel analysis, 95th percentile)"
    if k < 1:
        k = 1
        notes.append("No eigenvalue passed the rule for keeping a factor, which suggests the variables share little. One factor "
                     "was extracted so that there is something to look at.")
    if k > most:
        notes.append(f"{k} factors were asked for, but {p} variables can support at most {most} with this method; {most} "
                     f"{'was' if most == 1 else 'were'} extracted.")
        k = most

    # ---- extraction
    if extraction == "maximum_likelihood":
        loadings, unique, least, settled, held = maximum_likelihood(R, k)
        shared = 1 - unique
        if held:
            notes.append("A variable's unique variance was driven to its lower limit (a Heywood case): the model is trying to "
                         "explain that variable completely. This often means too many factors, or too few rows.")
    else:
        loadings, shared, used, settled, held = principal_axis(R, k)
        unique = 1 - shared
        least = float("nan")
        if held:
            notes.append("A variable's shared variance reached 100% and was held there (a Heywood case). This often means too many "
                         "factors, or too few rows.")
    if not settled:
        notes.append("The extraction did not settle; the loadings may be unreliable. Try fewer factors.")

    unrotated_ss = (loadings**2).sum(axis=0)
    pattern, phi = rotate(loadings, rotation, float(opt.promax_power))
    pattern, phi = tidy(pattern, phi)
    oblique = rotation in OBLIQUE and k >= 2
    structure = pattern @ phi
    ss = (pattern**2).sum(axis=0)
    factors = [f"Factor {j + 1}" for j in range(k)]

    tables = []
    # ---- is the data suited?
    kmo_all, kmo_each = kmo(R)
    chi2, df, p_sph = bartlett_sphericity(R, n)
    if opt.show_suitability:
        tables.append(ResultTable("Is the data suited to factor analysis?", ["Check", "Value", "Reading"], [
            ["Subjects (complete rows) and variables", f"{n} and {p}", f"{n / p:.1f} subjects per variable"],
            ["Sampling adequacy (Kaiser-Meyer-Olkin)", f"{kmo_all:.3f}", _verdict(kmo_all)],
            ["Bartlett's test that the variables are unrelated", f"chi-square({df}) = {chi2:.2f}, P = {p_text(p_sph)}",
             "they are related" if p_sph < opt.alpha else "no clear relation: nothing to factor"],
            ["Determinant of the correlation matrix", f"{np.exp(np.linalg.slogdet(R)[1]):.3g}",
             "near zero: some variables nearly duplicate others" if eigen[-1] < 0.01 else "fine"]]))

    # ---- loadings
    order = list(range(p))
    if opt.sort_loadings:
        home = np.argmax(np.abs(pattern), axis=1)
        order = sorted(range(p), key=lambda i: (home[i], -abs(pattern[i, home[i]])))
    cut = float(max(opt.hide_below, 0.0))

    def cell(v) -> str:
        return "" if abs(v) < cut else three(v)

    kind = "pattern loadings" if oblique else "loadings"
    title = f"Factor {kind} ({EXTRACTIONS[extraction]}, {ROTATIONS[rotation]})"
    tables.append(ResultTable(title, ["Variable", *factors, "Shared variance (communality)", "Unique variance"],
                              [[names[i], *[cell(v) for v in pattern[i]], three(shared[i]), three(unique[i])] for i in order]))

    variance_rows = [["Sum of squared loadings", *[three(v) for v in ss]],
                     ["Share of the total variance", *[f"{v / p:.1%}" for v in ss]]]
    if not oblique:
        variance_rows.append(["Running total", *[f"{v / p:.1%}" for v in np.cumsum(ss)]])
    tables.append(ResultTable("How much each factor accounts for", ["", *factors], variance_rows))
    if oblique:
        tables.append(ResultTable("Correlations between the factors", ["", *factors],
                                  [[factors[i], *[three(v) for v in phi[i]]] for i in range(k)]))
        if opt.show_structure:
            tables.append(ResultTable("Correlations of each variable with each factor (structure)", ["Variable", *factors],
                                      [[names[i], *[three(v) for v in structure[i]]] for i in order]))

    # ---- how many factors: the evidence
    tables.append(ResultTable("How many factors? Eigenvalues of the correlation matrix",
                              ["Component", "Eigenvalue", "Share of variance", "Running total", "What random tables give (95th percentile)", "Kept"],
                              [[str(j + 1), three(eigen[j]), f"{eigen[j] / p:.1%}", f"{eigen[: j + 1].sum() / p:.1%}", three(thresholds[j]),
                                "Yes" if j < k else "No"] for j in range(p)]))

    if opt.show_each_variable:
        smc = 1 - 1 / np.diag(np.linalg.inv(R))
        tables.append(ResultTable("Each variable", ["Variable", "Sampling adequacy", "Shared with the other variables (R²)",
                                                    "Shared variance the factors explain"],
                                  [[names[i], three(kmo_each[i]), three(smc[i]), three(shared[i])] for i in range(p)]))

    # ---- fit
    rebuilt = pattern @ phi @ pattern.T
    off = ~np.eye(p, dtype=bool)
    left = (R - rebuilt)[off]
    rmsr = float(np.sqrt(np.mean(left**2)))
    big = float(np.mean(np.abs(left) > 0.05))
    fit_text = ""
    if opt.show_fit:
        fit_rows = [["Typical size of a correlation left unexplained (root mean square residual)", f"{rmsr:.4f}"],
                    ["Correlations left unexplained by more than 0.05", f"{big:.0%}"],
                    ["Total variance the factors account for", f"{shared.sum() / p:.1%}"]]
        if extraction == "maximum_likelihood":
            dof = ((p - k) ** 2 - (p + k)) // 2
            if dof > 0:
                value = (n - 1 - (2 * p + 5) / 6 - 2 * k / 3) * least
                p_fit = float(stats.chi2.sf(value, dof))
                fit_rows.insert(0, [f"Test that {k} factor{'s are' if k != 1 else ' is'} enough", f"chi-square({dof}) = {value:.2f}, P = {p_text(p_fit)}"])
                fit_text = (f" The test that {k} factor{'s are' if k != 1 else ' is'} enough gives chi-square({dof}) = {value:.2f}, "
                            f"P = {p_fit:.4g}" + (": more are needed." if p_fit < opt.alpha else "."))
            else:
                fit_rows.insert(0, [f"Test that {k} factor{'s are' if k != 1 else ' is'} enough", "no degrees of freedom left to test it"])
        tables.append(ResultTable("How well the factors reproduce the correlations", ["Measure", "Value"], fit_rows))

    weights = np.linalg.solve(R, structure)            # regression (Thurstone) scores
    if opt.show_score_weights:
        tables.append(ResultTable("Weights that turn standardised variables into factor scores", ["Variable", *factors],
                                  [[names[i], *[num(v) for v in weights[i]]] for i in range(p)]))

    # ---- graphs
    charts = []
    if opt.graph_scree:
        xs = [float(j + 1) for j in range(p)]
        charts.append(ChartSpec("Scree plot: eigenvalues, and what chance alone gives", x_label="Component", y_label="Eigenvalue",
                                series=[ChartSeries("This data", xs, eigen.tolist(), kind="line_markers", color="#1f5aa6"),
                                        ChartSeries("Random tables (95th percentile)", xs, thresholds.tolist(), kind="line", color="#d1342f")],
                                reference_lines=[{"kind": "h", "value": 1.0}]))
    if opt.graph_loadings and k >= 2:
        for a, b in list(combinations(range(min(k, 4)), 2)):
            charts.append(ChartSpec(f"Loadings: {factors[a]} against {factors[b]}", x_label=factors[a], y_label=factors[b],
                                    series=[ChartSeries("", pattern[:, a].tolist(), pattern[:, b].tolist())],
                                    reference_lines=[{"kind": "h", "value": 0.0}, {"kind": "v", "value": 0.0}],
                                    annotations=[{"x": float(pattern[i, a]), "y": float(pattern[i, b]), "text": names[i]} for i in range(p)]
                                    if p <= 40 else [],
                                    x_limits=(-1.05, 1.05), y_limits=(-1.05, 1.05), group="factor_pairs",
                                    option=f"{factors[a]} against {factors[b]}"))
    if opt.graph_heatmap:
        charts.append(heatmap_chart("Loadings at a glance", pattern[order], [names[i] for i in order], factors, cmap="RdBu_r",
                                    color_limits=(-1.0, 1.0)))

    # ---- headline
    leaders = []
    home = np.argmax(np.abs(pattern), axis=1)
    for j in range(k):
        mine = [i for i in np.argsort(-np.abs(pattern[:, j])) if home[i] == j][:3]
        if mine:
            leaders.append(f"{factors[j]}: {', '.join(names[i] for i in mine)}")
    summary = (f"{k} factor{'s' if k != 1 else ''} ({rule}) account{'s' if k == 1 else ''} for {shared.sum() / p:.0%} of the variance "
               f"of {p} variables in {n} subjects. Sampling adequacy {kmo_all:.2f} ({_verdict(kmo_all)})."
               + (" Leading variables - " + "; ".join(leaders) + "." if leaders else "") + fit_text)

    # ---- notes
    dropped = int((~np.all(np.isnan(matrix), axis=1)).sum() - n)
    if dropped:
        notes.insert(0, f"{dropped} row{'s' if dropped != 1 else ''} with a blank in any variable {'were' if dropped != 1 else 'was'} left out.")
    if n < 5 * p or n < 100:
        notes.append(f"With {n} subjects for {p} variables the loadings are rough: a factor analysis is usually trusted from about "
                     "100 subjects and 5 to 10 per variable, more when the shared variances are low.")
    if kmo_all < 0.5:
        notes.append("The sampling adequacy is below 0.5: the variables do not share enough for factors to be meaningful. Look at "
                     "each variable's own value and consider leaving the lowest out.")
    notes.append("A loading is how strongly a variable follows a factor, from -1 to 1; values beyond about ±0.4 are usually read as "
                 "belonging to the factor. The shared variance (communality) is the part of a variable the factors explain; the "
                 "rest is its own.")
    if oblique:
        notes.append("This rotation lets the factors correlate. The loadings shown are pattern loadings - each factor's own "
                     "contribution to a variable, like regression weights - so they can pass 1, and the factors' shares of variance "
                     "overlap and do not add up.")
    elif rotation != "none" and k >= 2:
        notes.append(f"The {ROTATIONS[rotation]} rotation keeps the factors uncorrelated. If you expect them to be related, as traits "
                     "and abilities usually are, a rotation that lets them correlate (promax or oblimin) describes the data better.")
    notes.append("Rotation does not change how well the factors fit or any variable's shared variance; it only turns the axes so "
                 "that each variable leans on as few factors as possible.")
    how = ("unique variances found by maximum likelihood (Joreskog 1967), which assumes roughly normal variables"
           if extraction == "maximum_likelihood" else
           "iterated principal axis factoring, starting from each variable's squared multiple correlation with the others")
    notes.append(f"Worked from the correlation matrix of the complete rows; extraction by {how}; rows put on a common length before "
                 "rotating (Kaiser's normalisation). Parallel analysis follows Horn 1965"
                 + (f"; promax with power {opt.promax_power:g}" if rotation == "promax" else "") + ".")

    result = AnalysisResult(title="Factor analysis", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.make_scores:
        scores = np.full((matrix.shape[0], k), np.nan)
        scores[filled] = Zs @ weights
        used_rows = np.where(~np.all(np.isnan(matrix), axis=1))[0]
        last = int(used_rows.max()) + 1 if used_rows.size else 0
        result.new_sheet = NewSheetData("Factor scores", "Row", [float(i + 1) for i in range(last)],
                                        {factors[j]: scores[:last, j].tolist() for j in range(k)})
        notes.append("Each subject's factor scores (regression method) are on a new sheet, 'Factor scores'; rows with a blank have none.")
    return result
