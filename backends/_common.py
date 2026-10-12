"""Statistics shared by the comparison analyses, so that every one of them
corrects for multiple testing, checks its assumptions, reports effect sizes
and warns about small samples in exactly the same way.

Nothing here draws or formats beyond ResultTable rows; each backend decides
where the pieces go. The order results are shown in is fixed across
analyses: headline, the effect with its interval, the test, the checks on
its assumptions, then the notes.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

from ..backend_types import ResultTable

# ---------------------------------------------------------------- vocabulary
# One set of names for keeping many comparisons honest, used by every
# analysis that makes follow-up comparisons.
CORRECTIONS = {
    "auto": "Best for this design",
    "holm": "Holm",
    "tukey": "Tukey",
    "dunnett": "Dunnett",
    "games_howell": "Games-Howell",
    "dunn": "Dunn",
    "fdr": "Benjamini-Hochberg (false discovery rate)",
    "none": "No correction",
}

SMALL_N = 5  # below this per group, a note says how little the test can see


# ---------------------------------------------------------------- p-value adjustment
def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)[::-1]
    m = p.size
    adjusted = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        running = min(running, p[idx] * m / (m - rank))
        adjusted[idx] = running
    return np.clip(adjusted, 0, 1)


def adjust(p, method: str) -> np.ndarray:
    if method == "fdr":
        return benjamini_hochberg(p)
    if method == "none":
        return np.asarray(p, dtype=float)
    return holm(p)


# ---------------------------------------------------------------- follow-up comparisons
def resolve_correction(correction: str, scope: str, distribution: str, equal_var: bool, paired: bool = False) -> str:
    """What "Best for this design" means here, and a fall-back when a
    named method doesn't apply (Tukey on ranks, Dunnett on all pairs)."""
    ranks = distribution == "nonparametric"
    if correction == "auto":
        if paired:
            return "holm"
        if ranks:
            return "dunn"
        if scope == "vs_control":
            return "dunnett" if equal_var else "holm"
        return "tukey" if equal_var else "games_howell"
    if ranks and correction in ("tukey", "dunnett", "games_howell"):
        return "dunn"
    if correction == "tukey" and scope == "vs_control":
        return "dunnett"
    if correction == "dunnett" and scope != "vs_control":
        return "tukey"
    return correction


def pairwise(arrays: list[np.ndarray], labels: list[str], scope: str, control: int, method: str,
             confidence: float = 0.95, paired: bool = False, ranks: bool = False) -> tuple[ResultTable, str]:
    """Follow-up comparisons after an overall test, each with its difference,
    a confidence interval where the method gives one, and the adjusted P.

    ``method`` is already resolved (see resolve_correction). Returns the
    table and a one-line description of what was done."""
    k = len(arrays)
    control = max(0, min(control, k - 1))
    pairs = [(control, j) for j in range(k) if j != control] if scope == "vs_control" else list(combinations(range(k), 2))
    pct = int(round(confidence * 100))
    rows: list[list[str]] = []
    means = [float(np.mean(a)) for a in arrays]

    if method == "dunn":
        ranked = stats.rankdata(np.concatenate(arrays))
        n_total = ranked.size
        _vals, ties = np.unique(np.concatenate(arrays), return_counts=True)
        tie_term = float(np.sum(ties**3 - ties)) / (12 * (n_total - 1)) if n_total > 1 else 0.0
        bounds = np.cumsum([0] + [a.size for a in arrays])
        mean_ranks = [float(np.mean(ranked[bounds[i]:bounds[i + 1]])) for i in range(k)]
        raw = []
        for i, j in pairs:
            se = np.sqrt((n_total * (n_total + 1) / 12 - tie_term) * (1 / arrays[i].size + 1 / arrays[j].size))
            z = (mean_ranks[i] - mean_ranks[j]) / se if se else 0.0
            raw.append(float(2 * stats.norm.sf(abs(z))))
        adjusted = holm(raw)
        for (i, j), p_raw, p_adj in zip(pairs, raw, adjusted):
            rows.append([f"{labels[i]} vs {labels[j]}",
                         f"{np.median(arrays[i]) - np.median(arrays[j]):.4g}", "–", f"{p_adj:.4g}",
                         "Yes" if p_adj < 0.05 else "No"])
        return (ResultTable("Which groups differ (Dunn's test, Holm-adjusted)",
                            ["Comparison", "Difference in medians", f"{pct}% CI", "Adjusted P", "Significant"], rows),
                "Dunn's test on ranks, with Holm's adjustment.")

    if method == "tukey":
        # Tukey-Kramer: the pooled error term, one studentized range for every pair
        from . import _ptukey

        ns = [a.size for a in arrays]
        df = sum(ns) - k
        mse = sum((a.size - 1) * np.var(a, ddof=1) for a in arrays) / df
        q_crit = _ptukey.ppf(confidence, k, df)
        ses = [np.sqrt(mse / 2 * (1 / ns[i] + 1 / ns[j])) for i, j in pairs]
        diffs = [means[i] - means[j] for i, j in pairs]
        qs = np.array([abs(d) / se if se else np.inf for d, se in zip(diffs, ses)])
        ps = np.where(np.isfinite(qs), _ptukey.sf(np.where(np.isfinite(qs), qs, 0.0), k, df), 0.0)
        for (i, j), d, se, p in zip(pairs, diffs, ses, ps):
            half = q_crit * se
            rows.append([f"{labels[i]} vs {labels[j]}", f"{d:.4g}", f"{d - half:.4g} to {d + half:.4g}",
                         f"{p:.4g}", "Yes" if p < 0.05 else "No"])
        return (ResultTable("Which groups differ (Tukey)", ["Comparison", "Mean difference", f"{pct}% CI",
                                                              "Adjusted P", "Significant"], rows),
                "Tukey's test for every pair.")

    if method == "dunnett":
        others = [j for j in range(k) if j != control]
        res = stats.dunnett(*[arrays[j] for j in others], control=arrays[control])
        ci = res.confidence_interval(confidence_level=confidence)
        for n, j in enumerate(others):
            p = float(res.pvalue[n])
            rows.append([f"{labels[j]} vs {labels[control]}", f"{means[j] - means[control]:.4g}",
                         f"{ci.low[n]:.4g} to {ci.high[n]:.4g}", f"{p:.4g}", "Yes" if p < 0.05 else "No"])
        return (ResultTable(f"Which groups differ from {labels[control]} (Dunnett)",
                            ["Comparison", "Mean difference", f"{pct}% CI", "Adjusted P", "Significant"], rows),
                f"Dunnett's test, each group against {labels[control]}.")

    if method == "games_howell":
        from . import _ptukey

        for i, j in pairs:
            a, b = arrays[i], arrays[j]
            va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
            se = np.sqrt(va + vb)
            df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)) if va + vb > 0 else np.inf
            diff = means[i] - means[j]
            q = abs(diff) / se * np.sqrt(2) if se else np.inf
            p = float(_ptukey.sf(q, k, df)[0]) if np.isfinite(q) else 0.0
            q_crit = _ptukey.ppf(confidence, k, df)
            half = q_crit / np.sqrt(2) * se
            rows.append([f"{labels[i]} vs {labels[j]}", f"{diff:.4g}", f"{diff - half:.4g} to {diff + half:.4g}",
                         f"{p:.4g}", "Yes" if p < 0.05 else "No"])
        return (ResultTable("Which groups differ (Games-Howell)",
                            ["Comparison", "Mean difference", f"{pct}% CI", "Adjusted P", "Significant"], rows),
                "Games-Howell test for every pair - it doesn't assume equal scatter.")

    # a test between each pair (paired or not, t or rank), then Holm / FDR / none
    raw, diffs = [], []
    for i, j in pairs:
        a, b = arrays[i], arrays[j]
        if paired:
            n = min(a.size, b.size)
            test = stats.wilcoxon if ranks else stats.ttest_rel
            raw.append(float(test(a[:n], b[:n]).pvalue))
            diffs.append(float(np.median(a[:n] - b[:n]) if ranks else np.mean(a[:n] - b[:n])))
        else:
            raw.append(float((stats.mannwhitneyu(a, b) if ranks else stats.ttest_ind(a, b, equal_var=False)).pvalue))
            diffs.append(float(np.median(a) - np.median(b)) if ranks else means[i] - means[j])
    adjusted = adjust(raw, method)
    name = {"holm": "Holm", "fdr": "Benjamini-Hochberg", "none": "no correction"}.get(method, "Holm")
    for (i, j), d, p_raw, p_adj in zip(pairs, diffs, raw, adjusted):
        rows.append([f"{labels[i]} vs {labels[j]}", f"{d:.4g}", "–", f"{p_adj:.4g}", "Yes" if p_adj < 0.05 else "No"])
    kind = ("Wilcoxon signed rank tests" if paired else "Mann-Whitney tests") if ranks else (
        "paired t tests" if paired else "Welch t tests")
    return (ResultTable(f"Which groups differ ({kind}, {name})",
                        ["Comparison", "Median difference" if ranks else "Mean difference", f"{pct}% CI",
                         "Adjusted P" if method != "none" else "P",
                         "Significant"], rows),
            f"{kind[0].upper() + kind[1:]} between each pair, {name}.")


# ---------------------------------------------------------------- normality
def lilliefors_p(d_stat: float, n: int) -> float:
    """Kolmogorov-Smirnov P when the mean and SD were estimated from the
    same data (Dallal-Wilkinson approximation to Lilliefors). The plain KS
    P value is far too large in that case."""
    from .normality_testing import _dallal_wilkinson_lilliefors_p

    return _dallal_wilkinson_lilliefors_p(d_stat, n)


def normality_rows(values: np.ndarray) -> list[list[str]]:
    """Rows for "Are the residuals normal?", KS corrected for estimated parameters."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    rows = []
    if v.size >= 3:
        w, p = stats.shapiro(v)
        rows.append(["Shapiro-Wilk", f"W = {w:.4f}", f"{p:.4g}"])
    if v.size >= 8:
        k2, p = stats.normaltest(v)
        rows.append(["D'Agostino-Pearson", f"K2 = {k2:.4f}", f"{p:.4g}"])
    if v.size >= 8:
        ad = stats.anderson(v, dist="norm", method="interpolate")
        rows.append(["Anderson-Darling", f"A2 = {ad.statistic:.4f}", f"{ad.pvalue:.4g}"])
    if v.size >= 5:
        sd = np.std(v, ddof=1) or 1.0
        d_stat = float(stats.kstest((v - np.mean(v)) / sd, "norm").statistic)
        rows.append(["Kolmogorov-Smirnov (Lilliefors)", f"D = {d_stat:.4f}", f"{lilliefors_p(d_stat, v.size):.4g}"])
    return rows


# ---------------------------------------------------------------- assumption checks
def assumption_checks(groups: dict[str, np.ndarray], distribution: str, equal_var: bool | None,
                      paired_diffs: np.ndarray | None = None) -> tuple[ResultTable, list[str]]:
    """Shape and spread of each group, read in plain words against the test
    that was chosen. Returns the table and the advice lines."""
    rows, advice = [], []
    ns = {label: v.size for label, v in groups.items()}
    small = min(ns.values()) < 8 if ns else True

    shaped = {"Differences": paired_diffs} if paired_diffs is not None else groups
    skewed, not_normal = [], []
    for label, v in shaped.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 3:
            rows.append([label, str(v.size), "–", "–", "too few values to judge"])
            continue
        w, p = stats.shapiro(v)
        skew = float(stats.skew(v)) if v.size >= 3 else float("nan")
        rows.append([label, str(v.size), f"{np.std(v, ddof=1):.4g}", f"{p:.3g}",
                     "looks normal" if p >= 0.05 else "doesn't look normal"])
        if p < 0.05:
            not_normal.append(label)
        if skew > 1 and (v > 0).all():
            skewed.append(label)

    table = ResultTable("Checks on the assumptions",
                        ["Group", "N", "SD", "Shapiro-Wilk P", "Shape"], rows)

    if distribution == "gaussian" and not_normal:
        if skewed and len(skewed) == len(shaped):
            advice.append(f"{', '.join(not_normal)} {'is' if len(not_normal) == 1 else 'are'} right-skewed with only "
                          "positive values - 'normal once logged' is likely the better choice.")
        else:
            advice.append(f"{', '.join(not_normal)} {'does' if len(not_normal) == 1 else 'do'} not look normal - "
                          "consider 'assume nothing about their shape'.")
    if equal_var is not None and paired_diffs is None and len(groups) >= 2:
        arrays = [v for v in groups.values() if v.size >= 2]
        sds = [float(np.std(v, ddof=1)) for v in arrays]
        if len(arrays) >= 2 and min(sds) > 0:
            ratio = max(sds) / min(sds)
            bf_p = float(stats.levene(*arrays, center="median").pvalue)
            table.rows.append(["Spread", "", f"largest / smallest SD = {ratio:.2g}", f"{bf_p:.3g}",
                               "Brown-Forsythe test"])
            if equal_var and (ratio > 2 or bf_p < 0.05):
                advice.append(f"The SDs differ {ratio:.1f}-fold - turn off 'the groups scatter about equally' "
                              "so the test doesn't rely on it.")
    if small:
        advice.append("With fewer than 8 values per group these checks can't see much either way - lean on what "
                      "you know about this kind of measurement.")
    if not advice:
        advice.append("Nothing in these checks argues against the test you chose.")
    return table, advice


def small_n_note(ns: dict[str, int]) -> str | None:
    few = [f"{label} (n = {n})" for label, n in ns.items() if n < SMALL_N]
    if not few:
        return None
    return ("Very small samples: " + ", ".join(few) + ". A test this small can only detect large effects, "
            "and a non-significant P says little - read the confidence interval instead.")


# ---------------------------------------------------------------- effect sizes
def hedges_g(a: np.ndarray, b: np.ndarray, paired: bool = False, confidence: float = 0.95) -> tuple[float, float, float]:
    """Standardised difference (A - B) with a small-sample correction, and
    an approximate confidence interval."""
    z = stats.norm.ppf(0.5 + confidence / 2)
    if paired:
        d = a - b
        n = d.size
        sd = np.std(d, ddof=1)
        dz = float(np.mean(d) / sd) if sd else float("nan")
        g = dz * (1 - 3 / (4 * (n - 1) - 1)) if n > 2 else dz
        se = np.sqrt(1 / n + g**2 / (2 * n))
    else:
        n1, n2 = a.size, b.size
        pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
        d = float((np.mean(a) - np.mean(b)) / pooled) if pooled else float("nan")
        g = d * (1 - 3 / (4 * (n1 + n2) - 9))
        se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return g, g - z * se, g + z * se


_DIRECT_LIMIT = 2_000_000  # list every difference up to this many; count beyond it


def _kth_implicit(count_le, snap, lo: float, hi: float, m: int) -> float:
    """The m-th smallest (0-based) of a set too big to list, given
    count_le(x) = how many members are <= x and snap(x) = the largest member
    <= x. Bisection on the value, then snapped to the exact member."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if count_le(mid) >= m + 1:
            hi = mid
        else:
            lo = mid
    return snap(hi)


def _unpaired_order_stats(a: np.ndarray, b: np.ndarray, ms: list[int]) -> list[float]:
    """Order statistics of every difference a_i - b_j, without listing them."""
    a, b = np.sort(a), np.sort(b)
    n2 = b.size

    def count_le(x):
        return int(np.sum(n2 - np.searchsorted(b, a - x, side="left")))

    def snap(x):
        idx = np.searchsorted(b, a - x, side="left")
        ok = idx < n2
        return float(np.max(a[ok] - b[idx[ok]]))

    lo, hi = float(a[0] - b[-1]) - 1.0, float(a[-1] - b[0])
    return [_kth_implicit(count_le, snap, lo, hi, m) for m in ms]


def _walsh_order_stats(d: np.ndarray, ms: list[int]) -> list[float]:
    """Order statistics of the Walsh averages (d_i + d_j)/2, i <= j."""
    d = np.sort(d)
    idx_i = np.arange(d.size)

    def count_le(x):
        j = np.searchsorted(d, 2 * x - d, side="right")
        return int(np.sum(np.maximum(0, j - idx_i)))

    def snap(x):
        j = np.searchsorted(d, 2 * x - d, side="right") - 1
        ok = j >= idx_i
        return float(np.max((d[ok] + d[j[ok]]) / 2))

    lo, hi = float(d[0]) - 1.0, float(d[-1])
    return [_kth_implicit(count_le, snap, lo, hi, m) for m in ms]


def hodges_lehmann(a: np.ndarray, b: np.ndarray, paired: bool = False,
                   confidence: float = 0.95) -> tuple[float, float, float]:
    """Median difference (A - B) with its distribution-free confidence
    interval - the effect that goes with a rank test.

    Small samples list every difference; large ones (a 50 000-row sheet
    has 2.5 billion) find the needed order statistics by counting instead,
    so memory and time stay small and the answer is the same."""
    z = stats.norm.ppf(0.5 + confidence / 2)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if paired:
        d = a - b
        n = d.size
        total = n * (n + 1) // 2
        k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    else:
        n1, n2 = a.size, b.size
        total = n1 * n2
        k = int(np.floor(n1 * n2 / 2 - z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)))
    k = max(0, min(k, total - 1))
    mid = [(total - 1) // 2, total // 2]
    wanted = mid + [k, total - 1 - k]
    if total <= _DIRECT_LIMIT:
        if paired:
            i, j = np.triu_indices(d.size)
            values = np.sort((d[i] + d[j]) / 2)
        else:
            values = np.sort(np.subtract.outer(a, b).ravel())
        picked = [float(values[m]) for m in wanted]
    else:
        picked = _walsh_order_stats(d, wanted) if paired else _unpaired_order_stats(a, b, wanted)
    return 0.5 * (picked[0] + picked[1]), picked[2], picked[3]


def rank_biserial(a: np.ndarray, b: np.ndarray, paired: bool = False) -> float:
    if paired:
        d = a - b
        d = d[d != 0]
        if not d.size:
            return 0.0
        ranks = stats.rankdata(np.abs(d))
        plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
        return float((plus - minus) / (plus + minus))
    u = stats.mannwhitneyu(a, b).statistic
    return float(2 * u / (a.size * b.size) - 1)


def eta_omega(arrays: list[np.ndarray]) -> tuple[float, float]:
    """η² and ω² for a one-way layout (ω² is the less biased of the two)."""
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ss_between = sum(a.size * (a.mean() - grand) ** 2 for a in arrays)
    ss_total = float(np.sum((allv - grand) ** 2))
    k, n = len(arrays), allv.size
    ms_within = (ss_total - ss_between) / (n - k) if n > k else float("nan")
    eta = ss_between / ss_total if ss_total else float("nan")
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within) if ss_total else float("nan")
    return float(eta), float(max(omega, 0.0))


def magnitude(value: float, cuts: tuple[float, float, float]) -> str:
    """Conventional words for an effect size: negligible / small / medium / large."""
    v = abs(value)
    if not np.isfinite(v):
        return ""
    return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


# ---------------------------------------------------------------- Fisher's exact test beyond 2x2
def _log_table_prob(tables: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """log P of each table (last two axes) under fixed margins - the
    multivariate hypergeometric probability."""
    from scipy.special import gammaln

    n = rows.sum()
    const = gammaln(rows + 1).sum() + gammaln(cols + 1).sum() - gammaln(n + 1)
    return const - gammaln(tables + 1).sum(axis=(-2, -1))


def fisher_rxc(table: np.ndarray, n_resamples: int = 200_000, seed: int = 20240601) -> tuple[float, str]:
    """Freeman-Halton extension of Fisher's exact test: the P value is the
    total probability of every table with the same margins that is no more
    likely than the one observed.

    Exact (full enumeration) for 2 x c and r x 2 tables of moderate size;
    otherwise a Monte Carlo estimate from ``n_resamples`` random tables with
    a fixed seed, so the same data always give the same P. Returns
    (P, how it was computed)."""
    table = np.asarray(table, dtype=np.int64)
    if table.shape[0] != 2 and table.shape[1] == 2:
        table = table.T
    rows, cols = table.sum(axis=1), table.sum(axis=0)
    observed = float(_log_table_prob(table[None], rows, cols)[0])
    tol = 1e-7

    if table.shape[0] == 2:
        r1 = int(rows[0])
        c = cols.size
        ranges = [np.arange(0, int(cols[j]) + 1) for j in range(c - 1)]
        count = float(np.prod([r.size for r in ranges])) if ranges else 1.0
        if count <= 4_000_000:
            grids = np.meshgrid(*ranges, indexing="ij") if ranges else []
            first = np.stack([g.ravel() for g in grids], axis=1) if grids else np.zeros((1, 0), dtype=np.int64)
            last = r1 - first.sum(axis=1)
            ok = (last >= 0) & (last <= cols[-1])
            top = np.concatenate([first[ok], last[ok, None]], axis=1)
            bottom = cols[None, :] - top
            ok2 = (bottom >= 0).all(axis=1)
            tables = np.stack([top[ok2], bottom[ok2]], axis=1)
            logp = _log_table_prob(tables, rows, cols)
            p = float(np.exp(logp[logp <= observed + tol]).sum())
            return min(p, 1.0), "exact"

    from scipy import stats as _stats

    sampler = _stats.random_table(rows, cols, seed=np.random.default_rng(seed))
    hits = 0
    done = 0
    batch = 50_000
    while done < n_resamples:
        size = min(batch, n_resamples - done)
        logp = _log_table_prob(sampler.rvs(size=size).astype(np.int64), rows, cols)
        hits += int(np.sum(logp <= observed + tol))
        done += size
    return (hits + 1) / (n_resamples + 1), f"Monte Carlo, {n_resamples:,} tables"
