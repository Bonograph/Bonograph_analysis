"""Do the columns agree? Reliability of a scale and agreement between raters.

Every row is one subject; what the data sets are decides the measure:

  items of one scale       -> Cronbach's alpha, with what each item adds
  raters giving scores     -> intraclass correlations (Shrout and Fleiss 1979)
  raters giving categories -> Cohen's kappa (two raters, weighted if the
                              categories are ordered) or Fleiss' kappa (more)

Rows with a value missing in any data set are left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register


@dataclass
class ReliabilityOptions:
    kind: str = "items"            # "items" | "scores" | "categories"
    icc_raters: str = "sample"     # "sample": raters stand for others like them | "these": only these raters matter | "different": each subject has its own raters
    icc_agreement: bool = True     # raters must give the same number (not only rank the subjects alike)
    weights: str = "none"          # kappa for ordered categories: "none" | "linear" | "quadratic"
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def _complete(data: SheetData) -> tuple[list[str], np.ndarray, int]:
    names, matrix = data.first_replicate_matrix()
    if len(names) < 2:
        raise AnalysisError("This needs two data sets or more, with one subject on each row.")
    filled = np.isfinite(matrix).any(axis=1)
    keep = np.isfinite(matrix).all(axis=1)
    return names, matrix[keep], int(filled.sum() - keep.sum())


def cronbach(matrix: np.ndarray) -> float:
    k = matrix.shape[1]
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - matrix.var(axis=0, ddof=1).sum() / total_var))


def icc_forms(matrix: np.ndarray, level: float) -> dict:
    """The six intraclass correlations of Shrout and Fleiss (1979) with
    their confidence limits, from the two-way table of subjects by raters."""
    n, k = matrix.shape
    grand = matrix.mean()
    bms = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)                 # between subjects
    jms = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)                 # between raters
    wms = ((matrix - matrix.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))   # within subjects
    ems = (((matrix - grand) ** 2).sum() - (n - 1) * bms - (k - 1) * jms) / ((n - 1) * (k - 1))   # residual
    a = 1 - (1 - level) / 2
    out = {"ms": {"subjects": bms, "raters": jms, "within": wms, "residual": ems}, "n": n, "k": k}

    def simple(f0: float, df1: float, df2: float):
        fl = f0 / stats.f.ppf(a, df1, df2)
        fu = f0 * stats.f.ppf(a, df2, df1)
        return ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)), (1 - 1 / fl, 1 - 1 / fu), float(stats.f.sf(f0, df1, df2))

    if wms > 0:
        single, average, p = simple(bms / wms, n - 1, n * (k - 1))
        out["1"] = {"single": (bms - wms) / (bms + (k - 1) * wms), "single_ci": single, "average": (bms - wms) / bms, "average_ci": average,
                    "p": p, "f": bms / wms, "df": (n - 1, n * (k - 1))}
    if ems > 0:
        single, average, p = simple(bms / ems, n - 1, (n - 1) * (k - 1))
        out["3"] = {"single": (bms - ems) / (bms + (k - 1) * ems), "single_ci": single, "average": (bms - ems) / bms, "average_ci": average,
                    "p": p, "f": bms / ems, "df": (n - 1, (n - 1) * (k - 1))}
        rho = (bms - ems) / (bms + (k - 1) * ems + k * (jms - ems) / n)
        fj = jms / ems
        top = (k * rho * fj + n * (1 + (k - 1) * rho) - k * rho) ** 2
        bottom = (n - 1) * k**2 * rho**2 * fj**2 + (n * (1 + (k - 1) * rho) - k * rho) ** 2
        v = (k - 1) * (n - 1) * top / bottom if bottom > 0 else (n - 1) * (k - 1)
        f_up, f_low = stats.f.ppf(a, n - 1, v), stats.f.ppf(a, v, n - 1)
        low = n * (bms - f_up * ems) / (f_up * (k * jms + (k * n - k - n) * ems) + n * bms)
        high = n * (f_low * bms - ems) / (k * jms + (k * n - k - n) * ems + n * f_low * bms)
        out["2"] = {"single": rho, "single_ci": (low, high), "average": (bms - ems) / (bms + (jms - ems) / n),
                    "average_ci": (k * low / (1 + (k - 1) * low), k * high / (1 + (k - 1) * high)), "p": p, "f": bms / ems,
                    "df": (n - 1, (n - 1) * (k - 1))}
    return out


def weighted_kappa(table: np.ndarray, weights: str = "none") -> dict:
    """Cohen's kappa from a square table of counts, with the large-sample
    standard errors of Fleiss, Cohen and Everitt (1969): one for the
    confidence interval, one for the test against no agreement beyond chance."""
    n = table.sum()
    c = table.shape[0]
    p = table / n
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    i, j = np.indices((c, c))
    if weights == "linear" and c > 1:
        w = 1 - np.abs(i - j) / (c - 1)
    elif weights == "quadratic" and c > 1:
        w = 1 - (i - j) ** 2 / (c - 1) ** 2
    else:
        w = (i == j).astype(float)
    observed = float((w * p).sum())
    expected = float((w * np.outer(rows, cols)).sum())
    if expected >= 1:
        return {"kappa": float("nan"), "observed": observed, "expected": expected}
    kappa = (observed - expected) / (1 - expected)
    w_row = w @ cols                    # mean weight of each row category
    w_col = rows @ w                    # ... and of each column category
    spread = (p * (w - (w_row[:, None] + w_col[None, :]) * (1 - kappa)) ** 2).sum() - (kappa - expected * (1 - kappa)) ** 2
    se = float(np.sqrt(max(spread, 0.0) / (n * (1 - expected) ** 2)))
    null = (np.outer(rows, cols) * (w - (w_row[:, None] + w_col[None, :])) ** 2).sum() - expected**2
    se0 = float(np.sqrt(max(null, 0.0) / (n * (1 - expected) ** 2)))
    return {"kappa": float(kappa), "observed": observed, "expected": expected, "se": se, "se0": se0}


def fleiss_kappa(counts: np.ndarray) -> dict:
    """Fleiss' kappa (1971) from subjects x categories counts of how many raters chose each category."""
    n, _c = counts.shape
    m = counts[0].sum()
    share = counts.sum(axis=0) / (n * m)
    per_subject = ((counts**2).sum(axis=1) - m) / (m * (m - 1))
    observed, expected = float(per_subject.mean()), float((share**2).sum())
    if expected >= 1:
        return {"kappa": float("nan"), "observed": observed, "expected": expected}
    kappa = (observed - expected) / (1 - expected)
    se0 = float(np.sqrt(2 / (n * m * (m - 1))) * np.sqrt(max(expected - (2 * m - 3) * expected**2 + 2 * (m - 2) * (share**3).sum(), 0.0))
                / (1 - expected))
    return {"kappa": float(kappa), "observed": observed, "expected": expected, "se0": se0, "share": share}


def _strength(value: float) -> str:
    """Landis and Koch's (1977) words for a kappa."""
    if not np.isfinite(value):
        return ""
    for cut, word in ((0.0, "none beyond chance"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")):
        if value <= cut:
            return word
    return "almost perfect"


@register("reliability")
def run(data: SheetData, options: ReliabilityOptions | None = None) -> AnalysisResult:
    opt = options or ReliabilityOptions()
    names, matrix, dropped = _complete(data)
    n, k = matrix.shape
    if n < 3:
        raise AnalysisError("At least 3 rows with a value in every data set are needed.")
    level = opt.confidence_level
    pct = f"{round(level * 100)}%"
    sig = opt.significant_digits
    z = stats.norm.ppf(0.5 + level / 2)

    def num(value) -> str:
        return "" if value is None or not np.isfinite(value) else f"{value:.{sig}g}"

    def three(value) -> str:
        if not np.isfinite(value):
            return ""
        text = f"{value:.3f}"
        return text.lstrip("-") if float(text) == 0 else text      # never "-0.000"

    def span(low, high) -> str:
        return f"{three(low)} to {three(high)}"

    def p_text(p) -> str:
        return format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    left_out = [f"{dropped} row{'s' if dropped != 1 else ''} with a missing value {'were' if dropped != 1 else 'was'} left out."] if dropped else []

    # ── items of a scale ─────────────────────────────────────────────────────
    if opt.kind == "items":
        alpha = cronbach(matrix)
        if not np.isfinite(alpha):
            raise AnalysisError("The total score is the same for every subject, so reliability cannot be worked out.")
        corr = np.corrcoef(matrix, rowvar=False)
        mean_r = float(corr[np.triu_indices(k, 1)].mean())
        standardised = k * mean_r / (1 + (k - 1) * mean_r)
        df1, df2 = n - 1, (n - 1) * (k - 1)
        tail = (1 - level) / 2
        low, high = 1 - (1 - alpha) * stats.f.ppf(1 - tail, df1, df2), 1 - (1 - alpha) * stats.f.ppf(tail, df1, df2)
        total = matrix.sum(axis=1)
        item_rows, without, item_total = [], [], []
        for j in range(k):
            rest = total - matrix[:, j]
            r = float(np.corrcoef(matrix[:, j], rest)[0, 1]) if matrix[:, j].std() > 0 and rest.std() > 0 else float("nan")
            a_without = cronbach(np.delete(matrix, j, axis=1)) if k > 2 else float("nan")
            item_total.append(r)
            without.append(a_without)
            remark = ("runs against the rest: is it scored the other way round?" if np.isfinite(r) and r < 0 else
                      "the scale is more reliable without it" if np.isfinite(a_without) and a_without > alpha + 0.005 else
                      "weakly tied to the rest" if np.isfinite(r) and r < 0.3 else "")
            item_rows.append([names[j], num(matrix[:, j].mean()), num(matrix[:, j].std(ddof=1)), three(r), three(a_without), remark])
        tables = [ResultTable("Reliability of the scale", ["Measure", "Value", f"{pct} CI"],
                              [["Cronbach's alpha", three(alpha), span(low, high)], ["Alpha from the correlations (items put on one scale)", three(standardised), ""],
                               ["Mean correlation between items", three(mean_r), ""], ["Items", str(k), ""], ["Subjects", str(n), ""]]),
                  ResultTable("What each item adds", ["Item", "Mean", "SD", "Correlation with the rest", "Alpha without it", "Remark"], item_rows),
                  ResultTable("Correlations between the items", ["Item"] + names,
                              [[names[i]] + [three(corr[i, j]) for j in range(k)] for i in range(k)])]
        word = "low" if alpha < 0.6 else "questionable" if alpha < 0.7 else "acceptable" if alpha < 0.8 else "good" if alpha < 0.9 else "high"
        summary = (f"Cronbach's alpha = {alpha:.3f} ({pct} CI {span(low, high)}) for {k} items and {n} subjects - {word} by the usual "
                   "rough guide.")
        flagged = [names[j] for j in range(k) if np.isfinite(item_total[j]) and item_total[j] < 0]
        notes = left_out + [
            "Alpha says how consistently the items rank the subjects, taken together. It rises with the number of items as well "
            "as with their correlation, so a long scale can score high with weak items.",
            "It assumes every item measures the same thing on the same scale. Reverse-scored items must be turned round first"
            + (f": {', '.join(flagged)} run{'s' if len(flagged) == 1 else ''} against the rest." if flagged else "."),
            "'Alpha without it' above the alpha of the whole scale marks an item that lowers reliability. Above about 0.95, items "
            "are probably repeating each other.",
            "Rough guide: under 0.6 low, 0.7 acceptable, 0.8 good, 0.9 high. Interval by Feldt's method (1965). "
            "Reference: Cronbach 1951."]
        xs = [float(j) for j in range(k)]
        charts = [ChartSpec("Each item's correlation with the rest of the scale", x_label="Item", y_label="Correlation",
                            series=[ChartSeries("", xs, [0.0 if not np.isfinite(v) else v for v in item_total], kind="bar", color="#1f5aa6")],
                            x_categories=names, reference_lines=[{"kind": "h", "value": 0.3, "color": "#d1342f", "style": "--", "width": 0.9}])]
        if k > 2:
            charts.append(ChartSpec("Alpha with each item left out (line: all items)", x_label="Item left out", y_label="Cronbach's alpha",
                                    series=[ChartSeries("", xs, without, kind="bar", color="#7b9fd0")], x_categories=names,
                                    reference_lines=[{"kind": "h", "value": alpha, "color": "#222222", "style": "-", "width": 1.2}]))
        return AnalysisResult(title="Do the columns agree?", summary=summary, tables=tables, notes=notes, charts=charts)

    # ── raters giving scores ─────────────────────────────────────────────────
    if opt.kind == "scores":
        forms = icc_forms(matrix, level)
        if "3" not in forms and "1" not in forms:
            raise AnalysisError("The raters agree exactly on every subject, or nothing varies: there is no variation to split up.")
        which = {"different": "1", "sample": "2", "these": "3"}[opt.icc_raters]
        if which == "2" and not opt.icc_agreement:
            which = "3"                      # consistency for a sample of raters is numerically the fixed-rater form
        if which not in forms:
            which = "3" if "3" in forms else "1"
        described = {"1": ("ICC(1,1)", "ICC(1,k)", "Each subject scored by different raters"),
                     "2": ("ICC(2,1)", "ICC(2,k)", "The same raters, standing for raters in general; they must give the same number"),
                     "3": ("ICC(3,1)", "ICC(3,k)", "The same raters; they need only rank the subjects alike")}
        rows = []
        for key in ("1", "2", "3"):
            if key in forms:
                f = forms[key]
                mark = "  ← asked for" if key == which else ""
                rows.append([described[key][0] + mark, described[key][2], "one rater's score", three(f["single"]), span(*f["single_ci"])])
                rows.append([described[key][1].replace("k", str(k)) + mark, "", f"the mean of all {k} raters", three(f["average"]), span(*f["average_ci"])])
        chosen = forms[which]
        ms = forms["ms"]
        tables = [ResultTable("Intraclass correlations", ["Form", "When", "Reliability of", "ICC", f"{pct} CI"], rows),
                  ResultTable("Is there any agreement at all?", ["Test", "F", "df", "P value"],
                              [[f"Subjects differ more than raters disagree ({described[which][0]})", f"{chosen['f']:.4f}",
                                f"{chosen['df'][0]}, {chosen['df'][1]:g}", p_text(chosen["p"])]]),
                  ResultTable("Where the variation lies", ["Source", "Mean square", "df"],
                              [["Between subjects", num(ms["subjects"]), str(n - 1)], ["Between raters", num(ms["raters"]), str(k - 1)],
                               ["Residual", num(ms["residual"]), str((n - 1) * (k - 1))], ["Within subjects (raters and residual)", num(ms["within"]), str(n * (k - 1))]]),
                  ResultTable("Each rater", ["Rater", "Mean", "SD", "Mean minus the mean of all"],
                              [[names[j], num(matrix[:, j].mean()), num(matrix[:, j].std(ddof=1)), f"{matrix[:, j].mean() - matrix.mean():+.{sig}g}"]
                               for j in range(k)])]
        value = chosen["single"]
        word = "poor" if value < 0.5 else "moderate" if value < 0.75 else "good" if value < 0.9 else "excellent"
        summary = (f"{described[which][0]} = {value:.3f} ({pct} CI {span(*chosen['single_ci'])}) for one rater's score - {word} by the "
                   f"usual rough guide; {chosen['average']:.3f} for the mean of the {k} raters. {n} subjects.")
        notes = left_out + [
            "An intraclass correlation is the share of the variation in the scores that comes from real differences between "
            "subjects. Read the lower confidence limit, not only the estimate: with few subjects it can be far below.",
            "Which form is right depends on the study, not on the data. 'One rater's score' is the reliability of the method as "
            "it will be used if one rater scores each subject; 'the mean of all raters' applies only if every subject will get "
            "that many ratings.",
            "ICC(2,·) counts a rater who scores everyone higher as disagreement; ICC(3,·) does not. A large gap between the two "
            "means the raters differ in level: see 'Each rater'.",
            "Rough guide (Koo and Li 2016): under 0.5 poor, 0.5 to 0.75 moderate, 0.75 to 0.9 good, above 0.9 excellent.",
            "Forms and confidence limits from Shrout and Fleiss 1979; McGraw and Wong 1996."]
        order = np.argsort(matrix.mean(axis=1))
        from graphs.engine.style import PALETTES

        palette = PALETTES["classic"]
        xs = [float(i + 1) for i in range(n)]
        charts = [ChartSpec("Each rater's score, subjects in order of their mean score", x_label="Subject (ordered)", y_label="Score",
                            series=[ChartSeries(names[j], xs, matrix[order, j].tolist(), kind="line_markers" if n <= 40 else "line",
                                                color=palette[j % len(palette)]) for j in range(k)])]
        return AnalysisResult(title="Do the columns agree?", summary=summary, tables=tables, notes=notes, charts=charts)

    # ── raters giving categories ─────────────────────────────────────────────
    if not np.allclose(matrix, np.round(matrix)):
        raise AnalysisError("Categories must be whole-number codes (1, 2, 3 ...). For scores on a scale choose 'Raters giving scores'.")
    codes = np.unique(matrix)
    if codes.size < 2:
        raise AnalysisError("Every rating is the same category, so agreement beyond chance cannot be worked out.")
    if codes.size > 20:
        raise AnalysisError(f"{codes.size} different categories were found. For scores on a scale choose 'Raters giving scores'.")
    index = {c: i for i, c in enumerate(codes)}
    labels = [f"{c:g}" for c in codes]
    c = codes.size

    if k == 2:
        table = np.zeros((c, c))
        for a, b in matrix:
            table[index[a], index[b]] += 1
        res = weighted_kappa(table, opt.weights if c > 2 else "none")
        if not np.isfinite(res["kappa"]):
            raise AnalysisError("Both raters used one category only, so chance agreement is complete and kappa has no value.")
        kappa, se, se0 = res["kappa"], res["se"], res["se0"]
        p = float(2 * stats.norm.sf(abs(kappa) / se0)) if se0 > 0 else float("nan")
        weighted = opt.weights != "none" and c > 2
        name = f"Cohen's kappa, {opt.weights} weights" if weighted else "Cohen's kappa"
        rows = [[name, three(kappa), three(se), span(max(-1.0, kappa - z * se), min(1.0, kappa + z * se)), p_text(p), _strength(kappa)]]
        if weighted:
            plain = weighted_kappa(table, "none")
            rows.append(["Cohen's kappa, exact matches only", three(plain["kappa"]), three(plain["se"]),
                         span(max(-1.0, plain["kappa"] - z * plain["se"]), min(1.0, plain["kappa"] + z * plain["se"])), "", _strength(plain["kappa"])])
        tables = [ResultTable("Agreement beyond chance", ["Measure", "Value", "SE", f"{pct} CI", "P value (against chance)", "Strength"], rows),
                  ResultTable("How the agreement is made up", ["", "Share"],
                              [["Agreement seen" + (" (near misses count in part)" if weighted else ""), f"{res['observed']:.1%}"],
                               ["Agreement expected by chance", f"{res['expected']:.1%}"],
                               ["Same category given", f"{np.trace(table) / n:.1%}"], ["Subjects", str(n)]]),
                  ResultTable(f"Ratings: {names[0]} down, {names[1]} across", [names[0]] + labels + ["Total"],
                              [[labels[i]] + [str(int(v)) for v in table[i]] + [str(int(table[i].sum()))] for i in range(c)]
                              + [["Total"] + [str(int(v)) for v in table.sum(axis=0)] + [str(n)]])]
        summary = (f"{name} = {kappa:.3f} ({pct} CI {span(max(-1.0, kappa - z * se), min(1.0, kappa + z * se))}): {_strength(kappa)} agreement "
                   f"by Landis and Koch's guide. The raters gave the same category for {np.trace(table) / n:.0%} of {n} subjects; "
                   f"{res['expected']:.0%} was expected by chance.")
        notes = left_out + [
            "Kappa is the agreement beyond what two raters would reach by chance, as a share of the most they could reach: "
            "0 is chance, 1 is complete agreement.",
            "Kappa falls when one category is rare or when the raters use the categories in different proportions, even if they "
            "mostly agree. Report the raw agreement and the table beside it.",
            ("Weights give part credit for a near miss, which suits ordered categories; quadratic weights give a value close to "
             "an intraclass correlation." if c > 2 else ""),
            "Landis and Koch (1977): up to 0.20 slight, 0.40 fair, 0.60 moderate, 0.80 substantial, above that almost perfect - "
            "their own description of these cuts was 'clearly arbitrary'.",
            "Standard errors: Fleiss, Cohen and Everitt 1969. Reference: Cohen 1960, 1968."]
        charts = [ChartSpec(f"Ratings: {names[0]} (rows) against {names[1]} (columns)", kind="heatmap", matrix=table.tolist(),
                            row_labels=labels, col_labels=labels, cmap="Blues", x_label=names[1], y_label=names[0])]
        return AnalysisResult(title="Do the columns agree?", summary=summary, tables=tables, notes=[x for x in notes if x], charts=charts)

    counts = np.zeros((n, c))
    for i, row in enumerate(matrix):
        for value in row:
            counts[i, index[value]] += 1
    res = fleiss_kappa(counts)
    if not np.isfinite(res["kappa"]):
        raise AnalysisError("Every rater used one category only, so chance agreement is complete and kappa has no value.")
    kappa, se0 = res["kappa"], res["se0"]
    p = float(2 * stats.norm.sf(abs(kappa) / se0)) if se0 > 0 else float("nan")
    share = res["share"]
    per_category = []
    for j in range(c):
        pj = share[j]
        kj = 1 - (counts[:, j] * (k - counts[:, j])).sum() / (n * k * (k - 1) * pj * (1 - pj)) if 0 < pj < 1 else float("nan")
        per_category.append([labels[j], f"{pj:.1%}", three(kj), _strength(kj)])
    all_same = int((counts.max(axis=1) == k).sum())
    tables = [ResultTable("Agreement beyond chance", ["Measure", "Value", "SE (against chance)", "P value (against chance)", "Strength"],
                          [[f"Fleiss' kappa, {k} raters", three(kappa), three(se0), p_text(p), _strength(kappa)]]),
              ResultTable("How the agreement is made up", ["", "Value"],
                          [["Pairs of raters that agree, per subject", f"{res['observed']:.1%}"], ["Expected by chance", f"{res['expected']:.1%}"],
                           ["Subjects on which every rater agrees", f"{all_same} of {n}"]]),
              ResultTable("Category by category", ["Category", "Share of all ratings", "Kappa for this category against the rest", "Strength"], per_category)]
    summary = (f"Fleiss' kappa = {kappa:.3f} for {k} raters and {n} subjects: {_strength(kappa)} agreement by Landis and Koch's guide. "
               f"All raters agree on {all_same} subjects.")
    notes = left_out + [
        "Fleiss' kappa treats the raters as interchangeable: it is the chance-corrected share of rater pairs that agree. "
        "With two raters it is not the same as Cohen's kappa, which lets each rater have their own habits.",
        "It takes no account of order between categories: a near miss counts as a miss.",
        "The category table shows where the raters part: a low value means that category is used inconsistently.",
        "Landis and Koch (1977): up to 0.20 slight, 0.40 fair, 0.60 moderate, 0.80 substantial, above that almost perfect.",
        "Reference: Fleiss 1971; the standard error is the one for testing against chance agreement."]
    charts = [ChartSpec("How many raters chose each category, subject by subject", kind="heatmap", matrix=counts.T.tolist(),
                        row_labels=labels, x_label="Subject", y_label="Category", cmap="Blues")]
    return AnalysisResult(title="Do the columns agree?", summary=summary, tables=tables, notes=notes, charts=charts)
