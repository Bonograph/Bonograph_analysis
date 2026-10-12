"""Repeated measures when some values are missing: a mixed-effects model.

Each row is a subject, each data set a condition or time point. A
repeated-measures ANOVA can use only the subjects with every value; this
model uses every value there is, by treating each subject as having their
own level (a random intercept) around which their values scatter:

    value = mean of the condition + subject's own shift + residual

Fitted by restricted maximum likelihood (REML). With no values missing it
gives the same F as a repeated-measures ANOVA with sphericity assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize, stats

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, ResultTable
from ..data_source import SheetData
from ..p_values import format_p
from ..registry import register
from . import _common


@dataclass
class MixedModelOptions:
    compare: str = "all_pairs"     # "none" | "all_pairs" | "vs_control" | "consecutive"
    control_index: int = 0
    correction: str = "holm"       # "holm" | "fdr" | "none"
    show_subjects_graph: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    significant_digits: int = 4
    p_style: str = "number"


def fit_random_intercept(matrix: np.ndarray) -> dict:
    """REML fit of value = condition mean + subject + residual to a subjects x
    conditions matrix with NaN for missing values.

    For a ratio g of subject variance to residual variance, a subject with m
    values has V = s2 (I + g J), whose inverse is (1/s2)(I - g/(1 + m g) J) and
    whose determinant is s2^m (1 + m g). So the condition means, the residual
    variance and the restricted likelihood all follow from sums over subjects,
    and only g has to be searched for.
    """
    present = np.isfinite(matrix)
    rows = present.any(axis=1)
    present, filled = present[rows].astype(float), np.where(present[rows], matrix[rows], 0.0)
    k = matrix.shape[1]
    n_obs = int(present.sum())
    p = k
    counts = present.sum(axis=1)                 # values each subject has
    row_sums = filled.sum(axis=1)
    per_condition = present.sum(axis=0)
    sum_y, sum_yy = filled.sum(axis=0), float((filled**2).sum())
    n_subjects = int(rows.sum())

    def pieces(g: float):
        # the sums over subjects, taken for all subjects at once (a long table has many thousands)
        shrink = g / (1 + counts * g)
        a = np.diag(per_condition) - (present * shrink[:, None]).T @ present          # X' W X
        b = sum_y - (present * (shrink * row_sums)[:, None]).sum(axis=0)              # X' W y
        yy = sum_yy - float((shrink * row_sums**2).sum())                             # y' W y
        logdet = float(np.log1p(counts * g).sum())
        return a, b, yy, logdet

    def negative_reml(log_g: float) -> float:
        g = np.exp(log_g)
        a, b, yy, logdet = pieces(g)
        sign, logdet_a = np.linalg.slogdet(a)
        if sign <= 0:
            return np.inf
        beta = np.linalg.solve(a, b)
        s2 = (yy - b @ beta) / (n_obs - p)
        if s2 <= 0:
            return np.inf
        return 0.5 * ((n_obs - p) * np.log(s2) + logdet + logdet_a)

    best = optimize.minimize_scalar(negative_reml, bounds=(-14.0, 14.0), method="bounded", options={"xatol": 1e-10})
    g = float(np.exp(best.x))
    if negative_reml(-14.0) <= best.fun + 1e-9:        # no sign of subject-to-subject variation
        g = 0.0
    a, b, yy, _logdet = pieces(g)
    beta = np.linalg.solve(a, b)
    s2 = float((yy - b @ beta) / (n_obs - p))
    return {"means": beta, "cov": s2 * np.linalg.inv(a), "residual_var": s2, "subject_var": g * s2, "n_obs": n_obs,
            "n_subjects": n_subjects}


@register("mixed_model")
def run(data: SheetData, options: MixedModelOptions | None = None) -> AnalysisResult:
    opt = options or MixedModelOptions()
    names, matrix = data.first_replicate_matrix()
    if len(names) < 2:
        raise AnalysisError("This needs two conditions or more as data sets, with one subject on each row.")
    matrix = matrix[np.isfinite(matrix).any(axis=1)]
    used = np.isfinite(matrix).any(axis=0)
    names, matrix = [n for n, u in zip(names, used) if u], matrix[:, used]
    k = len(names)
    n_subjects = matrix.shape[0]
    present = np.isfinite(matrix)
    n_obs = int(present.sum())
    df_error = n_obs - n_subjects - (k - 1)
    if k < 2 or n_subjects < 3:
        raise AnalysisError("At least 3 subjects (rows) and 2 conditions (data sets) are needed.")
    if df_error < 1 or int((present.sum(axis=1) >= 2).sum()) < 2:
        raise AnalysisError("Too few subjects were measured more than once: at least two rows need values in two or more data sets.")
    sig = opt.significant_digits

    def num(value) -> str:
        return f"{value:.{sig}g}"

    def p_text(p) -> str:
        return format_p(p, opt.p_style, digits=sig, alpha=opt.alpha)

    fit = fit_random_intercept(matrix)
    means, cov = fit["means"], fit["cov"]
    contrast = np.column_stack([np.ones(k - 1), -np.eye(k - 1)])            # first condition against each other
    gap = contrast @ means
    f_value = float(gap @ np.linalg.solve(contrast @ cov @ contrast.T, gap) / (k - 1))
    p_value = float(stats.f.sf(f_value, k - 1, df_error))
    t_crit = stats.t.ppf(0.5 + opt.confidence_level / 2, df_error)
    pct = f"{round(opt.confidence_level * 100)}%"
    se = np.sqrt(np.diag(cov))

    complete = int(present.all(axis=1).sum())
    missing = n_subjects * k - n_obs
    tables = [ResultTable("Do the conditions differ?", ["Effect", "F", "df", "P value"],
                          [["Condition", f"{f_value:.4f}", f"{k - 1}, {df_error}", p_text(p_value)]])]
    with np.errstate(all="ignore"):
        raw_means = np.nanmean(matrix, axis=0)
    tables.append(ResultTable("Mean of each condition",
                              ["Condition", "Values", "Missing", "Mean of the values there", "Mean from the model", "SE", f"{pct} CI"],
                              [[names[j], str(int(present[:, j].sum())), str(int(n_subjects - present[:, j].sum())), num(raw_means[j]),
                                num(means[j]), num(se[j]), f"{num(means[j] - t_crit * se[j])} to {num(means[j] + t_crit * se[j])}"]
                               for j in range(k)]))

    if opt.compare != "none":
        if opt.compare == "vs_control":
            c = max(0, min(opt.control_index, k - 1))
            pairs = [(j, c) for j in range(k) if j != c]
        elif opt.compare == "consecutive":
            pairs = [(j + 1, j) for j in range(k - 1)]
        else:
            pairs = list(combinations(range(k), 2))
        rows, raw = [], []
        for i, j in pairs:
            diff = means[i] - means[j]
            se_d = float(np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j]))
            t = diff / se_d
            p = float(2 * stats.t.sf(abs(t), df_error))
            raw.append(p)
            rows.append([f"{names[i]} − {names[j]}", num(diff), num(se_d), f"{num(diff - t_crit * se_d)} to {num(diff + t_crit * se_d)}",
                         f"{t:.4f}", p_text(p)])
        headers = ["Comparison", "Difference", "SE", f"{pct} CI", "t", "P value"]
        if len(pairs) > 1 and opt.correction != "none":
            for row, a in zip(rows, _common.adjust(raw, opt.correction)):
                row += [p_text(float(a)), "Yes" if a < opt.alpha else "No"]
            headers += [f"P value ({'Benjamini-Hochberg' if opt.correction == 'fdr' else 'Holm'})", f"Significant (α = {opt.alpha:g})"]
        tables.append(ResultTable("Pairs of conditions", headers, rows))

    subject_var, residual_var = fit["subject_var"], fit["residual_var"]
    icc = subject_var / (subject_var + residual_var) if subject_var + residual_var > 0 else 0.0
    tables.append(ResultTable("Where the variation lies", ["Source", "Variance", "SD", "Share"],
                              [["Between subjects", num(subject_var), num(np.sqrt(subject_var)), f"{icc:.1%}"],
                               ["Within a subject (residual)", num(residual_var), num(np.sqrt(residual_var)), f"{1 - icc:.1%}"]]))
    tables.append(ResultTable("The data that went in", ["", ""],
                              [["Subjects (rows with a value)", str(n_subjects)], ["Conditions", str(k)], ["Values", str(n_obs)],
                               ["Missing values", f"{missing} ({missing / (n_subjects * k):.1%})"],
                               ["Subjects with every value", f"{complete} - all a repeated-measures ANOVA could have used"]]))

    summary = (f"The conditions {'differ' if p_value < opt.alpha else 'do not differ clearly'}: F({k - 1}, {df_error}) = {f_value:.3f}, "
               f"P = {p_value:.4g}. All {n_subjects} subjects were used ({n_obs} values, {missing} missing); "
               + (f"a repeated-measures ANOVA could have used only {complete}." if missing else "none was missing, so this equals "
                  "a repeated-measures ANOVA with sphericity assumed."))

    notes = ["Each subject is given their own level, and the conditions are compared within subjects. A subject with one value "
             "missing still adds what their other values say.",
             "This is right when values are missing for reasons unconnected with what they would have been (a lost sample, a "
             "missed visit). If subjects drop out because they are doing badly, no analysis of the values left can be trusted.",
             "The model assumes that any two conditions are equally correlated within a subject (compound symmetry) - the same "
             "assumption as sphericity. With time points far apart, where neighbours are closer than distant ones, treat a "
             "borderline P value with care.",
             f"{icc:.0%} of the variation is between subjects." + (" None was found, so the subjects add nothing here and the test "
                                                                 "is that of an ordinary one-way ANOVA." if subject_var == 0 else ""),
             f"Restricted maximum likelihood; F and t tests use {df_error} residual degrees of freedom (values - subjects - "
             "conditions + 1). Programs that use Satterthwaite's or Kenward-Roger's degrees of freedom give slightly different "
             "P values when values are missing.",
             "References: Laird and Ware 1982; Pinheiro and Bates, Mixed-Effects Models in S and S-PLUS (2000)."]

    xs = [float(j) for j in range(k)]
    charts = [ChartSpec(f"Mean of each condition from the model, with {pct} confidence intervals", x_label="Condition", y_label="Mean",
                        series=[ChartSeries("", xs, means.tolist(), kind="line_markers", color="#1f5aa6", y_err=(t_crit * se).tolist())],
                        x_categories=names, x_limits=(-0.5, k - 0.5))]
    if opt.show_subjects_graph:
        step = max(1, int(np.ceil(n_subjects / 60)))
        lines = []
        for row in matrix[::step]:
            cols = np.flatnonzero(np.isfinite(row))
            if cols.size >= 2:
                lines.append({"kind": "path", "points": [[float(c), float(row[c])] for c in cols], "color": "#9aa9c4", "style": "-", "width": 0.8})
        alone = [(float(c), float(matrix[i, c])) for i in range(0, n_subjects, step) for c in np.flatnonzero(present[i]) if present[i].sum() == 1]
        series = [ChartSeries("Mean from the model", xs, means.tolist(), kind="line_markers", color="#d1342f")]
        if alone:
            series.append(ChartSeries("Subjects with one value", [a[0] for a in alone], [a[1] for a in alone], kind="scatter", color="#9aa9c4"))
        charts.append(ChartSpec("Each subject across the conditions" + (f" (every {step}th subject)" if step > 1 else ""),
                                x_label="Condition", y_label="Value", series=series, reference_lines=lines, x_categories=names,
                                x_limits=(-0.5, k - 0.5)))
    return AnalysisResult(title="Repeated measures with gaps", summary=summary, tables=tables, notes=notes, charts=charts)
