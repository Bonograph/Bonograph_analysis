"""How well a measurement tells two groups apart (ROC curve): pick which group is the control (negative) class and which is
the patient (positive) class, an AUC confidence interval method, and how
to report the curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backend_types import AnalysisError, AnalysisResult, ResultTable
from ..data_source import SheetData
from ..registry import register



def _p(value, opt) -> str:
    """One P value, written the way the user asked for (analysis/p_values.py)."""
    from ..p_values import format_p

    return format_p(value, getattr(opt, "p_style", "number"),
                    digits=getattr(opt, "significant_digits", 4),
                    alpha=getattr(opt, "alpha", 0.05))

@dataclass
class RocOptions:
    control_index: int = 0
    patient_index: int = 1
    confidence_level: float = 0.95
    ci_method: str = "delong"  # "delong" | "hanley_mcneil"
    report_as: str = "percentage"  # "fraction" | "percentage"
    significant_digits: int = 4
    p_style: str = "number"  # how P values are written (analysis/p_values.py)


def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _delong_se(patient: np.ndarray, control: np.ndarray) -> float:
    """DeLong, DeLong & Clarke-Pearson (1988): the nonparametric standard error
    of the area, from each subject's placement among the other group (ties
    count half). Computed from mid-ranks (Sun & Xu 2014), so it is exact and
    fast however many subjects there are."""
    m, n = patient.size, control.size
    both = stats.rankdata(np.concatenate([patient, control]))
    # V10: the share of controls each patient outscores; V01: the share of patients outscoring each control
    v10 = (both[:m] - stats.rankdata(patient)) / n
    v01 = 1.0 - (both[m:] - stats.rankdata(control)) / m
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    return float(np.sqrt(max(var, 0.0)))


@register("roc_curve_analysis")
def run(data: SheetData, options: RocOptions | None = None) -> AnalysisResult:
    from ..ml_utils import roc_auc_score, roc_curve

    opt = options or RocOptions()
    groups = data.all_group_arrays()
    labels = list(groups)
    if len(labels) < 2:
        raise AnalysisError(
            f"ROC analysis needs two groups (control and patient values); this sheet has {len(labels)}."
        )

    ci = max(0, min(opt.control_index, len(labels) - 1))
    pi = max(0, min(opt.patient_index, len(labels) - 1))
    if ci == pi:
        raise AnalysisError("Control values and patient values must be two different data sets.")

    control_label, patient_label = labels[ci], labels[pi]
    control, patient = groups[control_label], groups[patient_label]
    if control.size < 2 or patient.size < 2:
        raise AnalysisError("Each of control and patient values needs at least 2 values.")

    scores = np.concatenate([patient, control])
    truth = np.concatenate([np.ones_like(patient), np.zeros_like(control)])
    fpr, tpr, thresholds = roc_curve(truth, scores)
    auc = float(roc_auc_score(truth, scores))

    se = _hanley_mcneil_se(auc, patient.size, control.size) if opt.ci_method == "hanley_mcneil" else _delong_se(patient, control)
    z = stats.norm.ppf(0.5 + opt.confidence_level / 2)
    lo, hi = max(0.0, auc - z * se), min(1.0, auc + z * se)
    z_stat = (auc - 0.5) / se if se else float("inf")
    p_value = float(2 * stats.norm.sf(abs(z_stat)))

    sig = opt.significant_digits
    scale = 100 if opt.report_as == "percentage" else 1
    unit = "%" if opt.report_as == "percentage" else ""
    pct = int(round(opt.confidence_level * 100))
    method_name = "Hanley-McNeil" if opt.ci_method == "hanley_mcneil" else "DeLong"

    auc_table = ResultTable(
        title=f"'{patient_label}' (patient) vs '{control_label}' (control): area under ROC curve",
        headers=["AUC", f"{pct}% CI ({method_name})", "P value (vs AUC = 0.5)"],
        rows=[[f"{auc:.{sig}g}", f"{lo:.{sig}g} to {hi:.{sig}g}", _p(p_value, opt)]],
    )

    step = max(1, len(thresholds) // 20)
    curve_rows = [
        [f"{thresholds[i]:.{sig}g}", f"{fpr[i]*scale:.{sig}g}{unit}", f"{tpr[i]*scale:.{sig}g}{unit}"]
        for i in range(0, len(thresholds), step)
    ]
    curve_table = ResultTable(
        title="ROC curve (sampled)",
        headers=["Threshold", f"False positive rate{(' (' + unit + ')') if unit else ''}", f"True positive rate{(' (' + unit + ')') if unit else ''}"],
        rows=curve_rows,
    )

    summary = f"'{patient_label}' (patient, n={patient.size}) vs '{control_label}' (control, n={control.size}): AUC = {auc:.{sig}g} ({pct}% CI {lo:.{sig}g} to {hi:.{sig}g})."
    notes = [f"'{patient_label}' was treated as the positive (patient) class and '{control_label}' as the negative (control) class; higher values are assumed to indicate the patient class."]

    from ..chart_helpers import roc_chart

    chart = roc_chart(f"ROC curve: '{patient_label}' vs '{control_label}'", fpr, tpr, auc, percent=scale == 100)
    return AnalysisResult(title="How well a test separates", summary=summary, tables=[auc_table, curve_table], notes=notes, charts=[chart])
