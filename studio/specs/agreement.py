"""Agreement and accuracy: how well a measurement separates two groups, and
whether two methods agree."""

from __future__ import annotations

from ...backends.bland_altman import BlandAltmanOptions
from ...backends.reliability import ReliabilityOptions
from ...backends.roc_curve_analysis import RocOptions
from ..spec import Chips, Choose, Custom, Heading, Note, Opt, Pick, Row, Sentence, Spec
from .common import columns, confidence
from .sketches import agreement_scene, sketch

ROC = Spec(
    "roc_curve_analysis", RocOptions,
    [
        Heading("Which values come from whom?"),
        Row("Subjects without the condition", Pick("control_index", columns)),
        Row("Subjects with the condition", Pick("patient_index", columns)),
        Note("Choose two different data sets.", tone="warn", when=lambda s, c: s["control_index"] == s["patient_index"]),
        Note("Higher values are taken to point towards the condition.", tone="faint"),
        Heading("Report"),
        Choose("report_as", [Opt("percentage", "Rates as percentages"), Opt("fraction", "Rates as fractions")],
               style="segmented"),
        Row("Standard error of the area by", Pick("ci_method", [("delong", "DeLong - makes no assumption (usual)"),
                                                                ("hanley_mcneil", "Hanley & McNeil - assumes a binormal shape")])),
    ],
    on_sheet=lambda p: p.suggest({"patient_index": 1 if len(p.ctx.names) > 1 else 0}),
    methods=lambda s, c: (f"A ROC curve was drawn for {c.name(s['patient_index'])} against {c.name(s['control_index'])}; "
                          f"the area under it is given with a {confidence()} confidence interval "
                          f"({'DeLong' if s['ci_method'] == 'delong' else 'Hanley-McNeil'})."),
)

_CALC = {"diff": "diff_a_minus_b", "ratio": "ratio_a_over_b", "pct": "pct_diff_a_minus_b"}


def _ba_finish(s, ctx) -> None:
    s["calculate"] = _CALC[s.pop("_measure", "diff")]


AGREE = Spec(
    "bland_altman", BlandAltmanOptions,
    [
        Heading("Compare two methods, pair by pair", "Say it as a sum. The order you write it in is the order it is worked out in."),
        Sentence(["For each subject, take", Pick("method_a_index", columns),
                  Pick("_measure", [("diff", "minus"), ("ratio", "divided by"), ("pct", "minus, as a % of the pair's mean,")]),
                  Pick("method_b_index", columns)]),
        Note("Choose two different data sets.", tone="warn", when=lambda s, c: s["method_a_index"] == s["method_b_index"]),
        Custom(lambda p: sketch(agreement_scene, 190, "With two methods measured on the same subjects, each pair is drawn "
                                                      "here: the gap between the methods against the pair's mean.")),
        Note("If the dots fan out to the right, the gap grows with the size of the value: divide, or take the gap as a "
             "percentage, instead of subtracting.", tone="faint"),
        Chips([("", [("create_new_graph", "Add the agreement plot to the results")])]),
    ],
    virtual={"_measure": "diff"},
    finish=_ba_finish,
    on_sheet=lambda p: p.suggest({"method_b_index": 1 if len(p.ctx.names) > 1 else 0}),
    methods=lambda s, c: (f"Agreement between {c.name(s['method_a_index'])} and {c.name(s['method_b_index'])} was "
                          "assessed by Bland-Altman analysis: the bias and 95% limits of agreement."),
)

# ── items and raters ────────────────────────────────────────────────────────


def _reliability_methods(s, ctx) -> str:
    if s["kind"] == "items":
        return f"Internal consistency was measured with Cronbach's alpha, with a {confidence()} confidence interval by Feldt's method."
    if s["kind"] == "scores":
        form = {"different": "ICC(1,1), one-way random effects",
                "sample": "ICC(2,1), two-way random effects, absolute agreement" if s["icc_agreement"] else "ICC(3,1), two-way, consistency",
                "these": "ICC(3,1), two-way mixed effects, consistency"}[s["icc_raters"]]
        return (f"Agreement between raters was measured with the intraclass correlation ({form}; Shrout and Fleiss 1979), "
                f"with {confidence()} confidence intervals.")
    if ctx.groups > 2:
        return "Agreement between the raters was measured with Fleiss' kappa."
    weights = {"none": "", "linear": ", with linear weights", "quadratic": ", with quadratic weights"}[s["weights"]]
    return f"Agreement between the two raters was measured with Cohen's kappa{weights}, with a {confidence()} confidence interval."


RELIABILITY = Spec(
    "reliability", ReliabilityOptions,
    [
        Note("Each row is one subject. Rows with a blank in any data set are left out."),
        Heading("What are the data sets?"),
        Choose("kind", [
            Opt("items", "Items of one scale", "Questions meant to measure the same thing - Cronbach's alpha.", phrase="Cronbach's alpha"),
            Opt("scores", "Raters giving scores", "Several raters or instruments measuring each subject - intraclass correlation.",
                phrase="intraclass correlation"),
            Opt("categories", "Raters giving categories", "Each rater puts each subject in a category (1, 2, 3 ...) - kappa.",
                phrase="kappa"),
        ], columns=3),
        Heading("Who are the raters?", when=lambda s, c: s["kind"] == "scores"),
        Choose("icc_raters", [
            Opt("sample", "A sample of raters", "The result should hold for other raters like them."),
            Opt("these", "The only raters", "These raters are the ones that matter."),
            Opt("different", "Different for each subject", "No rater scores every subject."),
        ], columns=3, when=lambda s, c: s["kind"] == "scores"),
        Choose("icc_agreement", [
            Opt(True, "They must give the same number", "A rater who scores everyone higher counts as disagreeing."),
            Opt(False, "They need only rank alike", "A constant difference between raters is not held against them."),
        ], when=lambda s, c: s["kind"] == "scores" and s["icc_raters"] == "sample"),
        Heading("Are the categories in order?", when=lambda s, c: s["kind"] == "categories" and c.groups == 2),
        Choose("weights", [
            Opt("none", "No order", "Only an exact match counts."),
            Opt("linear", "In order", "A near miss counts in part, by how far apart."),
            Opt("quadratic", "In order, far misses matter most", "Part credit falls off with the square of the distance."),
        ], columns=3, when=lambda s, c: s["kind"] == "categories" and c.groups == 2),
        Note("With three raters or more, Fleiss' kappa is used; it takes no account of order.",
             when=lambda s, c: s["kind"] == "categories" and c.groups > 2),
    ],
    methods=_reliability_methods,
)

SPECS = [ROC, AGREE, RELIABILITY]
