"""Build ChartSpecs for the graphs analyses commonly offer (QQ, residual,
survival, ROC, heat map). Plain numbers in, plain numbers out - backends
stay free of matplotlib; graphs/engine/charts.py does the drawing."""

from __future__ import annotations

import numpy as np

from .backend_types import ChartSeries, ChartSpec


def unavailable_chart(title: str, reason: str) -> ChartSpec:
    """Stands in, among the graphs, for a graph the user asked for but this
    data can't support - says why, and what the data would need."""
    return ChartSpec(title=title, kind="message", message=reason)


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def qq_chart(title: str, theoretical, actual, y_label: str = "Actual value") -> ChartSpec:
    """Normal quantile plot with the usual reference line through the quartiles."""
    t = np.asarray(theoretical, dtype=float)
    a = np.asarray(actual, dtype=float)
    refs = []
    if t.size >= 2:
        t25, t75 = np.percentile(t, [25, 75])
        a25, a75 = np.percentile(a, [25, 75])
        if t75 > t25:
            refs.append({"kind": "line", "points": [[float(t25), float(a25)], [float(t75), float(a75)]]})
    return ChartSpec(title=title, series=[ChartSeries("", _floats(t), _floats(a))],
                     x_label="Theoretical quantile (normal)", y_label=y_label, reference_lines=refs)


def residual_chart(title: str, predicted, residuals, groups=None, absolute: bool = False) -> ChartSpec:
    """Residual (or |residual|) against predicted Y, coloured by group when given."""
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if absolute:
        r = np.abs(r)
    if groups is None:
        series = [ChartSeries("", _floats(p), _floats(r))]
    else:
        names = [str(g) for g in groups]
        series = []
        for name in dict.fromkeys(names):
            mask = np.array([g == name for g in names])
            series.append(ChartSeries(name, _floats(p[mask]), _floats(r[mask])))
    return ChartSpec(title=title, series=series, x_label="Predicted Y",
                     y_label="|Residual|" if absolute else "Residual",
                     reference_lines=[] if absolute else [{"kind": "h", "value": 0.0}])


def km_chart(title: str, groups: dict, ascending: bool = False) -> ChartSpec:
    """Kaplan-Meier steps per group from {label: (time, event 1/0)};
    censored subjects are open symbols."""
    from graphs.engine.stats import km_curve_arrays

    def shown(s):
        return (1.0 - np.asarray(s)) * 100 if ascending else np.asarray(s) * 100

    series = []
    for label, (time, event) in groups.items():
        curve = km_curve_arrays(time, event)
        series.append(ChartSeries(str(label), _floats(curve.times), _floats(shown(curve.survival)), kind="step",
                                  hollow_x=_floats(curve.censor_times), hollow_y=_floats(shown(curve.censor_survival))))
    return ChartSpec(title=title, series=series, x_label="Time",
                     y_label="Percent with the event" if ascending else "Percent surviving", y_limits=(-2.0, 105.0))


KM_VIEWS = (("survival", "Still event-free"), ("event", "Had the event"), ("hazard", "Cumulative hazard"),
            ("log", "Log of the share event-free"))


def km_charts(groups: dict, *, first: str = "survival", ci_style: str = "asymmetric", band: bool = True,
              median_lines: bool = True, at_risk: dict | None = None) -> list[ChartSpec]:
    """The survival graph in four views, as one graph with a choice (they
    share ChartSpec.group): the share still event-free, the share who had
    the event, the cumulative hazard (Nelson-Aalen) and the log of survival.

    ``groups`` is {label: (time, event 1/0)}. The first two views can carry a
    confidence band and lines at the median; ``at_risk`` ({"times": [...],
    "rows": [{"label", "counts"}]}) is printed under their time axis."""
    from graphs.engine.stats import km_curve_arrays, km_interval
    from graphs.engine.style import PALETTES

    palette = PALETTES["classic"]
    curves = {label: km_curve_arrays(time, event) for label, (time, event) in groups.items()}
    colours = {label: palette[i % len(palette)] for i, label in enumerate(curves)}
    risk = None
    if at_risk and at_risk.get("times"):
        risk = {"times": at_risk["times"], "title": "Number at risk",
                "rows": [dict(row, color=colours.get(row["label"], "#333333")) for row in at_risk["rows"]]}

    def percent_view(key: str) -> ChartSpec:
        rising = key == "event"

        def shown(values):
            v = np.asarray(values, dtype=float) * 100
            return 100 - v if rising else v

        steps, bands, lines = [], [], []
        for label, curve in curves.items():
            colour = colours[label]
            steps.append(ChartSeries(str(label), _floats(curve.times), _floats(shown(curve.survival)), kind="step", color=colour,
                                     hollow_x=_floats(curve.censor_times), hollow_y=_floats(shown(curve.censor_survival))))
            if band and curve.times.size > 1:
                lo, hi = km_interval(curve.survival, curve.se, ci_style)
                lower, upper = (shown(hi), shown(lo)) if rising else (shown(lo), shown(hi))
                bands.append(ChartSeries("", _floats(curve.times), _floats(lower), kind="step_band", color=colour,
                                         y_upper=_floats(upper)))
            if median_lines and curve.median is not None:
                lines.append({"kind": "path", "points": [[0.0, 50.0], [curve.median, 50.0], [curve.median, -2.0]],
                              "color": colour, "style": ":", "width": 1.0})
        title = "Share who had the event" if rising else "Survival curves"
        return ChartSpec(title=title, series=steps + bands, x_label="Time", y_label="Percent with the event" if rising else "Percent surviving",
                         y_limits=(-2.0, 105.0), reference_lines=lines, group="km_view", option=dict(KM_VIEWS)[key],
                         extra={"at_risk": risk} if risk else {})

    def hazard_view() -> ChartSpec:
        series = []
        for label, (time, event) in groups.items():
            time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=int)
            xs, ys, total = [0.0], [0.0], 0.0
            for t in np.unique(time[event == 1]):
                total += float(np.sum((time == t) & (event == 1))) / float(np.sum(time >= t))
                xs.append(float(t))
                ys.append(total)
            if time.size and time.max() > xs[-1]:
                xs.append(float(time.max()))
                ys.append(total)
            series.append(ChartSeries(str(label), xs, ys, kind="step", color=colours[label]))
        return ChartSpec(title="Cumulative hazard", series=series, x_label="Time", y_label="Cumulative hazard",
                         group="km_view", option=dict(KM_VIEWS)["hazard"])

    def log_view() -> ChartSpec:
        series = []
        for label, curve in curves.items():
            keep = curve.survival > 0            # the log of 0 has no place on the graph
            series.append(ChartSeries(str(label), _floats(curve.times[keep]), _floats(np.log(curve.survival[keep])), kind="step",
                                      color=colours[label]))
        return ChartSpec(title="Log of the share still event-free", series=series, x_label="Time", y_label="Natural log of the share event-free",
                         group="km_view", option=dict(KM_VIEWS)["log"])

    order = [first] + [key for key, _name in KM_VIEWS if key != first]
    build = {"survival": lambda: percent_view("survival"), "event": lambda: percent_view("event"),
             "hazard": hazard_view, "log": log_view}
    return [build[key]() for key in order]


def roc_chart(title: str, fpr, tpr, auc: float, percent: bool = False) -> ChartSpec:
    k = 100.0 if percent else 1.0
    return ChartSpec(
        title=title,
        series=[ChartSeries(f"AUC = {auc:.3g}", _floats(np.asarray(fpr) * k), _floats(np.asarray(tpr) * k), kind="line")],
        x_label="False positive rate, 1 \u2212 specificity (%)" if percent else "False positive rate, 1 \u2212 specificity",
        y_label="True positive rate, sensitivity (%)" if percent else "True positive rate, sensitivity",
        reference_lines=[{"kind": "line", "points": [[0.0, 0.0], [k, k]]}],
        x_limits=(0.0, k * 1.02),
        y_limits=(0.0, k * 1.02),
    )


def heatmap_chart(title: str, matrix, row_labels, col_labels, cmap: str = "viridis", color_limits=None) -> ChartSpec:
    m = np.asarray(matrix, dtype=float)
    return ChartSpec(title=title, kind="heatmap", matrix=[_floats(row) for row in m],
                     row_labels=[str(v) for v in row_labels], col_labels=[str(v) for v in col_labels], cmap=cmap,
                     color_limits=color_limits)


def xy_chart(title: str, x, series: dict, kind: str = "line_markers", x_label: str = "", y_label: str = "",
             reference_lines=None) -> ChartSpec:
    """Simple chart of one or more Y series against shared X. Use the key
    "" for a single unlabelled series (no legend)."""
    return ChartSpec(title=title, series=[ChartSeries(str(k), _floats(x), _floats(v), kind=kind) for k, v in series.items()],
                     x_label=x_label, y_label=y_label, reference_lines=list(reference_lines or []))


def dendrogram_chart(title: str, linkage_matrix, leaf_labels, y_label: str = "Distance") -> ChartSpec:
    """Tree from a scipy linkage matrix, as plain line segments."""
    from scipy.cluster.hierarchy import dendrogram

    tree = dendrogram(np.asarray(linkage_matrix, dtype=float), no_plot=True, labels=[str(v) for v in leaf_labels])
    # scipy puts leaf i at x = 10 i + 5; rescale so leaves sit on category positions 0..n-1
    series = [ChartSeries("", [(x - 5) / 10 for x in xs], _floats(ys), kind="line", color="#1f5aa6")
              for xs, ys in zip(tree["icoord"], tree["dcoord"])]
    leaves = [str(v) for v in tree["ivl"]]
    return ChartSpec(title=title, series=series, y_label=y_label, x_categories=leaves if len(leaves) <= 60 else None)


def bland_altman_chart(title: str, means, differences, bias: float, lower: float, upper: float, y_label: str) -> ChartSpec:
    m = _floats(means)
    return ChartSpec(
        title=title,
        series=[ChartSeries("", m, _floats(differences))],
        x_label="Average of the two methods",
        y_label=y_label,
        reference_lines=[{"kind": "h", "value": float(bias)}, {"kind": "h", "value": float(lower)}, {"kind": "h", "value": float(upper)}],
        annotations=[
            {"x": max(m), "y": float(bias), "text": f"Bias {bias:.3g}"},
            {"x": max(m), "y": float(upper), "text": f"+95% LoA {upper:.3g}"},
            {"x": max(m), "y": float(lower), "text": f"-95% LoA {lower:.3g}"},
        ],
    )


def survival_curves_chart(title: str, times, curves: dict) -> ChartSpec:
    """Model-predicted survival (fractions) per scenario, drawn as steps."""
    t = [0.0] + _floats(times)
    series = [ChartSeries(str(label), t, [100.0] + _floats(np.asarray(s) * 100), kind="step") for label, s in curves.items()]
    return ChartSpec(title=title, series=series, x_label="Time", y_label="Percent surviving", y_limits=(-2.0, 105.0))
