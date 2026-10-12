"""What analyses Bonograph offers, arranged by the question being asked.

The picker used to be organised the way a statistics package's menu usually
is - by the shape of the table the data sits in. That tells you where an
analysis lives only if you already know its name. Here the top level is the
question ("Is the difference real?", "What model describes it?"), each entry
says in one line what it answers, and the method names sit underneath as
supporting detail for people who already know what they want.

``key`` is the name the backend registered itself under (analysis/registry.py)
and never changes; ``title`` and ``question`` are what the user reads, and can
be reworded freely without touching a single backend.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Needs:
    """What an analysis needs from a sheet before it can run.

    The picker checks these against the sheet you are on, so an analysis
    that cannot work with the data in front of you says so before you run
    it rather than after.
    """
    min_groups: int = 1       # data sets (columns of values)
    min_rows: int = 3         # filled rows
    numeric_x: bool = False   # a real X column, not just row order
    replicates: bool = False  # side-by-side repeats within each data set
    counts: bool = False      # whole, non-negative numbers
    binary: bool = False      # a column holding only 0 and 1
    no_data: bool = False     # generates its own data; the sheet doesn't matter
    row_factor: bool = False  # rows are levels of a factor (row titles, or repeats side by side)
    title_factors: int = 0    # data-set titles that split into this many factors ("Male-Low-Fed")


@dataclass(frozen=True)
class Analysis:
    key: str        # the registry name - what actually runs
    title: str      # what the picker shows
    question: str   # the question it answers, in one plain line
    methods: str    # the named tests behind it
    needs: str      # the shape of data it wants, in words
    tags: str = ""  # extra words to catch in a search
    req: "Needs | None" = None  # checked against the sheet; None = the group's own


@dataclass(frozen=True)
class Group:
    id: str
    title: str
    question: str
    glyph: str      # which small drawing the rail shows (see the dialog)
    entries: tuple[Analysis, ...]
    req: Needs = None      # what its analyses need, unless one says otherwise
    produces: str = ""     # what you get back, in one line
    tools: tuple = ()      # what the Analyze window lists (see Tool, below)


GROUPS: tuple[Group, ...] = (
    Group("compare", "Compare groups", "Is the difference real?", "compare", (
        Analysis("two_group_comparison", "Two groups",
                 "Do two sets of measurements really differ?",
                 "Welch's t test, Student's t test, paired t test, Mann-Whitney, Wilcoxon matched pairs",
                 "two columns of numbers", "ttest student welch nonparametric"),
        Analysis("one_way_anova", "Three groups or more",
                 "Does any group stand out from the rest?",
                 "Welch's ANOVA, one-way ANOVA, Kruskal-Wallis, Friedman · Tukey, Dunnett, Games-Howell, Dunn",
                 "three or more columns", "anova posthoc tukey dunnett", req=Needs(min_groups=3)),
        Analysis("one_sample_test", "One group against a value",
                 "Is this group's average different from a number you expect?",
                 "One-sample t test, Wilcoxon signed rank",
                 "one column of numbers", "hypothesised mean", req=Needs(min_groups=1)),
        Analysis("two_way_anova", "Two factors at once",
                 "Do two factors each matter, and do they interact?",
                 "Two-way ANOVA (Type III), repeated measures with Geisser-Greenhouse · Tukey, Holm",
                 "a grouped table: rows one factor, columns the other", "interaction repeated measures",
                 req=Needs(min_groups=2, min_rows=2, row_factor=True)),
        Analysis("three_way_anova", "Three factors at once",
                 "The same question with a third factor in play.",
                 "Three-way ANOVA",
                 "data-set titles naming three factors, e.g. Male-Low-Fed", "interaction",
                 req=Needs(min_groups=8, min_rows=2, title_factors=3)),
        Analysis("multiple_comparisons", "Many pairs at once",
                 "Test the pairs directly, skipping the overall test - when the pairs are the question.",
                 "Welch t tests or Mann-Whitney tests, Holm or FDR adjusted",
                 "several columns", "bonferroni holm fdr"),
        Analysis("nested_t_test", "Two groups, subsamples inside",
                 "Two groups where each subject was measured several times.",
                 "Nested t test",
                 "a concentric table with two groups", "hierarchical pseudoreplication", req=Needs(min_groups=2, replicates=True)),
        Analysis("nested_one_way_anova", "Several groups, subsamples inside",
                 "The same, with three or more groups.",
                 "Nested one-way ANOVA · Holm, Benjamini-Hochberg",
                 "a concentric table", "hierarchical pseudoreplication", req=Needs(min_groups=2, replicates=True)),
        Analysis("effect_sizes", "How large is the difference?",
                 "The size of a difference in standard units, to report beside the P value.",
                 "Cohen's d, Hedges' g, Glass's delta, eta and omega squared, rank-biserial correlation · noncentral intervals",
                 "two or more columns of numbers", "effect size cohen hedges eta omega standardised magnitude"),
        Analysis("ancova", "Groups, allowing for a covariate",
                 "Compare groups at the same baseline, age or dose.",
                 "Analysis of covariance (Type III), equal-slopes test, adjusted means",
                 "X holds the covariate; each data set one group's outcome", "ancova covariate baseline adjusted means",
                 req=Needs(min_groups=2, min_rows=4, numeric_x=True)),
        Analysis("mixed_model", "Repeated measures with gaps",
                 "The same subjects under several conditions, when some values are missing.",
                 "Mixed-effects model (random intercept, REML)",
                 "one subject per row, one condition per column; blanks allowed", "mixed model missing repeated reml longitudinal",
                 req=Needs(min_groups=2, min_rows=3)),
        Analysis("power_sample_size", "How many subjects?",
                 "Plan a study: the number needed, the power you have, or the effect you can find.",
                 "Power and sample size for t tests, ANOVA, two proportions and a correlation",
                 "nothing - you describe the study", "power sample size planning n subjects", req=Needs(no_data=True)),
    ),
       req=Needs(min_groups=2),
       produces="A P value, the size of the difference, and a confidence interval."),
    Group("describe", "Describe the data", "What does it look like?", "describe", (
        Analysis("descriptive_statistics", "Summarise each column",
                 "Means, spread, confidence intervals, column by column.",
                 "Mean, median, SD, SEM, 95% CI, quartiles, CV, geometric mean",
                 "any column of numbers", "average summary statistics"),
        Analysis("row_wise_statistics", "Summarise each row",
                 "The same figures taken across a row instead of down a column.",
                 "Mean, SD, SEM and n per row",
                 "replicate subcolumns", "average across replicates", req=Needs(min_groups=1, replicates=True)),
        Analysis("frequency_distribution", "How values are spread",
                 "Count how often values fall in each range.",
                 "Histogram counts, cumulative and relative frequencies",
                 "one or more columns of numbers", "histogram bins", req=Needs(min_groups=1)),
        Analysis("normality_testing", "Is it normally distributed?",
                 "Check the assumption most tests quietly rely on.",
                 "Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, Kolmogorov-Smirnov (Lilliefors)",
                 "a column with a handful of values or more", "gaussian assumption qq normality normal"),
        Analysis("variance_tests", "Is the spread the same?",
                 "Check whether the groups scatter equally - what Student's t test and ANOVA assume.",
                 "Brown-Forsythe, Levene, Bartlett, Fligner-Killeen, F test of two variances",
                 "two or more columns of numbers", "variance homogeneity levene bartlett homoscedasticity equal sd",
                 req=Needs(min_groups=2)),
        Analysis("outlier_detection", "Any value out of line?",
                 "Find points far enough out to be worth a second look.",
                 "Generalized ESD (Rosner), Grubbs' test, box-plot rule (Tukey), robust z-scores",
                 "a column of numbers", "extreme suspect esd rosner grubbs tukey fences boxplot iqr mad robust outlier"),
        Analysis("batch_pvalue_analysis", "Many P values at once",
                 "Sort many results out at once without inflating false positives.",
                 "Benjamini-Hochberg, Benjamini-Krieger-Yekutieli, Benjamini-Yekutieli, Holm-Sidak, Bonferroni",
                 "a column of p values", "fdr multiple testing benjamini", req=Needs(min_groups=1, min_rows=2)),
    ),
       req=Needs(min_groups=1),
       produces="A table of figures for each data set, and the graphs that go with them."),
    Group("fit", "Fit a curve", "What model describes it?", "fit", (
        Analysis("nonlinear_regression", "Fit a curve",
                 "Dose-response, Michaelis-Menten, binding, decay and more.",
                 "Nonlinear regression: Michaelis-Menten, dose-response (4PL, 5PL, EC50), exponential, Gaussian",
                 "X against one or more Y columns", "ec50 ic50 km vmax kd hill dose response michaelis menten"),
        Analysis("global_fit", "Fit data sets together",
                 "One model for several data sets, sharing the parameters they have in common - and a test of whether they differ.",
                 "Global nonlinear regression with shared parameters, extra sum-of-squares F test, AICc",
                 "X against two or more Y columns", "global shared parameters compare curves ec50 differ simultaneous",
                 req=Needs(min_groups=2, min_rows=4, numeric_x=True)),
        Analysis("compare_models", "Which model fits better?",
                 "Two models on the same data: is the fuller one worth its extra parameters?",
                 "Extra sum-of-squares F test, AICc and Akaike weights",
                 "X against one or more Y columns", "compare models aic akaike nested f test one site two site",
                 req=Needs(min_groups=1, min_rows=4, numeric_x=True)),
        Analysis("peak_fitting", "Peaks in a trace",
                 "Find the peaks of a spectrum or chromatogram and fit them all at once.",
                 "Peak detection by prominence, multi-peak fit: Gaussian, Lorentzian, Voigt",
                 "X against a trace of Y values", "peaks spectrum chromatogram gaussian lorentzian voigt deconvolution fwhm area",
                 req=Needs(min_groups=1, min_rows=8, numeric_x=True)),
        Analysis("baseline_spectrum", "Flatten the baseline",
                 "Take a drifting baseline out from under the peaks of a trace.",
                 "Asymmetric least squares, iterated polynomial, line between the ends",
                 "X against a trace of Y values", "baseline drift background spectrum als detrend subtract",
                 req=Needs(min_groups=1, min_rows=5, numeric_x=True)),
        Analysis("fourier_filter", "Frequencies in a trace",
                 "See which rhythms a trace holds, or filter some of them out.",
                 "Fourier transform (FFT), low-pass, high-pass and band filters",
                 "X in equal steps against a trace of Y values", "fft fourier spectrum frequency filter lowpass highpass bandpass noise period",
                 req=Needs(min_groups=1, min_rows=16, numeric_x=True)),
        Analysis("linear_regression", "Straight-line fit",
                 "Slope, intercept, and how sure you can be of them.",
                 "Linear regression, 95% confidence bands, runs test",
                 "X against Y", "slope intercept r squared"),
        Analysis("logistic_regression", "Fit a yes/no outcome",
                 "How an outcome's odds change along one predictor.",
                 "Binary logistic regression",
                 "X against a 0/1 outcome", "odds ratio binary probability", req=Needs(min_groups=1, numeric_x=True, binary=True, min_rows=4)),
        Analysis("deming_regression", "Line when both axes err",
                 "A fit that admits X was measured with error too.",
                 "Deming regression",
                 "X against Y, both measured", "method comparison errors in variables"),
        Analysis("spline_lowess_smoothing", "Smooth without a model",
                 "Draw the trend when no equation is worth defending.",
                 "Smoothing spline, LOWESS",
                 "X against Y", "loess smoother trend"),
        Analysis("standard_curve_interpolation", "Read off a standard curve",
                 "Turn readings from unknowns into concentrations.",
                 "Fit standards, then interpolate unknowns with error",
                 "standards plus unknown readings", "elisa calibration unknowns"),
        Analysis("area_under_curve", "Area under a curve",
                 "Total exposure, or the size of a peak.",
                 "Trapezoidal area, peak by peak, with baseline",
                 "X against Y", "auc integral peak"),
        Analysis("curve_smoothing_diff_integrate", "Smooth a trace, or take its slope or area",
                 "Reshape a trace: clean it, find its slope, or its running total.",
                 "Smoothing, first and second derivative, integral",
                 "a trace: X against Y", "derivative slope kinetics trace"),
        Analysis("plot_function", "Draw an equation",
                 "Put a formula on the page, with no data needed.",
                 "Any equation over a range you choose",
                 "nothing - you give the equation", "simulate formula theoretical", req=Needs(no_data=True)),
    ),
       req=Needs(min_groups=1, numeric_x=True, min_rows=4),
       produces="Fitted parameters with confidence intervals, and the curve drawn through your points."),
    Group("relate", "Find relationships", "Do they move together?", "relate", (
        Analysis("correlation_analysis", "Two variables together",
                 "How strongly two measurements track each other.",
                 "Pearson r, Spearman rank correlation, 95% CI",
                 "two columns of numbers", "r correlation coefficient rank"),
        Analysis("correlation_matrix", "Every pair at once",
                 "The same, for every pair of columns in the table.",
                 "Pearson or Spearman correlation matrix",
                 "several columns of numbers", "heatmap pairwise", req=Needs(min_groups=3)),
        Analysis("partial_correlation", "Two variables, others held fixed",
                 "Do two measurements still go together once other variables are allowed for?",
                 "Partial and semipartial correlation, Pearson or Spearman",
                 "three or more columns, one row per subject", "partial controlling confounder adjusted semipartial",
                 req=Needs(min_groups=3, min_rows=5)),
        Analysis("multiple_linear_regression", "Several predictors, a number",
                 "How several variables together explain one measured outcome.",
                 "Multiple linear regression, with model fit and diagnostics",
                 "one outcome column and several predictors", "multivariable covariates adjusted", req=Needs(min_groups=3)),
        Analysis("multiple_logistic_regression", "Several predictors, yes/no",
                 "The same when the outcome is yes or no.",
                 "Multiple logistic regression, odds ratios",
                 "a 0/1 outcome and several predictors", "odds ratio risk factors", req=Needs(min_groups=3, binary=True)),
        Analysis("count_regression", "Several predictors, a count",
                 "The same when the outcome counts events, cells or visits.",
                 "Poisson, quasi-Poisson and negative binomial regression · rate ratios, exposure, a check for extra spread",
                 "a column of whole-number counts and one or more predictors",
                 "poisson negative binomial rate ratio overdispersion incidence exposure offset glm",
                 req=Needs(min_groups=2, min_rows=6)),
        Analysis("ordinal_regression", "Several predictors, ordered categories",
                 "Explain a grade, stage or rating whose categories come in an order.",
                 "Ordinal logistic regression (proportional odds), ordinal probit · Brant test",
                 "an outcome coded 1, 2, 3 ... in order, and one or more predictors",
                 "ordinal ordered cumulative logit proportional odds likert severity stage polr",
                 req=Needs(min_groups=2, min_rows=10)),
        Analysis("multinomial_regression", "Several predictors, unordered categories",
                 "Explain which of several categories a subject falls in.",
                 "Multinomial logistic regression · odds ratios against a reference category, likelihood-ratio tests",
                 "an outcome coded 1, 2, 3 ... and one or more predictors",
                 "multinomial polytomous nominal categories choice baseline category logit",
                 req=Needs(min_groups=2, min_rows=10)),
    ),
       req=Needs(min_groups=2),
       produces="The strength of the relationship, with a confidence interval and a P value."),
    Group("counts", "Counts and proportions", "Are the categories linked?", "counts", (
        Analysis("chi_square_test", "Compare counts",
                 "Whether an outcome depends on the group it came from.",
                 "Chi-square, Fisher's exact (Freeman-Halton), chi-square for trend · relative risk, odds ratio, Cramér's V",
                 "a cross-tab of whole counts", "contingency odds ratio relative risk"),
        Analysis("stratified_tables", "Compare counts within layers",
                 "Does a link between two categories hold up once a third factor is allowed for?",
                 "Cochran-Mantel-Haenszel test, Mantel-Haenszel common odds ratio, Breslow-Day test",
                 "a 2×2 table for each layer (four columns of counts), or three columns of codes",
                 "stratified layers crosstab confounding simpson paradox cmh common odds ratio adjusted third variable",
                 req=Needs(min_groups=3, min_rows=1, counts=True)),
        Analysis("goodness_of_fit", "Counts against expectations",
                 "Do observed counts match the proportions you predicted?",
                 "Chi-square goodness of fit, binomial test",
                 "observed counts and expected proportions", "expected observed", req=Needs(min_groups=1, min_rows=2, counts=True)),
        Analysis("paired_proportions", "Yes/no, same subjects",
                 "Has the proportion changed between conditions measured on the same subjects?",
                 "McNemar's test (exact, mid-P), Cochran's Q",
                 "one subject per row, a 1/0 code in each column", "mcnemar cochran paired binary before after proportion",
                 req=Needs(min_groups=2, min_rows=2, binary=True)),
        Analysis("percentage_of_total", "Turn counts into percentages",
                 "Express each value as its share of the whole.",
                 "Percentage by row, column or grand total",
                 "any table of counts or amounts", "fraction share proportion", req=Needs(min_groups=1, min_rows=1)),
    ),
       req=Needs(min_groups=2, min_rows=2, counts=True),
       produces="A P value, an effect size such as an odds ratio, and the percentages behind them."),
    Group("time", "Time to event", "How long until it happens?", "time", (
        Analysis("kaplan_meier", "Survival curves",
                 "Follow groups over time, counting subjects who leave early.",
                 "Kaplan-Meier, log-rank, Gehan-Breslow (generalised Wilcoxon), hazard ratio",
                 "a time and a 1/0 outcome per subject", "survival censored logrank km"),
        Analysis("cox_regression", "What changes the risk",
                 "Which factors speed the event up or slow it down.",
                 "Cox proportional hazards regression, hazard ratios",
                 "time, outcome and predictor columns", "hazard ratio survival covariates", req=Needs(min_groups=2, min_rows=4, numeric_x=True, binary=True)),
        Analysis("life_table", "Survival by intervals",
                 "Survival summarised period by period - for large studies, or follow-up that only exists as counts.",
                 "Life table (actuarial method) · hazard rate and median time, tests on the intervals",
                 "a time and a 1/0 outcome per subject, or events and withdrawals per interval",
                 "life table actuarial intervals grouped survival hazard withdrawn",
                 req=Needs(min_groups=1, min_rows=2, numeric_x=True)),
        Analysis("cox_time_varying", "Risk that changes over time",
                 "When an effect fades or grows during follow-up, or a treatment starts part-way.",
                 "Cox regression with time-dependent covariates · hazard ratios that change with time",
                 "time, outcome and predictor columns; optionally a column of start times",
                 "time dependent time varying covariate non proportional hazards immortal time transplant",
                 req=Needs(min_groups=2, min_rows=6, numeric_x=True, binary=True)),
    ),
       req=Needs(min_groups=1, min_rows=4, numeric_x=True, binary=True),
       produces="Survival curves, median survival, and a test comparing the groups."),
    Group("agreement", "Agreement and accuracy", "Can it tell things apart?", "agreement", (
        Analysis("roc_curve_analysis", "How well a test separates",
                 "Whether a measurement can tell two groups apart, and where to cut.",
                 "ROC curve, AUC, sensitivity, specificity",
                 "a measurement and a known 0/1 grouping", "sensitivity specificity auc diagnostic", req=Needs(min_groups=2)),
        Analysis("bland_altman", "Do two methods agree?",
                 "Compare two ways of measuring the same thing.",
                 "Bland-Altman bias and limits of agreement",
                 "two columns measuring the same subjects", "method comparison bias limits", req=Needs(min_groups=2)),
        Analysis("reliability", "Do the columns agree?",
                 "Reliability of a scale's items, or agreement between raters.",
                 "Cronbach's alpha, intraclass correlation, Cohen's and Fleiss' kappa",
                 "one subject per row; a column per item or rater", "cronbach alpha icc kappa raters agreement reliability",
                 req=Needs(min_groups=2, min_rows=3)),
    ),
       req=Needs(min_groups=2),
       produces="A curve or plot, with the numbers that summarise how well it does."),
    Group("patterns", "Patterns across variables", "What structure is in there?", "patterns", (
        Analysis("pca", "Find the main axes",
                 "Boil many variables down to the few that carry the variation.",
                 "PCA with scores, loadings and scree",
                 "several numeric columns, one row per subject", "dimension reduction components biplot"),
        Analysis("factor_analysis", "Find hidden factors",
                 "Are a few underlying factors behind many correlated variables?",
                 "Exploratory factor analysis: principal axis, maximum likelihood · varimax, quartimax, promax, oblimin · KMO, Bartlett's test",
                 "several numeric columns, one row per subject",
                 "factor analysis efa latent questionnaire scale loadings rotation communality"),
        Analysis("discriminant_analysis", "Tell known groups apart",
                 "Which measurements separate the groups, and how well can a new subject be placed?",
                 "Linear discriminant analysis · Wilks' lambda, Box's M, leave-one-out classification",
                 "a column of group codes and one or more measurements",
                 "discriminant lda classify classification canonical groups predict membership",
                 req=Needs(min_groups=2, min_rows=6)),
        Analysis("hierarchical_clustering", "Group by similarity",
                 "Build a tree of which rows resemble which.",
                 "Hierarchical clustering with a dendrogram",
                 "several numeric columns", "dendrogram linkage similarity"),
        Analysis("kmeans_clustering", "Split into k groups",
                 "Divide the rows into a set number of clusters.",
                 "K-means, with silhouette scores",
                 "several numeric columns", "cluster kmeans silhouette"),
    ),
       req=Needs(min_groups=3),
       produces="The structure found in the data, as a table and as a graph."),
    Group("reshape", "Reshape and clean", "Get the table ready", "reshape", (
        Analysis("transform_data", "Transform values",
                 "Change every value by a chain of steps - a log, a scaling, anything.",
                 "Steps you pick, in the order you put them",
                 "any numeric column", "log reciprocal square root equation"),
        Analysis("transform_x_concentration", "Doses to a log scale",
                 "Turn concentrations into the log scale a dose-response wants.",
                 "log, -log, or back again",
                 "an X column of concentrations", "molar log dose"),
        Analysis("normalize_values", "Put on a common scale",
                 "Rescale so runs or plates can be compared.",
                 "Percent of control, of a range, or of a chosen row",
                 "any numeric columns", "percent scale baseline control"),
        Analysis("baseline_correction", "Subtract a baseline",
                 "Take a blank away, or combine columns arithmetically.",
                 "Subtract or divide by a baseline column or value",
                 "at least two columns", "blank subtract ratio background"),
        Analysis("remove_rows", "Drop rows you don't want",
                 "Filter out blanks, or rows failing a condition.",
                 "Remove by condition, by position, or blanks",
                 "any table", "filter delete exclude"),
        Analysis("transpose_xy", "Turn the table on its side",
                 "Swap rows for columns when the layout is the wrong way round.",
                 "Transpose into a new sheet",
                 "any table", "swap rotate flip"),
        Analysis("extract_rearrange", "Rebuild the table",
                 "Pull a subset out, or lay it out in another shape.",
                 "Extract, stack or unstack into a new sheet",
                 "any table", "subset stack unstack pivot"),
        Analysis("select_transform", "Pick and convert variables",
                 "Choose which variables to carry forward, and recode them.",
                 "Select, rename and transform columns",
                 "a multiple-variables table", "recode subset columns"),
    ),
       req=Needs(min_groups=1, min_rows=1),
       produces="A new sheet: your data is left exactly as it was."),
    Group("make", "Make data", "Generate values to try", "make", (
        Analysis("simulate_xy_data", "Simulated XY data",
                 "Make a curve with realistic scatter, to test a fit.",
                 "Any equation plus noise you specify",
                 "nothing - you set the model", "simulate synthetic mock", req=Needs(no_data=True)),
        Analysis("simulate_column_data", "Simulated groups",
                 "Make groups with known means, to see what a test would say.",
                 "Normal, lognormal, Poisson and more",
                 "nothing - you set the distribution", "simulate synthetic power", req=Needs(no_data=True)),
        Analysis("simulate_contingency_table", "Simulated counts",
                 "Make a 2x2 table of counts from probabilities you choose.",
                 "Random draws from set proportions",
                 "nothing - you set the proportions", "simulate synthetic contingency"),
        Analysis("monte_carlo_simulation", "Run it many times over",
                 "Repeat an experiment thousands of times to see how often it works.",
                 "Monte Carlo runs with a summary of the outcomes",
                 "nothing - you set the experiment", "power simulation repeat", req=Needs(no_data=True)),
    ),
       req=Needs(no_data=True),
       produces="A new sheet of generated values, ready to analyse."),
)

BY_KEY: dict[str, Analysis] = {a.key: a for g in GROUPS for a in g.entries}


# ── Tools: what the Analyze window and menu actually list ───────────────────
# Close relatives are one tool with modes, not separate entries: comparing
# groups is one job whether there are two or five, and simulating data is one
# job whatever shape the data takes. The mode that suits the sheet is picked
# for you (``rule``) and can be switched at the top of the settings.

@dataclass(frozen=True)
class Tool:
    id: str
    title: str
    question: str
    methods: str
    variants: tuple[tuple[str, str], ...]  # (registry key, the mode's short name)
    rule: str = ""                          # how the mode is picked from the sheet

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(k for k, _label in self.variants)

    def label_of(self, key: str) -> str:
        return dict(self.variants).get(key, "")

    def pick(self, shape: "SheetShape | None") -> str:
        """The mode that suits this sheet: the first one unless the sheet says otherwise."""
        first = self.variants[0][0]
        if shape is None or len(self.variants) < 2:
            return first
        if self.rule == "groups3" and shape.groups >= 3:
            return self.variants[1][0]
        if self.rule == "factors3" and shape.title_factors >= 3:
            return self.variants[1][0]
        if self.rule == "binary" and shape.binary:
            return self.variants[1][0]
        return first


def _solo(key: str) -> Tool:
    a = BY_KEY[key]
    return Tool(key, a.title, a.question, a.methods, ((key, ""),))


_TOOL_PLAN: dict[str, tuple[Tool, ...]] = {
    "compare": (
        Tool("compare_groups", "Compare groups", "Do the groups really differ?",
             "Welch or Student t test, Mann-Whitney, Wilcoxon · Welch ANOVA, one-way ANOVA, Kruskal-Wallis, Friedman",
             (("two_group_comparison", "Two groups"), ("one_way_anova", "Three or more")), "groups3"),
        _solo("one_sample_test"),
        Tool("factors", "Two or three factors", "Do several factors each matter, and do they interact?",
             "Two-way ANOVA (Type III), repeated measures · three-way ANOVA",
             (("two_way_anova", "Two factors"), ("three_way_anova", "Three factors")), "factors3"),
        _solo("multiple_comparisons"),
        Tool("subsamples", "Subsamples within subjects",
             "Each subject measured several times - the subject, not the measurement, is the unit.",
             "Nested t test, nested one-way ANOVA",
             (("nested_t_test", "Two groups"), ("nested_one_way_anova", "Three or more")), "groups3"),
        _solo("effect_sizes"),
        _solo("ancova"),
        _solo("mixed_model"),
        _solo("power_sample_size"),
    ),
    "describe": (
        Tool("summarise", "Summarise", "Means, spread and confidence intervals, down columns or across rows.",
             "Mean, median, SD, SEM, 95% CI, quartiles, CV, geometric mean",
             (("descriptive_statistics", "Each column"), ("row_wise_statistics", "Each row"))),
        Tool("shape", "Shape of the values", "How are the values spread out, and is that close to normal?",
             "Histogram counts, cumulative frequencies · Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling",
             (("frequency_distribution", "Histogram"), ("normality_testing", "Normal?"))),
        _solo("variance_tests"),
        _solo("outlier_detection"),
        _solo("batch_pvalue_analysis"),
    ),
    "fit": (
        _solo("nonlinear_regression"),
        Tool("curves_compared", "Compare fitted curves", "Do the data sets share a curve - and which model fits better?",
             "Global fit with shared parameters · extra sum-of-squares F test · AICc",
             (("global_fit", "Data sets together"), ("compare_models", "Two models"))),
        Tool("straight_line", "Straight-line fit", "Slope, intercept, and how sure you can be of them.",
             "Least squares with confidence bands and a runs test · Deming regression",
             (("linear_regression", "Y measured with error"), ("deming_regression", "Both measured with error"))),
        _solo("logistic_regression"),
        _solo("standard_curve_interpolation"),
        Tool("trace", "Work with a trace", "Its area, its slope, or its underlying trend.",
             "Area under a curve · derivatives and running integral · splines, LOWESS",
             (("area_under_curve", "Area"), ("curve_smoothing_diff_integrate", "Slope"),
              ("spline_lowess_smoothing", "Trend"))),
        Tool("signals", "Peaks and signals", "Find and fit peaks, flatten a baseline, or see and filter the frequencies.",
             "Multi-peak fit: Gaussian, Lorentzian, Voigt · baseline by asymmetric least squares · Fourier transform and filters",
             (("peak_fitting", "Peaks"), ("baseline_spectrum", "Baseline"), ("fourier_filter", "Frequencies"))),
    ),
    "relate": (
        Tool("correlate", "Correlation", "How strongly do measurements track each other?",
             "Pearson r, Spearman rank correlation, 95% CI · correlation matrix · partial correlation",
             (("correlation_analysis", "Two variables"), ("correlation_matrix", "Every pair"),
              ("partial_correlation", "Others held fixed"))),
        Tool("predictors", "Several predictors", "How do several variables together explain an outcome?",
             "Multiple linear regression · multiple logistic regression, odds ratios · Poisson and negative binomial regression",
             (("multiple_linear_regression", "A measured outcome"), ("multiple_logistic_regression", "A yes/no outcome"),
              ("count_regression", "A count")),
             "binary"),
        Tool("category_outcome", "Several predictors, categories", "How do several variables explain which category a subject falls in?",
             "Ordinal logistic regression (proportional odds), Brant test · multinomial logistic regression",
             (("ordinal_regression", "In an order"), ("multinomial_regression", "No order"))),
    ),
    "counts": (_solo("chi_square_test"), _solo("stratified_tables"), _solo("goodness_of_fit"), _solo("paired_proportions"),
               _solo("percentage_of_total")),
    "time": (_solo("kaplan_meier"), _solo("life_table"), _solo("cox_regression"), _solo("cox_time_varying")),
    "agreement": (_solo("roc_curve_analysis"), _solo("bland_altman"), _solo("reliability")),
    "patterns": (
        _solo("pca"),
        _solo("factor_analysis"),
        Tool("clusters", "Find clusters", "Which rows belong together?",
             "Hierarchical clustering with a dendrogram · k-means with silhouette scores",
             (("hierarchical_clustering", "A tree"), ("kmeans_clustering", "A set number"))),
        _solo("discriminant_analysis"),
    ),
    "reshape": (
        Tool("transform", "Transform values", "Change every value step by step, or put doses on a log scale.",
             "A chain of steps you build · log, -log, or back again",
             (("transform_data", "Step by step"), ("transform_x_concentration", "Doses to logs"))),
        Tool("rescale", "Rescale", "Put runs, plates or subjects on a common footing.",
             "Percent of a range or a control · take off a baseline, or scale by it",
             (("normalize_values", "0 to 100%"), ("baseline_correction", "Against a baseline"))),
        _solo("remove_rows"),
        Tool("restructure", "Reshape the table", "Lay the same values out another way, on a new sheet.",
             "Transpose · stack or unstack · pick and recode variables",
             (("transpose_xy", "Swap rows and columns"), ("extract_rearrange", "Stack or unstack"),
              ("select_transform", "Pick variables"))),
    ),
    "make": (
        Tool("simulate", "Simulate data", "Make data with a known answer, to see what an analysis finds.",
             "A curve plus noise · groups from a distribution · a 2x2 table of counts",
             (("simulate_xy_data", "A curve"), ("simulate_column_data", "Groups"),
              ("simulate_contingency_table", "Counts"))),
        _solo("plot_function"),
        _solo("monte_carlo_simulation"),
    ),
}

# every group now holds its tools, and its analyses in the tools' order
GROUPS = tuple(
    dataclasses.replace(g, tools=_TOOL_PLAN[g.id],
                        entries=tuple(BY_KEY[k] for t in _TOOL_PLAN[g.id] for k in t.keys))
    for g in GROUPS
)
TOOL_OF: dict[str, Tool] = {k: t for g in GROUPS for t in g.tools for k in t.keys}
TOOLS: dict[str, Tool] = {t.id: t for g in GROUPS for t in g.tools}
GROUP_OF: dict[str, Group] = {a.key: g for g in GROUPS for a in g.entries}

# What to put in front of someone the moment they open the picker, given the
# kind of table they are on. Deliberately short - four suggestions people will
# actually read, not a second copy of the menu.
SUGGESTED: dict[str, tuple[str, ...]] = {
    "Bivariate": ("nonlinear_regression", "linear_regression",
                  "correlation_analysis", "area_under_curve"),
    "Column": ("two_group_comparison", "one_way_anova",
               "descriptive_statistics", "normality_testing"),
    "Grouped": ("two_way_anova", "multiple_comparisons",
                "descriptive_statistics", "row_wise_statistics"),
    "Cross-tab": ("chi_square_test", "goodness_of_fit", "percentage_of_total"),
    "Kaplan–Meier": ("kaplan_meier", "cox_regression"),
    "Proportion": ("percentage_of_total", "goodness_of_fit"),
    "Many variables": ("multiple_linear_regression", "correlation_matrix",
                           "pca", "kmeans_clustering"),
    "Concentric": ("nested_t_test", "nested_one_way_anova", "descriptive_statistics"),
    # names earlier versions used for the same kinds of table
    "Multiple variables": ("multiple_linear_regression", "correlation_matrix", "pca", "kmeans_clustering"),
    "XY": ("nonlinear_regression", "linear_regression",
           "correlation_analysis", "area_under_curve"),
    "Contingency": ("chi_square_test", "goodness_of_fit", "percentage_of_total"),
    "Survival": ("kaplan_meier", "cox_regression"),
    "Parts of whole": ("percentage_of_total", "goodness_of_fit"),
    "Nested": ("nested_t_test", "nested_one_way_anova", "descriptive_statistics"),
}


@dataclass(frozen=True)
class SheetShape:
    """What the sheet in front of you actually holds.

    Worked out once when the sheet is chosen, then checked against every
    analysis's Needs - so the picker can say "this one wants three data
    sets; this sheet has two" before anything is run.
    """
    groups: int = 0
    rows: int = 0
    replicates: int = 1
    numeric_x: bool = False
    counts: bool = False
    binary: bool = False
    row_factor: bool = False
    title_factors: int = 0

    @classmethod
    def read(cls, sheet) -> "SheetShape":
        import numpy as np

        labels = sheet.group_labels()
        values, replicates, binary = [], 1, False
        for label in labels:
            reps, matrix = sheet.group_replicate_matrix(label)
            replicates = max(replicates, len(reps))
            finite = matrix[np.isfinite(matrix)] if matrix.size else matrix
            if finite.size:
                values.append(finite)
                if np.isin(finite, (0.0, 1.0)).all():
                    binary = True
        pooled = np.concatenate(values) if values else np.empty(0)
        x = np.asarray(sheet.x, dtype=float) if sheet.x is not None else np.empty(0)
        names = [str(n).strip() for n in (getattr(sheet, "row_names", None) or [])]
        titled = sum(1 for n in names if n and not _is_number(n))
        return cls(
            groups=len(labels),
            rows=int(sheet.n_rows or 0),
            replicates=replicates,
            numeric_x=bool(np.isfinite(x).sum() >= 2),
            counts=bool(pooled.size and (pooled >= 0).all() and np.all(pooled == np.round(pooled))),
            binary=binary,
            row_factor=bool(replicates >= 2 or titled >= 2),
            title_factors=_title_factors(labels),
        )


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _title_factors(labels: list[str]) -> int:
    """How many factors the data-set titles spell out: "Male-Low-Fed" and
    "Female-High-Fasted" give 3 when every title splits the same way and
    each part takes at least two values."""
    best = 0
    for sep in ("-", "_", "|", ":", "/"):
        parts = [label.split(sep) for label in labels]
        if len(parts) < 2 or len({len(p) for p in parts}) != 1 or len(parts[0]) < 2:
            continue
        if all(len({p[i].strip() for p in parts}) >= 2 for i in range(len(parts[0]))):
            best = max(best, len(parts[0]))
    return best


def requirement(analysis: Analysis) -> Needs:
    return analysis.req or GROUP_OF[analysis.key].req or Needs()


def verdict(analysis: Analysis, shape: "SheetShape | None") -> tuple[bool, str]:
    """(can it run, why not). An unreadable or unknown sheet is never called
    a failure - the picker simply says nothing about fit."""
    need = requirement(analysis)
    if need.no_data:
        return True, "Makes its own data - the sheet doesn't matter."
    if shape is None:
        return True, ""
    if shape.groups < need.min_groups:
        return False, f"Needs {need.min_groups}+ data sets; this sheet has {shape.groups}."
    if shape.rows < need.min_rows:
        return False, f"Needs {need.min_rows}+ rows; this sheet has {shape.rows}."
    if need.replicates and shape.replicates < 2:
        return False, "Needs side-by-side repeats; this sheet has one value per point."
    if need.numeric_x and not shape.numeric_x:
        return False, "Needs numbers in the X column; this sheet has none."
    if need.counts and not shape.counts:
        return False, "Needs whole counts; this sheet has fractions or negatives."
    if need.binary and not shape.binary:
        return False, "Needs a 0/1 outcome column; this sheet has none."
    if need.row_factor and not shape.row_factor:
        return False, "Needs rows that are a second factor (row titles or side-by-side repeats); these rows are plain repeats."
    if need.title_factors and shape.title_factors < need.title_factors:
        return False, (f"Needs data-set titles naming {need.title_factors} factors, like Male-Low-Fed; "
                       "these titles don't split that way.")
    return True, "This sheet has what it needs."


def suggested_for(table_kind: str | None) -> list[Analysis]:
    return [BY_KEY[k] for k in SUGGESTED.get(table_kind or "", ()) if k in BY_KEY]


def _norm(value: str) -> str:
    """Lower case, apostrophes and dashes dropped: "Fisher's" -> "fishers",
    "t-test" -> "t test", "Mann-Whitney" -> "mann whitney"."""
    import re

    value = value.lower().replace("\u2019", "'")
    value = re.sub(r"'s\b", "s", value)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", value).split())


def _squash(value: str) -> str:
    return _norm(value).replace(" ", "")


def search(text: str) -> list[Analysis]:
    """Analyses matching ``text``, best first.

    A named test comes first: typing "t test", "Tukey" or "log-rank" puts
    the analyses that run that test at the top, those whose test name
    *starts* with it above those that merely mention it. Otherwise the
    title, the question, the tests and a few extra words (so "ec50", "km"
    or "dendrogram" find the right thing) are searched, title matches
    first. Spelling is forgiving - dashes, spaces and apostrophes don't
    matter - and words of three letters or fewer must match whole words,
    so "cox" doesn't find Wilcoxon.
    """
    import re

    query = _norm(text)
    words = query.split()
    if not words:
        return []
    squashed_query = query.replace(" ", "")
    scored = []
    for order, analysis in enumerate(BY_KEY.values()):
        group = GROUP_OF[analysis.key]
        # the named tests, one by one ("Welch's t test", "Tukey", ...)
        tests = [_norm(t) for t in re.split(r"[,·:;()]| or | and ", analysis.methods) if t.strip()]
        tests.append(_norm(analysis.title))  # "Fit a curve" is as good a name as any test's
        phrase = 0
        if any(t.startswith(query) or (len(squashed_query) >= 4 and t.replace(" ", "").startswith(squashed_query))
               for t in tests):
            phrase = 100
        elif any(re.search(rf"\b{re.escape(query)}\b", t) for t in tests):
            phrase = 70
        elif len(squashed_query) >= 4 and any(squashed_query in t.replace(" ", "") for t in tests):
            phrase = 40

        fields = [(_norm(analysis.title), 5), (_norm(analysis.question), 3), (_norm(analysis.methods), 2),
                  (_norm(analysis.tags), 2), (_norm(analysis.key), 2), (_norm(analysis.needs), 1),
                  (_norm(f"{group.title} {group.question}"), 2)]
        score = 0
        for word in words:
            if len(word) <= 3:
                best = max((w for value, w in fields if re.search(rf"\b{re.escape(word)}\b", value)), default=0)
            else:
                best = max((w for value, w in fields if word in value or word in value.replace(" ", "")),
                           default=0)
            if not best:
                score = 0
                break  # every word has to appear somewhere
            score += best
        if not score and len(squashed_query) >= 4:
            # the whole query run together: "mannwhitney", "ttest", "chisquare"
            score = max((w for value, w in fields if squashed_query in value.replace(" ", "")), default=0)
        if requirement(analysis).no_data:
            phrase = 0  # "normal" means the normality test, not the simulation's distribution list
        total = phrase + score
        if total:
            # ties keep the order the catalog lists them in, so a group's
            # headline analysis comes before its specialised ones
            scored.append((-total, order, analysis))
    return [a for _s, _o, a in sorted(scored)]
