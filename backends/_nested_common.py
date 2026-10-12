"""Shared variance-decomposition math for the nested designs
(nested_t_test.py, nested_one_way_anova.py) - not a backend itself (no
@register call), just the arithmetic both of them need.

Both "subjects" (rows) and "replicates" (the sheet's Y1..Yn subcolumns)
matter here in a way flat analyses like one_way_anova.py don't: since
subjects are nested *within* groups and are what was actually randomized/
sampled, the correct F-test for a group effect uses the between-subjects-
within-groups mean square as its denominator - not the pooled replicate
residual naive one-way ANOVA would use. Skipping that distinction is
exactly the kind of bug that silently double/triple-counts replicate
measurements as if they were independent subjects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend_types import AnalysisError
from ..data_source import SheetData


@dataclass
class NestedDecomposition:
    labels: list[str]
    group_means: list[float]
    group_n_subjects: list[int]
    ss_between: float
    df_between: int
    ss_subjects: float
    df_subjects: int
    grand_mean: float
    n_total: int

    @property
    def ms_between(self) -> float:
        return self.ss_between / self.df_between

    @property
    def ms_subjects(self) -> float:
        return self.ss_subjects / self.df_subjects

    @property
    def f_statistic(self) -> float:
        return self.ms_between / self.ms_subjects if self.ms_subjects else float("inf")


def decompose(data: SheetData, min_groups: int) -> NestedDecomposition:
    labels = data.group_labels()

    per_group: list[tuple[str, list[tuple[float, int]]]] = []
    for label in labels:
        _reps, matrix = data.group_replicate_matrix(label)
        subjects = []
        for row in matrix:
            valid = row[~np.isnan(row)]
            if valid.size >= 1:
                subjects.append((float(np.mean(valid)), int(valid.size)))
        if len(subjects) >= 2:
            per_group.append((label, subjects))

    if len(per_group) < min_groups:
        raise AnalysisError(
            f"A nested design needs at least {min_groups} group(s), each "
            f"with 2+ subjects (rows) of data; found {len(per_group)} "
            "usable group(s)."
        )

    group_means, group_n, ss_subjects, df_subjects = [], [], 0.0, 0
    weighted_sum, total_n = 0.0, 0
    for _label, subjects in per_group:
        n_i = sum(m for _mean, m in subjects)
        mean_i = sum(mean * m for mean, m in subjects) / n_i
        group_means.append(mean_i)
        group_n.append(len(subjects))
        ss_subjects += sum(m * (mean - mean_i) ** 2 for mean, m in subjects)
        df_subjects += len(subjects) - 1
        weighted_sum += mean_i * n_i
        total_n += n_i

    if df_subjects < 1:
        raise AnalysisError(
            "Not enough subjects per group to estimate subject-to-subject "
            "variability - each group needs at least 2 rows of data."
        )

    grand_mean = weighted_sum / total_n
    ss_between = sum(
        n_i * (mean_i - grand_mean) ** 2
        for (_label, subjects), mean_i in zip(per_group, group_means)
        for n_i in [sum(m for _mean, m in subjects)]
    )

    return NestedDecomposition(
        labels=[label for label, _ in per_group],
        group_means=group_means,
        group_n_subjects=group_n,
        ss_between=ss_between,
        df_between=len(per_group) - 1,
        ss_subjects=ss_subjects,
        df_subjects=df_subjects,
        grand_mean=grand_mean,
        n_total=total_n,
    )
