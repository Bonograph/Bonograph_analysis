"""Small NumPy/SciPy implementations of a few routines several analyses
share: the ROC curve and its area, silhouette scores, principal components,
and an L2-regularised logistic regression.

They are written here, on NumPy and SciPy alone, so that the analyses need no
further compiled library (one more thing that can fail to load on a locked-
down computer). The conventions are the usual ones: the first ROC threshold
is infinity, each principal component's sign is fixed by its largest
loading, and the logistic regression's penalty is the common default. Each
routine is checked against independent reference calculations in
test/test_ml_utils.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------
def _binary_truth(y_true) -> np.ndarray:
    y = np.asarray(y_true, dtype=float).ravel()
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("Only one class present in y_true. ROC is not defined in that case.")
    return (y == classes[1]).astype(float)


def roc_curve(y_true, y_score, drop_intermediate: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds); thresholds[0] is +inf so the curve starts at
    (0, 0). Tied scores are treated as one step, and with
    ``drop_intermediate`` points lying on a straight segment are omitted."""
    truth = _binary_truth(y_true)
    score = np.asarray(y_score, dtype=float).ravel()
    if truth.size != score.size:
        raise ValueError("y_true and y_score must have the same length.")

    order = np.argsort(-score, kind="mergesort")
    score, truth = score[order], truth[order]
    # last index of each run of tied scores
    distinct = np.where(np.diff(score))[0]
    idx = np.r_[distinct, truth.size - 1]
    tps = np.cumsum(truth)[idx]
    fps = (1 + idx) - tps
    thresholds = score[idx]

    if drop_intermediate and fps.size > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps, tps, thresholds = fps[keep], tps[keep], thresholds[keep]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, thresholds]
    return fps / fps[-1], tps / tps[-1], thresholds


def roc_auc_score(y_true, y_score) -> float:
    """Area under the ROC curve (trapezoid rule over the full, untrimmed
    curve - equal to the Mann-Whitney U statistic divided by n1*n0)."""
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------
def silhouette_samples(X, labels, block: int = 512) -> np.ndarray:
    """Silhouette coefficient of every sample (Euclidean distance). A sample
    alone in its cluster scores 0. Computed in row blocks so a large data
    set never needs the full n x n distance matrix in memory."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    labels = np.asarray(labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    k = classes.size
    if not 2 <= k <= X.shape[0] - 1:
        raise ValueError(f"Number of labels is {k}. Valid values are 2 to n_samples - 1 (inclusive).")
    onehot = np.zeros((X.shape[0], k))
    onehot[np.arange(X.shape[0]), inverse] = 1.0
    sizes = onehot.sum(axis=0)

    out = np.empty(X.shape[0])
    for start in range(0, X.shape[0], block):
        stop = min(start + block, X.shape[0])
        cluster_sums = cdist(X[start:stop], X) @ onehot  # (rows, k): total distance to each cluster
        own = inverse[start:stop]
        rows = np.arange(stop - start)
        own_size = sizes[own]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = cluster_sums[rows, own] / (own_size - 1)
            mean_other = cluster_sums / sizes
        mean_other[rows, own] = np.inf
        b = mean_other.min(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (b - a) / np.maximum(a, b)
        s = np.where(own_size > 1, s, 0.0)
        out[start:stop] = np.nan_to_num(s, nan=0.0)
    return out


def silhouette_score(X, labels) -> float:
    return float(np.mean(silhouette_samples(X, labels)))


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
class PCA:
    """Principal components by SVD of the centred data. Attributes:
    ``components_`` (one row per component), ``explained_variance_``
    (ddof = 1) and ``explained_variance_ratio_``. Each component's sign is
    fixed so that its largest-magnitude loading is positive; without a rule
    the sign is arbitrary, and results would not be reproducible."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        k = min(n, p) if self.n_components is None else int(self.n_components)
        if not 1 <= k <= min(n, p):
            raise ValueError(f"n_components={k} must be between 1 and min(n_samples, n_features)={min(n, p)}")
        self.mean_ = X.mean(axis=0)
        _u, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        vt = vt * signs[:, None]
        variance = s**2 / (n - 1)
        total = variance.sum()
        self.components_ = vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / total if total > 0 else np.zeros(k)
        self.n_components_ = k
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Logistic regression (L2-regularised)
# ---------------------------------------------------------------------------
class LogisticRegression:
    """Binary logistic regression minimising
    ``0.5 * ||w||^2 + C * sum(log-loss)`` (the intercept is not penalised),
    with C = 1 unless set. Solved by damped Newton iterations to
    full precision, so the optimum is exact rather than stopping at a
    loose tolerance."""

    def __init__(self, C: float = 1.0, max_iter: int = 200, tol: float = 1e-12):
        self.C, self.max_iter, self.tol = C, max_iter, tol

    def _objective(self, params, Xb, y, penalty):
        z = Xb @ params
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        return 0.5 * np.sum(penalty * params**2) + self.C * loss

    def fit(self, X, y) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("This solver needs samples of exactly 2 classes in the data.")
        self.classes_ = classes
        t = (y == classes[1]).astype(float)
        Xb = np.column_stack([X, np.ones(X.shape[0])])
        penalty = np.ones(Xb.shape[1])
        penalty[-1] = 0.0  # never penalise the intercept
        w = np.zeros(Xb.shape[1])
        f = self._objective(w, Xb, t, penalty)
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
            grad = penalty * w + self.C * (Xb.T @ (p - t))
            if np.max(np.abs(grad)) < self.tol:
                break
            hess = np.diag(penalty) + self.C * (Xb.T * (p * (1 - p))) @ Xb
            step = np.linalg.solve(hess + 1e-12 * np.eye(hess.shape[0]), grad)
            scale = 1.0
            while scale > 1e-12:  # backtrack until the objective really decreases
                candidate = w - scale * step
                f_new = self._objective(candidate, Xb, t, penalty)
                if f_new <= f:
                    break
                scale /= 2.0
            else:
                break
            converged = abs(f - f_new) < 1e-15 * max(1.0, abs(f))
            w, f = candidate, f_new
            if converged:
                break
        self.coef_ = w[:-1].reshape(1, -1)
        self.intercept_ = w[-1:].copy()
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
