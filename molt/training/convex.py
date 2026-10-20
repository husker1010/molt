"""Penalized linear scorers fitted by convex optimisation (elastic-net logistic models).

The solver minimises the weighted logistic negative log-likelihood plus an elastic-net
penalty,

    (1 / sum w) sum_i w_i [log(1 + exp(eta_i)) - y_i eta_i]
        + lambda * sum_j f_j [alpha |beta_j| + (1 - alpha) / 2 beta_j^2],

with ``eta = X beta`` and per-column penalty factors ``f_j`` (``0`` leaves a column, such as
an intercept or a bin indicator, unpenalized). It uses accelerated proximal gradient
descent (FISTA with adaptive restart) along a decreasing, warm-started penalty path and is
deterministic. Two model families are built on it:

* **binary** - logistic elastic net with an unpenalized intercept;
* **discrete time** - person-period logistic hazard model with unpenalized bin-specific
  intercepts and penalized covariate effects (time-invariant log-odds shift ``x' gamma``).

:func:`cv_path` chooses the penalty (and mixing) by K-fold cross-validated deviance
(participant-level folds, stratified by the outcome) and returns the cross-validated linear
predictor of every training participant at the chosen penalty, which is what a stacked
model needs for its training rows.

Example:
    >>> fit = cv_path(X, y, family="binary", alphas=(0.5,), n_lambdas=12, folds=5, seed=0)
    >>> eta_new = fit.predict(X_new)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConvexFit", "cv_path", "elastic_net_path", "person_period_rows"]

_EPS = 1e-12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _logistic_nll(eta: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean logistic negative log-likelihood."""
    return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / max(w.sum(), _EPS))


def _spectral_bound(X: np.ndarray, w: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """Largest eigenvalue of ``X' diag(w) X / sum(w)`` by power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v) + _EPS
    lam = 0.0
    sw = max(w.sum(), _EPS)
    for _ in range(iters):
        u = X.T @ (w * (X @ v)) / sw
        lam = float(np.linalg.norm(u))
        if lam <= _EPS:
            return 1.0
        v = u / lam
    return lam * 1.05


def _fista(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    pf: np.ndarray,
    lam: float,
    alpha: float,
    beta: np.ndarray,
    lipschitz: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Solve one penalized problem from the warm start ``beta``."""
    sw = max(w.sum(), _EPS)
    l1 = lam * alpha * pf
    l2 = lam * (1.0 - alpha) * pf
    step = 1.0 / (lipschitz / 4.0 + float(l2.max(initial=0.0)))
    x_old = beta.copy()
    z = beta.copy()
    t = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        grad = X.T @ (w * (_sigmoid(X @ z) - y)) / sw + l2 * z
        x_new = z - step * grad
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - step * l1, 0.0)
        delta = x_new - x_old
        if float(np.dot(grad, delta)) > 0.0:  # adaptive restart
            t = 1.0
            z = x_new
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * delta
            t = t_new
        x_old = x_new
        if np.max(np.abs(delta)) <= tol * (1.0 + np.max(np.abs(x_new))):
            break
    return x_old, it


def elastic_net_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float],
    *,
    alpha: float = 0.5,
    weights: np.ndarray | None = None,
    penalty_factor: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-5,
) -> np.ndarray:
    """Coefficient path ``(len(lambdas), p)`` of the penalized logistic model.

    ``X`` must contain any intercept or unpenalized columns explicitly (penalty factor 0).
    ``lambdas`` should be decreasing; every solution warm-starts the next one.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    pf = np.ones(X.shape[1]) if penalty_factor is None else np.asarray(penalty_factor, float)
    lipschitz = _spectral_bound(X, w)
    beta = np.zeros(X.shape[1])
    path = np.zeros((len(lambdas), X.shape[1]))
    for i, lam in enumerate(lambdas):
        beta, _ = _fista(X, y, w, pf, float(lam), float(alpha), beta, lipschitz, max_iter, tol)
        path[i] = beta
    return path


def lambda_max(
    X: np.ndarray, y: np.ndarray, pf: np.ndarray, alpha: float, w: np.ndarray | None = None
) -> float:
    """Smallest penalty with all penalized coefficients zero (unpenalized part fitted)."""
    w = np.ones(len(y)) if w is None else w
    free = pf == 0
    beta = np.zeros(X.shape[1])
    if free.any():
        sub = elastic_net_path(
            X[:, free], y, [0.0], alpha=1.0, weights=w, penalty_factor=np.zeros(free.sum())
        )
        beta[free] = sub[0]
    grad = X.T @ (w * (_sigmoid(X @ beta) - y)) / max(w.sum(), _EPS)
    pen = ~free
    if not pen.any():
        return 1.0
    return float(np.max(np.abs(grad[pen]) / pf[pen]) / max(alpha, 1e-3)) + 1e-8


def person_period_rows(
    bins: np.ndarray, events: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Person-period expansion: participant row, bin index and outcome of every at-risk bin.

    Participant ``i`` with observed bin ``k_i`` contributes bins ``0..k_i``; the outcome is 1
    in bin ``k_i`` for an event and 0 otherwise.
    """
    bins = np.asarray(bins, dtype=int)
    rows = np.repeat(np.arange(len(bins)), bins + 1)
    k = np.concatenate([np.arange(b + 1) for b in bins]) if len(bins) else np.zeros(0, int)
    last = np.zeros(len(rows), dtype=bool)
    last[np.cumsum(bins + 1) - 1] = True
    outcome = (last & (np.asarray(events)[rows] > 0)).astype(np.float64)
    k = np.minimum(k, n_bins - 1)
    return rows, k, outcome


@dataclass
class ConvexFit:
    """A fitted penalized linear scorer.

    Attributes:
        family: ``binary`` or ``discrete_time``.
        coef: Covariate coefficients on the standardized design.
        intercepts: Intercept (binary) or bin-specific intercepts (discrete time).
        center, scale: Column standardization of the training design.
        alpha, lam: Chosen elastic-net mixing and penalty.
        oof: Cross-validated linear predictor of the training rows, ``(n, n_blocks)``: the
            contribution of every column block (one block = ``x' beta``).
        cv_loss: Cross-validated deviance at the chosen penalty, per alpha and lambda.
    """

    family: str
    coef: np.ndarray
    intercepts: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    alpha: float
    lam: float
    oof: np.ndarray
    cv_loss: dict[str, list[float]] = field(default_factory=dict)
    lambdas: list[float] = field(default_factory=list)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """Training standardization; missing values become the training mean (0)."""
        Z = (np.asarray(X, dtype=np.float64) - self.center) / self.scale
        return np.where(np.isfinite(Z), Z, 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Linear predictor ``x' beta`` (without intercepts)."""
        return self.standardize(X) @ self.coef

    @property
    def n_nonzero(self) -> int:
        return int(np.count_nonzero(self.coef))


def _design(family: str, Z: np.ndarray, y: np.ndarray, extra: dict) -> tuple[np.ndarray, ...]:
    """Expanded design, outcome, penalty factors and participant index of every row."""
    n, p = Z.shape
    if family == "binary":
        X = np.column_stack([np.ones(n), Z])
        pf = np.r_[0.0, np.ones(p)]
        return X, np.asarray(y, float), pf, np.arange(n)
    n_bins = int(extra["n_bins"])
    rows, k, outcome = person_period_rows(extra["bins"], extra["events"], n_bins)
    onehot = np.zeros((len(rows), n_bins))
    onehot[np.arange(len(rows)), k] = 1.0
    X = np.column_stack([onehot, Z[rows]])
    pf = np.r_[np.zeros(n_bins), np.ones(p)]
    return X, outcome, pf, rows


def _row_loss(X: np.ndarray, y: np.ndarray, beta: np.ndarray, rows: np.ndarray, n: int) -> float:
    """Deviance per participant (sum over its person-period rows, mean over participants)."""
    eta = X @ beta
    ll = np.logaddexp(0.0, eta) - y * eta
    return float(np.bincount(rows, weights=ll, minlength=n).sum() / max(n, 1))


def _strata_folds(strata: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(strata), dtype=int)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def cv_path(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    family: str = "binary",
    bins: np.ndarray | None = None,
    events: np.ndarray | None = None,
    n_bins: int | None = None,
    alphas: Sequence[float] = (0.5,),
    n_lambdas: int = 12,
    lambda_ratio: float = 0.01,
    folds: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
    tol: float = 1e-5,
    blocks: Sequence[int] | None = None,
) -> ConvexFit:
    """Fit a penalized scorer with the penalty chosen by K-fold cross-validated deviance.

    Args:
        X: ``(n, p)`` covariates (NaN = missing, imputed by the training mean).
        y: Binary outcome (``family="binary"``).
        family: ``binary`` or ``discrete_time``.
        bins, events, n_bins: Observed bin, event indicator and number of bins
            (``family="discrete_time"``).
        alphas: Elastic-net mixing values searched jointly with the penalty.
        n_lambdas: Length of the geometric penalty path below ``lambda_max``.
        lambda_ratio: Smallest penalty as a fraction of ``lambda_max``.
        folds: Number of cross-validation folds (stratified by outcome / event).
        seed: Fold-assignment seed.
        blocks: Widths of consecutive column blocks; the cross-validated predictor is then
            returned per block (default one block).

    Returns:
        The refitted scorer with the cross-validated linear predictor of every row.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    center = np.nanmean(X, axis=0) if n else np.zeros(X.shape[1])
    center = np.nan_to_num(center)
    scale = np.nanstd(X, axis=0) if n else np.ones(X.shape[1])
    scale = np.where(np.nan_to_num(scale) > 1e-8, np.nan_to_num(scale), 1.0)
    Z = np.where(np.isfinite(X), (X - center) / scale, 0.0)
    keep = np.nanstd(X, axis=0) > 1e-8 if n else np.zeros(X.shape[1], bool)
    extra = {"bins": bins, "events": events, "n_bins": n_bins}
    if family == "binary":
        strata = np.asarray(y, float)
    elif family == "discrete_time":
        strata = np.asarray(events, float)
    else:
        raise ValueError(f"Unknown family {family!r}")
    widths = list(blocks) if blocks else [X.shape[1]]
    block_of = np.repeat(np.arange(len(widths)), widths)[keep]
    Zk = Z[:, keep]
    Xd, yd, pf, rows = _design(family, Zk, y, extra)
    fold_of = _strata_folds(strata, max(2, min(folds, n)), seed)
    best: tuple[float, float, float] | None = None
    cv_loss: dict[str, list[float]] = {}
    oof_by: dict[tuple[float, int], np.ndarray] = {}
    lam_by: dict[float, np.ndarray] = {}
    for alpha in alphas:
        lmax = lambda_max(Xd, yd, pf, float(alpha))
        lams = lmax * np.geomspace(1.0, lambda_ratio, n_lambdas)
        lam_by[float(alpha)] = lams
        losses = np.zeros(n_lambdas)
        oof = np.zeros((n_lambdas, n, len(widths)))
        for f in range(int(fold_of.max()) + 1):
            test_p = fold_of == f
            tr_rows, te_rows = ~test_p[rows], test_p[rows]
            path = elastic_net_path(
                Xd[tr_rows],
                yd[tr_rows],
                lams,
                alpha=float(alpha),
                penalty_factor=pf,
                max_iter=max_iter,
                tol=tol,
            )
            te_index = np.flatnonzero(test_p)
            remap = np.full(n, -1)
            remap[te_index] = np.arange(len(te_index))
            for i in range(n_lambdas):
                losses[i] += (
                    _row_loss(
                        Xd[te_rows], yd[te_rows], path[i], remap[rows[te_rows]], len(te_index)
                    )
                    * len(te_index)
                    / n
                )
                b = path[i][pf > 0]
                for g in range(len(widths)):
                    sel = block_of == g
                    oof[i, te_index, g] = Zk[te_index][:, sel] @ b[sel]
        cv_loss[str(float(alpha))] = losses.tolist()
        i_best = int(np.argmin(losses))
        for i in range(n_lambdas):
            oof_by[(float(alpha), i)] = oof[i]
        if best is None or losses[i_best] < best[0] - 1e-12:
            best = (float(losses[i_best]), float(alpha), float(i_best))
    assert best is not None
    _, alpha, i_best = best
    lams = lam_by[alpha]
    path = elastic_net_path(
        Xd, yd, lams[: int(i_best) + 1], alpha=alpha, penalty_factor=pf, max_iter=max_iter, tol=tol
    )
    beta = path[-1]
    coef = np.zeros(X.shape[1])
    coef[keep] = beta[pf > 0]
    intercepts = beta[pf == 0]
    return ConvexFit(
        family=family,
        coef=coef,
        intercepts=intercepts,
        center=center,
        scale=scale,
        alpha=alpha,
        lam=float(lams[int(i_best)]),
        oof=oof_by[(alpha, int(i_best))],
        cv_loss=cv_loss,
        lambdas=lams.tolist(),
    )
