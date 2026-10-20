"""Small numerical helpers shared by the metric modules.

The functions here implement weighted binomial generalized linear models (logit and
complementary log-log links) by iteratively reweighted least squares, restricted cubic spline
bases and a few array utilities. They are deliberately dependency-free (NumPy only) so that
they can be called thousands of times inside bootstrap loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EPS",
    "GLMFit",
    "as_1d_float",
    "cloglog",
    "fit_binomial_glm",
    "inv_cloglog",
    "logit",
    "rcs_basis",
    "rcs_knots",
]

EPS = 1e-6
_ETA_CLIP = 30.0


def as_1d_float(x: object, name: str = "array") -> np.ndarray:
    """Convert ``x`` to a contiguous one-dimensional float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Log-odds of probabilities clipped to ``[eps, 1 - eps]``."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def cloglog(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Complementary log-log transform ``log(-log(1 - p))`` of clipped probabilities."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(-np.log1p(-p))


def inv_cloglog(eta: np.ndarray) -> np.ndarray:
    """Inverse complementary log-log link ``1 - exp(-exp(eta))``."""
    eta = np.clip(np.asarray(eta, dtype=np.float64), -_ETA_CLIP, np.log(_ETA_CLIP * 2))
    return -np.expm1(-np.exp(eta))


@dataclass(frozen=True)
class GLMFit:
    """Result of :func:`fit_binomial_glm`.

    Attributes:
        coef: Estimated coefficients (one per design-matrix column).
        cov: Model-based covariance matrix (inverse Fisher information).
        converged: Whether the IRLS iterations converged.
        n_iter: Number of iterations used.
        link: Link function name.
    """

    coef: np.ndarray
    cov: np.ndarray
    converged: bool
    n_iter: int
    link: str

    def linear_predictor(self, X: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """Linear predictor ``X @ coef + offset`` for a new design matrix."""
        eta = np.asarray(X, dtype=np.float64) @ self.coef
        return eta if offset is None else eta + offset

    def predict(self, X: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """Mean response for a new design matrix."""
        eta = self.linear_predictor(X, offset)
        if self.link == "logit":
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -_ETA_CLIP, _ETA_CLIP)))
        return inv_cloglog(eta)


def _link_parts(eta: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and derivative d(mu)/d(eta) for the supported links."""
    if link == "logit":
        eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
        mu = 1.0 / (1.0 + np.exp(-eta))
        dmu = mu * (1.0 - mu)
    elif link == "cloglog":
        eta = np.clip(eta, -_ETA_CLIP, np.log(_ETA_CLIP * 2))
        e = np.exp(eta)
        mu = -np.expm1(-e)
        dmu = np.exp(eta - e)
    else:  # pragma: no cover - guarded by the public function
        raise ValueError(f"unsupported link {link!r}")
    return mu, dmu


def fit_binomial_glm(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    link: str = "logit",
    max_iter: int = 100,
    tol: float = 1e-10,
) -> GLMFit:
    """Fit a weighted binomial GLM by iteratively reweighted least squares.

    The response may be any real number (for example jackknife pseudo-observations), in which
    case the fit is a quasi-likelihood estimating-equation solution with binomial variance.

    Args:
        X: Design matrix of shape ``(n, k)``; include a column of ones for an intercept.
        y: Response of shape ``(n,)``.
        weights: Non-negative prior weights (for example inverse-probability-of-censoring
            weights); ``None`` means unit weights.
        offset: Optional offset added to the linear predictor.
        link: ``"logit"`` or ``"cloglog"``.
        max_iter: Maximum number of IRLS iterations.
        tol: Convergence tolerance on the maximum absolute coefficient change.

    Returns:
        A :class:`GLMFit`; coefficients are NaN when the system is singular.
    """
    if link not in ("logit", "cloglog"):
        raise ValueError("link must be 'logit' or 'cloglog'")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = as_1d_float(y)
    n, k = X.shape
    w = np.ones(n) if weights is None else as_1d_float(weights)
    off = np.zeros(n) if offset is None else as_1d_float(offset)
    keep = (w > 0) & np.isfinite(y) & np.isfinite(off) & np.all(np.isfinite(X), axis=1)
    X, y, w, off = X[keep], y[keep], w[keep], off[keep]
    nan_fit = GLMFit(np.full(k, np.nan), np.full((k, k), np.nan), False, 0, link)
    if X.shape[0] <= k or w.sum() <= 0:
        return nan_fit

    mu0 = np.clip((y + 0.5) / 2.0, 0.05, 0.95)
    eta = np.log(mu0 / (1 - mu0)) if link == "logit" else np.log(-np.log1p(-mu0))
    coef = np.zeros(k)
    converged = False
    it = 0
    info = np.eye(k)
    for it in range(1, max_iter + 1):
        mu, dmu = _link_parts(eta, link)
        var = np.clip(mu * (1.0 - mu), 1e-12, None)
        dmu = np.where(np.abs(dmu) < 1e-12, 1e-12, dmu)
        irls_w = w * dmu**2 / var
        z = eta - off + (y - mu) / dmu
        info = X.T @ (X * irls_w[:, None])
        try:
            new_coef = np.linalg.solve(info, X.T @ (irls_w * z))
        except np.linalg.LinAlgError:
            return nan_fit
        if not np.all(np.isfinite(new_coef)):
            return nan_fit
        step = np.max(np.abs(new_coef - coef))
        coef = new_coef
        eta = X @ coef + off
        if step < tol * (1.0 + np.max(np.abs(coef))):
            converged = True
            break
    mu, dmu = _link_parts(eta, link)
    var = np.clip(mu * (1.0 - mu), 1e-12, None)
    info = X.T @ (X * (w * dmu**2 / var)[:, None])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return GLMFit(coef, cov, converged, it, link)


def rcs_knots(x: np.ndarray, n_knots: int = 4) -> np.ndarray:
    """Knot locations for a restricted cubic spline at Harrell's recommended quantiles.

    Args:
        x: Covariate values.
        n_knots: Number of knots (3 to 7). Fewer knots are returned when ``x`` has too few
            distinct values.

    Returns:
        Sorted unique knot locations.
    """
    quantiles = {
        3: (0.10, 0.50, 0.90),
        4: (0.05, 0.35, 0.65, 0.95),
        5: (0.05, 0.275, 0.50, 0.725, 0.95),
        6: (0.05, 0.23, 0.41, 0.59, 0.77, 0.95),
        7: (0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975),
    }
    if n_knots not in quantiles:
        raise ValueError("n_knots must be between 3 and 7")
    x = as_1d_float(x)
    x = x[np.isfinite(x)]
    return np.unique(np.quantile(x, quantiles[n_knots])) if x.size else np.array([])


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, linear beyond the boundary knots.

    Args:
        x: Covariate values of shape ``(n,)``.
        knots: Sorted knot locations; with fewer than three knots only ``x`` is returned.

    Returns:
        Basis matrix of shape ``(n, len(knots) - 1)`` whose first column is ``x``.
    """
    x = as_1d_float(x)
    knots = np.asarray(knots, dtype=np.float64)
    k = knots.size
    if k < 3:
        return x[:, None]
    t_last, t_prev = knots[-1], knots[-2]
    scale = (t_last - knots[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        tj = knots[j]
        term = (
            np.clip(x - tj, 0, None) ** 3
            - np.clip(x - t_prev, 0, None) ** 3 * (t_last - tj) / (t_last - t_prev)
            + np.clip(x - t_last, 0, None) ** 3 * (t_prev - tj) / (t_last - t_prev)
        )
        cols.append(term / scale)
    return np.column_stack(cols)
