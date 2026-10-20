"""Recalibration, operating thresholds, selective prediction and conformal prediction.

The functions in this module operate on plain NumPy arrays and are shared by the neural
model and the classical comparators so that every model is post-processed identically.

* :class:`LogisticRecalibrator` - two-parameter (intercept and slope) logistic
  recalibration of binary probabilities on the logit scale.
* :class:`KaplanMeierCensoring` and :func:`ipcw_weights` - inverse probability of
  censoring weights at a fixed horizon from a Kaplan-Meier fit of the censoring
  distribution.
* :class:`CloglogRecalibrator` - recalibration of a predicted survival curve through
  ``cloglog{1 - S*(t)} = a(t) + b * cloglog{1 - S(t)}``: the slope ``b`` is estimated at one
  horizon with IPCW weights; the intercept is either that horizon's (single map) or
  re-estimated at every reported horizon from IPCW pseudo-observations and interpolated.
* :func:`sensitivity_threshold` - largest cut-off that attains a target sensitivity.
* :func:`combine_uncertainty`, :func:`abstention_cutoff` - selective prediction at a target
  coverage from ensemble spread and Monte Carlo dropout.
* :func:`mondrian_conformal_pvalues` - class-conditional (Mondrian) cross-conformal
  p-values and prediction sets for a binary endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import optimize

__all__ = [
    "EPS",
    "CloglogRecalibrator",
    "KaplanMeierCensoring",
    "LogisticRecalibrator",
    "abstain",
    "abstention_cutoff",
    "binary_log_loss",
    "cloglog",
    "combine_uncertainty",
    "conformal_sets",
    "cross_fitted_log_loss",
    "format_conformal_set",
    "horizon_status",
    "inv_cloglog",
    "ipcw_pseudo_observations",
    "ipcw_weights",
    "logit",
    "mondrian_conformal_pvalues",
    "sensitivity_threshold",
    "sigmoid",
]

EPS = 1e-6
"""Probabilities are clipped to ``[EPS, 1 - EPS]`` before any logit/cloglog transform."""


def _as_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def logit(p: Any) -> np.ndarray:
    """Log-odds of ``p`` (clipped to ``[EPS, 1 - EPS]``)."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: Any) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def cloglog(r: Any) -> np.ndarray:
    """Complementary log-log transform ``log(-log(1 - r))`` of a risk ``r``."""
    r = np.clip(np.asarray(r, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(-np.log1p(-r))


def inv_cloglog(eta: Any) -> np.ndarray:
    """Inverse of :func:`cloglog`: ``1 - exp(-exp(eta))``."""
    eta = np.clip(np.asarray(eta, dtype=np.float64), -30.0, 30.0)
    return -np.expm1(-np.exp(eta))


def binary_log_loss(p: Any, y: Any, sample_weight: Any | None = None) -> float:
    """Mean (optionally weighted) binary cross-entropy over finite labels."""
    p, y = _as_float(p), _as_float(y)
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    w = np.ones_like(p) if sample_weight is None else _as_float(sample_weight)
    pc = np.clip(p[ok], EPS, 1.0 - EPS)
    ll = -(y[ok] * np.log(pc) + (1.0 - y[ok]) * np.log1p(-pc))
    return float(np.sum(w[ok] * ll) / max(np.sum(w[ok]), EPS))


# --------------------------------------------------------------------------------------
# Binary recalibration
# --------------------------------------------------------------------------------------


@dataclass
class LogisticRecalibrator:
    """Two-parameter logistic recalibration ``p* = sigmoid(a + b * logit(p))``.

    The parameters are maximum-likelihood estimates obtained by Newton-Raphson with a
    negligible ridge term for numerical stability. When the calibration data contain fewer
    than ``min_per_class`` observations of either class the map stays the identity.

    Attributes:
        intercept: Calibration intercept ``a``.
        slope: Calibration slope ``b``.
        fitted: Whether :meth:`fit` estimated non-identity parameters.
        n: Number of observations used in the fit.
    """

    intercept: float = 0.0
    slope: float = 1.0
    fitted: bool = False
    n: int = 0
    min_per_class: int = 2

    def fit(
        self,
        p: Any,
        y: Any,
        sample_weight: Any | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-9,
        ridge: float = 1e-6,
    ) -> LogisticRecalibrator:
        """Estimate intercept and slope from probabilities ``p`` and binary labels ``y``."""
        p, y = _as_float(p), _as_float(y)
        w = np.ones_like(p) if sample_weight is None else _as_float(sample_weight)
        ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x, y, w = logit(p[ok]), y[ok], w[ok]
        self.n = int(ok.sum())
        self.intercept, self.slope, self.fitted = 0.0, 1.0, False
        if min(int((y == 1).sum()), int((y == 0).sum())) < self.min_per_class:
            return self
        design = np.column_stack([np.ones_like(x), x])
        prior = np.array([0.0, 1.0])
        beta = prior.copy()

        def objective(b: np.ndarray) -> float:
            eta = design @ b
            ll = y * eta - np.logaddexp(0.0, eta)
            return float(-np.sum(w * ll) + 0.5 * ridge * np.sum((b - prior) ** 2))

        current = objective(beta)
        for _ in range(max_iter):
            mu = sigmoid(design @ beta)
            grad = design.T @ (w * (y - mu)) - ridge * (beta - prior)
            hess = (design * (w * mu * (1.0 - mu))[:, None]).T @ design + ridge * np.eye(2)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            while scale > 1e-6:
                candidate = beta + scale * step
                value = objective(candidate)
                if np.isfinite(value) and value <= current + 1e-12:
                    break
                scale *= 0.5
            else:
                break
            beta, current = candidate, value
            if np.max(np.abs(scale * step)) < tol:
                break
        if np.all(np.isfinite(beta)):
            self.intercept, self.slope, self.fitted = float(beta[0]), float(beta[1]), True
        return self

    def transform(self, p: Any) -> np.ndarray:
        """Apply the recalibration map; non-finite inputs stay non-finite."""
        p = np.asarray(p, dtype=np.float64)
        out = sigmoid(self.intercept + self.slope * logit(p))
        return np.where(np.isfinite(p), out, np.nan)

    def get_params(self) -> dict[str, float]:
        """Fitted ``intercept`` and ``slope``."""
        return {"intercept": self.intercept, "slope": self.slope}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticRecalibrator:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def cross_fitted_log_loss(
    p: Any, y: Any, folds: Any, *, recalibrate: bool = True
) -> tuple[float, np.ndarray]:
    """Log-loss of out-of-fold probabilities after cross-fitted logistic recalibration.

    For every fold ``f`` a :class:`LogisticRecalibrator` is fitted on the labelled
    predictions of the other folds and applied to fold ``f``; the log-loss is pooled over all
    labelled predictions. With a single fold the map is fitted in-sample.

    Args:
        p: Out-of-fold probabilities.
        y: Binary labels (NaN = unlabelled, ignored).
        folds: Fold index of every prediction.
        recalibrate: If ``False`` the raw probabilities are scored.

    Returns:
        ``(log_loss, recalibrated_probabilities)``.
    """
    p, y = _as_float(p), _as_float(y)
    folds = np.asarray(folds).reshape(-1)
    out = p.copy()
    if recalibrate:
        unique = np.unique(folds)
        for f in unique:
            test = folds == f
            train = ~test if len(unique) > 1 else np.ones_like(test)
            cal = LogisticRecalibrator().fit(p[train], y[train])
            out[test] = cal.transform(p[test])
    return binary_log_loss(out, y), out


# --------------------------------------------------------------------------------------
# Censoring weights
# --------------------------------------------------------------------------------------


@dataclass
class KaplanMeierCensoring:
    """Kaplan-Meier estimate ``G(t) = P(C > t)`` of the censoring distribution.

    Ties between events and censorings are resolved with events first, i.e. a participant
    with an event at ``t`` is not counted as censored at ``t``.

    Attributes:
        times: Distinct observed times (ascending).
        survival: ``G`` evaluated at ``times`` (right-continuous step function).
        min_prob: Lower bound applied to ``G`` when used as a weight denominator.
    """

    times: np.ndarray
    survival: np.ndarray
    min_prob: float = 0.05

    @classmethod
    def fit(cls, time: Any, event: Any, *, min_prob: float = 0.05) -> KaplanMeierCensoring:
        """Fit on observed times and event indicators (1 = event, 0 = censored)."""
        time, event = _as_float(time), _as_float(event)
        ok = np.isfinite(time) & np.isfinite(event)
        time, censored = time[ok], 1.0 - event[ok]
        if time.size == 0:
            return cls(np.zeros(0), np.zeros(0), min_prob)
        uniq, inverse = np.unique(time, return_inverse=True)
        at_risk = time.size - np.searchsorted(np.sort(time), uniq, side="left")
        n_cens = np.bincount(inverse, weights=censored, minlength=uniq.size)
        factors = 1.0 - n_cens / np.maximum(at_risk, 1)
        return cls(uniq, np.cumprod(factors), min_prob)

    def __call__(self, t: Any, *, left: bool = False, clip: bool = True) -> np.ndarray:
        """Evaluate ``G(t)`` (or the left limit ``G(t-)`` when ``left``)."""
        t = np.asarray(t, dtype=np.float64)
        if self.times.size == 0:
            return np.ones_like(t)
        side = "left" if left else "right"
        idx = np.searchsorted(self.times, t, side=side) - 1
        values = np.where(idx >= 0, self.survival[np.clip(idx, 0, None)], 1.0)
        return np.maximum(values, self.min_prob) if clip else values

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "survival": self.survival.tolist(),
            "min_prob": self.min_prob,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KaplanMeierCensoring:
        return cls(
            np.asarray(data["times"], dtype=np.float64),
            np.asarray(data["survival"], dtype=np.float64),
            float(data.get("min_prob", 0.05)),
        )


def horizon_status(time: Any, event: Any, horizon: float) -> np.ndarray:
    """Binary status at ``horizon``: 1 = event by horizon, 0 = event-free beyond it, NaN = censored before."""
    time, event = _as_float(time), _as_float(event)
    out = np.full(time.shape, np.nan)
    out[(time <= horizon) & (event == 1)] = 1.0
    out[time > horizon] = 0.0
    return out


def ipcw_weights(
    time: Any, event: Any, horizon: float, censoring: KaplanMeierCensoring
) -> np.ndarray:
    """Inverse probability of censoring weights for the binary status at ``horizon``.

    Cases (event at ``T <= horizon``) receive ``1 / G(T-)``, controls (``T > horizon``)
    receive ``1 / G(horizon)`` and participants censored before the horizon receive 0.
    """
    time, event = _as_float(time), _as_float(event)
    w = np.zeros(time.shape)
    case = np.isfinite(time) & (time <= horizon) & (event == 1)
    control = np.isfinite(time) & (time > horizon)
    if case.any():
        w[case] = 1.0 / censoring(time[case], left=True)
    if control.any():
        w[control] = 1.0 / censoring(np.array([horizon]))[0]
    return w


# --------------------------------------------------------------------------------------
# Survival recalibration
# --------------------------------------------------------------------------------------


def ipcw_pseudo_observations(
    time: Any, event: Any, horizon: float, censoring: KaplanMeierCensoring
) -> np.ndarray:
    """IPCW pseudo-observations of the event indicator by ``horizon``.

    ``Y*_i = 1{T_i <= horizon, event_i = 1} / G(T_i-)`` for every participant (0 for event-free
    and early-censored participants). Their mean over all participants estimates the marginal
    cumulative incidence ``P(T <= horizon)``, and ``E[Y*_i | x_i] = F(horizon | x_i)`` under
    independent censoring, so they can stand in for the unobserved horizon status.
    """
    time, event = _as_float(time), _as_float(event)
    out = np.zeros(time.shape)
    case = np.isfinite(time) & (time <= horizon) & (event == 1)
    if case.any():
        out[case] = 1.0 / censoring(time[case], left=True)
    return np.where(np.isfinite(time) & np.isfinite(event), out, np.nan)


@dataclass
class CloglogRecalibrator:
    """Recalibrate survival curves via ``cloglog{1-S*(t)} = a(t) + b * cloglog{1-S(t)}``.

    ``a`` and ``b`` are estimated at the reference ``horizon`` by IPCW-weighted maximum
    likelihood of the binary horizon status under a complementary log-log link. The slope is
    constrained to be positive so that the map is increasing in the predicted risk.

    *Single map* (``knots`` empty): ``a(t) = a`` at every time point of the curve.

    *Horizon intercepts* (:meth:`fit` with ``horizon_risk``): the slope ``b`` is kept and the
    intercept is re-estimated at every reported horizon ``h`` (e.g. 12, 24 and 36 months) so
    that calibration-in-the-large holds there,
    ``mean_i inv_cloglog(a_h + b * cloglog r_i(h)) = mean_i Y*_i(h)``, with IPCW
    pseudo-observations ``Y*`` (:func:`ipcw_pseudo_observations`). Between horizons ``a(t)`` is
    interpolated linearly, outside them it is held constant, and the recalibrated survival
    curve is made non-increasing in time (running minimum), since a time-varying intercept
    alone does not guarantee monotone curves.
    """

    intercept: float = 0.0
    slope: float = 1.0
    horizon: float = 24.0
    fitted: bool = False
    n: int = 0
    min_per_class: int = 2
    knots: list[float] = field(default_factory=list)
    knot_intercepts: list[float] = field(default_factory=list)

    def fit(
        self,
        risk: Any,
        time: Any,
        event: Any,
        *,
        weights: Any | None = None,
        censoring: KaplanMeierCensoring | None = None,
        horizon_risk: Mapping[float, Any] | None = None,
    ) -> CloglogRecalibrator:
        """Fit on predicted horizon risks and observed follow-up.

        Args:
            risk: Predicted ``1 - S(horizon)`` for every participant.
            time: Observed time (same unit as ``horizon``).
            event: Event indicator.
            weights: Optional precomputed IPCW weights; otherwise computed from
                ``censoring`` (fitted on the same data when ``None``).
            censoring: Censoring distribution used to derive the weights and the
                pseudo-observations.
            horizon_risk: Optional ``{horizon: predicted risk at that horizon}``; when given,
                the intercept is re-estimated at every horizon with the slope fixed (see the
                class docstring). Omitted, the single map is fitted.
        """
        risk, time, event = _as_float(risk), _as_float(time), _as_float(event)
        if weights is None:
            if censoring is None:
                censoring = KaplanMeierCensoring.fit(time, event)
            weights = ipcw_weights(time, event, self.horizon, censoring)
        w = _as_float(weights)
        y = horizon_status(time, event, self.horizon)
        ok = np.isfinite(risk) & np.isfinite(y) & (w > 0)
        self.n = int(ok.sum())
        self.intercept, self.slope, self.fitted = 0.0, 1.0, False
        self.knots, self.knot_intercepts = [], []
        if min(int((y[ok] == 1).sum()), int((y[ok] == 0).sum())) < self.min_per_class:
            return self
        x, y, w = cloglog(risk[ok]), y[ok], w[ok]

        def nll(params: np.ndarray) -> float:
            f = np.clip(inv_cloglog(params[0] + params[1] * x), EPS, 1.0 - EPS)
            return float(-np.sum(w * (y * np.log(f) + (1.0 - y) * np.log1p(-f))) / np.sum(w))

        res = optimize.minimize(
            nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-10.0, 10.0), (0.05, 20.0)]
        )
        if res.success or np.isfinite(res.fun):
            self.intercept, self.slope, self.fitted = float(res.x[0]), float(res.x[1]), True
        if horizon_risk and self.fitted:
            if censoring is None:
                censoring = KaplanMeierCensoring.fit(time, event)
            self._fit_horizon_intercepts(horizon_risk, time, event, censoring)
        return self

    def _fit_horizon_intercepts(
        self,
        horizon_risk: Mapping[float, Any],
        time: np.ndarray,
        event: np.ndarray,
        censoring: KaplanMeierCensoring,
    ) -> None:
        """Calibration-in-the-large intercept at every horizon with the slope held fixed."""
        knots, values = [], []
        for h in sorted(float(k) for k in horizon_risk):
            r = _as_float(horizon_risk[h])
            pseudo = ipcw_pseudo_observations(time, event, h, censoring)
            ok = np.isfinite(r) & np.isfinite(pseudo)
            if ok.sum() < 2 * self.min_per_class or not (pseudo[ok] > 0).any():
                continue
            x, target = cloglog(r[ok]), float(np.clip(np.mean(pseudo[ok]), EPS, 1.0 - EPS))

            def gap(a: float, x: np.ndarray = x, target: float = target) -> float:
                return float(np.mean(inv_cloglog(a + self.slope * x)) - target)

            lo, hi = -15.0, 15.0
            if gap(lo) > 0 or gap(hi) < 0:
                continue
            knots.append(h)
            values.append(float(optimize.brentq(gap, lo, hi, xtol=1e-10)))
        self.knots, self.knot_intercepts = knots, values

    def intercept_at(self, times: Any) -> np.ndarray:
        """Intercept ``a(t)`` at ``times`` (constant ``intercept`` for a single map)."""
        times = np.asarray(times, dtype=np.float64)
        if not self.knots:
            return np.full(times.shape, self.intercept)
        return np.interp(times, self.knots, self.knot_intercepts)

    def transform_risk(self, risk: Any, times: Any | None = None) -> np.ndarray:
        """Recalibrate risks ``1 - S(t)`` (any shape).

        Args:
            risk: Predicted risks; with ``times`` the last axis indexes the time points.
            times: Time of every entry of the last axis of ``risk`` (enables the horizon
                intercepts and the monotonicity constraint); ``None`` applies the reference
                map ``a`` point-wise.
        """
        risk = np.asarray(risk, dtype=np.float64)
        if times is None or not self.knots:
            out = inv_cloglog(self.intercept + self.slope * cloglog(risk))
        else:
            out = inv_cloglog(self.intercept_at(times) + self.slope * cloglog(risk))
            order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
            ranked = np.take(np.where(np.isfinite(out), out, -np.inf), order, axis=-1)
            ranked = np.maximum.accumulate(ranked, axis=-1)
            back = np.empty_like(order)
            back[order] = np.arange(order.size)
            out = np.take(ranked, back, axis=-1)
        return np.where(np.isfinite(risk), out, np.nan)

    def transform(self, risk: Any, times: Any | None = None) -> np.ndarray:
        """Alias of :meth:`transform_risk`."""
        return self.transform_risk(risk, times)

    def transform_survival(self, survival: Any, times: Any | None = None) -> np.ndarray:
        """Recalibrate a survival curve array ``S`` (see :meth:`transform_risk`).

        With ``times`` (the time of every column, e.g. the bin boundaries) the horizon
        intercepts are interpolated to each column and the curve is made non-increasing.
        """
        survival = np.asarray(survival, dtype=np.float64)
        return 1.0 - self.transform_risk(1.0 - survival, times)

    def get_params(self) -> dict[str, Any]:
        """Fitted ``intercept``, ``slope``, the estimation ``horizon`` and horizon intercepts."""
        out: dict[str, Any] = {
            "intercept": self.intercept,
            "slope": self.slope,
            "horizon": self.horizon,
        }
        for h, a in zip(self.knots, self.knot_intercepts):
            out[f"intercept_{h:g}m"] = a
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloglogRecalibrator:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# --------------------------------------------------------------------------------------
# Operating points and selective prediction
# --------------------------------------------------------------------------------------


def sensitivity_threshold(p: Any, y: Any, target: float = 0.90) -> float:
    """Largest cut-off ``t`` such that ``P(p >= t | y = 1) >= target``.

    Returns ``nan`` when there are no positive labels.
    """
    p, y = _as_float(p), _as_float(y)
    pos = np.sort(p[(y == 1) & np.isfinite(p)])[::-1]
    if pos.size == 0:
        return float("nan")
    k = max(1, math.ceil(target * pos.size - 1e-9))
    return float(pos[min(k, pos.size) - 1])


def combine_uncertainty(ensemble_sd: Any, mc_sd: Any | None = None) -> np.ndarray:
    """Total predictive spread ``sqrt(ensemble_sd**2 + mc_sd**2)`` (missing terms count as 0)."""
    a = np.nan_to_num(np.asarray(ensemble_sd, dtype=np.float64), nan=0.0)
    if mc_sd is None:
        return np.abs(a)
    b = np.nan_to_num(np.asarray(mc_sd, dtype=np.float64), nan=0.0)
    return np.sqrt(a**2 + b**2)


def abstention_cutoff(uncertainty: Any, coverage: float = 0.90) -> float:
    """Smallest cut-off ``c`` with ``P(u <= c) >= coverage`` on the reference predictions."""
    u = _as_float(uncertainty)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float("inf")
    return float(np.quantile(u, coverage, method="inverted_cdf"))


def abstain(uncertainty: Any, cutoff: float, flags: Any | None = None) -> np.ndarray:
    """Boolean abstention decision: uncertainty above ``cutoff`` or a raised quality flag."""
    u = _as_float(uncertainty)
    out = ~(u <= cutoff)
    if flags is not None:
        out |= np.asarray(flags, dtype=bool).reshape(-1)
    return out


# --------------------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------------------


def mondrian_conformal_pvalues(
    calibration: Sequence[tuple[Any, Any]],
    test_probs: Sequence[Any],
) -> np.ndarray:
    """Class-conditional (Mondrian) cross-conformal p-values for a binary endpoint.

    With nonconformity score ``s_c(x) = 1 - p_c(x)``, the class-``c`` p-value of a test
    point pools the calibration folds,

    ``p_c(x) = (1 + sum_k #{i in V_k : y_i = c, s_i^(k) >= s_c^(k)(x)}) / (n_c + 1)``,

    where fold ``k`` is scored by its own model. A single calibration fold gives the usual
    split-conformal p-value.

    Args:
        calibration: One ``(p_positive, y)`` pair per calibration fold, scored by the model of
            that fold (unlabelled entries are ignored).
        test_probs: For every fold, the positive-class probabilities that the same model
            assigns to the test points (equal lengths across folds).

    Returns:
        Array ``(n_test, 2)`` with the p-values of class 0 and class 1.
    """
    if len(calibration) != len(test_probs):
        raise ValueError("calibration and test_probs must have one entry per fold")
    n_test = len(np.asarray(test_probs[0]).reshape(-1)) if test_probs else 0
    counts = np.zeros((n_test, 2))
    n_class = np.zeros(2)
    for (p_cal, y_cal), p_test in zip(calibration, test_probs):
        p_cal, y_cal, p_test = _as_float(p_cal), _as_float(y_cal), _as_float(p_test)
        ok = np.isfinite(p_cal) & np.isfinite(y_cal)
        for c in (0, 1):
            pc_cal = p_cal[ok & (y_cal == c)]
            pc_cal = pc_cal if c == 1 else 1.0 - pc_cal
            scores = np.sort(1.0 - pc_cal)
            pt = p_test if c == 1 else 1.0 - p_test
            s_test = 1.0 - pt
            counts[:, c] += scores.size - np.searchsorted(scores, s_test - 1e-12, side="left")
            n_class[c] += scores.size
    pvals = (1.0 + counts) / (n_class[None, :] + 1.0)
    return np.clip(pvals, 0.0, 1.0)


def conformal_sets(pvalues: Any, alpha: float = 0.05) -> np.ndarray:
    """Boolean membership ``(n, 2)`` of each class in the prediction set ``{c : p_c > alpha}``."""
    return np.asarray(pvalues, dtype=np.float64) > alpha


def format_conformal_set(members: Any) -> list[str]:
    """Render boolean set memberships as ``"{}"``, ``"{0}"``, ``"{1}"`` or ``"{0,1}"``."""
    out = []
    for row in np.asarray(members, dtype=bool).reshape(-1, 2):
        labels = [str(c) for c in (0, 1) if row[c]]
        out.append("{" + ",".join(labels) + "}")
    return out
