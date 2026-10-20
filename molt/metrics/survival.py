"""Time-to-event evaluation metrics with inverse-probability-of-censoring weighting.

Conventions:

* ``time`` is the observed follow-up time (any unit, months throughout MOLT).
* For single-risk functions ``event`` is 1 for an event and 0 for censoring.
* For competing-risk functions ``event`` is 0 for censoring, ``cause`` (default 1) for the
  event of interest and any other positive code for a competing event.
* ``risk`` is the predicted cumulative risk ``F(horizon | x) = 1 - S(horizon | x)`` when a
  horizon is involved; concordance indices accept any score where higher means higher risk.
* ``censoring`` selects the censoring survival function ``G`` used for IPCW: ``None``
  estimates it by Kaplan-Meier on the evaluation data (so it is re-estimated inside every
  bootstrap resample), a ``(time, event)`` pair estimates it on those data, and a
  :class:`StepFunction` is used as given.

Event weights use the left limit ``G(T_i-)``; participants still at risk at the horizon use
``G(horizon)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from scipy import stats

from molt.metrics._glm import (
    as_1d_float,
    cloglog,
    fit_binomial_glm,
    rcs_basis,
    rcs_knots,
)
from molt.metrics.binary import weighted_auroc, weighted_average_precision

__all__ = [
    "StepFunction",
    "aalen_johansen",
    "brier_score",
    "brier_scores",
    "censoring_survival",
    "competing_risk_auroc",
    "decision_curve_survival",
    "harrell_c",
    "horizon_status",
    "integrated_brier_score",
    "kaplan_meier",
    "km_risk_by_stratum",
    "net_benefit_crude",
    "net_benefit_survival",
    "pseudo_observations",
    "survival_calibration",
    "survival_tier",
    "time_dependent_auprc",
    "time_dependent_auroc",
    "uno_c",
    "wolbers_concordance",
]


@dataclass(frozen=True)
class StepFunction:
    """Right-continuous step function equal to 1 before the first jump time.

    Attributes:
        times: Sorted unique jump times.
        values: Function value from each jump time onwards.
        n_risk: Number at risk at each time (Kaplan-Meier fits only).
        n_event: Number of events at each time (Kaplan-Meier fits only).
    """

    times: np.ndarray
    values: np.ndarray
    n_risk: np.ndarray = field(default_factory=lambda: np.array([]))
    n_event: np.ndarray = field(default_factory=lambda: np.array([]))
    initial: float = 1.0

    def __call__(self, t: object, left: bool = False) -> np.ndarray:
        """Evaluate at ``t``; ``left=True`` returns the left limit ``f(t-)``."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.times, t, side="left" if left else "right") - 1
        safe = np.clip(idx, 0, max(self.values.size - 1, 0))
        vals = self.values[safe] if self.values.size else np.full(t.shape, self.initial)
        return np.where(idx >= 0, vals, self.initial)


Censoring = StepFunction | tuple | None


def _clean(*arrays: object) -> list[np.ndarray]:
    """Convert to float arrays of equal length and drop rows with any non-finite value."""
    arrs = [as_1d_float(a) for a in arrays]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("all inputs must have the same length")
    keep = np.ones(n, dtype=bool)
    for a in arrs:
        keep &= np.isfinite(a)
    return [a[keep] for a in arrs]


def kaplan_meier(time: object, event: object) -> StepFunction:
    """Kaplan-Meier estimate of the survival function.

    Args:
        time: Follow-up times.
        event: Event indicator (non-zero = event).

    Returns:
        :class:`StepFunction` with numbers at risk and events at every distinct time.
    """
    time, event = _clean(time, event)
    if time.size == 0:
        return StepFunction(np.array([]), np.array([]))
    uniq, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    d = np.bincount(inverse, weights=(event != 0).astype(float), minlength=uniq.size)
    n_risk = time.size - np.cumsum(counts) + counts
    surv = np.cumprod(1.0 - d / n_risk)
    return StepFunction(uniq, surv, n_risk.astype(float), d)


def censoring_survival(time: object, event: object) -> StepFunction:
    """Kaplan-Meier estimate of the censoring survival function ``G`` (censoring = event)."""
    time, event = _clean(time, event)
    return kaplan_meier(time, (event == 0).astype(float))


def _resolve_censoring(censoring: Censoring, time: np.ndarray, event: np.ndarray) -> StepFunction:
    if censoring is None:
        return censoring_survival(time, event)
    if isinstance(censoring, StepFunction):
        return censoring
    c_time, c_event = censoring
    return censoring_survival(c_time, c_event)


def _inv(g: np.ndarray) -> np.ndarray:
    """Safe reciprocal of censoring survival probabilities (0 where ``g == 0``)."""
    g = np.asarray(g, dtype=np.float64)
    out = np.zeros_like(g)
    np.divide(1.0, g, out=out, where=g > 0)
    return out


def _horizon_weights(
    time: np.ndarray, event: np.ndarray, horizon: float, G: StepFunction, cause: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Case mask and IPCW case/control weights at a horizon.

    Controls are participants event-free at the horizon and, in the competing-risk setting,
    participants with a competing event by the horizon.
    """
    if cause is None:
        case = (time <= horizon) & (event != 0)
        comp = np.zeros_like(case)
    else:
        case = (time <= horizon) & (event == cause)
        comp = (time <= horizon) & (event != 0) & (event != cause)
    alive = time > horizon
    g_left = _inv(G(time, left=True))
    w_case = np.where(case, g_left, 0.0)
    w_ctrl = np.where(alive, _inv(G(horizon)), 0.0) + np.where(comp, g_left, 0.0)
    return case, w_case, w_ctrl


def horizon_status(time: object, event: object, horizon: float) -> np.ndarray:
    """Binary horizon status: 1 = event by the horizon, 0 = event-free beyond it, NaN otherwise."""
    time = as_1d_float(time)
    event = as_1d_float(event)
    out = np.full(time.size, np.nan)
    out[(time <= horizon) & (event != 0)] = 1.0
    out[time > horizon] = 0.0
    return out


def survival_tier(time: object, event: object, horizon: float) -> str:
    """Reporting tier at a horizon (events by the horizon vs participants event-free through it)."""
    from molt.metrics.binary import reporting_tier

    status = horizon_status(time, event, horizon)
    return reporting_tier(int(np.nansum(status == 1)), int(np.sum(status == 0)))


# --------------------------------------------------------------------------- discrimination


def time_dependent_auroc(
    time: object, event: object, risk: object, horizon: float, censoring: Censoring = None
) -> float:
    """Cumulative/dynamic time-dependent AUROC with IPCW (Uno 2007; Blanche 2013).

    Cases have an event by ``horizon`` and are weighted by ``1 / G(T_i-)``; controls are
    event-free beyond ``horizon`` and share weight ``1 / G(horizon)``. Participants censored
    before the horizon contribute only through ``G``. Without censoring this equals the
    binary AUROC for the label ``T <= horizon``.

    Args:
        time: Follow-up times.
        event: Event indicator (0/1).
        risk: Predicted risk by the horizon (or any monotone risk score).
        horizon: Evaluation horizon.
        censoring: Censoring-distribution specification (see module docstring).

    Returns:
        AUROC at the horizon.
    """
    time, event, risk = _clean(time, event, risk)
    G = _resolve_censoring(censoring, time, event)
    case, w_case, w_ctrl = _horizon_weights(time, event, horizon, G, None)
    return weighted_auroc(risk, case, w_case, w_ctrl)


def competing_risk_auroc(
    time: object,
    event: object,
    risk: object,
    horizon: float,
    cause: int = 1,
    censoring: Censoring = None,
) -> float:
    """Cause-specific cumulative/dynamic AUROC under competing risks (Blanche 2013, def. 2).

    Cases have the event of interest by the horizon; controls are participants event-free at
    the horizon and participants with a competing event by the horizon, each weighted by
    the inverse censoring probability.

    Args:
        time: Follow-up times.
        event: 0 = censored, ``cause`` = event of interest, other positive = competing event.
        risk: Predicted cumulative incidence of ``cause`` by the horizon.
        horizon: Evaluation horizon.
        cause: Code of the event of interest.
        censoring: Censoring-distribution specification.

    Returns:
        Competing-risk AUROC at the horizon.
    """
    time, event, risk = _clean(time, event, risk)
    G = _resolve_censoring(censoring, time, event)
    case, w_case, w_ctrl = _horizon_weights(time, event, horizon, G, cause)
    return weighted_auroc(risk, case, w_case, w_ctrl)


def time_dependent_auprc(
    time: object, event: object, risk: object, horizon: float, censoring: Censoring = None
) -> float:
    """Censoring-adjusted area under the precision-recall curve at a horizon.

    Precision and recall are computed with the same IPCW case and control weights as
    :func:`time_dependent_auroc`, and summarized as weighted average precision.
    """
    time, event, risk = _clean(time, event, risk)
    G = _resolve_censoring(censoring, time, event)
    case, w_case, w_ctrl = _horizon_weights(time, event, horizon, G, None)
    return weighted_average_precision(risk, case, w_case + w_ctrl)


def _structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    out = np.empty(time.size, dtype=[("event", bool), ("time", np.float64)])
    out["event"] = event != 0
    out["time"] = time
    return out


def harrell_c(time: object, event: object, risk: object) -> float:
    """Harrell's concordance index (``sksurv.metrics.concordance_index_censored``)."""
    from sksurv.metrics import concordance_index_censored

    time, event, risk = _clean(time, event, risk)
    if not np.any(event != 0):
        return float("nan")
    return float(concordance_index_censored(event != 0, time, risk)[0])


def wolbers_concordance(
    time: object,
    event: object,
    risk: object,
    cause: int = 1,
    tau: float | None = None,
    censoring: Censoring = None,
    ipcw: bool = True,
    chunk: int = 512,
) -> float:
    """Concordance index for the event of interest under competing risks (Wolbers 2014).

    A participant ``i`` with the event of interest at ``T_i <= tau`` is compared with every
    participant ``j`` who either has ``T_j > T_i`` or a competing event at ``T_j <= T_i``. With
    ``ipcw=True`` pairs are weighted by ``1 / (G(T_i-) G(T_i))`` and ``1 / (G(T_i-) G(T_j-))``
    respectively. Without competing events this is Uno's truncated concordance.

    Args:
        time: Follow-up times.
        event: 0 = censored, ``cause`` = event of interest, other positive = competing event.
        risk: Risk score for ``cause`` (higher = earlier event).
        cause: Code of the event of interest.
        tau: Truncation time (``None`` = no truncation).
        censoring: Censoring-distribution specification.
        ipcw: Apply inverse-probability-of-censoring weights.
        chunk: Number of cases processed per vectorized block.

    Returns:
        The concordance index.
    """
    time, event, risk = _clean(time, event, risk)
    tau = np.inf if tau is None else float(tau)
    if ipcw:
        G = _resolve_censoring(censoring, time, event)
        g_left = G(time, left=True)
        g_right = G(time)
    else:
        g_left = g_right = np.ones_like(time)
    comp = (event != 0) & (event != cause)
    cases = np.flatnonzero((event == cause) & (time <= tau) & (g_left > 0) & (g_right > 0))
    inv_gl_all = _inv(g_left)
    num = den = 0.0
    for start in range(0, cases.size, chunk):
        idx = cases[start : start + chunk]
        ti, ri = time[idx, None], risk[idx, None]
        a = time[None, :] > ti
        b = (time[None, :] <= ti) & comp[None, :]
        w = np.where(a, (1.0 / (g_left[idx] * g_right[idx]))[:, None], 0.0)
        w = w + np.where(b, inv_gl_all[None, :] / g_left[idx, None], 0.0)
        conc = (ri > risk[None, :]) + 0.5 * (ri == risk[None, :])
        num += float(np.sum(w * conc))
        den += float(np.sum(w))
    return num / den if den > 0 else float("nan")


def uno_c(
    time: object, event: object, risk: object, tau: float, censoring: Censoring = None
) -> float:
    """Uno's IPCW concordance truncated at ``tau`` (``sksurv.metrics.concordance_index_ipcw``).

    The censoring distribution is estimated on ``censoring`` data when a ``(time, event)``
    pair is supplied and on the evaluation data otherwise. When scikit-survival cannot
    evaluate the weights (for example a zero censoring survival probability at an event time
    in a bootstrap resample) the equivalent single-risk :func:`wolbers_concordance` is used.
    """
    from sksurv.metrics import concordance_index_ipcw

    time, event, risk = _clean(time, event, risk)
    if not np.any((event != 0) & (time < tau)):
        return float("nan")
    test = _structured(time, event)
    if isinstance(censoring, StepFunction):
        return wolbers_concordance(time, (event != 0).astype(float), risk, 1, tau, censoring)
    if censoring is None:
        train = test
    else:
        c_time, c_event = _clean(*censoring)
        train = _structured(c_time, c_event)
    try:
        return float(concordance_index_ipcw(train, test, risk, tau=tau)[0])
    except ValueError:
        G = _resolve_censoring(censoring, time, event)
        return wolbers_concordance(time, (event != 0).astype(float), risk, 1, tau, G)


# ------------------------------------------------------------------------------ Brier scores


def brier_score(
    time: object, event: object, risk: object, horizon: float, censoring: Censoring = None
) -> float:
    """IPCW Brier score at a horizon (Graf 1999) for predicted risks ``F(horizon | x)``."""
    time, event, risk = _clean(time, event, risk)
    G = _resolve_censoring(censoring, time, event)
    _, w_case, w_ctrl = _horizon_weights(time, event, horizon, G, None)
    return float(np.mean(w_case * (1.0 - risk) ** 2 + w_ctrl * risk**2))


def brier_scores(
    time: object,
    event: object,
    surv: np.ndarray,
    times: Sequence[float],
    censoring: Censoring = None,
) -> np.ndarray:
    """IPCW Brier scores of predicted survival curves at several times.

    Args:
        time: Follow-up times.
        event: Event indicator (0/1).
        surv: Predicted survival probabilities, shape ``(n, len(times))``.
        times: Evaluation times.
        censoring: Censoring-distribution specification.

    Returns:
        Brier score at each time.
    """
    time_a, event_a = as_1d_float(time), as_1d_float(event)
    surv = np.asarray(surv, dtype=np.float64)
    keep = np.isfinite(time_a) & np.isfinite(event_a) & np.all(np.isfinite(surv), axis=1)
    time_a, event_a, surv = time_a[keep], event_a[keep], surv[keep]
    G = _resolve_censoring(censoring, time_a, event_a)
    times = np.asarray(times, dtype=np.float64)
    case = (time_a[:, None] <= times[None, :]) & (event_a[:, None] != 0)
    alive = time_a[:, None] > times[None, :]
    w_case = np.where(case, _inv(G(time_a, left=True))[:, None], 0.0)
    w_alive = np.where(alive, _inv(G(times))[None, :], 0.0)
    return np.mean(w_case * surv**2 + w_alive * (1.0 - surv) ** 2, axis=0)


def integrated_brier_score(
    time: object,
    event: object,
    surv: np.ndarray,
    times: Sequence[float],
    censoring: Censoring = None,
    include_origin: bool = True,
) -> float:
    """Integrated Brier score over ``[0, times[-1]]`` (trapezoidal rule, divided by the span).

    Args:
        time: Follow-up times.
        event: Event indicator (0/1).
        surv: Predicted survival probabilities, shape ``(n, len(times))``.
        times: Increasing evaluation grid.
        censoring: Censoring-distribution specification.
        include_origin: Prepend time zero (where every prediction is 1 and the score is 0) so
            the integral covers ``[0, tau]``; with ``False`` the integral covers
            ``[times[0], times[-1]]`` as in :func:`sksurv.metrics.integrated_brier_score`.

    Returns:
        The integrated Brier score.
    """
    times = np.asarray(times, dtype=np.float64)
    bs = brier_scores(time, event, surv, times, censoring)
    if include_origin and times[0] > 0:
        times = np.r_[0.0, times]
        bs = np.r_[0.0, bs]
    span = times[-1] - times[0]
    return float(np.trapezoid(bs, times) / span) if span > 0 else float("nan")


# -------------------------------------------------------------------------------- calibration


def pseudo_observations(
    time: object, event: object, horizon: float, chunk: int = 1024
) -> np.ndarray:
    """Jackknife pseudo-observations of the cumulative risk ``1 - S(horizon)`` (Andersen 2003).

    ``theta_i = n * F(horizon) - (n - 1) * F_{-i}(horizon)`` where ``F_{-i}`` is the
    Kaplan-Meier risk without participant ``i``.

    Args:
        time: Follow-up times (no missing values).
        event: Event indicator (0/1).
        horizon: Time at which the risk is evaluated.
        chunk: Rows processed per vectorized block.

    Returns:
        Pseudo-observation per participant.
    """
    time = as_1d_float(time)
    event = as_1d_float(event) != 0
    n = time.size
    km = kaplan_meier(time, event)
    sel = (km.times <= horizon) & (km.n_event > 0)
    u, n_r, d = km.times[sel], km.n_risk[sel], km.n_event[sel]
    full = 1.0 - float(np.prod(1.0 - d / n_r))
    out = np.empty(n)
    for start in range(0, n, chunk):
        ti = time[start : start + chunk, None]
        ei = event[start : start + chunk, None]
        nr = n_r[None, :] - (ti >= u[None, :])
        dd = d[None, :] - (ei & (ti == u[None, :]))
        fac = np.ones_like(nr, dtype=np.float64)
        np.divide(nr - dd, nr, out=fac, where=nr > 0)
        out[start : start + chunk] = n * full - (n - 1) * (1.0 - np.prod(fac, axis=1))
    return out


def survival_calibration(
    time: object,
    event: object,
    risk: object,
    horizon: float,
    estimator: str = "ipcw",
    smoother: str = "spline",
    n_knots: int = 4,
    censoring: Censoring = None,
    n_grid: int = 50,
) -> dict[str, object]:
    """Horizon-specific calibration on the complementary log-log scale.

    The observed horizon outcome is represented either by IPCW-weighted horizon labels
    (``estimator="ipcw"``: events by the horizon weighted ``1 / G(T-)``, participants
    event-free beyond it weighted ``1 / G(horizon)``) or by jackknife pseudo-observations
    (``estimator="pseudo"``). The calibration intercept is fitted with ``cloglog(risk)`` as
    offset and the slope as the coefficient of ``cloglog(risk)``, both with a cloglog link.
    The integrated calibration index (ICI; Austin 2020) is the mean absolute difference
    between predicted risk and a smoothed observed risk: ``smoother="spline"`` fits a
    restricted cubic spline of ``cloglog(risk)`` with the same estimator, and
    ``smoother="hazard"`` fits a Cox proportional-hazards regression on that spline basis.

    Args:
        time: Follow-up times.
        event: Event indicator (0/1).
        risk: Predicted risk by the horizon.
        horizon: Calibration horizon.
        estimator: ``"ipcw"`` or ``"pseudo"``.
        smoother: ``"spline"`` or ``"hazard"``.
        n_knots: Number of spline knots.
        censoring: Censoring-distribution specification (IPCW estimator only).
        n_grid: Number of points of the returned calibration curve.

    Returns:
        Dict with ``intercept``, ``slope``, ``ici``, ``e50``, ``e90``, ``emax`` and ``curve``
        (dict of ``predicted`` and ``observed`` arrays).
    """
    time, event, risk = _clean(time, event, risk)
    x = cloglog(risk)
    if estimator == "ipcw":
        G = _resolve_censoring(censoring, time, event)
        case, w_case, w_ctrl = _horizon_weights(time, event, horizon, G, None)
        y, w = case.astype(float), w_case + w_ctrl
    elif estimator == "pseudo":
        y, w = pseudo_observations(time, event, horizon), np.ones_like(time)
    else:
        raise ValueError("estimator must be 'ipcw' or 'pseudo'")
    ones = np.ones((time.size, 1))
    a = fit_binomial_glm(ones, y, weights=w, offset=x, link="cloglog").coef[0]
    slope_fit = fit_binomial_glm(np.column_stack([ones, x]), y, weights=w, link="cloglog")

    lo, hi = np.quantile(risk, [0.01, 0.99]) if risk.size else (0.0, 1.0)
    grid = np.linspace(lo, hi, n_grid)
    knots = rcs_knots(x, n_knots) if np.unique(x).size > 2 * n_knots else np.array([])
    basis, grid_basis = rcs_basis(x, knots), rcs_basis(cloglog(grid), knots)
    if smoother == "spline":
        fit = fit_binomial_glm(np.column_stack([ones, basis]), y, weights=w, link="cloglog")
        observed = fit.predict(np.column_stack([ones, basis]))
        curve = fit.predict(np.column_stack([np.ones(n_grid), grid_basis]))
    elif smoother == "hazard":
        observed, curve = _cox_smoother(time, event, basis, grid_basis, horizon)
    else:
        raise ValueError("smoother must be 'spline' or 'hazard'")
    diff = np.abs(risk - observed)
    finite = np.isfinite(diff)
    stat = lambda f: float(f(diff[finite])) if finite.any() else float("nan")
    return {
        "intercept": float(a),
        "slope": float(slope_fit.coef[1]),
        "ici": stat(np.mean),
        "e50": stat(np.median),
        "e90": stat(lambda v: np.quantile(v, 0.9)),
        "emax": stat(np.max),
        "curve": {"predicted": grid, "observed": curve},
    }


def _cox_smoother(
    time: np.ndarray, event: np.ndarray, basis: np.ndarray, grid_basis: np.ndarray, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Observed horizon risk from a Cox model on a spline basis of the predicted risk."""
    import pandas as pd
    from lifelines import CoxPHFitter

    cols = [f"b{j}" for j in range(basis.shape[1])]
    df = pd.DataFrame(basis, columns=cols)
    df["time"], df["event"] = time, (event != 0).astype(int)
    cph = CoxPHFitter(penalizer=1e-6)
    try:
        cph.fit(df, duration_col="time", event_col="event")
    except Exception:  # noqa: BLE001 - lifelines raises several convergence error types
        nan = np.full(time.size, np.nan)
        return nan, np.full(grid_basis.shape[0], np.nan)
    obs = 1.0 - cph.predict_survival_function(df[cols], times=[horizon]).to_numpy()[0]
    grid = (
        1.0
        - cph.predict_survival_function(
            pd.DataFrame(grid_basis, columns=cols), times=[horizon]
        ).to_numpy()[0]
    )
    return obs, grid


# --------------------------------------------------------------- cumulative incidence and NB


def aalen_johansen(time: object, event: object, cause: int = 1) -> StepFunction:
    """Aalen-Johansen cumulative incidence of ``cause`` in the presence of competing events.

    Returns:
        :class:`StepFunction` of the cumulative incidence (initial value 0).
    """
    time, event = _clean(time, event)
    if time.size == 0:
        return StepFunction(np.array([]), np.array([]), initial=0.0)
    uniq, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    n_risk = time.size - np.cumsum(counts) + counts
    d_all = np.bincount(inverse, weights=(event != 0).astype(float), minlength=uniq.size)
    d_k = np.bincount(inverse, weights=(event == cause).astype(float), minlength=uniq.size)
    surv = np.cumprod(1.0 - d_all / n_risk)
    surv_prev = np.r_[1.0, surv[:-1]]
    cif = np.cumsum(surv_prev * d_k / n_risk)
    return StepFunction(uniq, cif, n_risk.astype(float), d_k, initial=0.0)


def net_benefit_survival(
    time: object, event: object, risk: object, horizon: float, threshold: float
) -> float:
    """Net benefit at a risk threshold for a time-to-event outcome (Vickers 2008).

    ``NB = (1 - S_+(h)) * pi - S_+(h) * pi * pt / (1 - pt)``, where ``pi`` is the fraction
    with ``risk >= pt`` and ``S_+`` is their Kaplan-Meier survival at the horizon.
    """
    time, event, risk = _clean(time, event, risk)
    if time.size == 0 or not 0 < threshold < 1:
        return float("nan")
    high = risk >= threshold
    pi = float(high.mean())
    if pi == 0:
        return 0.0
    s = float(kaplan_meier(time[high], event[high] != 0)(horizon))
    return (1 - s) * pi - s * pi * threshold / (1 - threshold)


def net_benefit_crude(
    time: object,
    event: object,
    risk: object,
    horizon: float,
    threshold: float,
    cause: int = 1,
) -> float:
    """Net benefit using the Aalen-Johansen crude risk of ``cause`` among flagged participants."""
    time, event, risk = _clean(time, event, risk)
    if time.size == 0 or not 0 < threshold < 1:
        return float("nan")
    high = risk >= threshold
    pi = float(high.mean())
    if pi == 0:
        return 0.0
    f = float(aalen_johansen(time[high], event[high], cause)(horizon))
    return f * pi - (1 - f) * pi * threshold / (1 - threshold)


def decision_curve_survival(
    time: object,
    event: object,
    risk: object,
    horizon: float,
    thresholds: Sequence[float],
    competing: bool = False,
    cause: int = 1,
) -> dict[str, np.ndarray]:
    """Decision curve for a time-to-event outcome with treat-all and treat-none references.

    Args:
        time: Follow-up times.
        event: Event codes (0/1, or competing-risk codes when ``competing``).
        risk: Predicted risk by the horizon.
        horizon: Evaluation horizon.
        thresholds: Threshold probabilities.
        competing: Use the Aalen-Johansen crude risk instead of Kaplan-Meier.
        cause: Event of interest for the crude-risk version.

    Returns:
        Dict of arrays ``threshold``, ``model``, ``treat_all`` and ``treat_none``.
    """
    time, event, risk = _clean(time, event, risk)
    th = np.asarray(thresholds, dtype=np.float64)
    if competing:
        model = [net_benefit_crude(time, event, risk, horizon, t, cause) for t in th]
        f_all = float(aalen_johansen(time, event, cause)(horizon))
    else:
        model = [net_benefit_survival(time, event, risk, horizon, t) for t in th]
        f_all = 1.0 - float(kaplan_meier(time, event != 0)(horizon))
    odds = th / (1 - th)
    return {
        "threshold": th,
        "model": np.asarray(model),
        "treat_all": f_all - (1 - f_all) * odds,
        "treat_none": np.zeros_like(th),
    }


def km_risk_by_stratum(
    time: object,
    event: object,
    risk: object,
    horizon: float,
    cutpoints: Sequence[float] = (0.20, 0.40),
    alpha: float = 0.05,
    cause: int = 1,
) -> list[dict[str, float]]:
    """Kaplan-Meier observed risk at the horizon per predicted-risk stratum.

    Strata are ``[0, c1), [c1, c2), ..., [ck, 1]`` (bounds are reported clipped to [0, 1]). Competing events (codes other than 0
    and ``cause``) are censored for the Kaplan-Meier net risk and handled by Aalen-Johansen
    for the reported crude risk.

    Args:
        time: Follow-up times.
        event: Event codes.
        risk: Predicted risk by the horizon.
        horizon: Evaluation horizon.
        cutpoints: Increasing risk cut-offs.
        alpha: Error level of the log-log Greenwood interval.
        cause: Event of interest.

    Returns:
        One dict per stratum with ``lower``, ``upper``, ``n``, ``events``, ``mean_predicted``,
        ``observed``, ``ci_low``, ``ci_high`` and ``crude_risk``.
    """
    time, event, risk = _clean(time, event, risk)
    edges = np.r_[-np.inf, np.asarray(cutpoints, dtype=np.float64), np.inf]
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for lo, hi in pairwise(edges):
        m = (risk >= lo) & (risk < hi)
        row = {"lower": max(float(lo), 0.0), "upper": min(float(hi), 1.0), "n": int(m.sum())}
        if not m.any():
            row.update(
                events=0,
                mean_predicted=np.nan,
                observed=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                crude_risk=np.nan,
            )
            rows.append(row)
            continue
        t, e = time[m], event[m]
        km = kaplan_meier(t, e == cause)
        s = float(km(horizon))
        sel = (km.times <= horizon) & (km.n_event > 0)
        nr, d = km.n_risk[sel], km.n_event[sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            gw = float(np.sum(d / (nr * (nr - d))))
        if 0 < s < 1 and np.isfinite(gw):
            se = np.sqrt(gw) / abs(np.log(s))
            s_lo = np.exp(-np.exp(np.log(-np.log(s)) + z * se))
            s_hi = np.exp(-np.exp(np.log(-np.log(s)) - z * se))
            ci = (1 - s_hi, 1 - s_lo)
        else:
            ci = (np.nan, np.nan)
        row.update(
            events=int(np.sum((t <= horizon) & (e == cause))),
            mean_predicted=float(risk[m].mean()),
            observed=1.0 - s,
            ci_low=float(ci[0]),
            ci_high=float(ci[1]),
            crude_risk=float(aalen_johansen(t, e, cause)(horizon)),
        )
        rows.append(row)
    return rows
