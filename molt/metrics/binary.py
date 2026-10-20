"""Discrimination, calibration and clinical-utility metrics for binary outcomes.

All functions take a binary label vector ``y`` (0/1) and a vector of predicted
probabilities or scores ``p``; rows with a missing label or prediction are dropped. The
implementations are vectorized and avoid per-call overhead so that they can be used inside
participant-level bootstrap loops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from molt.metrics._glm import as_1d_float, fit_binomial_glm, logit

__all__ = [
    "CalibrationResult",
    "auprc",
    "auroc",
    "auroc_variance_delong",
    "brier",
    "calibration_intercept_slope",
    "clopper_pearson",
    "confusion_counts",
    "decision_curve",
    "expected_calibration_error",
    "false_positive_rate",
    "net_benefit",
    "net_benefit_treat_all",
    "observed_expected",
    "percentile_ranks",
    "reporting_tier",
    "threshold_for_sensitivity",
    "threshold_metrics",
    "tiered_discrimination",
    "weighted_auroc",
    "weighted_average_precision",
]


def _clean(y: object, p: object, w: object | None = None) -> tuple[np.ndarray, ...]:
    """Drop rows with missing values and return float arrays ``(y, p[, w])``."""
    y = as_1d_float(y, "y")
    p = as_1d_float(p, "p")
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same length ({y.size} != {p.size})")
    keep = np.isfinite(y) & np.isfinite(p)
    if w is not None:
        w = as_1d_float(w, "w")
        keep &= np.isfinite(w)
        return y[keep], p[keep], w[keep]
    return y[keep], p[keep]


# --------------------------------------------------------------------------- discrimination


def weighted_auroc(
    score: np.ndarray, is_case: np.ndarray, w_case: np.ndarray, w_control: np.ndarray
) -> float:
    """Weighted area under the ROC curve with ties counted as one half.

    Each observation can contribute to the case group, the control group or neither through
    its weights, which allows inverse-probability-of-censoring and competing-risk weighting.

    Args:
        score: Predicted scores, higher meaning more likely to be a case.
        is_case: Boolean case indicator (used only to zero the opposite group's weights).
        w_case: Case weight per observation (ignored where ``is_case`` is False).
        w_control: Control weight per observation (ignored where ``is_case`` is True).

    Returns:
        The weighted AUROC, or NaN when either group has zero total weight.
    """
    score = as_1d_float(score)
    is_case = np.asarray(is_case, dtype=bool)
    wp = np.where(is_case, w_case, 0.0)
    wn = np.where(is_case, 0.0, w_control)
    keep = np.isfinite(score) & ((wp > 0) | (wn > 0))
    score, wp, wn = score[keep], wp[keep], wn[keep]
    tot_p, tot_n = wp.sum(), wn.sum()
    if score.size == 0 or tot_p <= 0 or tot_n <= 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    gp = np.add.reduceat(wp[order], starts)
    gn = np.add.reduceat(wn[order], starts)
    below = np.cumsum(gn) - gn
    return float(np.sum(gp * (below + 0.5 * gn)) / (tot_p * tot_n))


def auroc(y: object, p: object, sample_weight: object | None = None) -> float:
    """Area under the ROC curve (Mann-Whitney estimator, ties count one half).

    Args:
        y: Binary labels.
        p: Predicted probabilities or scores.
        sample_weight: Optional non-negative observation weights.

    Returns:
        AUROC, or NaN if only one class is present.
    """
    if sample_weight is None:
        y, p = _clean(y, p)
        w = np.ones_like(y)
    else:
        y, p, w = _clean(y, p, sample_weight)
    case = y > 0.5
    return weighted_auroc(p, case, w, w)


def weighted_average_precision(score: np.ndarray, is_case: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision (step-wise area under the precision-recall curve).

    Matches :func:`sklearn.metrics.average_precision_score` with ``sample_weight``.

    Args:
        score: Predicted scores.
        is_case: Boolean case indicator.
        w: Non-negative observation weights.

    Returns:
        Average precision, or NaN when the total case weight is zero.
    """
    score = as_1d_float(score)
    is_case = np.asarray(is_case, dtype=bool)
    w = as_1d_float(w)
    keep = np.isfinite(score) & (w > 0)
    score, is_case, w = score[keep], is_case[keep], w[keep]
    if score.size == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    tp = np.cumsum(np.where(is_case[order], w[order], 0.0))
    fp = np.cumsum(np.where(is_case[order], 0.0, w[order]))
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp, fp = tp[last], fp[last]
    if tp[-1] <= 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, np.finfo(float).tiny)
    recall = tp / tp[-1]
    d_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(d_recall * precision))


def auprc(y: object, p: object, sample_weight: object | None = None) -> float:
    """Area under the precision-recall curve as average precision.

    Args:
        y: Binary labels.
        p: Predicted probabilities or scores.
        sample_weight: Optional observation weights.

    Returns:
        Average precision, or NaN if there are no positives.
    """
    if sample_weight is None:
        y, p = _clean(y, p)
        w = np.ones_like(y)
    else:
        y, p, w = _clean(y, p, sample_weight)
    return weighted_average_precision(p, y > 0.5, w)


def auroc_variance_delong(y: object, p: object) -> tuple[float, float]:
    """AUROC and its DeLong variance computed from midrank structural components.

    Args:
        y: Binary labels.
        p: Predicted scores.

    Returns:
        ``(auroc, variance)``; the variance is NaN when either class has fewer than two
        observations.
    """
    y, p = _clean(y, p)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    m, n = pos.size, neg.size
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    r_all = stats.rankdata(np.r_[pos, neg])
    v10 = (r_all[:m] - stats.rankdata(pos)) / n
    v01 = 1.0 - (r_all[m:] - stats.rankdata(neg)) / m
    auc = float(v10.mean())
    if m < 2 or n < 2:
        return auc, float("nan")
    return auc, float(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)


def percentile_ranks(y: object, p: object) -> np.ndarray:
    """Percentile rank of each positive's score within the negatives' score distribution.

    The rank of positive ``i`` is ``100 * (#neg < p_i + 0.5 * #neg == p_i) / n_neg``; the mean
    of the ranks equals ``100 * AUROC``. Useful for sparse-positive reporting.

    Args:
        y: Binary labels.
        p: Predicted scores.

    Returns:
        Array of percentile ranks, in the order the positives appear.
    """
    y, p = _clean(y, p)
    neg = np.sort(p[y <= 0.5])
    pos = p[y > 0.5]
    if neg.size == 0:
        return np.full(pos.size, np.nan)
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return 100.0 * (lo + 0.5 * (hi - lo)) / neg.size


def reporting_tier(n_pos: int | object, n_neg: int | None = None) -> str:
    """Reporting tier from class counts.

    Tier ``"A"`` requires at least 10 observations in both classes (full inferential
    reporting), tier ``"B"`` has 5-9 in the smaller class (descriptive AUROC), and tier ``"C"``
    has fewer than five in either class (counts and positive-case score ranks only).

    Args:
        n_pos: Number of positives, or a label vector when ``n_neg`` is None.
        n_neg: Number of negatives.

    Returns:
        ``"A"``, ``"B"`` or ``"C"``.
    """
    if n_neg is None:
        y = as_1d_float(n_pos)
        y = y[np.isfinite(y)]
        n_pos, n_neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
    smaller = min(int(n_pos), int(n_neg))
    if smaller >= 10:
        return "A"
    if smaller >= 5:
        return "B"
    return "C"


def tiered_discrimination(y: object, p: object) -> dict[str, object]:
    """Discrimination summary that follows the reporting tier of the class counts.

    Args:
        y: Binary labels.
        p: Predicted scores.

    Returns:
        Dict with ``tier``, ``n_pos``, ``n_neg``, ``auroc`` (NaN for tier C),
        ``percentile_ranks`` (list, reported for tiers B and C) and ``median_percentile``.
    """
    y, p = _clean(y, p)
    n_pos, n_neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
    tier = reporting_tier(n_pos, n_neg)
    ranks = percentile_ranks(y, p)
    return {
        "tier": tier,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": auroc(y, p) if tier in ("A", "B") else float("nan"),
        "percentile_ranks": [] if tier == "A" else [float(r) for r in ranks],
        "median_percentile": float(np.median(ranks)) if ranks.size else float("nan"),
    }


# ------------------------------------------------------------------------------ calibration


def brier(y: object, p: object) -> float:
    """Brier score (mean squared error of predicted probabilities)."""
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def expected_calibration_error(y: object, p: object, n_bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins.

    ``ECE = sum_b (n_b / n) * |mean(y_b) - mean(p_b)|`` where bins contain (as nearly as
    possible) equal numbers of participants after sorting by predicted probability.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        n_bins: Number of bins (capped at the number of observations).

    Returns:
        ECE in probability units.

    Note:
        ECE is an absolute-deviation statistic and is biased upward: resampling adds noise
        to every bin mean, so bootstrap replicates are systematically larger than the
        estimate and a percentile interval can exclude the estimate itself (e.g. 0.012 with
        a percentile interval of 0.018-0.052). :func:`molt.metrics.report.evaluate_binary`
        therefore reports the basic (bias-corrected) bootstrap interval
        ``[2 ECE - q_97.5, 2 ECE - q_2.5]``, clipped at 0 and widened to contain the point
        estimate (``ci_method = basic_bootstrap``; see
        :meth:`molt.metrics.bootstrap.BootstrapResult.basic_ci`).
    """
    y, p = _clean(y, p)
    n = y.size
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    bins = min(int(n_bins), n)
    edges = np.linspace(0, n, bins + 1).round().astype(int)
    starts = edges[:-1][np.diff(edges) > 0]
    sum_y = np.add.reduceat(y[order], starts)
    sum_p = np.add.reduceat(p[order], starts)
    return float(np.sum(np.abs(sum_y - sum_p)) / n)


@dataclass(frozen=True)
class CalibrationResult:
    """Logistic recalibration summary.

    Attributes:
        intercept: Calibration-in-the-large (intercept with the logit of ``p`` as offset).
        slope: Coefficient of ``logit(p)`` in a logistic model with free intercept.
        slope_intercept: Intercept of the model that estimates the slope.
        slope_se: Model-based standard error of the slope.
    """

    intercept: float
    slope: float
    slope_intercept: float
    slope_se: float


def calibration_intercept_slope(y: object, p: object) -> CalibrationResult:
    """Calibration intercept and slope by logistic recalibration of ``logit(p)``.

    Args:
        y: Binary labels.
        p: Predicted probabilities.

    Returns:
        :class:`CalibrationResult`; an ideal model has intercept 0 and slope 1.
    """
    y, p = _clean(y, p)
    lp = logit(p)
    ones = np.ones((y.size, 1))
    fit_int = fit_binomial_glm(ones, y, offset=lp, link="logit")
    fit_slope = fit_binomial_glm(np.column_stack([ones, lp]), y, link="logit")
    return CalibrationResult(
        intercept=float(fit_int.coef[0]),
        slope=float(fit_slope.coef[1]),
        slope_intercept=float(fit_slope.coef[0]),
        slope_se=float(np.sqrt(fit_slope.cov[1, 1])),
    )


def observed_expected(y: object, p: object, alpha: float = 0.05) -> dict[str, float]:
    """Observed-to-expected event ratio with an exact Poisson confidence interval.

    The observed count ``O`` is treated as Poisson and the expected count ``E = sum(p)`` as
    fixed, giving ``[chi2(alpha/2, 2O) / 2E, chi2(1 - alpha/2, 2O + 2) / 2E]``.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        alpha: Two-sided error level.

    Returns:
        Dict with ``oe``, ``ci_low``, ``ci_high``, ``observed`` and ``expected``.
    """
    y, p = _clean(y, p)
    obs, exp = float(y.sum()), float(p.sum())
    if exp <= 0:
        return {"oe": np.nan, "ci_low": np.nan, "ci_high": np.nan, "observed": obs, "expected": exp}
    lo = stats.chi2.ppf(alpha / 2, 2 * obs) / 2 if obs > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * obs + 2) / 2
    return {
        "oe": obs / exp,
        "ci_low": lo / exp,
        "ci_high": hi / exp,
        "observed": obs,
        "expected": exp,
    }


# ----------------------------------------------------------------- threshold-based metrics


def clopper_pearson(k: float, n: float, alpha: float = 0.05) -> tuple[float, float, float]:
    """Proportion ``k / n`` with its exact (Clopper-Pearson) confidence interval.

    Args:
        k: Number of successes.
        n: Number of trials.
        alpha: Two-sided error level.

    Returns:
        ``(proportion, ci_low, ci_high)``; all NaN when ``n == 0``.
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    lo = 0.0 if k <= 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return float(k / n), lo, hi


def confusion_counts(y: object, p: object, threshold: float) -> dict[str, int]:
    """True/false positive/negative counts when predicting positive for ``p >= threshold``."""
    y, p = _clean(y, p)
    pred = p >= threshold
    pos = y > 0.5
    return {
        "tp": int(np.sum(pred & pos)),
        "fp": int(np.sum(pred & ~pos)),
        "tn": int(np.sum(~pred & ~pos)),
        "fn": int(np.sum(~pred & pos)),
    }


def threshold_metrics(
    y: object, p: object, threshold: float, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Sensitivity, specificity, PPV, NPV and FPR at a threshold with exact binomial CIs.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        threshold: Participants with ``p >= threshold`` are classified positive.
        alpha: Two-sided error level.

    Returns:
        Dict ``metric -> {"value", "ci_low", "ci_high", "k", "n"}`` for ``sensitivity``,
        ``specificity``, ``ppv``, ``npv``, ``fpr`` and ``flag_rate`` (fraction classified
        positive).
    """
    c = confusion_counts(y, p, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    spec = {
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "ppv": (tp, tp + fp),
        "npv": (tn, tn + fn),
        "fpr": (fp, fp + tn),
        "flag_rate": (tp + fp, tp + fp + tn + fn),
    }
    out = {}
    for name, (k, n) in spec.items():
        val, lo, hi = clopper_pearson(k, n, alpha)
        out[name] = {"value": val, "ci_low": lo, "ci_high": hi, "k": k, "n": n}
    return out


def false_positive_rate(y: object, p: object, threshold: float) -> float:
    """Fraction of negatives with ``p >= threshold``."""
    c = confusion_counts(y, p, threshold)
    n = c["fp"] + c["tn"]
    return c["fp"] / n if n else float("nan")


def threshold_for_sensitivity(y: object, p: object, target: float = 0.90) -> float:
    """Largest threshold whose sensitivity (``p >= threshold``) is at least ``target``.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        target: Required sensitivity.

    Returns:
        The selected threshold (one of the positives' scores), or NaN without positives.
    """
    y, p = _clean(y, p)
    pos = np.sort(p[y > 0.5])[::-1]
    if pos.size == 0:
        return float("nan")
    k = int(np.ceil(target * pos.size - 1e-12))
    return float(pos[max(k, 1) - 1])


def net_benefit(y: object, p: object, threshold: float) -> float:
    """Net benefit of treating participants with ``p >= threshold``.

    ``NB = TP/n - FP/n * pt / (1 - pt)`` (Vickers and Elkin decision-curve analysis).
    """
    y, p = _clean(y, p)
    n = y.size
    if n == 0 or not 0 < threshold < 1:
        return float("nan")
    pred = p >= threshold
    tp = np.sum(pred & (y > 0.5))
    fp = np.sum(pred & (y <= 0.5))
    return float(tp / n - fp / n * threshold / (1 - threshold))


def net_benefit_treat_all(y: object, threshold: float) -> float:
    """Net benefit of the treat-all strategy; the treat-none reference is zero."""
    y = as_1d_float(y)
    y = y[np.isfinite(y)]
    if y.size == 0 or not 0 < threshold < 1:
        return float("nan")
    prev = float(np.mean(y > 0.5))
    return prev - (1 - prev) * threshold / (1 - threshold)


def decision_curve(
    y: object, p: object, thresholds: Sequence[float] | np.ndarray
) -> dict[str, np.ndarray]:
    """Decision curve: model, treat-all and treat-none net benefit over thresholds.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        thresholds: Threshold probabilities in (0, 1).

    Returns:
        Dict of arrays ``threshold``, ``model``, ``treat_all`` and ``treat_none``.
    """
    y, p = _clean(y, p)
    th = np.asarray(thresholds, dtype=np.float64)
    n = y.size
    pos = y > 0.5
    pred = p[None, :] >= th[:, None]
    tp = (pred & pos[None, :]).sum(axis=1) / n
    fp = (pred & ~pos[None, :]).sum(axis=1) / n
    odds = th / (1 - th)
    prev = pos.mean()
    return {
        "threshold": th,
        "model": tp - fp * odds,
        "treat_all": prev - (1 - prev) * odds,
        "treat_none": np.zeros_like(th),
    }
