"""Model selection losses, cross-fitted post-processing and prediction tables.

Given the out-of-fold predictions of the inner-fold models of one outer training set, this
module

* scores a hyperparameter configuration (:func:`selection_loss`): for EGFR the log-loss after
  cross-fitted two-parameter logistic recalibration, for survival endpoints the censored
  discrete-time negative log-likelihood on a bin grid common to all candidate configurations;
* fits the post-processing of the selected configuration on the pooled inner out-of-fold
  predictions (:func:`fit_postprocessing`): recalibration map, sensitivity-targeted EGFR
  threshold and abstention cut-off;
* applies it to the cross-fitted outer-test predictions (:func:`apply_postprocessing`),
  including Mondrian cross-conformal p-values for EGFR, and renders the prediction table
  (:func:`prediction_frame`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from molt.calibration import (
    CloglogRecalibrator,
    KaplanMeierCensoring,
    LogisticRecalibrator,
    abstain,
    abstention_cutoff,
    combine_uncertainty,
    conformal_sets,
    cross_fitted_log_loss,
    format_conformal_set,
    horizon_status,
    mondrian_conformal_pvalues,
    sensitivity_threshold,
)
from molt.training.cohort_data import analytic_sets
from molt.training.preprocess import discretize, survival_bins
from molt.training.trainer import EnsemblePrediction

__all__ = [
    "Labels",
    "PostProcessing",
    "apply_postprocessing",
    "fit_postprocessing",
    "labels_for",
    "nll_on_grid",
    "prediction_frame",
    "risk_stratum",
    "selection_loss",
    "selection_objective",
    "survival_at",
]


@dataclass
class Labels:
    """Endpoint labels of an ordered list of participants (raw, unbinned)."""

    ids: list[str]
    y: np.ndarray
    time: np.ndarray
    event: np.ndarray
    in_set: np.ndarray

    def take(self, idx: np.ndarray) -> Labels:
        return Labels(
            [self.ids[i] for i in idx],
            self.y[idx],
            self.time[idx],
            self.event[idx],
            self.in_set[idx],
        )


def labels_for(
    table: pd.DataFrame, ids: Sequence[str], task: str, *, require_modality: bool = True
) -> Labels:
    """Labels of ``ids`` for ``task``; ``in_set`` marks the analytic set of the endpoint."""
    sub = table.loc[list(ids)]
    sets = analytic_sets(sub, require_modality=require_modality)
    ep = "pfi" if task == "egfr" else task
    y = pd.to_numeric(sub["egfr_status"], errors="coerce").to_numpy(float)
    in_egfr = sub.index.isin(sets["egfr"])
    y = np.where(in_egfr, y, np.nan)
    time = pd.to_numeric(sub[f"{ep}_time"], errors="coerce").to_numpy(float)
    event = pd.to_numeric(sub[f"{ep}_event"], errors="coerce").to_numpy(float)
    in_surv = sub.index.isin(sets[ep])
    in_set = in_egfr if task == "egfr" else in_surv
    return Labels(
        list(map(str, ids)),
        y,
        np.where(in_surv, time, np.nan),
        np.where(in_surv, event, np.nan),
        np.asarray(in_set, bool),
    )


def survival_at(surv: np.ndarray, edges: np.ndarray, t: float) -> np.ndarray:
    """Survival at time ``t`` from curves at the bin boundaries ``edges[1:]``.

    Between boundaries the hazard is taken as constant within the bin.
    """
    surv = np.asarray(surv, dtype=np.float64)
    full = np.concatenate([np.ones((surv.shape[0], 1)), surv], axis=1)
    if t <= 0:
        return full[:, 0]
    if t >= edges[-1]:
        return full[:, -1]
    j = int(np.searchsorted(edges, t, side="right") - 1)
    frac = (t - edges[j]) / (edges[j + 1] - edges[j])
    if frac <= 1e-9:
        return full[:, j]
    ratio = np.clip(full[:, j + 1] / np.maximum(full[:, j], 1e-12), 1e-12, 1.0)
    return full[:, j] * ratio**frac


def nll_on_grid(
    surv: np.ndarray, edges: np.ndarray, time: np.ndarray, event: np.ndarray, width: float
) -> float:
    """Censored discrete-time NLL of predicted curves evaluated on a grid of ``width``.

    Using a grid shared by all candidate configurations makes the likelihood comparable across
    bin widths; ``width`` must be a multiple of the native bin width.
    """
    grid = survival_bins(width, float(edges[-1]))
    s = np.column_stack([survival_at(surv, edges, g) for g in grid])
    hz = np.clip(1.0 - s[:, 1:] / np.maximum(s[:, :-1], 1e-12), 1e-6, 1 - 1e-6)
    ok = np.isfinite(time) & np.isfinite(event)
    if not ok.any():
        return float("nan")
    k, ev, valid = discretize(time[ok], event[ok], grid)
    hz = hz[ok][valid]
    k, ev = k[valid], ev[valid]
    if not len(k):
        return float("nan")
    cols = np.arange(hz.shape[1])[None, :]
    log_s = np.log1p(-hz)
    before = cols < k[:, None]
    at = cols == k[:, None]
    ll_event = (log_s * before).sum(1) + (np.log(hz) * at).sum(1)
    ll_cens = (log_s * (before | at)).sum(1)
    return float(-(ev * ll_event + (1 - ev) * ll_cens).mean())


def selection_loss(
    task: str,
    val_preds: Sequence[EnsemblePrediction],
    val_labels: Sequence[Labels],
    edges: np.ndarray,
    *,
    grid_width: float | None = None,
) -> float:
    """Inner out-of-fold loss of one configuration (lower is better)."""
    if task == "egfr":
        p = np.concatenate([pr.p for pr in val_preds])
        y = np.concatenate([lb.y for lb in val_labels])
        folds = np.concatenate([np.full(len(lb.ids), j) for j, lb in enumerate(val_labels)])
        loss, _ = cross_fitted_log_loss(p, y, folds)
        return loss
    surv = np.concatenate([pr.survival for pr in val_preds])
    time = np.concatenate([np.where(lb.in_set, lb.time, np.nan) for lb in val_labels])
    event = np.concatenate([lb.event for lb in val_labels])
    width = grid_width or float(edges[1] - edges[0])
    return nll_on_grid(surv, edges, time, event, width)


def _selection_metric(
    task: str, val_preds: Sequence[EnsemblePrediction], val_labels: Sequence[Labels]
) -> float:
    """Pooled inner out-of-fold AUROC (EGFR) or Harrell's C of the horizon risk (survival)."""
    from molt.metrics.binary import auroc
    from molt.metrics.survival import harrell_c

    if task == "egfr":
        p = np.concatenate([pr.p for pr in val_preds])
        y = np.concatenate([lb.y for lb in val_labels])
        ok = np.isfinite(y)
        return float(auroc(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else float("nan")
    risk = np.concatenate([1.0 - pr.survival[:, pr.horizon_index] for pr in val_preds])
    time = np.concatenate([np.where(lb.in_set, lb.time, np.nan) for lb in val_labels])
    event = np.concatenate([lb.event for lb in val_labels])
    ok = np.isfinite(time) & np.isfinite(event)
    return float(harrell_c(time[ok], event[ok], risk[ok])) if ok.sum() > 1 else float("nan")


def selection_objective(
    task: str,
    preds_by_mode: Mapping[str, Sequence[EnsemblePrediction]],
    val_labels: Sequence[Labels],
    edges: np.ndarray,
    *,
    mode_weights: Mapping[str, float] | None = None,
    metric_weight: float = 0.0,
    grid_width: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Model-selection objective over prediction modes (lower is better).

    ``sum_m w_m [L_m - metric_weight * D_m]`` over the modes ``m`` of ``mode_weights``
    (default ``{"full": 1}``), where ``L_m`` is :func:`selection_loss` and ``D_m`` the pooled
    inner out-of-fold discrimination (AUROC for EGFR, Harrell's C for survival endpoints).

    Returns:
        The objective and the per-mode components (``loss_<mode>``, ``metric_<mode>``).
    """
    weights = dict(mode_weights or {"full": 1.0})
    total, parts = 0.0, {}
    for mode, w in weights.items():
        if float(w) == 0.0:
            continue
        preds = preds_by_mode[mode]
        loss = selection_loss(task, preds, val_labels, edges, grid_width=grid_width)
        parts[f"loss_{mode}"] = loss
        value = loss
        if metric_weight:
            metric = _selection_metric(task, preds, val_labels)
            parts[f"metric_{mode}"] = metric
            value = loss - float(metric_weight) * (metric if np.isfinite(metric) else 0.5)
        total += float(w) * value
    return float(total), parts


@dataclass
class PostProcessing:
    """Post-processing fitted on pooled inner out-of-fold predictions of one mode."""

    task: str
    mode: str
    recalibration: dict[str, Any]
    threshold: float = float("nan")
    abstain_cutoff: float = float("inf")
    horizon: float = 24.0
    conformal: list[dict[str, list[float]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "mode": self.mode,
            "recalibration": self.recalibration,
            "threshold": self.threshold,
            "abstain_cutoff": self.abstain_cutoff,
            "horizon": self.horizon,
            "conformal": self.conformal,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostProcessing:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def recalibrator(self) -> LogisticRecalibrator | CloglogRecalibrator:
        if self.task == "egfr":
            return LogisticRecalibrator.from_dict(self.recalibration)
        return CloglogRecalibrator.from_dict(self.recalibration)


def _uncertainty(pred: EnsemblePrediction, task: str) -> np.ndarray:
    if task == "egfr":
        return combine_uncertainty(pred.p_ensemble_sd, pred.p_mc)
    return combine_uncertainty(pred.risk_ensemble_sd, pred.risk_mc)


def fit_postprocessing(
    task: str,
    mode: str,
    val_preds: Sequence[EnsemblePrediction],
    val_labels: Sequence[Labels],
    edges: np.ndarray,
    *,
    horizon: float = 24.0,
    sensitivity_target: float = 0.90,
    coverage: float = 0.90,
    intercepts: str = "horizon",
    horizons: Sequence[float] = (12.0, 24.0, 36.0),
) -> PostProcessing:
    """Fit recalibration, threshold, abstention cut-off and conformal calibration scores.

    For survival endpoints the cloglog slope is estimated at ``horizon``; with
    ``intercepts="horizon"`` (``recalibration.pfi_intercepts``) the intercept is re-estimated
    at every reported horizon in ``horizons`` and interpolated between them, with
    ``"single"`` one map estimated at ``horizon`` is applied to the whole curve (see
    :class:`~molt.calibration.CloglogRecalibrator`).
    """
    if intercepts not in ("horizon", "single"):
        raise ValueError(f"intercepts must be 'horizon' or 'single', got {intercepts!r}")
    unc = np.concatenate([_uncertainty(pr, task) for pr in val_preds])
    in_set = np.concatenate([lb.in_set for lb in val_labels])
    if task == "egfr":
        p = np.concatenate([pr.p for pr in val_preds])
        y = np.concatenate([lb.y for lb in val_labels])
        recal = LogisticRecalibrator().fit(p, y)
        threshold = sensitivity_threshold(recal.transform(p), y, sensitivity_target)
        conformal = [
            {"p": pr.p.tolist(), "y": lb.y.tolist()} for pr, lb in zip(val_preds, val_labels)
        ]
        return PostProcessing(
            task,
            mode,
            recal.to_dict(),
            threshold,
            abstention_cutoff(unc[in_set], coverage),
            horizon,
            conformal,
        )
    surv = np.concatenate([pr.survival for pr in val_preds])
    time = np.concatenate([lb.time for lb in val_labels])
    event = np.concatenate([lb.event for lb in val_labels])
    ok = in_set & np.isfinite(time) & np.isfinite(event)
    risk = 1.0 - survival_at(surv, edges, horizon)
    censoring = KaplanMeierCensoring.fit(time[ok], event[ok])
    horizon_risk = None
    if intercepts == "horizon":
        horizon_risk = {
            float(h): 1.0 - survival_at(surv[ok], edges, float(h))
            for h in sorted({*map(float, horizons), float(horizon)})
            if 0 < float(h) <= edges[-1]
        }
    recal = CloglogRecalibrator(horizon=horizon).fit(
        risk[ok], time[ok], event[ok], censoring=censoring, horizon_risk=horizon_risk
    )
    return PostProcessing(
        task, mode, recal.to_dict(), float("nan"), abstention_cutoff(unc[ok], coverage), horizon, []
    )


def apply_postprocessing(
    post: PostProcessing,
    test_preds: Sequence[EnsemblePrediction],
    edges: np.ndarray,
    *,
    flags: np.ndarray | None = None,
    alpha: float = 0.05,
    horizons: Sequence[float] = (12.0, 24.0, 36.0),
) -> dict[str, Any]:
    """Average the inner-fold predictions of the test participants and post-process them.

    Returns a dictionary of arrays (``p_raw``, ``p_cal``, risks, uncertainty, abstention,
    conformal p-values, reliability weights, survival curves).
    """
    task = post.task
    n = len(test_preds[0].ids)
    unc_ens = np.mean(
        [
            combine_uncertainty(pr.p_ensemble_sd if task == "egfr" else pr.risk_ensemble_sd)
            for pr in test_preds
        ],
        axis=0,
    )
    unc_mc = np.mean([pr.p_mc if task == "egfr" else pr.risk_mc for pr in test_preds], axis=0)
    unc = np.mean([_uncertainty(pr, task) for pr in test_preds], axis=0)
    out: dict[str, Any] = {
        "uncertainty": unc,
        "unc_ensemble": unc_ens,
        "unc_mc": unc_mc,
        "abstain_cutoff": np.full(n, post.abstain_cutoff),
        "abstain": abstain(unc, post.abstain_cutoff, flags),
        "rho": np.mean([pr.rho for pr in test_preds], axis=0),
    }
    recal = post.recalibrator()
    if task == "egfr":
        p_raw = np.mean([pr.p for pr in test_preds], axis=0)
        p_cal = recal.transform(p_raw)
        out.update(
            p_raw=p_raw,
            p_cal=p_cal,
            threshold=np.full(n, post.threshold),
            pred_pos=(p_cal >= post.threshold).astype(float),
        )
        if post.conformal and len(post.conformal) == len(test_preds):
            cal = [(np.asarray(c["p"], float), np.asarray(c["y"], float)) for c in post.conformal]
            pvals = mondrian_conformal_pvalues(cal, [pr.p for pr in test_preds])
            out.update(
                conformal_p0=pvals[:, 0],
                conformal_p1=pvals[:, 1],
                conformal_set=format_conformal_set(conformal_sets(pvals, alpha)),
            )
        return out
    surv_raw = np.mean([pr.survival for pr in test_preds], axis=0)
    surv_cal = np.clip(recal.transform_survival(surv_raw, np.asarray(edges)[1:]), 0.0, 1.0)
    out.update(
        p_raw=1.0 - survival_at(surv_raw, edges, post.horizon),
        p_cal=1.0 - survival_at(surv_cal, edges, post.horizon),
        surv_raw=surv_raw,
        surv_cal=surv_cal,
    )
    for h in horizons:
        out[f"risk_{int(h)}m"] = 1.0 - survival_at(surv_cal, edges, h)
    return out


def risk_stratum(risk: np.ndarray, cutoffs: Sequence[float] = (0.20, 0.40)) -> list[str | None]:
    """Low / intermediate / high stratum of an absolute risk."""
    lo, hi = cutoffs
    out: list[str | None] = []
    for r in np.asarray(risk, dtype=float):
        out.append(
            None
            if not np.isfinite(r)
            else ("low" if r < lo else "high" if r >= hi else "intermediate")
        )
    return out


def prediction_frame(
    *,
    ids: Sequence[str],
    table: pd.DataFrame,
    task: str,
    mode: str,
    outer_fold: int | np.ndarray,
    arrays: Mapping[str, Any],
    labels: Labels,
    config_id: int | None = None,
    flags: np.ndarray | None = None,
    risk_cutoffs: Sequence[float] = (0.20, 0.40),
    horizon: float = 24.0,
) -> pd.DataFrame:
    """Prediction table rows (one per participant) in the run-artefact schema."""
    sub = table.loc[list(ids)]
    n = len(ids)
    nan = np.full(n, np.nan)
    rho = np.asarray(arrays.get("rho", np.zeros((n, 3))))
    df = pd.DataFrame(
        {
            "participant_id": list(ids),
            "cohort": sub["cohort"].astype(str).to_numpy(),
            "outer_fold": np.broadcast_to(np.asarray(outer_fold), (n,)).astype(int),
            "task": task,
            "mode": mode,
            "config_id": -1 if config_id is None else int(config_id),
            "p_raw": arrays.get("p_raw", nan),
            "p_cal": arrays.get("p_cal", nan),
            "risk_12m": arrays.get("risk_12m", nan),
            "risk_24m": arrays.get("risk_24m", nan),
            "risk_36m": arrays.get("risk_36m", nan),
            "threshold": arrays.get("threshold", nan),
            "pred_pos": arrays.get("pred_pos", nan),
            "uncertainty": arrays.get("uncertainty", nan),
            "unc_ensemble": arrays.get("unc_ensemble", nan),
            "unc_mc": arrays.get("unc_mc", nan),
            "abstain_cutoff": arrays.get("abstain_cutoff", nan),
            "abstain": np.asarray(arrays.get("abstain", np.zeros(n, bool)), bool),
            "conformal_p0": arrays.get("conformal_p0", nan),
            "conformal_p1": arrays.get("conformal_p1", nan),
            "conformal_set": arrays.get("conformal_set", [None] * n),
            "rho_clinical": rho[:, 0],
            "rho_rna": rho[:, 1],
            "rho_ct": rho[:, 2],
            "has_clinical": sub["has_clinical_record"].astype(bool).to_numpy(),
            "has_rna": sub["has_rna"].astype(bool).to_numpy(),
            "has_ct": sub["has_ct"].astype(bool).to_numpy(),
            "rna_quality_flag": nan if flags is None else np.asarray(flags, float),
            "in_analytic_set": labels.in_set,
            "y_egfr": labels.y,
            "time": labels.time,
            "event": labels.event,
            "status_24m": horizon_status(labels.time, labels.event, horizon),
        }
    )
    df["risk_stratum"] = (
        risk_stratum(df["risk_24m"].to_numpy(), risk_cutoffs) if task != "egfr" else None
    )
    if "surv_cal" in arrays:
        df["surv_raw"] = [np.round(r, 6).tolist() for r in np.asarray(arrays["surv_raw"])]
        df["surv_cal"] = [np.round(r, 6).tolist() for r in np.asarray(arrays["surv_cal"])]
    return df
