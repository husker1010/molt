"""Run-level summaries of pooled out-of-fold or locked predictions (``summary.json``).

For every mode the summary contains the metric panel of :mod:`molt.metrics` (with
participant-level bootstrap intervals) on the analytic set of the endpoint, plus operating
point, selective-prediction and conformal summaries computed from the prediction table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from molt.training.trainer import harrell_c
from molt.utils.logging import get_logger

__all__ = ["quick_metrics", "summarize_predictions"]

log = get_logger(__name__)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y[ok], p[ok]))


def quick_metrics(df: pd.DataFrame, task: str) -> dict[str, float]:
    """Point estimates without resampling (used for per-fold logging)."""
    sub = df[df["in_analytic_set"].astype(bool)] if "in_analytic_set" in df else df
    if task == "egfr":
        y, p = sub["y_egfr"].to_numpy(float), sub["p_cal"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        pc = np.clip(p[ok], 1e-6, 1 - 1e-6)
        return {
            "auroc": _auroc(y, p),
            "log_loss": (
                float(-np.mean(y[ok] * np.log(pc) + (1 - y[ok]) * np.log1p(-pc)))
                if ok.any()
                else np.nan
            ),
            "brier": float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else np.nan,
            "n": int(ok.sum()),
            "n_pos": int(np.nansum(y[ok])),
        }
    t, e, r = (sub[c].to_numpy(float) for c in ("time", "event", "risk_24m"))
    st = sub["status_24m"].to_numpy(float)
    return {
        "c_index": harrell_c(t, e, r),
        "auroc_24m_naive": _auroc(st, r),
        "n": int(np.isfinite(t).sum()),
        "events": int(np.nansum(e)),
    }


def _proportion(k: int, n: int) -> dict[str, Any]:
    try:
        from molt.metrics.binary import clopper_pearson

        value, lo, hi = clopper_pearson(k, n)
    except Exception:  # noqa: BLE001 - fall back to the point estimate
        value, lo, hi = (k / n if n else float("nan")), float("nan"), float("nan")
    return {"value": value, "ci_low": lo, "ci_high": hi, "k": int(k), "n": int(n)}


def _egfr_extras(df: pd.DataFrame) -> dict[str, Any]:
    y = df["y_egfr"].to_numpy(float)
    pos = df["pred_pos"].to_numpy(float)
    out: dict[str, Any] = {}
    if np.isfinite(pos).any():
        tp = int(((pos == 1) & (y == 1)).sum())
        fn = int(((pos == 0) & (y == 1)).sum())
        tn = int(((pos == 0) & (y == 0)).sum())
        fp = int(((pos == 1) & (y == 0)).sum())
        out["sensitivity"] = _proportion(tp, tp + fn)
        out["specificity"] = _proportion(tn, tn + fp)
        out["ppv"] = _proportion(tp, tp + fp)
        out["npv"] = _proportion(tn, tn + fn)
        out["flag_rate"] = _proportion(tp + fp, len(y))
    keep = ~df["abstain"].astype(bool).to_numpy()
    out["coverage"] = _proportion(int(keep.sum()), len(keep))
    out["auroc_retained"] = _auroc(y[keep], df["p_cal"].to_numpy(float)[keep])
    if "conformal_set" in df and df["conformal_set"].notna().any():
        sets = df["conformal_set"].fillna("{}").astype(str).to_numpy()
        covered = np.array([str(int(v)) in s for v, s in zip(y, sets)])
        out["conformal_coverage"] = _proportion(int(covered.sum()), len(y))
        for c in (0, 1):
            m = y == c
            out[f"conformal_coverage_class{c}"] = _proportion(int(covered[m].sum()), int(m.sum()))
        out["conformal_ambiguous_rate"] = _proportion(int((sets == "{0,1}").sum()), len(sets))
        out["conformal_empty_rate"] = _proportion(int((sets == "{}").sum()), len(sets))
    return out


def _surv_extras(df: pd.DataFrame, horizons: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from molt.metrics.survival import time_dependent_auroc
    except Exception:  # noqa: BLE001
        return out
    t, e = df["time"].to_numpy(float), df["event"].to_numpy(float)
    for h in horizons:
        col = f"risk_{int(h)}m"
        if col in df:
            try:
                out[f"auroc_{int(h)}m"] = float(
                    time_dependent_auroc(t, e, df[col].to_numpy(float), h)
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("time-dependent AUROC at %s failed: %s", h, exc)
    keep = ~df["abstain"].astype(bool).to_numpy()
    out["coverage"] = _proportion(int(keep.sum()), len(keep))
    if "risk_stratum" in df:
        out["strata"] = df["risk_stratum"].value_counts().to_dict()
    return out


def summarize_predictions(
    predictions: pd.DataFrame,
    task: str,
    *,
    n_boot: int = 200,
    seed: int = 20240613,
    horizon: float = 24.0,
    horizons: Sequence[float] = (12.0, 24.0, 36.0),
    risk_cutoffs: Sequence[float] = (0.20, 0.40),
) -> dict[str, Any]:
    """Metric panels per mode on the analytic set of ``task``."""
    summary: dict[str, Any] = {"task": task, "modes": {}}
    for mode, df in predictions.groupby("mode", sort=False):
        df = df[df["in_analytic_set"].astype(bool)]
        entry: dict[str, Any] = {"n": len(df), "quick": quick_metrics(df, task)}
        try:
            from molt.metrics import evaluate_binary, evaluate_survival

            if task == "egfr":
                entry["panel"] = evaluate_binary(
                    df,
                    y_col="y_egfr",
                    p_col="p_cal",
                    fold_col="outer_fold",
                    n_boot=n_boot,
                    seed=seed,
                )
            else:
                entry["panel"] = evaluate_survival(
                    df,
                    time_col="time",
                    event_col="event",
                    risk_col="risk_24m",
                    horizon=horizon,
                    tau=max(horizons),
                    fold_col="outer_fold",
                    thresholds=tuple(risk_cutoffs),
                    n_boot=n_boot,
                    seed=seed,
                )
        except Exception as exc:  # noqa: BLE001 - the quick panel is always available
            log.warning("metric panel for %s/%s failed: %s", task, mode, exc)
            entry["panel_error"] = str(exc)
        entry["extras"] = _egfr_extras(df) if task == "egfr" else _surv_extras(df, horizons)
        summary["modes"][str(mode)] = entry
    return _jsonable(summary)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
