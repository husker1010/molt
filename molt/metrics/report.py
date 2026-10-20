"""High-level evaluation reports and persistence of results tables.

:func:`evaluate_binary` and :func:`evaluate_survival` compute the standard MOLT metric panel
for a prediction table with participant-level bootstrap intervals and return tidy
dictionaries::

    {"metrics": {"auroc": {"value": 0.81, "ci_low": 0.76, "ci_high": 0.86, "ci_method": "bootstrap"},
                 ...},
     "info": {"n": 507, "n_pos": 65, "tier": "A", ...}}

:func:`panel_comparison_rows` turns a panel into rows of the cross-model comparison table
(shared metric names for every model), :func:`append_comparison` upserts such rows and
:func:`fill_registry` fills keys of the results registry (named report values). Both files are JSON and
are written atomically under an exclusive file lock shared with the command-line
helpers that manage the same files. The caller supplies the state directory explicitly.
"""

from __future__ import annotations

import datetime as dt
from filelock import FileLock
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

from molt.metrics import binary as B
from molt.metrics import survival as S
from molt.metrics.bootstrap import DEFAULT_N_BOOT, bootstrap, make_strata
from molt.utils.seed import DEFAULT_SEED

__all__ = [
    "COMPARISON_METRICS",
    "append_comparison",
    "comparison_metric_name",
    "comparison_rows",
    "evaluate_binary",
    "evaluate_survival",
    "fill_registry",
    "fill_registry_from_result",
    "load_comparison",
    "load_registry",
    "panel_comparison_rows",
    "state_dir",
    "to_jsonable",
]

_DEFAULT_STATE_DIR = None
_PROPORTIONS = ("sensitivity", "specificity", "ppv", "npv", "fpr", "flag_rate")


def _fmt_threshold(t: float) -> str:
    return f"{t:.2f}"


# --------------------------------------------------------------------------- binary panel


def _binary_stats(
    d: dict[str, np.ndarray],
    threshold: float | None,
    nb_thresholds: Sequence[float],
    ece_bins: int,
) -> dict[str, float]:
    y, p = d["y"], d["p"]
    cal = B.calibration_intercept_slope(y, p)
    out = {
        "auroc": B.auroc(y, p),
        "auprc": B.auprc(y, p),
        "brier": B.brier(y, p),
        "ece": B.expected_calibration_error(y, p, ece_bins),
        "cal_intercept": cal.intercept,
        "cal_slope": cal.slope,
    }
    for t in nb_thresholds:
        out[f"net_benefit@{_fmt_threshold(t)}"] = B.net_benefit(y, p, t)
    if threshold is not None:
        out["net_benefit@threshold"] = B.net_benefit(y, p, threshold)
    return out


def evaluate_binary(
    df: pandas.DataFrame,  # noqa: F821
    y_col: str = "y",
    p_col: str = "p_cal",
    fold_col: str | None = "outer_fold",
    threshold: float | None = None,
    nb_thresholds: Sequence[float] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    cluster_col: str | None = None,
    ece_bins: int = 10,
    enforce_tiers: bool = True,
) -> dict[str, dict]:
    """Evaluate binary predictions (for example EGFR status) with bootstrap intervals.

    Args:
        df: Prediction table with one row per participant.
        y_col: Column with the 0/1 label (rows with a missing label are dropped).
        p_col: Column with the predicted probability.
        fold_col: Column with the outer-fold id used, with the label, to stratify resamples.
        threshold: Optional operating threshold for sensitivity, specificity, PPV, NPV, FPR
            (exact binomial intervals) and net benefit.
        nb_thresholds: Additional threshold probabilities for net benefit.
        n_boot: Number of bootstrap resamples.
        seed: Random seed of the resampling.
        alpha: Two-sided error level.
        n_jobs: Bootstrap worker processes.
        cluster_col: Use a cluster bootstrap over this column (for example ``site``).
        ece_bins: Number of equal-frequency bins for the expected calibration error.
        enforce_tiers: Withhold rank-based metrics (NaN) in reporting tier C.

    Returns:
        ``{"metrics": {name: {value, ci_low, ci_high, ci_method}}, "info": {...}}``.
    """
    cols = [y_col, p_col] + [c for c in (fold_col, cluster_col) if c and c in df.columns]
    sub = df[cols].dropna(subset=[y_col, p_col])
    y = sub[y_col].to_numpy(dtype=float)
    p = sub[p_col].to_numpy(dtype=float)
    strata = make_strata(sub[fold_col].to_numpy(), y) if fold_col in sub.columns else y
    clusters = sub[cluster_col].to_numpy() if cluster_col and cluster_col in sub.columns else None
    stat = partial(
        _binary_stats, threshold=threshold, nb_thresholds=tuple(nb_thresholds), ece_bins=ece_bins
    )
    res = bootstrap(
        stat,
        {"y": y, "p": p},
        n_boot=n_boot,
        strata=strata,
        clusters=clusters,
        seed=seed,
        alpha=alpha,
        n_jobs=n_jobs,
    )
    ci_method = "cluster_bootstrap" if clusters is not None else "bootstrap"
    metrics = {k: {**_tidy(v), "ci_method": ci_method} for k, v in res.summary().items()}
    if "ece" in metrics:
        # ECE is biased upward under resampling: basic interval (see expected_calibration_error).
        lo, hi = res.basic_ci("ece", lower=0.0)
        metrics["ece"].update(ci_low=lo, ci_high=hi, ci_method=f"basic_{ci_method}")

    oe = B.observed_expected(y, p, alpha)
    metrics["oe_ratio"] = {
        "value": oe["oe"],
        "ci_low": oe["ci_low"],
        "ci_high": oe["ci_high"],
        "ci_method": "exact_poisson",
    }
    if threshold is not None:
        for name, val in B.threshold_metrics(y, p, threshold, alpha).items():
            metrics[name] = {
                "value": val["value"],
                "ci_low": val["ci_low"],
                "ci_high": val["ci_high"],
                "ci_method": "clopper_pearson",
                "k": val["k"],
                "n": val["n"],
            }
    tiered = B.tiered_discrimination(y, p)
    if enforce_tiers and tiered["tier"] == "C":
        for name in ("auroc", "auprc", "cal_intercept", "cal_slope"):
            metrics[name] = {
                "value": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "ci_method": "withheld_tier_c",
            }
    info = {
        "n": int(y.size),
        "n_pos": tiered["n_pos"],
        "n_neg": tiered["n_neg"],
        "prevalence": float(np.mean(y > 0.5)) if y.size else float("nan"),
        "tier": tiered["tier"],
        "percentile_ranks": tiered["percentile_ranks"],
        "threshold": threshold,
        "observed": oe["observed"],
        "expected": oe["expected"],
        "n_boot": n_boot,
        "seed": seed,
        "treat_all_net_benefit": {
            _fmt_threshold(t): B.net_benefit_treat_all(y, t)
            for t in list(nb_thresholds) + ([threshold] if threshold is not None else [])
        },
    }
    return {"metrics": metrics, "info": info}


# ------------------------------------------------------------------------- survival panel


def _survival_stats(
    d: dict[str, np.ndarray],
    horizon: float,
    tau: float,
    thresholds: Sequence[float],
    surv_times: Sequence[float] | None,
    calibration_estimator: str,
    competing: bool,
    cause: int,
) -> dict[str, float]:
    t, e_raw, r = d["time"], d["event"], d["risk"]
    e = (e_raw == cause).astype(float) if competing else (e_raw != 0).astype(float)
    cal = S.survival_calibration(t, e, r, horizon, estimator=calibration_estimator)
    out = {
        "auroc": S.time_dependent_auroc(t, e, r, horizon),
        "auprc": S.time_dependent_auprc(t, e, r, horizon),
        "uno_c": S.uno_c(t, e, r, tau),
        "harrell_c": S.harrell_c(t, e, r),
        "brier": S.brier_score(t, e, r, horizon),
        "cal_intercept": cal["intercept"],
        "cal_slope": cal["slope"],
        "ici": cal["ici"],
    }
    if surv_times is not None:
        out["ibs"] = S.integrated_brier_score(t, e, d["surv"], surv_times)
    for th in thresholds:
        out[f"net_benefit@{_fmt_threshold(th)}"] = S.net_benefit_survival(t, e, r, horizon, th)
    if competing:
        out["cr_auroc"] = S.competing_risk_auroc(t, e_raw, r, horizon, cause)
        out["wolbers_c"] = S.wolbers_concordance(t, e_raw, r, cause, tau)
        for th in thresholds:
            out[f"net_benefit_crude@{_fmt_threshold(th)}"] = S.net_benefit_crude(
                t, e_raw, r, horizon, th, cause
            )
    return out


def evaluate_survival(
    df: pandas.DataFrame,  # noqa: F821
    time_col: str = "time",
    event_col: str = "event",
    risk_col: str = "risk_24m",
    horizon: float = 24.0,
    tau: float = 36.0,
    fold_col: str | None = "outer_fold",
    thresholds: Sequence[float] = (0.20, 0.40),
    surv_cols: Mapping[float, str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    cluster_col: str | None = None,
    calibration_estimator: str = "ipcw",
    competing: bool = False,
    cause: int = 1,
) -> dict[str, dict]:
    """Evaluate time-to-event risk predictions with bootstrap intervals.

    Censoring weights are estimated by Kaplan-Meier on the evaluation data and re-estimated
    in every resample. Resampling is stratified by outer fold and event indicator.

    Args:
        df: Prediction table with one row per participant.
        time_col: Follow-up time column.
        event_col: Event column (0/1, or competing-risk codes when ``competing``).
        risk_col: Predicted risk by ``horizon``.
        horizon: Evaluation horizon for AUROC, AUPRC, Brier, calibration and net benefit.
        tau: Truncation time for Uno's concordance and the integrated Brier score.
        fold_col: Outer-fold column used for stratification.
        thresholds: Risk thresholds for net benefit and the observed-risk strata.
        surv_cols: Optional mapping ``time -> column`` of predicted survival probabilities
            for the integrated Brier score over ``[0, tau]``.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.
        alpha: Two-sided error level.
        n_jobs: Bootstrap worker processes.
        cluster_col: Use a cluster bootstrap over this column.
        calibration_estimator: ``"ipcw"`` or ``"pseudo"`` (see
            :func:`molt.metrics.survival.survival_calibration`).
        competing: Add competing-risk AUROC, Wolbers' concordance and crude-risk net benefit;
            single-risk metrics then treat competing events as censored.
        cause: Code of the event of interest.

    Returns:
        ``{"metrics": {...}, "info": {...}}`` with the Kaplan-Meier observed risk per
        predicted-risk stratum and treat-all net benefit in ``info``.
    """
    cols = [time_col, event_col, risk_col]
    grid = sorted((float(k), c) for k, c in (surv_cols or {}).items() if float(k) <= tau)
    times = [g[0] for g in grid]
    cols += [g[1] for g in grid]
    extra = [c for c in (fold_col, cluster_col) if c and c in df.columns]
    sub = df[cols + extra].dropna(subset=cols)
    data = {
        "time": sub[time_col].to_numpy(dtype=float),
        "event": sub[event_col].to_numpy(dtype=float),
        "risk": sub[risk_col].to_numpy(dtype=float),
    }
    if times:
        data["surv"] = sub[cols[3:]].to_numpy(dtype=float)
    e_main = (data["event"] == cause) if competing else (data["event"] != 0)
    strata = (
        make_strata(sub[fold_col].to_numpy(), e_main)
        if fold_col in sub.columns
        else e_main.astype(int)
    )
    clusters = sub[cluster_col].to_numpy() if cluster_col and cluster_col in sub.columns else None
    stat = partial(
        _survival_stats,
        horizon=horizon,
        tau=tau,
        thresholds=tuple(thresholds),
        surv_times=times or None,
        calibration_estimator=calibration_estimator,
        competing=competing,
        cause=cause,
    )
    res = bootstrap(
        stat,
        data,
        n_boot=n_boot,
        strata=strata,
        clusters=clusters,
        seed=seed,
        alpha=alpha,
        n_jobs=n_jobs,
    )
    ci_method = "cluster_bootstrap" if clusters is not None else "bootstrap"
    metrics = {k: {**_tidy(v), "ci_method": ci_method} for k, v in res.summary().items()}

    t, r = data["time"], data["risk"]
    e01 = e_main.astype(float)
    status = S.horizon_status(t, e01, horizon)
    n_case, n_ctrl = int(np.sum(status == 1)), int(np.sum(status == 0))
    tier = B.reporting_tier(n_case, n_ctrl)
    if tier == "C":
        for name in ("auroc", "auprc", "cal_intercept", "cal_slope", "cr_auroc"):
            if name in metrics:
                metrics[name] = {
                    "value": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ci_method": "withheld_tier_c",
                }
    km_all = S.kaplan_meier(t, e01)
    risk_all = 1.0 - float(km_all(horizon))
    info = {
        "n": int(t.size),
        "n_events": int(e01.sum()),
        "events_by_horizon": n_case,
        "event_free_through_horizon": n_ctrl,
        "censored_before_horizon": int(t.size - n_case - n_ctrl),
        "tier": tier,
        "horizon": horizon,
        "tau": tau,
        "km_risk_at_horizon": risk_all,
        "risk_strata": S.km_risk_by_stratum(
            t, data["event"] if competing else e01, r, horizon, thresholds, alpha, cause
        ),
        "treat_all_net_benefit": {
            _fmt_threshold(th): risk_all - (1 - risk_all) * th / (1 - th) for th in thresholds
        },
        "n_boot": n_boot,
        "seed": seed,
    }
    return {"metrics": metrics, "info": info}


def _tidy(summary: Mapping[str, float]) -> dict[str, float]:
    return {"value": summary["value"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"]}


# ------------------------------------------------------------------------- persistence


def state_dir(path: str | os.PathLike | None = None) -> Path:
    """Require an explicit directory for persistent evaluation summaries."""
    if not path:
        raise ValueError("An explicit state directory is required")
    return Path(path).expanduser().resolve()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _json_value(v: object) -> object:
    """Convert NumPy scalars to Python and non-finite floats to ``None``."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def to_jsonable(obj: object) -> object:
    """Recursively convert an evaluation result to strict JSON types.

    NumPy scalars and arrays become Python numbers and lists, and non-finite floats become
    ``None``, so the output can be written with ``json.dump(..., allow_nan=False)``.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return _json_value(obj)


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        yield


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _write_json_atomic(path: Path, obj: object) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def comparison_rows(
    result: Mapping[str, Mapping],
    task: str,
    cohort: str,
    model: str,
    mode: str,
    run_id: str = "",
    metrics: Sequence[str] | None = None,
) -> list[dict]:
    """Convert an evaluation result into comparison-table rows.

    Args:
        result: Output of :func:`evaluate_binary` or :func:`evaluate_survival`.
        task: Task id (``egfr``, ``pfi``, ``os`` ...).
        cohort: Cohort id.
        model: Model name.
        mode: Inference mode (``full`` or ``light``).
        run_id: Source run id.
        metrics: Subset of metric names (default: all).

    Returns:
        List of row dicts with the comparison-table schema.
    """
    n = result.get("info", {}).get("n")
    rows = []
    for name, m in result["metrics"].items():
        if metrics is not None and name not in metrics:
            continue
        rows.append(
            {
                "task": task,
                "cohort": cohort,
                "model": model,
                "mode": mode,
                "metric": name,
                "value": _json_value(m.get("value")),
                "ci_low": _json_value(m.get("ci_low")),
                "ci_high": _json_value(m.get("ci_high")),
                "n": _json_value(n),
                "run_id": run_id,
            }
        )
    return rows


COMPARISON_METRICS: dict[str, tuple[str, ...]] = {
    "binary": ("auroc", "auprc", "brier", "ece", "cal_intercept", "cal_slope", "oe_ratio"),
    "survival": (
        "auroc",
        "auprc",
        "brier",
        "harrell_c",
        "uno_c",
        "cal_intercept",
        "cal_slope",
        "ici",
    ),
}
"""Panel metrics reported in the cross-model comparison table, per endpoint type."""

_CONCORDANCE_NAMES = {"harrell_c": "c_harrell", "uno_c": "c_uno"}


def comparison_metric_name(metric: str, horizon: float | None = None) -> str:
    """Comparison-table name of a panel metric.

    Binary metrics keep their names. Survival metrics evaluated at a horizon carry it as a
    suffix (``auroc`` -> ``auroc_24m``); concordance indices, which are not
    horizon-specific, are named ``c_harrell`` and ``c_uno``.
    """
    if horizon is None:
        return metric
    if metric in _CONCORDANCE_NAMES:
        return _CONCORDANCE_NAMES[metric]
    h = float(horizon)
    return f"{metric}_{int(h) if h.is_integer() else h}m"


def panel_comparison_rows(
    result: Mapping[str, Mapping],
    task: str,
    cohort: str,
    model: str,
    mode: str,
    run_id: str = "",
) -> list[dict]:
    """Comparison rows of the standard metrics (:data:`COMPARISON_METRICS`) of a panel.

    Every model is reported through the same panel functions, so rows of different models
    share metric names and estimators.

    Args:
        result: Output of :func:`evaluate_binary` or :func:`evaluate_survival` (detected
            from the ``horizon`` entry of ``info``).
        task: Task id.
        cohort: Cohort id.
        model: Model name.
        mode: Inference mode.
        run_id: Source run id.
    """
    horizon = (result.get("info") or {}).get("horizon")
    kind = "binary" if horizon is None else "survival"
    rows = comparison_rows(
        result, task, cohort, model, mode, run_id, metrics=COMPARISON_METRICS[kind]
    )
    for row in rows:
        row["metric"] = comparison_metric_name(row["metric"], horizon)
    return rows


def load_comparison(path: str | os.PathLike | None = None) -> dict:
    """Read the comparison table (``{"updated", "rows"}``)."""
    file = state_dir(path) / "comparison.json"
    data = _read_json(file, {"updated": None, "rows": []})
    return data if isinstance(data, dict) else {"updated": None, "rows": []}


def append_comparison(rows: Sequence[Mapping], path: str | os.PathLike | None = None) -> int:
    """Upsert rows into ``comparison.json`` keyed by (task, cohort, model, mode, metric).

    Args:
        rows: Rows as produced by :func:`comparison_rows`.
        path: State directory (default :func:`state_dir`).

    Returns:
        Number of rows in the table after the update.
    """
    sdir = state_dir(path)
    file = sdir / "comparison.json"
    keys = ("task", "cohort", "model", "mode", "metric")
    with _locked(sdir / ".comparison.lock"):
        table = load_comparison(sdir)
        index = {tuple(r.get(k) for k in keys): i for i, r in enumerate(table["rows"])}
        for row in rows:
            clean = {k: _json_value(v) for k, v in row.items()}
            key = tuple(clean.get(k) for k in keys)
            if key in index:
                table["rows"][index[key]] = clean
            else:
                index[key] = len(table["rows"])
                table["rows"].append(clean)
        table["updated"] = _now()
        _write_json_atomic(file, table)
    return len(table["rows"])


def load_registry(path: str | os.PathLike | None = None) -> dict:
    """Read the results registry."""
    data = _read_json(state_dir(path) / "results_registry.json", {})
    return data if isinstance(data, dict) else {}


def fill_registry(
    entries: Mapping[str, object] | str,
    value: object = None,
    ci_low: object = None,
    ci_high: object = None,
    source: str = "",
    section: str | None = None,
    description: str | None = None,
    fmt: str | None = None,
    path: str | os.PathLike | None = None,
) -> int:
    """Fill one or several keys of ``results_registry.json``.

    Either call ``fill_registry("key", value, ci_low, ci_high, source=...)`` or pass a mapping
    ``{key: value | (value, ci_low, ci_high) | {"value", "ci_low", "ci_high", ...}}``. Filled
    entries get ``status="filled"``, a timestamp and the source run; existing section,
    anchor, description and format fields are kept unless overridden.

    Returns:
        Number of keys written.
    """
    if isinstance(entries, str):
        entries = {entries: {"value": value, "ci_low": ci_low, "ci_high": ci_high}}
    sdir = state_dir(path)
    file = sdir / "results_registry.json"
    with _locked(sdir / ".registry.lock"):
        reg = load_registry(sdir)
        for key, spec in entries.items():
            if isinstance(spec, Mapping):
                vals = dict(spec)
            elif isinstance(spec, (tuple, list)):
                vals = dict(zip(("value", "ci_low", "ci_high"), spec))
            else:
                vals = {"value": spec}
            entry = reg.get(key) or {
                "section": section or "unknown",
                "anchor": "",
                "description": description or key,
                "fmt": fmt or "{}",
            }
            for field, override in (
                ("section", section),
                ("description", description),
                ("fmt", fmt),
            ):
                if vals.get(field) or override:
                    entry[field] = vals.get(field) or override
            entry["value"] = _json_value(vals.get("value"))
            entry["ci_low"] = _json_value(vals.get("ci_low"))
            entry["ci_high"] = _json_value(vals.get("ci_high"))
            entry["status"] = "filled" if entry["value"] is not None else "placeholder"
            entry["updated"] = _now()
            entry["source"] = vals.get("source", source)
            reg[key] = entry
        _write_json_atomic(file, reg)
    return len(entries)


def fill_registry_from_result(
    result: Mapping[str, Mapping],
    mapping: Mapping[str, str],
    source: str = "",
    path: str | os.PathLike | None = None,
) -> int:
    """Fill registry keys from an evaluation result.

    Args:
        result: Output of :func:`evaluate_binary` or :func:`evaluate_survival`.
        mapping: ``registry_key -> metric name`` (``info`` fields such as ``n`` are allowed).
        source: Source run id.
        path: State directory.

    Returns:
        Number of keys written.
    """
    entries: dict[str, object] = {}
    for key, name in mapping.items():
        if name in result["metrics"]:
            m = result["metrics"][name]
            entries[key] = (m.get("value"), m.get("ci_low"), m.get("ci_high"))
        elif name in result.get("info", {}):
            entries[key] = result["info"][name]
        else:
            raise KeyError(f"metric {name!r} not found in result")
    return fill_registry(entries, source=source, path=path)
