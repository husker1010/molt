"""Locked-ensemble prediction for cohorts that never entered the development folds.

The locked model of an endpoint and mode is the ensemble of the outer-fold predictors of a
nested cross-validation run. Each outer predictor applies its own preprocessing, averages
its inner-fold ensembles and applies its own recalibration map; the locked output is the
mean recalibrated probability (EGFR) or the mean recalibrated survival curve (PFI/OS).
The EGFR threshold and the abstention cut-off were derived once on the pooled recalibrated
outer out-of-fold predictions and are applied unchanged; split-conformal p-values use the
pooled outer out-of-fold predictions as the calibration set.

Example:
    >>> from molt.training.locked import predict_locked
    >>> df = predict_locked("molt_egfr_v1", "tcga_lusc", device="cuda:0")
"""

from __future__ import annotations

import json
import dataclasses
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molt.calibration import (
    abstain,
    conformal_sets,
    format_conformal_set,
    mondrian_conformal_pvalues,
)
from molt.config import load_config
from molt.training.ct_patches import CTPatchSampler
from molt.training.data import MOLTDataset
from molt.training.evaluate import summarize_predictions
from molt.training.model_factory import model_builder, resolve_backend
from molt.training.nested_cv import (
    MODE_OMIT,
    comparison_label,
    ct_feature_store,
    load_bundle_from_config,
    publish_comparison,
)
from molt.training.postprocess import (
    PostProcessing,
    apply_postprocessing,
    labels_for,
    prediction_frame,
    survival_at,
)
from molt.training.settings import run_settings
from molt.training.trainer import Ensemble
from molt.utils.logging import get_logger
from molt.utils.status import MetricsLogger, RunStatus, resolve_runs_root

__all__ = ["LockedEnsemble", "predict_locked"]

log = get_logger(__name__)


class LockedEnsemble:
    """The locked outer-fold predictors of a nested cross-validation run.

    Args:
        run_dir: Directory of the development run (contains ``locked/manifest.json``).
        device: Device used for prediction.
    """

    def __init__(self, run_dir: str | Path, device: str = "cpu") -> None:
        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / "locked" / "manifest.json").read_text())
        self.cfg = load_config(self.run_dir / "config.yaml", with_defaults=False)
        self.task = str(self.manifest["task"])
        self.settings = run_settings(self.cfg, self.task)
        self.device = device

    @property
    def modes(self) -> list[str]:
        return list(self.manifest["modes"])

    def _inner_models(self, directory: Path) -> tuple[Any, Ensemble, dict[str, Any]]:
        with (directory / "preprocessor.pkl").open("rb") as fh:
            pre = pickle.load(fh)
        with (directory / "model_config.pkl").open("rb") as fh:
            meta = pickle.load(fh)
        backend = resolve_backend(meta.get("backend", "auto"))
        meta["config"] = dataclasses.replace(meta["config"], swin_checkpoint=None)
        builder = model_builder(
            backend, meta["config"], freeze_ct_backbone=bool(meta.get("ct_feature_cache"))
        )
        ens = Ensemble.load(directory, builder, device=self.device, amp=self.settings.train.amp)
        return pre, ens, {**meta, "backend_obj": backend}

    def predict(self, bundle: Any, ids: Sequence[str] | None = None) -> pd.DataFrame:
        """Locked predictions (all modes) for ``ids`` of ``bundle`` (default: all participants)."""
        st = self.settings
        ids = [str(i) for i in (ids if ids is not None else bundle.table.index)]
        sampler = CTPatchSampler(st.ct, seed=st.seed) if st.use_ct else None
        store = None
        per_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in self.modes}
        edges_all: list[np.ndarray] = []
        flags = np.zeros(len(ids))
        for outer in self.manifest["outer"]:
            edges = np.asarray(outer["bin_edges"], dtype=float)
            edges_all.append(edges)
            inner_preds: dict[str, list[Any]] = {m: [] for m in self.modes}
            for directory in outer["inner_models"]:
                model_dir = Path(directory)
                if not model_dir.is_absolute():
                    model_dir = self.run_dir / model_dir
                pre, ens, meta = self._inner_models(model_dir)
                if store is None and meta.get("ct_feature_cache"):
                    store = ct_feature_store(
                        bundle,
                        ids,
                        st,
                        meta["backend_obj"],
                        meta["config"],
                        self.device,
                        views=1,
                        model_state=ens.states[0],
                    )
                feats = pre.transform(bundle, ids)
                if not st.gene_targets:
                    feats.rna_genes = None
                flags = feats.rna_flag
                ds = MOLTDataset(feats, sampler, train=False, ct_store=store)
                for mode in self.modes:
                    inner_preds[mode].append(
                        ens.predict(
                            ds,
                            omit=MODE_OMIT[mode],
                            mc_samples=st.train.mc_samples,
                            batch_size=st.train.eval_batch_size,
                            num_workers=st.train.num_workers,
                        )
                    )
            for mode in self.modes:
                post = PostProcessing.from_dict(outer["postprocessing"][mode])
                post.conformal = []
                arrays = apply_postprocessing(post, inner_preds[mode], edges, horizons=st.horizons)
                arrays["edges"] = edges
                per_mode[mode].append(arrays)
            log.info(
                "locked: outer predictor %s scored %d participants", outer["outer_fold"], len(ids)
            )
        labels = labels_for(
            bundle.table,
            ids,
            self.task,
            require_modality=st.preprocess.require_modality_for_survival,
        )
        frames = []
        for mode in self.modes:
            arrays = self._combine(per_mode[mode], mode, flags)
            frame = prediction_frame(
                ids=ids,
                table=bundle.table,
                task=self.task,
                mode=mode,
                outer_fold=-1,
                arrays=arrays,
                labels=labels,
                flags=flags,
                risk_cutoffs=st.risk_cutoffs,
                horizon=st.preprocess.horizon,
            )
            if self.task != "egfr":
                frame["bin_width"] = float(arrays["edges"][1] - arrays["edges"][0])
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _combine(self, parts: list[dict[str, Any]], mode: str, flags: np.ndarray) -> dict[str, Any]:
        st = self.settings
        lock = self.manifest["modes"][mode]
        unc = np.mean([p["uncertainty"] for p in parts], axis=0)
        use_flags = flags if (st.abstain_on_quality_flag and mode != "light") else None
        out: dict[str, Any] = {
            "uncertainty": unc,
            "unc_ensemble": np.mean([p["unc_ensemble"] for p in parts], axis=0),
            "unc_mc": np.mean([p["unc_mc"] for p in parts], axis=0),
            "abstain_cutoff": np.full(len(unc), lock["abstain_cutoff"]),
            "abstain": abstain(unc, lock["abstain_cutoff"], use_flags),
            "rho": np.mean([p["rho"] for p in parts], axis=0),
            "p_raw": np.mean([p["p_raw"] for p in parts], axis=0),
        }
        if self.task == "egfr":
            p_cal = np.mean([p["p_cal"] for p in parts], axis=0)
            thr = float(lock["threshold"])
            out.update(
                p_cal=p_cal,
                threshold=np.full(len(p_cal), thr),
                pred_pos=(p_cal >= thr).astype(float),
            )
            cal = [
                (np.asarray(c["p"], float), np.asarray(c["y"], float))
                for c in lock.get("conformal", [])
            ]
            if cal:
                pvals = mondrian_conformal_pvalues(cal, [p_cal] * len(cal))
                out.update(
                    conformal_p0=pvals[:, 0],
                    conformal_p1=pvals[:, 1],
                    conformal_set=format_conformal_set(
                        conformal_sets(pvals, self.manifest["conformal_alpha"])
                    ),
                )
            return out
        edges = min((p["edges"] for p in parts), key=lambda e: e[1] - e[0])
        grid = edges[1:]

        def on_grid(curve: np.ndarray, e: np.ndarray) -> np.ndarray:
            return np.column_stack([survival_at(curve, e, t) for t in grid])

        surv_cal = np.mean([on_grid(p["surv_cal"], p["edges"]) for p in parts], axis=0)
        surv_raw = np.mean([on_grid(p["surv_raw"], p["edges"]) for p in parts], axis=0)
        out.update(
            surv_cal=surv_cal,
            surv_raw=surv_raw,
            edges=edges,
            p_cal=1.0 - survival_at(surv_cal, edges, st.preprocess.horizon),
        )
        for h in st.horizons:
            out[f"risk_{int(h)}m"] = 1.0 - survival_at(surv_cal, edges, h)
        return out


def predict_locked(
    run_id: str,
    cohort: str,
    *,
    runs_root: str | Path | None = None,
    device: str = "cpu",
    out_run_id: str | None = None,
    n_boot: int = 200,
    cohorts_dir: str | Path | None = None,
    ct_dir: str | Path | None = None,
    pathways_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Score ``cohort`` with the locked ensemble of ``run_id`` and write a locked-evaluation run.

    Outputs go to ``<runs_root>/<out_run_id>/`` (default ``<run_id>-locked-<cohort>``):
    ``status.json``, ``metrics.jsonl``, ``predictions.parquet`` and ``summary.json``. The
    standard metrics of the summary are upserted into the comparison table under the model
    label of the development run and ``cohort``.
    """
    root = resolve_runs_root(runs_root)
    locked = LockedEnsemble(root / run_id, device=device)
    if cohorts_dir is None:
        raise ValueError("cohorts_dir must be supplied for prediction")
    for key, value in {
        "cohorts_dir": cohorts_dir,
        "ct_dir": ct_dir,
        "pathways_dir": pathways_dir,
    }.items():
        locked.cfg.set_path(f"paths.{key}", str(Path(value).resolve()) if value else None)
    if locked.settings.ct is not None:
        locked.settings.ct.cache_dir = None
    locked.settings.ct_feature_dir = None
    out_id = out_run_id or f"{run_id}-locked-{cohort}"
    status = RunStatus(
        out_id,
        task=locked.task,
        model=f"{locked.cfg.get('experiment', {}).get('model', 'molt')}_locked",
        cohorts=[cohort],
        runs_root=root,
        gpu=(
            int(device.split(":")[1])
            if device.startswith("cuda:")
            else (0 if device == "cuda" else None)
        ),
    )
    status.start(f"locked evaluation of {run_id} on {cohort}")
    try:
        bundle = load_bundle_from_config(locked.cfg, [cohort], locked.settings)
        df = locked.predict(bundle)
        out_dir = root / out_id
        df.to_parquet(out_dir / "predictions.parquet", index=False)
        summary = summarize_predictions(
            df,
            locked.task,
            n_boot=n_boot,
            seed=locked.settings.seed,
            horizon=locked.settings.preprocess.horizon,
            horizons=locked.settings.horizons,
            risk_cutoffs=locked.settings.risk_cutoffs,
        )
        summary.update(run_id=out_id, source_run=run_id, cohort=cohort)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        publish_comparison(
            summary,
            locked.cfg,
            task=locked.task,
            cohort=cohort,
            model=comparison_label(
                locked.cfg, "stub" if locked.manifest.get("backend") == "stub" else None
            ),
            run_id=out_id,
        )
        metrics = MetricsLogger(out_id, runs_root=root)
        for mode, entry in summary.get("modes", {}).items():
            for name, value in entry.get("quick", {}).items():
                if name not in ("n", "n_pos", "events"):
                    metrics.log(name, value, split="test", member=mode)
        index = root / run_id / "locked" / "evaluations.json"
        evaluations = json.loads(index.read_text()) if index.exists() else {}
        evaluations[cohort] = {"run_id": out_id, "n": int(df["participant_id"].nunique())}
        index.write_text(json.dumps(evaluations, indent=1))
        status.finish(f"locked predictions for {df['participant_id'].nunique()} participants")
        return df
    except BaseException as exc:
        status.fail(f"{type(exc).__name__}: {exc}")
        raise
