"""Nested cross-validation with random search, cross-fitting and a locked ensemble.

For each outer fold ``k`` and each of the sampled hyperparameter configurations ``c``, one
deep ensemble is trained per inner fold ``j`` (preprocessing fitted on the inner training
set, early stopping on the inner validation fold). The configuration with the lowest pooled
inner out-of-fold loss is selected (EGFR: log-loss after cross-fitted logistic
recalibration; survival: discrete-time NLL). The three inner models of the selected
configuration provide out-of-fold predictions for every outer-training participant, on which
recalibration, thresholds, abstention cut-offs and conformal scores are fitted, and their
average scores outer test fold ``k`` (cross-fitting). The full and light modes are produced
by the same fitted networks (light = transcriptomic expert omitted). ``search.per_outer_fold``
(``{outer: overrides}``) fixes one configuration per outer fold instead of the search, and
``data.rna.batch_correction: combat`` corrects expression per inner fit (:mod:`molt.training.combat`).

Artefacts in ``<runs_root>/<run_id>/``::

    config.yaml  status.json  metrics.jsonl  search_space.json  folds.parquet
    selection.json        per outer fold: loss of every configuration and the selection
    predictions.parquet   pooled outer-test predictions (all modes)
    inner_oof.parquet     inner out-of-fold predictions of the selected configurations
    summary.json          metric panels of the pooled predictions
    checkpoints/outer{k}/inner{j}/  member*.pt, preprocessor.pkl, model_config.pkl, meta.json
    locked/manifest.json  the locked ensemble (5 outer predictors + post-processing)

The standard metrics of the pooled panels are also upserted into the cross-model comparison
table (``comparison.json`` in ``paths.state_dir``, explicit caller-supplied directory; disabled with
``evaluation.comparison: false``). Rows are labelled with ``experiment.model`` (suffixed by the
ablation name for ablation runs) unless ``experiment.comparison_label`` is set.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import pickle
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molt.calibration import abstention_cutoff, sensitivity_threshold
from molt.config import Config, apply_overrides
from molt.splits import NestedFolds, default_splits_path, load_or_create_folds
from molt.training.cohort_data import CohortBundle, development_universe, load_bundle
from molt.training.combat import fold_bundle, load_batches
from molt.training.ct_features import CTFeatureStore, build_feature_store, feature_cache_key
from molt.training.ct_patches import CTPatchSampler
from molt.training.data import MOLTDataset
from molt.training.evaluate import quick_metrics, summarize_predictions
from molt.training.gene_panel import bundle_gene_universe
from molt.training.model_factory import (
    build_loss_weights,
    build_model_config,
    model_builder,
    model_kwargs,
    resolve_backend,
)
from molt.training.parallel import parse_devices, run_jobs_parallel
from molt.training.postprocess import (
    apply_postprocessing,
    fit_postprocessing,
    labels_for,
    prediction_frame,
    selection_objective,
)
from molt.training.preprocess import Preprocessor
from molt.training.publish import comparison_label, publish_comparison
from molt.training.rna_augment import RNAAugmenter
from molt.training.settings import RunSettings, get, run_settings
from molt.training.trainer import EnsemblePrediction, Trainer, load_trainable_state
from molt.utils.hashing import stable_hash
from molt.utils.logging import get_logger
from molt.utils.seed import derive_seed, seed_everything
from molt.utils.status import MetricsLogger, RunStatus, resolve_runs_root

__all__ = [
    "MODE_OMIT",
    "InnerOutputs",
    "NestedCV",
    "comparison_label",
    "ct_feature_store",
    "development_folds",
    "load_bundle_from_config",
    "publish_comparison",
    "sample_configs",
]

log = get_logger(__name__)

MODE_OMIT: dict[str, tuple[str, ...]] = {
    "full": (),
    "light": ("rna",),
    "no_ct": ("ct",),
    "clinical": ("rna", "ct"),
    "no_gene": ("gene",),
    "no_lin": ("clinical_lin", "gene_lin", "rna_lin"),
}
"""Experts omitted at inference in each prediction mode (``rna`` omits every transcriptomic
expert: pathway, gene-level and stacked transcriptomic scores; ``gene`` the gene-level
experts; ``no_lin`` the stacked linear experts)."""


_VOLATILE_EXPERIMENT = ("parallel_jobs", "worker_threads", "outer_folds_subset", "keep_work")
_VOLATILE_KEYS = {
    "experiment": (*_VOLATILE_EXPERIMENT, "keep_all_checkpoints"),
    "train": ("num_workers", "eval_batch_size"),
}
"""Settings that do not change fitted models and are ignored when resuming a run."""


def sample_configs(grid: Mapping[str, Sequence[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Sample ``n`` distinct configurations (dotted overrides) from a grid, deterministically."""
    keys = sorted(grid)
    if not keys:
        return [{}]
    combos = list(itertools.product(*(list(grid[k]) for k in keys)))
    rng = np.random.default_rng(derive_seed(seed, "search"))
    pick = (
        range(len(combos))
        if len(combos) <= n
        else sorted(rng.choice(len(combos), n, replace=False))
    )
    return [dict(zip(keys, combos[i])) for i in pick]


@dataclass
class InnerOutputs:
    """Predictions of one inner-fold ensemble of one configuration."""

    outer_fold: int
    config_id: int
    inner_fold: int
    val_ids: list[str]
    test_ids: list[str]
    preds: dict[str, dict[str, EnsemblePrediction]]
    val_flags: np.ndarray
    test_flags: np.ndarray
    bin_edges: np.ndarray
    best_epochs: list[int]
    seconds: float
    checkpoint_dir: str


def load_bundle_from_config(
    cfg: Mapping[str, Any], cohorts: Sequence[str], settings: RunSettings
) -> CohortBundle:
    """Load cohorts using the paths of ``cfg`` and the modality switches of ``settings``.

    The full gene universe is kept when the gene-level expert is enabled in the configuration
    or its search grid (``data.rna.gene_panel.size > 0``), otherwise only pathway genes. The
    module-adjacency prior is ``model.pathway_prior`` (``strict``, ``curated_reactome`` or
    ``full``, default ``full``; see :func:`~molt.training.cohort_data.load_pathways`).
    """
    extra: list[str] = []
    sizes = [settings.preprocess.gene_panel_size]
    sizes += list(dict(get(cfg, "search.grid", {})).get("data.rna.gene_panel.size", []))
    if settings.preprocess.use_rna and any(int(v or 0) > 0 for v in sizes):
        extra = bundle_gene_universe(cohorts, get(cfg, "paths.cohorts_dir"))
    return load_bundle(
        cohorts,
        cohorts_dir=get(cfg, "paths.cohorts_dir"),
        ct_dir=get(cfg, "paths.ct_dir"),
        pathways_dir=get(cfg, "paths.pathways_dir"),
        use_expression=settings.preprocess.use_rna,
        use_ct=settings.use_ct,
        extra_genes=extra,
        pathway_prior=get(cfg, "model.pathway_prior", "full"),
        ct_selection=dict(get(cfg, "data.ct", {})),
    )


def development_folds(
    cfg: Mapping[str, Any],
    bundle: CohortBundle,
    universe: pd.Index,
    cohorts: Sequence[str],
    settings: RunSettings,
) -> NestedFolds:
    """Load (or create once) the shared nested fold assignment of the development universe.

    The file lives in ``paths.splits_dir`` (default ``<processed_dir>/splits``) unless
    ``cv.path`` names it explicitly; ``cv.rebuild: true`` regenerates it.
    """
    splits_dir = get(cfg, "paths.splits_dir")
    if not splits_dir and not get(cfg, "cv.path"):
        raise ValueError("Supply paths.splits_dir or cv.path")
    n_outer, n_inner = int(get(cfg, "cv.outer_folds", 5)), int(get(cfg, "cv.inner_folds", 3))
    seed = int(get(cfg, "cv.seed", settings.seed))
    path = get(cfg, "cv.path") or default_splits_path(splits_dir, cohorts, n_outer, n_inner, seed)
    table = bundle.table.loc[universe].reset_index()
    extra = get(cfg, "cv.extra_indicator")
    if extra and not str(extra).startswith("re:") and extra not in table.columns:
        extra = None
    return load_or_create_folds(
        table,
        path,
        n_outer=n_outer,
        n_inner=n_inner,
        seed=seed,
        rebuild=bool(get(cfg, "cv.rebuild", False)),
        endpoints=tuple(get(cfg, "cv.stratify_on", ["egfr_status", "pfi_event", "os_event"])),
        horizon_months=settings.preprocess.horizon,
        extra=extra,
    )


def ct_feature_store(
    bundle: CohortBundle,
    ids: Sequence[str],
    settings: RunSettings,
    backend: Any,
    model_cfg: Any | None,
    device: str,
    *,
    views: int | None = None,
    model_state: Mapping[str, Any] | None = None,
) -> CTFeatureStore | None:
    """Frozen-backbone CT features of the participants of ``ids`` that have a selected CT.

    ``model_cfg`` defines the backbone; when ``None`` it is derived from a preprocessor fitted
    on ``ids`` (only the imaging settings matter). Returns ``None`` if the backend has no
    imaging encoder.
    """
    if not (settings.use_ct and settings.ct_feature_cache):
        return None
    table = bundle.table.loc[list(ids)]
    items = [
        (pid, str(path))
        for pid, path, ok in zip(table.index, table["ct_path"], table["has_ct"])
        if ok and isinstance(path, str)
    ]
    sampler = CTPatchSampler(settings.ct, seed=settings.seed)
    if model_state is None and (
        settings.model.get("ct_backbone") != "swin_vit" or not settings.model.get("swin_checkpoint")
    ):
        raise ValueError(
            "Frozen feature caching requires a caller-supplied local Swin state; use feature_cache=false to train the backbone"
        )
    if model_cfg is None:
        pre = Preprocessor(settings.preprocess).fit(bundle, list(ids))
        model_cfg = build_model_config(backend, model_kwargs(pre, settings.model, use_ct=True))
    if not getattr(model_cfg, "use_ct", False):
        return None
    builder = model_builder(backend, model_cfg)

    def build_cached_model():
        model = builder()
        if model_state is not None:
            load_trainable_state(model, model_state)
        return model

    try:
        return build_feature_store(
            build_cached_model,
            sampler,
            items,
            views=views or settings.ct_feature_views,
            device=device,
            cache_dir=settings.ct_feature_dir if model_state is None else None,
            key=feature_cache_key(model_cfg, sampler) if model_state is None else None,
            amp=settings.train.amp,
        )
    except ValueError as exc:
        log.warning("CT feature cache unavailable (%s); sampling patches instead", exc)
        return None


def _device_index(device: str) -> int | None:
    if device.startswith("cuda"):
        return int(device.split(":")[1]) if ":" in device else 0
    return None


class _Progress:
    def __init__(self, status: RunStatus, total_units: float) -> None:
        self.status = status
        self.total = max(total_units, 1.0)
        self.done = 0.0
        self.prefix = ""
        self._last = 0.0

    def advance(self, units: float, message: str | None = None, force: bool = False) -> None:
        self.done += units
        now = time.time()
        if force or now - self._last > 2.0:
            self._last = now
            self.status.update(
                progress=min(self.done / self.total, 0.999), message=message or self.prefix
            )


class NestedCV:
    """Run the nested cross-validation of one endpoint.

    Args:
        cfg: Resolved configuration (see :mod:`molt.training.settings`).
        task: ``egfr``, ``pfi`` or ``os``.
        run_id: Run identifier (directory under ``runs_root``).
        cohorts: Development cohorts (default ``cfg.cohorts.development``).
        device: Torch device string.
        runs_root: Explicit root of run directories (or ``paths.runs_root`` in the configuration).
        bundle: Preloaded cohort bundle (optional).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        task: str,
        run_id: str,
        cohorts: Sequence[str] | None = None,
        device: str = "cpu",
        runs_root: str | Path | None = None,
        bundle: CohortBundle | None = None,
    ) -> None:
        self.cfg = cfg
        self.task = task
        self.run_id = run_id
        self.cohorts = list(cohorts or get(cfg, "cohorts.development", []))
        if not self.cohorts:
            raise ValueError("Supply cohorts or cohorts.development")
        self.device_spec = device
        self.device = parse_devices(device)[0]
        self.parallel_jobs = int(get(cfg, "experiment.parallel_jobs", 1) or 1)
        self.worker_threads = int(get(cfg, "experiment.worker_threads", 0) or 0)
        self.runs_root = resolve_runs_root(runs_root or get(cfg, "paths.runs_root"))
        self.run_dir = self.runs_root / run_id
        self.settings = run_settings(cfg, task)
        self.bundle = bundle
        self.n_outer = int(get(cfg, "cv.outer_folds", 5))
        self.n_inner = int(get(cfg, "cv.inner_folds", 3))
        self.keep_all = bool(get(cfg, "experiment.keep_all_checkpoints", False))
        grid = dict(get(cfg, "search.grid", {}))
        self.configs = sample_configs(
            grid, int(get(cfg, "search.n_configs", 8)), self.settings.seed
        )
        # ``search.per_outer_fold``: one fixed configuration per outer fold (no inner search)
        fixed = {int(k): dict(v) for k, v in dict(get(cfg, "search.per_outer_fold") or {}).items()}
        self.fixed: dict[int, int] | None = None
        if fixed:
            self.configs = [
                v for i, v in enumerate(fixed.values()) if v not in list(fixed.values())[:i]
            ]
            self.fixed = {k: self.configs.index(v) for k, v in fixed.items()}
        self.batch_correction = str(get(cfg, "data.rna.batch_correction", "none") or "none")
        widths = [float(v) for v in grid.get("data.survival.bin_width_months", [])]
        widths = widths or [self.settings.preprocess.bin_width]
        common = float(math.lcm(*[int(round(w)) for w in widths]))  # noqa: RUF046
        self.grid_width = (
            common if all(abs(common / w - round(common / w)) < 1e-9 for w in widths) else None
        )
        subset = get(cfg, "experiment.outer_folds_subset")
        self.outer_folds = (
            list(range(self.n_outer)) if subset in (None, [], "all") else [int(k) for k in subset]
        )
        self.backend = resolve_backend(self.settings.model_impl)

    # ------------------------------------------------------------------ setup
    def _write_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(self.run_dir / "config.yaml")
        (self.run_dir / "search_space.json").write_text(
            json.dumps(
                {"configs": self.configs, "grid_width": self.grid_width}, indent=1, default=str
            )
        )

    def _folds(self, universe: pd.Index) -> NestedFolds:
        folds = development_folds(self.cfg, self.bundle, universe, self.cohorts, self.settings)
        folds.save(self.run_dir / "folds.parquet")
        return folds

    @property
    def work_dir(self) -> Path:
        """Resumable intermediate results, keyed by a hash of the resolved configuration."""
        cfg = self.cfg.to_dict()
        for section, keys in _VOLATILE_KEYS.items():
            node = cfg.get(section)
            if isinstance(node, dict):
                for key in keys:
                    node.pop(key, None)
        cfg.pop("hardware", None)
        cfg.pop("evaluation", None)
        key = stable_hash({"cfg": cfg, "task": self.task, "cohorts": self.cohorts}, 12)
        return self.run_dir / "work" / key

    def _work_file(self, k: int, c: int, j: int) -> Path:
        return self.work_dir / f"outer{k}" / f"cfg{c}_inner{j}.pkl"

    def outer_configs(self, k: int) -> list[int]:
        """Configuration ids trained in outer fold ``k`` (every sampled one unless fixed)."""
        return list(range(len(self.configs))) if self.fixed is None else [self.fixed[k]]

    def _fold_bundle(self, train_ids: Sequence[str]) -> CohortBundle:
        """The bundle of one inner fit (``data.rna.batch_correction: combat``: corrected copy)."""
        if self.batch_correction == "none":
            return self.bundle
        if self.batch_correction != "combat":
            raise ValueError(f"unknown data.rna.batch_correction {self.batch_correction!r}")
        if getattr(self, "_batches", None) is None:
            column = str(get(self.cfg, "data.rna.batch_column", "plate"))
            self._batches = load_batches(get(self.cfg, "paths.cohorts_dir"), self.cohorts, column)
        return fold_bundle(self.bundle, train_ids, self._batches)

    def _config_for(self, c: int) -> tuple[Config, RunSettings]:
        ccfg = copy.deepcopy(self.cfg)
        apply_overrides(ccfg, self.configs[c])
        return ccfg, run_settings(ccfg, self.task)

    # ------------------------------------------------------------------ inner job
    def _fit_inner(self, k: int, c: int, j: int, folds: NestedFolds) -> InnerOutputs:
        work = self._work_file(k, c, j)
        ckpt = self.run_dir / "checkpoints" / f"outer{k}" / f"cfg{c}" / f"inner{j}"
        if work.exists():
            with work.open("rb") as fh:
                return pickle.load(fh)
        _, st = self._config_for(c)
        tr_ids, va_ids, te_ids = folds.inner_train(k, j), folds.inner_val(k, j), folds.outer_test(k)
        bundle = self._fold_bundle(tr_ids)
        pre = Preprocessor(st.preprocess).fit(bundle, tr_ids)
        sampler = CTPatchSampler(st.ct, seed=st.seed) if st.use_ct else None

        def dataset(ids: list[str], train: bool) -> MOLTDataset:
            feats = pre.transform(bundle, ids)
            augmenter = RNAAugmenter.build(st.rna_augment, pre, feats) if train else None
            if not st.gene_targets:
                feats.rna_genes = None
            return MOLTDataset(
                feats, sampler, train=train, ct_store=self.ct_store, augmenter=augmenter
            )

        ds_tr, ds_va, ds_te = dataset(tr_ids, True), dataset(va_ids, False), dataset(te_ids, False)
        use_ct = st.use_ct and bool(ds_tr.features.ct_present.any())
        mcfg = build_model_config(
            self.backend, model_kwargs(pre, st.model, use_ct=use_ct, gene_targets=st.gene_targets)
        )
        weights = build_loss_weights(
            self.backend,
            st.loss,
            self.task,
            pos_weight=pre.pos_weight,
            horizon_index=max(pre.horizon_bin() - 1, 0),
        )
        members = int(st.train.members)
        tag = f"c{c}"
        prefix = (
            f"outer {k + 1}/{self.n_outer} · config {c + 1}/{len(self.configs)} · "
            f"inner {j + 1}/{self.n_inner}"
        )

        def on_epoch(member: int, epoch: int, max_epochs: int) -> None:
            self.progress.advance(
                1.0, f"{prefix} · member {member + 1}/{members} · epoch {epoch + 1}/{max_epochs}"
            )

        builder = model_builder(self.backend, mcfg, freeze_ct_backbone=self.ct_store is not None)
        trainer = Trainer(
            builder,
            weights,
            st.train,
            task=self.task,
            device=self.device,
            metrics=self.metrics,
            context={"outer_fold": k, "inner_fold": j, "tag": tag},
            on_epoch=on_epoch,
        )
        seeds = [derive_seed(st.seed, "member", k, c, j, m) for m in range(members)]
        start = time.time()
        ensemble, results = trainer.fit_ensemble(ds_tr, ds_va, seeds)
        self.progress.advance(sum(st.train.max_epochs - r.epochs_run for r in results))
        preds: dict[str, dict[str, EnsemblePrediction]] = {}
        modes = list(dict.fromkeys(["full", *st.modes]))
        for mode in modes:
            omit = MODE_OMIT[mode]
            preds[mode] = {
                split: ensemble.predict(
                    ds,
                    omit=omit,
                    mc_samples=st.train.mc_samples,
                    batch_size=st.train.eval_batch_size,
                    num_workers=st.train.num_workers,
                )
                for split, ds in (("val", ds_va), ("test", ds_te))
            }
        ensemble.save(ckpt, extra={"backend": self.backend.name})
        with (ckpt / "preprocessor.pkl").open("wb") as fh:
            pickle.dump(pre, fh)
        with (ckpt / "model_config.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "backend": self.backend.name,
                    "config": mcfg,
                    "loss_weights": weights,
                    "ct_feature_cache": self.ct_store is not None,
                },
                fh,
            )
        (ckpt / "meta.json").write_text(
            json.dumps(
                {
                    "outer_fold": k,
                    "config_id": c,
                    "inner_fold": j,
                    "overrides": self.configs[c],
                    "seeds": seeds,
                    "best_epochs": [r.best_epoch for r in results],
                    "best_val": [r.best_val for r in results],
                    "preprocessor": pre.summary(),
                },
                indent=1,
                default=str,
            )
        )
        out = InnerOutputs(
            outer_fold=k,
            config_id=c,
            inner_fold=j,
            val_ids=va_ids,
            test_ids=te_ids,
            preds=preds,
            val_flags=ds_va.features.rna_flag.copy(),
            test_flags=ds_te.features.rna_flag.copy(),
            bin_edges=pre.bin_edges.copy(),
            best_epochs=[r.best_epoch for r in results],
            seconds=time.time() - start,
            checkpoint_dir=str(ckpt),
        )
        work.parent.mkdir(parents=True, exist_ok=True)
        tmp = work.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(out, fh)
        tmp.replace(work)
        return out

    # ------------------------------------------------------------------ outer fold
    def _outer(
        self, k: int, folds: NestedFolds
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        table = self.bundle.table
        losses: dict[int, float] = {}
        components: dict[int, dict[str, float]] = {}
        outputs: dict[int, list[InnerOutputs]] = {}
        mode_weights = dict(get(self.cfg, "search.selection_modes", {"full": 1.0}))
        for c in self.outer_configs(k):
            outs = [self._fit_inner(k, c, j, folds) for j in range(self.n_inner)]
            outputs[c] = outs
            labels = [
                labels_for(
                    table,
                    o.val_ids,
                    self.task,
                    require_modality=self.settings.preprocess.require_modality_for_survival,
                )
                for o in outs
            ]
            loss, components[c] = selection_objective(
                self.task,
                {m: [o.preds[m]["val"] for o in outs] for m in mode_weights},
                labels,
                outs[0].bin_edges,
                mode_weights=mode_weights,
                metric_weight=float(get(self.cfg, "search.selection_metric_weight", 0.0)),
                grid_width=self.grid_width,
            )
            losses[c] = loss if np.isfinite(loss) else math.inf
            self.metrics.log("selection_loss", loss, split="val", outer_fold=k, member=f"c{c}")
        best = min(losses, key=lambda c: (losses[c], c))
        log.info("outer fold %d: selected config %d (loss %.4f)", k, best, losses[best])
        chosen = outputs[best]
        final_dir = self.run_dir / "checkpoints" / f"outer{k}"
        for o in chosen:
            target = final_dir / f"inner{o.inner_fold}"
            if Path(o.checkpoint_dir).exists() and Path(o.checkpoint_dir) != target:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(o.checkpoint_dir, target)
            o.checkpoint_dir = str(target)
        if not self.keep_all:
            for c in range(len(self.configs)):
                shutil.rmtree(final_dir / f"cfg{c}", ignore_errors=True)

        st = self._config_for(best)[1]
        test_ids = chosen[0].test_ids
        test_labels = labels_for(
            table, test_ids, self.task, require_modality=st.preprocess.require_modality_for_survival
        )
        frames, oof_frames, posts = [], [], {}
        for mode in st.modes:
            val_preds = [o.preds[mode]["val"] for o in chosen]
            val_labels = [
                labels_for(
                    table,
                    o.val_ids,
                    self.task,
                    require_modality=self.settings.preprocess.require_modality_for_survival,
                )
                for o in chosen
            ]
            post = fit_postprocessing(
                self.task,
                mode,
                val_preds,
                val_labels,
                chosen[0].bin_edges,
                horizon=st.preprocess.horizon,
                sensitivity_target=st.sensitivity_target,
                coverage=st.coverage,
                intercepts=st.pfi_intercepts,
                horizons=st.horizons,
            )
            posts[mode] = post.to_dict()
            flags = (
                chosen[0].test_flags if (st.abstain_on_quality_flag and mode != "light") else None
            )
            arrays = apply_postprocessing(
                post,
                [o.preds[mode]["test"] for o in chosen],
                chosen[0].bin_edges,
                flags=flags,
                alpha=st.conformal_alpha,
                horizons=st.horizons,
            )
            frame = prediction_frame(
                ids=test_ids,
                table=table,
                task=self.task,
                mode=mode,
                outer_fold=k,
                arrays=arrays,
                labels=test_labels,
                config_id=best,
                flags=chosen[0].test_flags,
                risk_cutoffs=st.risk_cutoffs,
                horizon=st.preprocess.horizon,
            )
            if self.task != "egfr":
                frame["bin_width"] = float(chosen[0].bin_edges[1] - chosen[0].bin_edges[0])
            frames.append(frame)
            for metric, value in quick_metrics(frame, self.task).items():
                if metric not in ("n", "n_pos", "events"):
                    self.metrics.log(metric, value, split="test", outer_fold=k, member=mode)
            for o, pred, lab in zip(chosen, val_preds, val_labels):
                oof = apply_postprocessing(
                    post, [pred], o.bin_edges, alpha=st.conformal_alpha, horizons=st.horizons
                )
                oof_frames.append(
                    prediction_frame(
                        ids=o.val_ids,
                        table=table,
                        task=self.task,
                        mode=mode,
                        outer_fold=k,
                        arrays=oof,
                        labels=lab,
                        config_id=best,
                        flags=o.val_flags,
                        risk_cutoffs=st.risk_cutoffs,
                        horizon=st.preprocess.horizon,
                    ).assign(inner_fold=o.inner_fold)
                )
        selection = {
            "outer_fold": k,
            "selected": best,
            "overrides": self.configs[best],
            "losses": {str(c): (None if not np.isfinite(v) else v) for c, v in losses.items()},
            "loss_components": {str(c): v for c, v in components.items()},
            "inner_models": [str(Path(o.checkpoint_dir).relative_to(self.run_dir)) for o in chosen],
            "best_epochs": {str(o.inner_fold): o.best_epochs for o in chosen},
            "postprocessing": posts,
            "bin_edges": chosen[0].bin_edges.tolist(),
        }
        return (
            pd.concat(frames, ignore_index=True),
            pd.concat(oof_frames, ignore_index=True),
            selection,
        )

    # ------------------------------------------------------------------ locking
    def _lock(self, predictions: pd.DataFrame, selections: list[dict[str, Any]]) -> dict[str, Any]:
        st = self.settings
        modes: dict[str, Any] = {}
        for mode, df in predictions.groupby("mode", sort=False):
            df = df[df["in_analytic_set"].astype(bool)]
            entry: dict[str, Any] = {
                "abstain_cutoff": abstention_cutoff(df["uncertainty"].to_numpy(float), st.coverage)
            }
            if self.task == "egfr":
                y, p = df["y_egfr"].to_numpy(float), df["p_cal"].to_numpy(float)
                entry["threshold"] = sensitivity_threshold(p, y, st.sensitivity_target)
                entry["conformal"] = [{"p": p.tolist(), "y": y.tolist()}]
            modes[str(mode)] = entry
        manifest = {
            "run_id": self.run_id,
            "task": self.task,
            "cohorts": self.cohorts,
            "backend": self.backend.name,
            "modes": modes,
            "outer": selections,
            "horizon": st.preprocess.horizon,
            "horizons": list(st.horizons),
            "risk_cutoffs": list(st.risk_cutoffs),
            "conformal_alpha": st.conformal_alpha,
            "use_ct": st.use_ct,
            "ct_feature_cache": self.ct_store is not None,
        }
        out = self.run_dir / "locked" / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1, default=float))
        return manifest

    # ------------------------------------------------------------------ main
    def run(self) -> dict[str, Any]:
        """Execute the nested cross-validation and return the run summary."""
        seed_everything(self.settings.seed, deterministic=False)
        self._write_config()
        model_name = str(get(self.cfg, "experiment.model", "molt"))
        if self.backend.name == "stub":
            model_name = "stub"
        self.status = RunStatus(
            self.run_id,
            task=self.task,
            model=model_name,
            cohorts=self.cohorts,
            runs_root=self.runs_root,
            gpu=_device_index(self.device),
        )
        metrics_path = self.run_dir / "metrics.jsonl"
        if metrics_path.exists() and not self.work_dir.exists():
            # A fresh (non-resumed) run replaces the metric stream of an earlier run.
            metrics_path.replace(self.run_dir / "metrics.previous.jsonl")
        self.metrics = MetricsLogger(self.run_id, runs_root=self.runs_root)
        self.status.start("loading cohorts")
        try:
            if self.bundle is None:
                self.bundle = load_bundle_from_config(self.cfg, self.cohorts, self.settings)
            universe = development_universe(
                self.bundle.table,
                require_modality=self.settings.preprocess.require_modality_for_survival,
            )
            folds = self._folds(universe)
            self.ct_store = None
            if self.settings.use_ct and self.settings.ct_feature_cache:
                self.status.update(message="computing frozen-backbone CT features")
                self.ct_store = ct_feature_store(
                    self.bundle, list(universe), self.settings, self.backend, None, self.device
                )
            units = (
                sum(len(self.outer_configs(k)) for k in self.outer_folds)
                * self.n_inner
                * self.settings.train.members
                * self.settings.train.max_epochs
            )
            self.progress = _Progress(self.status, units)
            if self.parallel_jobs > 1:
                jobs = [
                    (k, c, j)
                    for k in self.outer_folds
                    for c in self.outer_configs(k)
                    for j in range(self.n_inner)
                    if not self._work_file(k, c, j).exists()
                ]
                run_jobs_parallel(self, jobs, self.parallel_jobs)
            frames, oof, selections = [], [], []
            for k in self.outer_folds:
                pred, inner, sel = self._outer(k, folds)
                frames.append(pred)
                oof.append(inner)
                selections.append(sel)
                self.progress.advance(0.0, f"outer fold {k + 1}/{self.n_outer} done", force=True)
            predictions = pd.concat(frames, ignore_index=True)
            predictions.to_parquet(self.run_dir / "predictions.parquet", index=False)
            pd.concat(oof, ignore_index=True).to_parquet(
                self.run_dir / "inner_oof.parquet", index=False
            )
            (self.run_dir / "selection.json").write_text(
                json.dumps(selections, indent=1, default=float)
            )
            self._lock(predictions, selections)
            self.status.update(message="summarizing")
            summary = summarize_predictions(
                predictions,
                self.task,
                n_boot=int(get(self.cfg, "evaluation.summary_bootstrap", 200)),
                seed=self.settings.seed,
                horizon=self.settings.preprocess.horizon,
                horizons=self.settings.horizons,
                risk_cutoffs=self.settings.risk_cutoffs,
            )
            summary.update(
                run_id=self.run_id,
                cohorts=self.cohorts,
                n_participants=len(universe),
                outer_folds=self.outer_folds,
                selected={s["outer_fold"]: s["selected"] for s in selections},
            )
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            publish_comparison(
                summary,
                self.cfg,
                task=self.task,
                cohort="+".join(self.cohorts),
                model=comparison_label(self.cfg, model_name),
                run_id=self.run_id,
            )
            if not bool(get(self.cfg, "experiment.keep_work", False)):
                shutil.rmtree(self.run_dir / "work", ignore_errors=True)
            self.status.finish(self._final_message(summary))
            return summary
        except BaseException as exc:
            self.status.fail(f"{type(exc).__name__}: {exc}")
            raise

    def _final_message(self, summary: Mapping[str, Any]) -> str:
        parts = []
        for mode, entry in summary.get("modes", {}).items():
            q = entry.get("quick", {})
            key = "auroc" if self.task == "egfr" else "c_index"
            if q.get(key) is not None:
                parts.append(f"{mode} {key}={q[key]:.3f}")
        return "done: " + ", ".join(parts) if parts else "done"
