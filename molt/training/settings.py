"""Translate a run configuration into the typed settings of the training components.

Configuration layout (YAML, see ``configs/molt_egfr.yaml``)::

    data:
      clinical: {numeric, categorical, winsor_limits, min_category_count, ...}
      rna:      {enabled, min_expr_fraction, mask_genes, exclude_genes, gene_targets,
                 sample_normalization, gene_panel: {size, min_mean, forced},
                 augment: {enabled, p, scale_sd, shift_sd, mask_min, mask_max, rank_sd}, ...}
      stacking: {folds, n_lambdas, lambda_ratio, components, experts: {name: {inputs, alphas}}}
      ct:       {enabled, n_lung, n_candidates, patch_size, augment, cache_dir, flags,
                 feature_cache, feature_views, feature_dir}
      survival: {endpoint, bin_width_months, max_months, primary_horizon_months, ...}
      domain:   {key, min_count}
    model:  {impl, latent_dim, dropout, ct_backbone, swin_checkpoint, ...}   # MOLTConfig fields
    loss:   {recon, domain, brier, focal_gamma, pos_weight, multitask, ...}
    train:  {lr, weight_decay, batch_size, max_epochs, patience, members, mc_samples, ...}
    calibration: {sensitivity_target, coverage, conformal_alpha, risk_cutoffs, horizons_months}
    recalibration: {pfi_intercepts}   # horizon (default) | single
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from molt.training.ct_patches import CTPatchConfig
from molt.training.preprocess import PreprocessConfig
from molt.training.rna_augment import RNAAugmentConfig
from molt.training.trainer import TrainSettings

__all__ = ["RunSettings", "get", "run_settings"]


def _plain(value: Any) -> Any:
    """Plain dict/list copy of a (possibly configuration-object) nested mapping."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Dotted lookup in nested mappings with a default."""
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return default
    return default if node is None else node


@dataclass
class RunSettings:
    """Typed settings of one training configuration."""

    task: str
    preprocess: PreprocessConfig
    ct: CTPatchConfig | None
    train: TrainSettings
    model: dict[str, Any]
    loss: dict[str, Any]
    model_impl: str = "auto"
    gene_targets: bool = True
    seed: int = 20240613
    sensitivity_target: float = 0.90
    coverage: float = 0.90
    conformal_alpha: float = 0.05
    risk_cutoffs: tuple[float, float] = (0.20, 0.40)
    horizons: tuple[float, ...] = (12.0, 24.0, 36.0)
    modes: tuple[str, ...] = ("full", "light")
    abstain_on_quality_flag: bool = True
    pfi_intercepts: str = "horizon"
    ct_feature_cache: bool = False
    ct_feature_views: int = 4
    ct_feature_dir: str | None = None
    rna_augment: RNAAugmentConfig = field(default_factory=RNAAugmentConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def use_ct(self) -> bool:
        return self.ct is not None and self.preprocess.use_ct


def run_settings(cfg: Mapping[str, Any], task: str) -> RunSettings:
    """Build :class:`RunSettings` for ``task`` (``egfr``, ``pfi`` or ``os``) from ``cfg``."""
    if task not in ("egfr", "pfi", "os"):
        raise ValueError(f"Unknown task {task!r}")
    clinical = dict(get(cfg, "data.clinical", {}))
    rna = dict(get(cfg, "data.rna", {}))
    panel = dict(rna.get("gene_panel") or {})
    ct = dict(get(cfg, "data.ct", {}))
    surv = dict(get(cfg, "data.survival", {}))
    domain = dict(get(cfg, "data.domain", {}))
    horizon = float(
        surv.get("primary_horizon_months", get(cfg, "survival.primary_horizon_months", 24))
    )
    endpoint = task if task in ("pfi", "os") else str(surv.get("endpoint", "pfi"))
    pre = PreprocessConfig.from_mapping(
        {
            **clinical,
            "hierarchy": _plain(clinical.get("hierarchy")),
            "coarsen_hide": _plain(clinical.get("coarsen_hide")),
            "use_rna": rna.get("enabled", True),
            "min_expr_fraction": rna.get("min_expr_fraction"),
            "mask_genes": rna.get("mask_genes") or (),
            "exclude_genes": rna.get("exclude_genes"),
            "standardize_pathways": rna.get("standardize_pathways"),
            "external_standardization": rna.get("external_standardization"),
            "sd_floor": rna.get("sd_floor"),
            "z_clip": rna.get("z_clip"),
            "rna_quality_missing_indicators": rna.get("quality_missing_indicators"),
            "sample_normalization": rna.get("sample_normalization"),
            "gene_panel_size": panel.get("size"),
            "gene_panel_min_mean": panel.get("min_mean"),
            "gene_panel_forced": panel.get("forced"),
            "stacking": _plain(get(cfg, "data.stacking")),
            "task": task,
            "use_ct": ct.get("enabled", True),
            "ct_flags": ct.get("flags"),
            "survival_endpoint": endpoint,
            "bin_width": surv.get("bin_width_months"),
            "max_time": surv.get("max_months"),
            "horizon": horizon,
            "require_modality_for_survival": surv.get("require_modality"),
            "ipcw_min_prob": surv.get("ipcw_min_prob"),
            "domain_key": domain.get("key"),
            "domain_min_count": domain.get("min_count"),
        }
    )
    ct_cfg = None
    if pre.use_ct:
        names = set(CTPatchConfig.__dataclass_fields__)
        values = {k: v for k, v in ct.items() if k in names and v is not None}
        if "hu_window" in values:
            values["hu_window"] = tuple(float(x) for x in values["hu_window"])
        ct_cfg = CTPatchConfig(**values)
    seed = int(get(cfg, "seed", 20240613))
    train = TrainSettings.from_mapping(get(cfg, "train", {}), seed=seed)
    model = {k: v for k, v in dict(get(cfg, "model", {})).items() if k != "impl"}
    calib = dict(get(cfg, "calibration", {}))
    return RunSettings(
        task=task,
        preprocess=pre,
        ct=ct_cfg,
        train=train,
        model=model,
        loss=dict(get(cfg, "loss", {})),
        model_impl=str(get(cfg, "model.impl", "auto")),
        gene_targets=bool(rna.get("gene_targets", True)),
        seed=seed,
        sensitivity_target=float(
            calib.get("sensitivity_target", get(cfg, "thresholds.egfr_sensitivity_target", 0.90))
        ),
        coverage=float(calib.get("coverage", 0.90)),
        conformal_alpha=float(calib.get("conformal_alpha", 0.05)),
        risk_cutoffs=tuple(
            calib.get("risk_cutoffs", get(cfg, "thresholds.risk_cutoffs", [0.2, 0.4]))
        ),
        horizons=tuple(
            float(h)
            for h in calib.get(
                "horizons_months", get(cfg, "survival.horizons_months", [12, 24, 36])
            )
        ),
        modes=tuple(get(cfg, "experiment.modes", ["full", "light"])),
        abstain_on_quality_flag=bool(calib.get("abstain_on_quality_flag", True)),
        pfi_intercepts=str(get(cfg, "recalibration.pfi_intercepts", "horizon")),
        ct_feature_cache=bool(ct.get("feature_cache", False)),
        ct_feature_views=int(ct.get("feature_views", 4)),
        ct_feature_dir=ct.get("feature_dir"),
        rna_augment=RNAAugmentConfig.from_mapping(rna.get("augment")),
    )
