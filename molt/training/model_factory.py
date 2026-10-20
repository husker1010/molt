"""Build networks and loss weights from fitted preprocessing and run configuration.

The trainer accepts any ``nn.Module`` that honours the MOLT interface
(``forward(batch, dropout_modalities)`` returning ``logit_egfr``, ``hazards``, ``survival``
and ``expert_weights``, plus ``compute_loss(outputs, batch, weights)``). A
:class:`ModelBackend` bundles such a model class with its configuration dataclass and
loss-weight dataclass; :func:`resolve_backend` returns :class:`molt.models.MOLT` without a fallback network.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from molt.training.preprocess import Preprocessor
from molt.utils.logging import get_logger

__all__ = [
    "ModelBackend",
    "build_loss_weights",
    "build_model_config",
    "model_builder",
    "model_kwargs",
    "resolve_backend",
]

log = get_logger(__name__)

# Keys derived from the data; user configuration cannot override them.
_DATA_KEYS = frozenset(
    {
        "n_num",
        "n_cat",
        "cat_cardinalities",
        "n_pathways",
        "pathway_adjacency",
        "n_survival_bins",
        "n_domains",
        "quality_dims",
        "quality_signs",
        "n_clinical_indicators",
        "n_genes",
        "n_panel_genes",
        "stacked_dims",
        "pathway_incidence",
        "primary_horizon_index",
        "cat_mixing",
        "cat_coarsen",
        "cat_coarsen_hide",
        "cat_coarsen_p",
    }
)


@dataclass(frozen=True)
class ModelBackend:
    """A model class with its configuration and loss-weight dataclasses."""

    name: str
    model_cls: type[nn.Module]
    config_cls: type
    loss_cls: type

    @property
    def config_fields(self) -> set[str]:
        return {f.name for f in dataclasses.fields(self.config_cls)}

    @property
    def loss_fields(self) -> set[str]:
        return {f.name for f in dataclasses.fields(self.loss_cls)}


def resolve_backend(impl: str = "molt") -> ModelBackend:
    """Return the MOLT backend; ``auto`` is a compatibility alias for MOLT."""
    if impl not in ("molt", "auto"):
        raise ValueError("This distribution supports only the MOLT backend")
    from molt.models import MOLT, LossWeights, MOLTConfig

    return ModelBackend("molt", MOLT, MOLTConfig, LossWeights)


def model_kwargs(
    pre: Preprocessor,
    model_cfg: Mapping[str, Any] | None = None,
    *,
    use_ct: bool = False,
    gene_targets: bool = True,
) -> dict[str, Any]:
    """Constructor arguments derived from the fitted preprocessor plus user settings."""
    n_pathways = pre.n_pathways
    use_rna = n_pathways > 0 and pre.n_genes > 0
    kwargs: dict[str, Any] = {
        "n_num": pre.n_num,
        "n_cat": pre.n_cat,
        "cat_cardinalities": pre.cat_cardinalities,
        "n_pathways": n_pathways if use_rna else 1,
        "pathway_adjacency": pre.adjacency if use_rna else None,
        "n_survival_bins": pre.n_bins,
        "n_domains": pre.n_domains,
        "use_ct": bool(use_ct),
        "use_rna": use_rna,
        "quality_dims": pre.quality_dims,
        "quality_signs": pre.quality_signs,
        "n_clinical_indicators": pre.n_indicators,
        "n_genes": pre.n_genes if (use_rna and gene_targets) else 0,
        "n_panel_genes": pre.n_panel_genes if use_rna else 0,
        "stacked_dims": dict(getattr(pre, "stacked_dims", {}) or {}) or None,
        "pathway_incidence": pre.incidence if (use_rna and gene_targets) else None,
        "primary_horizon_index": max(pre.horizon_bin() - 1, 0),
        "ct_hu_window": None,
    }
    hierarchy = getattr(pre, "category_hierarchy", None)
    if hierarchy is not None:
        cards = pre.cat_cardinalities
        mixing = hierarchy.mixing_matrix(cards)
        tables = hierarchy.coarsen_tables(cards, pre.n_num)
        kwargs["cat_mixing"] = None if mixing is None else mixing.tolist()
        if tables is not None:
            kwargs["cat_coarsen"] = tables[0].tolist()
            kwargs["cat_coarsen_hide"] = tables[1].astype(int).tolist()
            kwargs["cat_coarsen_p"] = float(hierarchy.coarsen_p)
    for key, value in dict(model_cfg or {}).items():
        if key == "gene_expert":
            # false: no neural gene-level expert; the panel still feeds the stacked scorers
            kwargs["n_panel_genes"] = kwargs["n_panel_genes"] if bool(value) else 0
            continue
        if key in ("impl",) or key in _DATA_KEYS:
            continue
        if key == "use_ct":
            kwargs[key] = bool(value) and bool(use_ct)
            continue
        kwargs[key] = value
    return kwargs


def build_model_config(backend: ModelBackend, kwargs: Mapping[str, Any]) -> Any:
    """Instantiate the backend configuration, dropping keys it does not define."""
    names = backend.config_fields
    unknown = sorted(set(kwargs) - names)
    if unknown:
        log.debug("%s ignores model settings %s", backend.config_cls.__name__, unknown)
    values = {k: v for k, v in kwargs.items() if k in names}
    for key in ("pathway_adjacency", "pathway_incidence"):
        if values.get(key) is not None:
            values[key] = np.asarray(values[key], dtype=np.float32)
    return backend.config_cls(**values)


def build_loss_weights(
    backend: ModelBackend,
    loss_cfg: Mapping[str, Any] | None,
    task: str,
    *,
    pos_weight: float | None = None,
    horizon_index: int | None = None,
) -> Any:
    """Loss weights for ``task`` (``egfr``, ``pfi`` or ``os``).

    Endpoint-specific fine-tuning activates only the endpoint being trained; with
    ``loss.multitask: true`` both endpoint terms are active. ``loss.pos_weight: auto`` uses the
    training-fold ratio of negatives to positives.
    """
    cfg = dict(loss_cfg or {})
    multitask = bool(cfg.get("multitask", False))
    egfr_w = float(cfg.get("egfr", 1.0))
    surv_w = float(cfg.get("surv", 1.0))
    values: dict[str, Any] = {
        "egfr": egfr_w if (task == "egfr" or multitask) else 0.0,
        "surv": surv_w if (task in ("pfi", "os") or multitask) else 0.0,
        "recon": float(cfg.get("recon", 0.1)),
        "domain": float(cfg.get("domain", 0.0)),
        "brier": float(cfg.get("brier", 0.0)),
        "attn_prior": float(cfg.get("attn_prior", 0.01)),
        "focal_gamma": float(cfg.get("focal_gamma", 0.0)),
        "horizon_index": horizon_index,
        "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
        "expert_aux": float(cfg.get("expert_aux", 0.0)),
        "rank": float(cfg.get("rank", 0.0)),
        "rank_scale": float(cfg.get("rank_scale", 1.0)),
        "gene_l1": float(cfg.get("gene_l1", 0.0)),
        "rna_precision": float(cfg.get("rna_precision", 0.0)),
    }
    pw = cfg.get("pos_weight", "auto")
    values["egfr_pos_weight"] = (
        float(pos_weight)
        if pw == "auto" and pos_weight is not None
        else (None if pw in (None, "none", "auto") else float(pw))
    )
    names = backend.loss_fields
    return backend.loss_cls(**{k: v for k, v in values.items() if k in names})


def model_builder(
    backend: ModelBackend, config: Any, *, freeze_ct_backbone: bool = False
) -> Callable[[], nn.Module]:
    """Zero-argument factory creating a fresh, randomly initialised model.

    Args:
        backend: Model backend.
        config: Backend configuration instance.
        freeze_ct_backbone: Disable gradients of the CT patch backbone (used with precomputed
            frozen-backbone features).
    """

    def build() -> nn.Module:
        model = backend.model_cls(config)
        imaging = getattr(model, "imaging", None)
        if (
            freeze_ct_backbone
            and imaging is not None
            and hasattr(imaging, "set_backbone_trainable")
        ):
            imaging.set_backbone_trainable(False)
        return model

    return build
