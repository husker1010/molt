"""The MOLT multimodal network: encoders, reliability-weighted fusion and task heads.

``MOLT`` consumes a batch dictionary with clinical variables, pathway activities and CT patches
(any subset may be missing per patient) and returns EGFR logits, discrete-time hazards and
survival curves computed from one fused Gaussian latent representation::

    model = MOLT(MOLTConfig(n_num=4, n_cat=10, cat_cardinalities=[...], n_pathways=42,
                            pathway_adjacency=adj, n_survival_bins=10, primary_horizon_index=3))
    outputs = model(batch, dropout_modalities=True)
    losses = model.compute_loss(outputs, batch, LossWeights.for_task("egfr"))
    losses["total"].backward()

The "light" configuration is the same network evaluated without the transcriptomic expert,
``model(batch, omit_modalities=("rna",))``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from molt.models.clinical import ClinicalEncoder
from molt.models.common import load_checkpoint_payload, make_mlp, sanitize
from molt.models.experts import (
    EXPERT_AUX_MODES,
    EXPERT_GROUPS,
    EXPERTS,
    STACKED_EXPERTS,
    STACKED_SOURCE,
    TRANSCRIPTOMIC_EXPERTS,
    ExpertAuxHeads,
    GeneEncoder,
    batch_size_of,
    expert_aux_outputs,
    gene_l1_penalty,
    modality_availability,
    scheduled_modality_dropout,
    stacked_shift,
)
from molt.models.fusion import MODALITIES, ProductOfExpertsFusion
from molt.models.heads import BinaryHead, DiscreteTimeSurvivalHead, DomainClassifier
from molt.models.imaging import ImagingEncoder
from molt.models.inference import mc_dropout, mc_dropout_predict, model_device
from molt.models.losses import LossWeights, compute_multitask_loss
from molt.models.pathway import PathwayEncoder

__all__ = [
    "EXPERTS",
    "MOLT",
    "CheckpointMixin",
    "LossWeights",
    "MOLTConfig",
    "batch_size_of",
    "mc_dropout",
    "mc_dropout_predict",
    "modality_availability",
    "model_device",
]

_ARRAY_FIELDS = ("pathway_adjacency", "pathway_incidence")
_NO_DECAY_TAGS = ("embedding", "_token", "node_id", "num_bias", "num_missing")
_TUPLE_FIELDS = ("fixed_reliability", "modality_dropout_exclude", "cnn_widths", "ct_hu_window")


@dataclass
class MOLTConfig:
    """Architecture and regularisation settings of :class:`MOLT`.

    Attributes:
        n_num: Number of continuous clinical variables.
        n_cat: Number of categorical clinical variables.
        cat_cardinalities: Vocabulary size per categorical variable, including the reserved
            missing (0) and out-of-vocabulary (1) indices.
        n_pathways: Number of pathway nodes ``P``.
        pathway_adjacency: ``(P, P)`` 0/1 adjacency prior (self-loops are added if absent).
        latent_dim: Dimension ``D`` of the fused latent space.
        n_survival_bins: Number of discrete time bins ``K``.
        n_domains: Number of domain classes for the gradient-reversal classifier (0 disables).
        use_ct: Build the CT branch.
        ct_backbone: ``"swin_vit"`` or ``"cnn3d"``.
        swin_checkpoint: Path to a caller-supplied Swin-ViT state or ``None``.
        quality_dims: Number of quality inputs per modality (keys ``clinical``, ``rna``, ``ct``).
        quality_signs: Optional orientation of each quality input (+1 adverse, -1 favourable).
        use_rna: Build the transcriptomic branch.
        n_clinical_indicators: Extra binary clinical indicators (``batch["clinical_indicators"]``).
        clinical_dim: Clinical token dimension.
        clinical_layers: Clinical transformer layers.
        clinical_heads: Clinical attention heads.
        clinical_indicator_tokens: Add per-variable missingness-indicator tokens.
        clinical_skip: Add a linear encoding of the flat clinical record to the clinical
            embedding (see :class:`~molt.models.clinical.ClinicalEncoder`).
        cat_mask_p: Stochastic masking probability of observed categories during training.
        num_mask_p: Stochastic masking probability of observed continuous values.
        pathway_dim: Pathway node dimension.
        pathway_layers: Graph attention layers.
        pathway_heads: Graph attention heads.
        node_id_dim: Pathway node-identity embedding size.
        pathway_graph_mode: ``"prior"``, ``"dense"`` or ``"mlp"``.
        pathway_skip: Add the graph-free encoding of the pathway activities to the pooled
            graph embedding (see :class:`~molt.models.pathway.PathwayEncoder`).
        n_rna_aux: Number of auxiliary transcriptomic features (``batch["rna_aux"]``, e.g.
            selected gene-level values) added to the pathway embedding through an MLP.
        n_genes: Genes predicted by the transcriptomic decoder (0 disables the masked-gene objective).
        pathway_incidence: Optional ``(n_genes, P)`` incidence for the masked-gene objective.
        gene_mask_ratio: Fraction of genes hidden in the masked-gene objective.
        gene_noise_std: Noise added to visible genes in the masked-gene objective.
        ct_embed_dim: CT patch/scan embedding dimension.
        mil_hidden: Width of the MIL attention network.
        mil_gated: Use gated MIL attention.
        ct_hu_window: HU window mapped to ``[0, 1]`` before the backbone (``None`` = inputs are
            already normalised).
        ct_chunk_size: Patches per backbone call.
        ct_grad_checkpoint: Recompute backbone activations in the backward pass.
        ct_recon: Enable the masked-patch objective.
        ct_recon_patches: Patches per scan used for the masked-patch objective.
        cnn_widths: Channel widths of the ``cnn3d`` backbone.
        swin_feature_size: Swin-ViT feature size.
        fusion_rule: ``"poe"`` or ``"mean"``.
        use_reliability: Learn reliability weights (``False`` fixes all ``rho`` to 1).
        fixed_reliability: Modalities with ``rho`` fixed to 1.
        expert_hidden: Hidden width of the expert networks (0 = linear experts).
        latent_sampling: Sample ``z ~ N(mu, Lambda^-1)`` during training instead of using ``mu``.
        head_hidden: Hidden width of the task heads.
        init_hazard: Initial per-bin hazard of the survival head.
        primary_horizon_index: Survival column index of the primary horizon (Brier term, risk).
        dropout: Encoder dropout.
        attn_dropout: Attention dropout.
        head_dropout: Task-head dropout.
        modality_dropout_p: Expert drop probability when ``dropout_modalities=True``.
        modality_dropout_exclude: Modalities never dropped.
        grl_alpha: Gradient-reversal strength.
        n_panel_genes: Genes of the gene-level expert panel (``batch["rna_panel"]``; 0 disables
            the gene-level expert). The expert shares the RNA availability and reliability.
        gene_hidden: Hidden width of the gene-level expert (0 = linear in the genes).
        gene_input_dropout: Probability of hiding each panel gene during training.
        gene_dropout: Dropout after the hidden layer of the gene-level expert.
        gene_weight_decay: Weight decay of the gene-level expert (``None`` = global value).
        expert_aux_heads: ``"none"``, ``"shared"`` (fused heads applied to every expert's
            single-expert posterior) or ``"own"`` (expert-specific heads); the auxiliary
            outputs are trained with the endpoint loss (``LossWeights.expert_aux``).
        modality_dropout_by: Expert-specific drop probabilities (default
            ``modality_dropout_p``), e.g. ``{"clinical": 0.5}``.
        modality_curriculum_epochs: Length of the transcriptomic curriculum: in epoch ``e`` a
            participant with an expression profile is trained on its transcriptomic experts
            only with probability ``max(0, 1 - e / epochs)``.
        stacked_dims: Input dimension of every stacked linear expert
            (``batch["stacked_<name>"]``, see :mod:`molt.training.stacking`); each enters the
            product of experts with its own Gaussian expert and the reliability weight of
            its source modality.
        modality_group_dropout: Joint drop probability of whole modality groups
            (``{"rna": 0.3}`` drops every transcriptomic expert at once, as in light mode).
        stacked_skip: Also add the stacked scores linearly to the EGFR logit and the hazard
            logits (``sum_m rho_m w_m' s_m`` over present stacked experts, ``w`` learned from 0).
        cat_mixing, cat_coarsen, cat_coarsen_hide, cat_coarsen_p: Partially specified
            categories (mixture matrix, coarsening table, hidden continuous variables and
            probability; see :class:`~molt.models.clinical.ClinicalEncoder`).
        rna_precision_experts: Experts shrunk by ``loss.rna_precision``: ``all``
            transcriptomic experts or the ``neural`` ones (pathway and gene-level).
        omit_experts: Experts removed from the product of experts in training and inference
            (component ablations, e.g. ``("rna",)`` for the pathway expert alone).
    """

    n_num: int
    n_cat: int
    cat_cardinalities: list[int]
    n_pathways: int = 42
    pathway_adjacency: np.ndarray | None = None
    latent_dim: int = 64
    n_survival_bins: int = 10
    n_domains: int = 0
    use_ct: bool = True
    ct_backbone: str = "cnn3d"
    swin_checkpoint: str | None = None
    quality_dims: dict[str, int] = field(default_factory=lambda: {"clinical": 0, "rna": 0, "ct": 0})
    quality_signs: dict[str, list[float]] | None = None
    use_rna: bool = True
    n_clinical_indicators: int = 0
    clinical_dim: int = 64
    clinical_layers: int = 2
    clinical_heads: int = 4
    clinical_indicator_tokens: bool = True
    clinical_skip: bool = False
    cat_mask_p: float = 0.10
    num_mask_p: float = 0.0
    pathway_dim: int = 64
    pathway_layers: int = 2
    pathway_heads: int = 4
    node_id_dim: int = 16
    pathway_graph_mode: str = "prior"
    pathway_skip: bool = True
    n_rna_aux: int = 0
    n_genes: int = 0
    pathway_incidence: np.ndarray | None = None
    gene_mask_ratio: float = 0.15
    gene_noise_std: float = 0.1
    ct_embed_dim: int = 128
    mil_hidden: int = 128
    mil_gated: bool = False
    ct_hu_window: tuple[float, float] | None = (-1024.0, 600.0)
    ct_chunk_size: int = 64
    ct_grad_checkpoint: bool = False
    ct_recon: bool = False
    ct_recon_patches: int = 4
    cnn_widths: tuple[int, ...] = (32, 64, 128, 256)
    swin_feature_size: int = 48
    fusion_rule: str = "poe"
    use_reliability: bool = True
    fixed_reliability: tuple[str, ...] = ()
    expert_hidden: int = 0
    latent_sampling: bool = False
    head_hidden: int = 64
    init_hazard: float = 0.05
    primary_horizon_index: int | None = None
    dropout: float = 0.1
    attn_dropout: float = 0.1
    head_dropout: float = 0.2
    modality_dropout_p: float = 0.2
    modality_dropout_exclude: tuple[str, ...] = ()
    grl_alpha: float = 1.0
    n_panel_genes: int = 0
    gene_hidden: int = 0
    gene_input_dropout: float = 0.5
    gene_dropout: float = 0.2
    gene_weight_decay: float | None = None
    expert_aux_heads: str = "none"
    modality_dropout_by: dict[str, float] | None = None
    modality_curriculum_epochs: int = 0
    stacked_dims: dict[str, int] | None = None
    modality_group_dropout: dict[str, float] | None = None
    stacked_skip: bool = False
    cat_mixing: Any = None
    cat_coarsen: Any = None
    cat_coarsen_hide: Any = None
    cat_coarsen_p: float = 0.0
    rna_precision_experts: str = "all"
    omit_experts: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.cat_cardinalities = [int(c) for c in self.cat_cardinalities]
        if self.n_cat != len(self.cat_cardinalities):
            raise ValueError(
                f"n_cat={self.n_cat} but {len(self.cat_cardinalities)} cardinalities were given"
            )
        for name in _ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, np.ndarray):
                setattr(self, name, np.asarray(value, dtype=np.float32))
        if self.pathway_adjacency is not None and self.pathway_adjacency.shape != (
            self.n_pathways,
            self.n_pathways,
        ):
            raise ValueError(
                f"pathway_adjacency must be ({self.n_pathways}, {self.n_pathways}), "
                f"got {self.pathway_adjacency.shape}"
            )
        if self.pathway_incidence is not None:
            if self.n_genes == 0:
                self.n_genes = int(self.pathway_incidence.shape[0])
            if self.pathway_incidence.shape != (self.n_genes, self.n_pathways):
                raise ValueError(
                    f"pathway_incidence must be ({self.n_genes}, {self.n_pathways}), "
                    f"got {self.pathway_incidence.shape}"
                )
        for name in _TUPLE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, tuple(value))
        dims = {m: 0 for m in MODALITIES}
        dims.update({k: int(v) for k, v in (self.quality_dims or {}).items()})
        unknown = set(dims) - set(MODALITIES)
        if unknown:
            raise ValueError(f"quality_dims has unknown modalities: {sorted(unknown)}")
        self.quality_dims = dims
        if self.ct_backbone not in ("swin_vit", "cnn3d"):
            raise ValueError(f"ct_backbone must be 'swin_vit' or 'cnn3d', got {self.ct_backbone!r}")
        if not 0.0 <= self.modality_dropout_p < 1.0:
            raise ValueError("modality_dropout_p must be in [0, 1)")
        if self.rna_precision_experts not in ("all", "neural"):
            raise ValueError("rna_precision_experts must be 'all' or 'neural'")
        if self.expert_aux_heads not in EXPERT_AUX_MODES:
            raise ValueError(f"expert_aux_heads must be one of {EXPERT_AUX_MODES}")
        by = {str(k): float(v) for k, v in dict(self.modality_dropout_by or {}).items()}
        if set(by) - set(EXPERTS) or any(not 0.0 <= v < 1.0 for v in by.values()):
            raise ValueError(f"modality_dropout_by must map experts {EXPERTS} to [0, 1)")
        self.modality_dropout_by = by or None
        stacked = {str(k): int(v) for k, v in dict(self.stacked_dims or {}).items()}
        if set(stacked) - set(STACKED_EXPERTS) or any(v < 1 for v in stacked.values()):
            raise ValueError(f"stacked_dims must map experts {STACKED_EXPERTS} to sizes >= 1")
        self.stacked_dims = stacked or None
        groups = {str(k): float(v) for k, v in dict(self.modality_group_dropout or {}).items()}
        if set(groups) - set(EXPERT_GROUPS) or any(not 0.0 <= v < 1.0 for v in groups.values()):
            raise ValueError(f"modality_group_dropout must map {sorted(EXPERT_GROUPS)} to [0, 1)")
        self.modality_group_dropout = groups or None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dictionary (arrays become nested lists)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MOLTConfig:
        """Build a configuration from :meth:`to_dict` output; unknown keys raise ``TypeError``."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise TypeError(f"Unknown MOLTConfig fields: {sorted(unknown)}")
        return cls(**dict(data))


class CheckpointMixin:
    """Save/load helpers for models built from a :class:`MOLTConfig`."""

    config: MOLTConfig

    def save_checkpoint(self, path: str | os.PathLike[str], **extra: Any) -> Path:
        """Save configuration, weights and optional metadata to ``path``."""
        from molt import __version__

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "molt-checkpoint",
            "molt_version": __version__,
            "model_class": type(self).__name__,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),  # type: ignore[attr-defined]
            "init_kwargs": dict(getattr(self, "init_kwargs", {})),
            "extra": extra,
        }
        torch.save(payload, out)
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ):
        """Create a model from :meth:`save_checkpoint` output.

        Backbone parameters are not re-read from disk: the saved state dict already
        contains them.
        """
        payload = load_checkpoint_payload(path, map_location)
        config = MOLTConfig.from_dict(payload["config"])
        kwargs = payload.get("init_kwargs") or {}
        model = cls(  # type: ignore[call-arg]
            dataclasses.replace(config, swin_checkpoint=None), **kwargs
        )
        model.config = config
        model.load_state_dict(payload["state_dict"], strict=strict)  # type: ignore[attr-defined]
        model.checkpoint_extra = payload.get("extra", {})  # type: ignore[attr-defined]
        return model


class MOLT(CheckpointMixin, nn.Module):
    """Molecular and Outcome Lung Transformer.

    Args:
        config: Architecture configuration.
    """

    def __init__(self, config: MOLTConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.clinical = ClinicalEncoder(
            c.n_num,
            c.cat_cardinalities,
            dim=c.clinical_dim,
            n_layers=c.clinical_layers,
            n_heads=c.clinical_heads,
            dropout=c.dropout,
            attn_dropout=c.attn_dropout,
            n_extra_indicators=c.n_clinical_indicators,
            indicator_tokens=c.clinical_indicator_tokens,
            linear_skip=c.clinical_skip,
            cat_mask_p=c.cat_mask_p,
            num_mask_p=c.num_mask_p,
            cat_mixing=c.cat_mixing,
            coarsen_table=c.cat_coarsen,
            coarsen_hide=c.cat_coarsen_hide,
            coarsen_p=c.cat_coarsen_p,
        )
        in_dims = {"clinical": self.clinical.out_dim}
        self.pathway: PathwayEncoder | None = None
        if c.use_rna:
            self.pathway = PathwayEncoder(
                c.n_pathways,
                c.pathway_adjacency,
                dim=c.pathway_dim,
                n_layers=c.pathway_layers,
                n_heads=c.pathway_heads,
                node_id_dim=c.node_id_dim,
                graph_mode=c.pathway_graph_mode,  # type: ignore[arg-type]
                dropout=c.dropout,
                attn_dropout=c.attn_dropout,
                n_genes=c.n_genes,
                incidence=c.pathway_incidence,
                gene_mask_ratio=c.gene_mask_ratio,
                gene_noise_std=c.gene_noise_std,
                activity_skip=c.pathway_skip,
            )
            in_dims["rna"] = self.pathway.out_dim
        self.rna_aux: nn.Module | None = None
        if c.use_rna and c.n_rna_aux > 0:
            self.rna_aux = make_mlp(
                2 * c.n_rna_aux, [c.pathway_dim], c.pathway_dim, dropout=c.dropout
            )
        self.gene_encoder: GeneEncoder | None = None
        if c.use_rna and c.n_panel_genes > 0:
            self.gene_encoder = GeneEncoder(
                c.n_panel_genes, c.gene_hidden, c.gene_input_dropout, c.gene_dropout
            )
            in_dims["gene"] = self.gene_encoder.out_dim
        self.imaging: ImagingEncoder | None = None
        if c.use_ct:
            self.imaging = ImagingEncoder(
                backbone=c.ct_backbone,  # type: ignore[arg-type]
                embed_dim=c.ct_embed_dim,
                mil_hidden=c.mil_hidden,
                mil_gated=c.mil_gated,
                dropout=c.dropout,
                swin_checkpoint=c.swin_checkpoint,
                swin_feature_size=c.swin_feature_size,
                cnn_widths=c.cnn_widths,
                hu_window=c.ct_hu_window,
                chunk_size=c.ct_chunk_size,
                grad_checkpoint=c.ct_grad_checkpoint,
                recon=c.ct_recon,
                recon_patches=c.ct_recon_patches,
            )
            in_dims["ct"] = self.imaging.out_dim
        self.stacked_names = [
            m for m in (c.stacked_dims or {}) if c.use_rna or STACKED_SOURCE[m] == "clinical"
        ]
        in_dims.update({m: int(c.stacked_dims[m]) for m in self.stacked_names})
        shared = {"gene": "rna", **{m: STACKED_SOURCE[m] for m in self.stacked_names}}
        self.fusion = ProductOfExpertsFusion(
            in_dims,
            c.latent_dim,
            quality_dims=c.quality_dims,
            quality_signs=c.quality_signs,
            use_reliability=c.use_reliability,
            fixed_reliability=[m for m in c.fixed_reliability if m in in_dims],
            rule=c.fusion_rule,  # type: ignore[arg-type]
            expert_hidden=c.expert_hidden,
            dropout=c.dropout,
            shared_reliability=shared,
        )
        self.egfr_head = BinaryHead(c.latent_dim, c.head_hidden, c.head_dropout)
        self.survival_head = DiscreteTimeSurvivalHead(
            c.latent_dim, c.n_survival_bins, c.head_hidden, c.head_dropout, c.init_hazard
        )
        self.domain_head = (
            DomainClassifier(c.latent_dim, c.n_domains, c.head_hidden, c.dropout, c.grl_alpha)
            if c.n_domains > 0
            else None
        )
        self.aux_heads = (
            ExpertAuxHeads(
                in_dims,
                c.latent_dim,
                c.n_survival_bins,
                c.head_hidden,
                c.head_dropout,
                c.init_hazard,
            )
            if c.expert_aux_heads == "own"
            else None
        )
        self.curriculum_weight = 0.0
        self.stacked_skip = (
            nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(c.stacked_dims[m])) for m in self.stacked_names}
            )
            if c.stacked_skip and self.stacked_names
            else None
        )

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.parameters()).device

    @property
    def modalities(self) -> list[str]:
        """Enabled modality branches."""
        return list(self.fusion.modalities)

    def set_grl_alpha(self, alpha: float) -> None:
        """Set the gradient-reversal strength of the domain classifier."""
        if self.domain_head is not None:
            self.domain_head.alpha = float(alpha)

    def set_epoch(self, epoch: int) -> None:
        """Advance the transcriptomic curriculum of the modality dropout to ``epoch``."""
        span = int(self.config.modality_curriculum_epochs)
        self.curriculum_weight = max(0.0, 1.0 - float(epoch) / span) if span > 0 else 0.0

    def param_groups(
        self,
        weight_decay: float = 1e-2,
        lr: float | None = None,
        ct_backbone_lr_scale: float = 1.0,
        head_lr_scale: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Optimizer parameter groups.

        Biases, normalisation parameters, embeddings and other parameters with fewer than two
        dimensions are exempt from weight decay; CT backbone parameters carry
        ``ct_backbone_lr_scale`` and the task heads (fused, auxiliary and domain heads)
        ``head_lr_scale``. Weight matrices of the gene-level expert use
        ``config.gene_weight_decay`` when it is set.

        Args:
            weight_decay: Decoupled weight decay for weight matrices.
            lr: Base learning rate; if given, each group's ``lr`` is ``lr * lr_scale``.
            ct_backbone_lr_scale: Learning-rate multiplier of the CT backbone.
            head_lr_scale: Learning-rate multiplier of the task heads.
        """
        kinds: dict[int, str] = {}
        if self.imaging is not None:
            kinds.update({id(p): "ct_backbone" for p in self.imaging.backbone.parameters()})
        for head in (self.egfr_head, self.survival_head, self.domain_head, self.aux_heads):
            if head is not None:
                kinds.update({id(p): "head" for p in head.parameters()})
        gene_modules = (
            [self.gene_encoder, self.fusion.experts["gene"]]
            if (self.gene_encoder is not None)
            else []
        )
        for module in gene_modules:
            kinds.update({id(p): "gene" for p in module.parameters()})
        scales = {"ct_backbone": ct_backbone_lr_scale, "head": head_lr_scale}
        gene_wd = self.config.gene_weight_decay
        groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim < 2 or any(tag in name for tag in _NO_DECAY_TAGS)
            key = (kinds.get(id(param), "model"), no_decay)
            groups.setdefault(key, []).append(param)
        out = []
        for (kind, no_decay), params in groups.items():
            scale = float(scales.get(kind, 1.0))
            decay = gene_wd if (kind == "gene" and gene_wd is not None) else weight_decay
            group: dict[str, Any] = {
                "params": params,
                "weight_decay": 0.0 if no_decay else float(decay),
                "lr_scale": scale,
                "name": f"{kind}{'_no_decay' if no_decay else ''}",
            }
            if lr is not None:
                group["lr"] = lr * scale
            out.append(group)
        return out

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        batch: Mapping[str, Any],
        dropout_modalities: bool = False,
        *,
        omit_modalities: Sequence[str] = (),
        compute_recon: bool | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Run the network on a batch.

        Args:
            batch: Batch dictionary (see the package documentation for keys and shapes).
                Optional extra keys: ``clinical_indicators`` ``(B, n_clinical_indicators)``,
                ``rna_aux`` ``(B, n_rna_aux)``,
                ``ct_patch_mask`` ``(B, N)`` and ``ct_features`` ``(B, N, F)`` (precomputed
                backbone features that replace ``ct_patches``).
            dropout_modalities: Randomly omit available experts with probability
                ``modality_dropout_p`` (at least one expert is kept per sample).
            omit_modalities: Modalities to omit for every sample (e.g. ``("rna",)`` for the
                light configuration).
            compute_recon: Compute the self-supervised masked-modelling losses (default: in training mode only).
            return_embeddings: Add the modality embeddings to the outputs.

        Returns:
            Dictionary with ``logit_egfr`` ``(B,)``, ``p_egfr`` ``(B,)``, ``hazard_logits``,
            ``hazards`` and ``survival`` ``(B, K)``, ``mu`` and ``precision_diag`` ``(B, D)``,
            ``expert_weights`` and ``expert_present`` per modality ``(B,)``, ``recon_loss``,
            ``attn_prior_penalty`` (scalars), ``domain_logits`` ``(B, n_domains)``,
            ``attention`` (``ct`` ``(B, N)``, ``rna`` ``(B, P)``, ``clinical`` in eval mode) and,
            when ``primary_horizon_index`` is set, ``risk_horizon`` ``(B,)``.
        """
        c = self.config
        device = self.device
        b = batch_size_of(batch)
        available = modality_availability(
            batch,
            device,
            use_rna=self.pathway is not None,
            use_ct=self.imaging is not None,
            omit=omit_modalities,
            use_gene=self.gene_encoder is not None,
            stacked=self.stacked_names,
            removed=c.omit_experts,
        )
        present = dict(available)
        if dropout_modalities:
            present = scheduled_modality_dropout(
                present,
                c.modality_dropout_p,
                p_by=c.modality_dropout_by,
                exclude=c.modality_dropout_exclude,
                curriculum_weight=self.curriculum_weight if self.training else 0.0,
                p_group=c.modality_group_dropout,
            )
        if compute_recon is None:
            compute_recon = self.training

        embeddings: dict[str, Tensor] = {}
        attention: dict[str, Tensor] = {}
        recon_terms: list[Tensor] = []
        penalty = torch.zeros((), device=device)

        num = batch.get("clinical_num")
        if num is None:
            num = torch.zeros(b, c.n_num, device=device)
        num_mask = batch.get("clinical_num_mask")
        if num_mask is None:
            num_mask = torch.isfinite(num)
        cat = batch.get("clinical_cat")
        if cat is None:
            cat = torch.zeros(b, c.n_cat, dtype=torch.long, device=device)
        h_clin, attn_clin = self.clinical(
            num.to(device),
            num_mask.to(device),
            cat.to(device),
            extra_indicators=batch.get("clinical_indicators"),
            return_attention=not self.training,
        )
        embeddings["clinical"] = h_clin
        if attn_clin is not None:
            attention["clinical"] = attn_clin

        if self.pathway is not None:
            rna = batch.get("rna")
            if rna is None:
                rna = torch.full((b, c.n_pathways), float("nan"), device=device)
            rna = rna.to(device)
            h_rna, attn_rna, penalty = self.pathway(rna)
            if self.rna_aux is not None:
                aux = batch.get("rna_aux")
                if aux is None:
                    aux = torch.full((b, c.n_rna_aux), float("nan"), device=device)
                aux = aux.to(device).float()
                aux_in = torch.cat([sanitize(aux), (~torch.isfinite(aux)).float()], dim=1)
                h_rna = h_rna + self.rna_aux(aux_in)
            embeddings["rna"] = h_rna
            attention["rna"] = attn_rna * present["rna"].unsqueeze(1).to(attn_rna.dtype)
            genes = batch.get("rna_genes")
            if compute_recon and self.pathway.decoder is not None and genes is not None:
                rows = available["rna"]
                if bool(rows.any()):
                    genes = genes.to(device)
                    recon_terms.append(
                        self.pathway.masked_gene_loss(
                            genes[rows],
                            activities=rna[rows],
                            h=None if self.pathway.incidence is not None else h_rna[rows],
                        )
                    )

        if self.gene_encoder is not None:
            panel = batch.get("rna_panel")
            if panel is None:
                panel = torch.full((b, c.n_panel_genes), float("nan"), device=device)
            embeddings["gene"] = self.gene_encoder(panel.to(device))
        for m in self.stacked_names:
            score = batch.get(f"stacked_{m}")
            if score is None:
                score = torch.zeros(b, c.stacked_dims[m], device=device)
            embeddings[m] = sanitize(score.to(device).float())

        n_patches = 0
        if self.imaging is not None:
            patches = batch.get("ct_patches")
            features = batch.get("ct_features")
            source = features if features is not None else patches
            if source is not None:
                n_patches = int(source.shape[1])
                h_img, attn_img, recon_ct = self.imaging(
                    patches,
                    present["ct"],
                    patch_mask=batch.get("ct_patch_mask"),
                    features=features,
                    compute_recon=bool(compute_recon),
                )
                if compute_recon and self.imaging.recon_head is not None:
                    recon_terms.append(recon_ct)
            else:
                h_img = torch.zeros(b, self.imaging.out_dim, device=device)
                attn_img = torch.zeros(b, 0, device=device)
            embeddings["ct"] = h_img
            attention["ct"] = attn_img
        if "ct" not in attention:
            attention["ct"] = torch.zeros(b, n_patches, device=device)

        fused = self.fusion(embeddings, present, batch.get("quality"))
        mu, precision = fused["mu"], fused["precision"]
        z = mu
        if c.latent_sampling and self.training:
            z = mu + torch.randn_like(mu) * precision.rsqrt()
        logit = self.egfr_head(z).float()
        hazard_logits, hazards, survival = self.survival_head(z)
        if self.stacked_skip is not None:
            shift = stacked_shift(embeddings, present, fused["rho"], self.stacked_skip)
            logit = logit + shift
            hazard_logits = hazard_logits + shift.unsqueeze(1)
            hazards = torch.sigmoid(hazard_logits)
            survival = torch.exp(torch.cumsum(-F.softplus(hazard_logits), dim=-1))
        if self.domain_head is not None:
            domain_logits = self.domain_head(z)
        else:
            domain_logits = torch.zeros(b, 0, device=device)

        recon_loss = (
            torch.stack(recon_terms).sum() if recon_terms else torch.zeros((), device=device)
        )
        rho = fused["rho"]
        outputs: dict[str, Any] = {
            "logit_egfr": logit,
            "p_egfr": torch.sigmoid(logit),
            "hazard_logits": hazard_logits,
            "hazards": hazards,
            "survival": survival,
            "mu": mu,
            "precision_diag": precision,
            "expert_weights": {
                m: (
                    rho[m] * present[m].to(rho[m].dtype)
                    if m in rho
                    else torch.zeros(b, device=device)
                )
                for m in EXPERTS
                if m in MODALITIES or m in rho
            },
            "expert_present": {
                m: present.get(m, torch.zeros(b, dtype=torch.bool, device=device))
                for m in EXPERTS
                if m in MODALITIES or m in present
            },
            "recon_loss": recon_loss,
            "attn_prior_penalty": penalty,
            "domain_logits": domain_logits,
            "attention": attention,
        }
        if c.primary_horizon_index is not None:
            idx = int(min(max(c.primary_horizon_index, 0), c.n_survival_bins - 1))
            outputs["risk_horizon"] = 1.0 - survival[:, idx]
        if c.expert_aux_heads != "none" and self.training:
            outputs["aux"] = expert_aux_outputs(self, fused, available)
        if self.gene_encoder is not None and self.training:
            outputs["gene_l1"] = gene_l1_penalty(self.gene_encoder, self.fusion.experts["gene"])
        if self.training:
            prec = fused["expert_precision"]
            shrunk = (
                ("rna", "gene") if c.rna_precision_experts == "neural" else TRANSCRIPTOMIC_EXPERTS
            )
            outputs["rna_precision"] = sum(
                (rho[m] * prec[m].mean(1) for m in shrunk if m in prec),
                torch.zeros(b, device=device),
            )
        if return_embeddings:
            outputs["embeddings"] = embeddings
        return outputs

    def compute_loss(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: LossWeights
    ) -> dict[str, Tensor]:
        """Weighted training objective; see :func:`molt.models.losses.compute_multitask_loss`."""
        return compute_multitask_loss(
            outputs, batch, weights, horizon_index=self.config.primary_horizon_index
        )
