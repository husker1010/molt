"""Gene-level transcriptomic expert, expert-wise auxiliary heads and modality-dropout schedules.

* :class:`GeneEncoder` maps a fold-filtered panel of standardized genes to the embedding of
  the gene-level expert: input dropout (genes are randomly hidden during training), then
  either the identity (the Gaussian expert of the fusion module is then a single linear map
  of the genes, i.e. a *linear* gene expert) or one hidden layer
  ``Linear -> LayerNorm -> GELU -> Dropout``. The hidden layer can be initialized from a
  denoising autoencoder trained on the training-fold profiles
  (:func:`molt.training.pretrain.pretrain_gene_encoder`).
* :func:`single_expert_posteriors` gives, for every expert ``m``, the latent posterior mean
  that the product of experts would produce if ``m`` were the only observed modality,
  ``z_m = rho_m lambda_m mu_m / (lambda_0 + rho_m lambda_m)``.
* :class:`ExpertAuxHeads` holds expert-specific EGFR and survival heads for deep supervision
  (``expert_aux_heads="own"``); with ``"shared"`` the fused heads are applied to ``z_m``.
* :func:`scheduled_modality_dropout` drops experts with modality-specific probabilities,
  drops whole modality groups jointly (for example every transcriptomic expert at once, the
  configuration of the light mode) and, during a curriculum phase, removes the
  non-transcriptomic experts of participants with an expression profile so that the
  transcriptomic experts fit the endpoint on their own first.
* Stacked linear experts (``clinical_lin``, ``gene_lin``, ``rna_lin``) take the scores of
  penalized linear models fitted in the training fold
  (:mod:`molt.training.stacking`) as their embedding; they belong to the modality group
  of their inputs (availability, omission, group dropout and reliability weight).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from molt.models.common import sanitize
from molt.models.heads import BinaryHead, DiscreteTimeSurvivalHead

__all__ = [
    "EXPERTS",
    "EXPERT_AUX_MODES",
    "EXPERT_GROUPS",
    "STACKED_EXPERTS",
    "STACKED_SOURCE",
    "TRANSCRIPTOMIC_EXPERTS",
    "ExpertAuxHeads",
    "GeneEncoder",
    "batch_size_of",
    "expert_aux_outputs",
    "gene_l1_penalty",
    "modality_availability",
    "scheduled_modality_dropout",
    "single_expert_posteriors",
    "stacked_shift",
]

STACKED_EXPERTS: tuple[str, ...] = ("clinical_lin", "gene_lin", "rna_lin")
"""Stacked linear experts (scores of penalized linear models fitted in the training fold)."""

STACKED_SOURCE: dict[str, str] = {"clinical_lin": "clinical", "gene_lin": "rna", "rna_lin": "rna"}
"""Modality whose availability and reliability weight a stacked expert uses."""

EXPERTS: tuple[str, ...] = ("clinical", "rna", "gene", "ct", *STACKED_EXPERTS)
"""Fusion experts: clinical, pathway (``rna``), gene-level (``gene``), CT and stacked."""

TRANSCRIPTOMIC_EXPERTS: tuple[str, ...] = ("rna", "gene", "gene_lin", "rna_lin")
"""Experts fed by the expression profile (pathway activities, gene panel, stacked scores)."""

EXPERT_GROUPS: dict[str, tuple[str, ...]] = {
    "clinical": ("clinical", "clinical_lin"),
    "rna": TRANSCRIPTOMIC_EXPERTS,
    "ct": ("ct",),
}
"""Modality groups: omitting or group-dropping a modality removes all of its experts."""

EXPERT_AUX_MODES = ("none", "shared", "own")
"""Accepted values of ``MOLTConfig.expert_aux_heads``."""


class GeneEncoder(nn.Module):
    """Embedding of the gene-level expert.

    Args:
        n_genes: Number of panel genes ``G``.
        hidden: Hidden width; ``0`` passes the (dropped-out) genes through unchanged so that
            the expert is linear in the genes.
        input_dropout: Probability of hiding each gene during training.
        dropout: Dropout after the hidden activation.
    """

    def __init__(
        self, n_genes: int, hidden: int = 0, input_dropout: float = 0.5, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.hidden = int(hidden)
        self.input_drop = nn.Dropout(float(input_dropout))
        self.encoder: nn.Sequential | None = None
        if self.hidden > 0:
            self.encoder = nn.Sequential(
                nn.Linear(self.n_genes, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )

    @property
    def out_dim(self) -> int:
        """Embedding dimension."""
        return self.hidden if self.hidden > 0 else self.n_genes

    @property
    def first_linear(self) -> nn.Linear | None:
        """The linear layer of the hidden variant (``None`` for the linear expert)."""
        return None if self.encoder is None else self.encoder[0]

    def forward(self, genes: Tensor) -> Tensor:
        """Encode ``(B, G)`` standardized genes (NaN = not measured, encoded as 0)."""
        x = self.input_drop(sanitize(genes.float()))
        return x if self.encoder is None else self.encoder(x)


def single_expert_posteriors(
    expert_mu: Mapping[str, Tensor],
    expert_precision: Mapping[str, Tensor],
    rho: Mapping[str, Tensor],
    prior_precision: float = 1.0,
) -> dict[str, Tensor]:
    """Posterior mean of the latent when only expert ``m`` is observed, for every ``m``."""
    out: dict[str, Tensor] = {}
    for m, mu in expert_mu.items():
        weight = rho[m].unsqueeze(1).to(mu.dtype) * expert_precision[m]
        out[m] = weight * mu / (prior_precision + weight)
    return out


class ExpertAuxHeads(nn.Module):
    """Expert-specific EGFR and discrete-time survival heads (deep supervision).

    Args:
        experts: Expert names.
        latent_dim: Latent dimension.
        n_bins: Survival bins.
        hidden: Hidden width of the heads.
        dropout: Head dropout.
        init_hazard: Initial hazard of the survival heads.
    """

    def __init__(
        self,
        experts: Iterable[str],
        latent_dim: int,
        n_bins: int,
        hidden: int = 64,
        dropout: float = 0.2,
        init_hazard: float = 0.05,
    ) -> None:
        super().__init__()
        names = list(experts)
        self.egfr = nn.ModuleDict({m: BinaryHead(latent_dim, hidden, dropout) for m in names})
        self.survival = nn.ModuleDict(
            {
                m: DiscreteTimeSurvivalHead(latent_dim, n_bins, hidden, dropout, init_hazard)
                for m in names
            }
        )


def scheduled_modality_dropout(
    present: Mapping[str, Tensor],
    p: float,
    *,
    p_by: Mapping[str, float] | None = None,
    exclude: Iterable[str] = (),
    curriculum_weight: float = 0.0,
    p_group: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Modality dropout with expert- and group-specific probabilities and a curriculum.

    First every modality group ``g`` of :data:`EXPERT_GROUPS` is dropped as a whole with
    probability ``p_group[g]`` (all of its experts at once; skipped for a sample when no
    other expert would remain). Then every available expert ``m`` is dropped independently
    with probability ``p_by[m]`` (default ``p``); at least one available expert is kept per
    sample. Before that, each participant with a transcriptomic expert is, with probability
    ``curriculum_weight``, reduced to its transcriptomic experts (clinical and CT experts
    removed).

    Args:
        present: Mapping expert -> ``(B,)`` availability.
        p: Default drop probability.
        p_by: Expert-specific drop probabilities.
        exclude: Experts never dropped by the random dropout.
        curriculum_weight: Probability of a transcriptome-only sample.
        p_group: Group-specific joint drop probabilities (keys of :data:`EXPERT_GROUPS`).
    """
    out = {k: v.clone() for k, v in present.items()}
    for group, prob in dict(p_group or {}).items():
        members = [m for m in EXPERT_GROUPS.get(group, ()) if m in out]
        if prob <= 0 or not members:
            continue
        ref = out[members[0]]
        inside = torch.stack([out[m] for m in members], 1).any(1)
        others = [v for m, v in out.items() if m not in members]
        outside = torch.stack(others, 1).any(1) if others else torch.zeros_like(inside)
        drop = inside & outside & (torch.rand(ref.shape, device=ref.device) < float(prob))
        for m in members:
            out[m] = out[m] & ~drop
    if curriculum_weight > 0:
        rna_names = [m for m in TRANSCRIPTOMIC_EXPERTS if m in out]
        if rna_names:
            has_rna = torch.stack([out[m] for m in rna_names], 1).any(1)
            ref = out[rna_names[0]]
            force = has_rna & (torch.rand(ref.shape, device=ref.device) < curriculum_weight)
            for m, value in list(out.items()):
                if m not in TRANSCRIPTOMIC_EXPERTS:
                    out[m] = value & ~force
    probs = {m: float((p_by or {}).get(m, p)) for m in out}
    if not any(v > 0 for v in probs.values()):
        return out
    names = list(out)
    excluded = set(exclude)
    avail = torch.stack([out[m].to(torch.bool) for m in names], dim=1)
    device = avail.device
    p_vec = torch.tensor(
        [0.0 if m in excluded else probs[m] for m in names], device=device, dtype=torch.float32
    )
    drop = torch.rand(avail.shape, device=device) < p_vec
    kept = avail & ~drop
    none_left = (kept.sum(1) == 0) & avail.any(1)
    scores = torch.rand(avail.shape, device=device).masked_fill(~avail, -1.0)
    restore = F.one_hot(scores.argmax(1), len(names)).to(torch.bool) & none_left.unsqueeze(1)
    kept = kept | restore
    return {m: kept[:, i] for i, m in enumerate(names)}


def gene_l1_penalty(encoder: GeneEncoder | None, expert_net: nn.Module | None) -> Tensor | None:
    """Mean per-unit L1 norm of the first linear map applied to the gene panel."""
    layer = encoder.first_linear if encoder is not None else None
    if layer is None and expert_net is not None:
        linears = [m for m in expert_net.modules() if isinstance(m, nn.Linear)]
        layer = linears[0] if linears else None
    if layer is None:
        return None
    return layer.weight.abs().sum(1).mean()


def batch_size_of(batch: Mapping[str, Any]) -> int:
    """Infer the batch size from the first tensor-valued standard key of ``batch``."""
    for key in (
        "clinical_num",
        "clinical_cat",
        "clinical_present",
        "rna",
        "rna_present",
        "ct_present",
        "y_egfr",
    ):
        value = batch.get(key)
        if isinstance(value, Tensor) and value.dim() >= 1:
            return int(value.shape[0])
    raise ValueError("Cannot infer the batch size from the batch")


def modality_availability(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    use_rna: bool = True,
    use_ct: bool = True,
    omit: Iterable[str] = (),
    use_gene: bool = False,
    stacked: Iterable[str] = (),
    removed: Iterable[str] = (),
) -> dict[str, Tensor]:
    """Per-sample availability ``{modality: (B,) bool}`` from the batch flags.

    A modality is available when its ``*_present`` flag is true, its inputs exist in the batch,
    the branch is enabled and it is not listed in ``omit``. With ``use_gene`` the gene-level
    expert (``"gene"``) is available with the RNA profile when ``rna_panel`` is in the batch.
    A stacked expert ``m`` of ``stacked`` is available with its source modality
    (:data:`STACKED_SOURCE`) when ``batch["stacked_<m>"]`` holds a finite score. Omitting
    ``"rna"`` or ``"clinical"`` omits every expert of that group (:data:`EXPERT_GROUPS`);
    omitting ``"gene"`` omits the gene-level experts ``gene`` and ``gene_lin``. Experts in
    ``removed`` are made unavailable individually, without the group expansion (``"rna"``
    then removes the pathway expert only).
    """
    b = batch_size_of(batch)
    omit = set(omit)
    removed = set(removed)
    unknown = (omit | removed) - set(EXPERTS)
    if unknown:
        raise ValueError(f"Unknown modalities to omit: {sorted(unknown)}")

    def flag(key: str, default: bool) -> Tensor:
        value = batch.get(key)
        if value is None:
            return torch.full((b,), default, dtype=torch.bool, device=device)
        return value.to(device=device, dtype=torch.bool)

    has_ct_input = batch.get("ct_patches") is not None or batch.get("ct_features") is not None
    present = {
        "clinical": flag("clinical_present", True),
        "rna": flag("rna_present", batch.get("rna") is not None)
        & bool(use_rna)
        & (batch.get("rna") is not None),
        "ct": flag("ct_present", has_ct_input) & bool(use_ct) & has_ct_input,
    }
    if use_gene:
        rna_flag = flag("rna_present", batch.get("rna_panel") is not None)
        present["gene"] = rna_flag & (batch.get("rna_panel") is not None)
    for m in stacked:
        score = batch.get(f"stacked_{m}")
        source = "clinical_present" if STACKED_SOURCE[m] == "clinical" else "rna_present"
        if score is None:
            present[m] = torch.zeros(b, dtype=torch.bool, device=device)
            continue
        finite = torch.isfinite(score.to(device)).all(1) if score.dim() > 1 else None
        present[m] = flag(source, True) & (finite if finite is not None else True)
    for group in ("rna", "clinical"):
        if group in omit:
            omit.update(EXPERT_GROUPS[group])
    if "gene" in omit:
        omit.add("gene_lin")
    for m in omit | removed:
        if m in present:
            present[m] = torch.zeros_like(present[m])
    return present


def expert_aux_outputs(
    model: nn.Module, fused: Mapping[str, Any], available: Mapping[str, Tensor]
) -> dict[str, dict[str, Tensor]]:
    """Endpoint outputs of every expert's single-expert posterior (deep supervision).

    Uses the expert-specific heads of ``model.aux_heads`` when present, otherwise the fused
    EGFR and survival heads of ``model``.
    """
    z_by = single_expert_posteriors(
        fused["expert_mu_raw"],
        fused["expert_precision_raw"],
        fused["rho"],
        model.fusion.prior_precision,
    )
    own = model.aux_heads
    out: dict[str, dict[str, Tensor]] = {"logit_egfr": {}, "hazard_logits": {}}
    out["hazards"], out["present"] = {}, {}
    for m, z in z_by.items():
        egfr_head = own.egfr[m] if own is not None else model.egfr_head
        surv_head = own.survival[m] if own is not None else model.survival_head
        out["logit_egfr"][m] = egfr_head(z).float()
        out["hazard_logits"][m], out["hazards"][m], _ = surv_head(z)
        out["present"][m] = available[m]
    return out


def stacked_shift(
    scores: Mapping[str, Tensor],
    present: Mapping[str, Tensor],
    rho: Mapping[str, Tensor],
    weights: Mapping[str, Tensor],
) -> Tensor:
    """Linear skip of the stacked scores: ``sum_m present_m rho_m (w_m' s_m)``, shape ``(B,)``."""
    total: Tensor | None = None
    for m, w in weights.items():
        s = scores[m]
        term = (s * w.to(s.dtype)).sum(1) * rho[m].to(s.dtype) * present[m].to(s.dtype)
        total = term if total is None else total + term
    assert total is not None
    return total.float()
