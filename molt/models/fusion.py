"""Reliability-weighted Gaussian product-of-experts fusion of modality embeddings.

Every available modality ``m`` produces a Gaussian expert with mean ``mu_m`` and a positive
diagonal precision ``lambda_m``. A reliability network maps the modality's quality inputs to a
weight ``rho_m`` in ``[0, 1]`` that is monotone non-increasing in every adverse quality flag. The
fused posterior over the shared latent ``z`` is

    Lambda = I + sum_m rho_m lambda_m,        mu = Lambda^{-1} sum_m rho_m lambda_m mu_m,

with the sums running over the observed modalities only, so missing modalities are omitted
without imputation and a patient without any modality falls back to the unit prior.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from molt.models.common import make_mlp, sanitize

__all__ = [
    "MODALITIES",
    "FusionRule",
    "GaussianExpert",
    "MonotoneReliability",
    "ProductOfExpertsFusion",
    "sample_modality_dropout",
]

MODALITIES: tuple[str, ...] = ("clinical", "rna", "ct")
"""Modality names used throughout the model."""

FusionRule = Literal["poe", "mean"]


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class GaussianExpert(nn.Module):
    """Map a modality embedding to a Gaussian expert ``(mu_m, lambda_m)``.

    Args:
        in_dim: Embedding dimension of the modality.
        latent_dim: Dimension ``D`` of the shared latent space.
        hidden: Hidden width (``0`` for a single linear map).
        dropout: Dropout on the hidden layer.
        min_precision: Lower bound added to the softplus precision.
        max_precision: Upper clamp of the precision.
        init_precision: Precision produced at initialisation for a zero embedding.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden: int = 0,
        dropout: float = 0.0,
        min_precision: float = 1e-3,
        max_precision: float = 1e3,
        init_precision: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = make_mlp(in_dim, [hidden] if hidden else [], 2 * latent_dim, dropout=dropout)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.bias[latent_dim:].fill_(_inv_softplus(max(init_precision - min_precision, 1e-4)))
        self.min_precision = float(min_precision)
        self.max_precision = float(max_precision)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return the expert mean and diagonal precision, both ``(B, D)``."""
        out = self.net(h).float()
        mu, raw = out.split(self.latent_dim, dim=-1)
        lam = (F.softplus(raw) + self.min_precision).clamp(max=self.max_precision)
        return mu, lam


class MonotoneReliability(nn.Module):
    """Scalar reliability ``rho in [0, 1]`` that is non-increasing in every adverse quality input.

    ``rho(q) = sigmoid(c - [g(s * q) - g(0)])`` where ``g`` is a two-layer network with
    non-negative weights (softplus-parametrised) and monotone activations, hence non-decreasing
    in each argument, and ``s`` holds the fixed input orientations (``+1`` for adverse flags
    where larger means worse, ``-1`` for quality scores where larger means better). ``rho(0)``
    equals ``sigmoid(c)`` with a learnable baseline ``c``.

    Args:
        n_inputs: Number of quality inputs ``Q`` (0 gives a learnable constant).
        hidden: Hidden width of ``g``.
        signs: Optional orientation per input (default all ``+1``).
        init_logit: Initial baseline ``c``.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden: int = 8,
        signs: Sequence[float] | None = None,
        init_logit: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.baseline = nn.Parameter(torch.tensor(float(init_logit)))
        if signs is None:
            signs = [1.0] * self.n_inputs
        if len(signs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} quality signs, got {len(signs)}")
        if any(s not in (-1, 1, -1.0, 1.0) for s in signs):
            raise ValueError("Quality signs must be +1 (adverse) or -1 (favourable)")
        self.register_buffer("signs", torch.tensor(list(signs), dtype=torch.float32))
        if self.n_inputs:
            self.w1 = nn.Parameter(torch.randn(hidden, self.n_inputs) * 0.5 - 1.0)
            self.b1 = nn.Parameter(torch.zeros(hidden))
            self.w2 = nn.Parameter(torch.randn(hidden) * 0.5)

    def _g(self, q: Tensor) -> Tensor:
        z = torch.tanh(q @ F.softplus(self.w1).T + self.b1)
        return z @ F.softplus(self.w2)

    def forward(self, quality: Tensor | None, batch_size: int | None = None) -> Tensor:
        """Return ``rho`` of shape ``(B,)``.

        Args:
            quality: ``(B, Q)`` quality inputs (NaN treated as 0, i.e. not adverse), or ``None``
                when ``Q == 0``.
            batch_size: Required when ``quality`` is ``None``.
        """
        if self.n_inputs == 0 or quality is None:
            if batch_size is None:
                if quality is None:
                    raise ValueError("batch_size is required without quality inputs")
                batch_size = quality.shape[0]
            return torch.sigmoid(self.baseline).expand(batch_size)
        q = sanitize(quality.to(self.baseline.device, torch.float32)) * self.signs
        shift = self._g(q) - self._g(torch.zeros_like(q[:1]))
        return torch.sigmoid(self.baseline - shift)


def sample_modality_dropout(
    present: Mapping[str, Tensor],
    p: float,
    *,
    exclude: Iterable[str] = (),
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Randomly omit available experts, keeping at least one per sample.

    Each available expert is dropped independently with probability ``p``; if all of a sample's
    experts were dropped, one of its originally available experts (chosen uniformly) is restored.
    Samples without any available expert are left unchanged.

    Args:
        present: Mapping modality -> ``(B,)`` boolean availability.
        p: Drop probability.
        exclude: Modalities that are never dropped.
        generator: Optional random generator.

    Returns:
        The new availability masks.
    """
    names = list(present)
    if not names or p <= 0:
        return {k: v.clone() for k, v in present.items()}
    avail = torch.stack([present[m].to(torch.bool) for m in names], dim=1)
    device = avail.device
    droppable = torch.tensor([m not in set(exclude) for m in names], device=device)
    drop = (torch.rand(avail.shape, device=device, generator=generator) < p) & droppable
    kept = avail & ~drop
    none_left = (kept.sum(1) == 0) & avail.any(1)
    scores = torch.rand(avail.shape, device=device, generator=generator).masked_fill(~avail, -1.0)
    restore = F.one_hot(scores.argmax(1), len(names)).to(torch.bool) & none_left.unsqueeze(1)
    kept = kept | restore
    return {m: kept[:, i] for i, m in enumerate(names)}


class ProductOfExpertsFusion(nn.Module):
    """Fuse modality embeddings into a Gaussian latent posterior.

    Args:
        in_dims: Embedding dimension per modality.
        latent_dim: Latent dimension ``D``.
        quality_dims: Number of quality inputs per modality.
        quality_signs: Optional orientation vectors per modality (see :class:`MonotoneReliability`).
        use_reliability: If ``False`` every ``rho_m`` is fixed to 1.
        fixed_reliability: Modalities whose ``rho_m`` is fixed to 1 (e.g. ``("ct",)``).
        rule: ``"poe"`` (product of experts) or ``"mean"`` (unweighted average of expert means,
            used as an ablation).
        expert_hidden: Hidden width of the expert networks.
        dropout: Dropout in the expert networks.
        prior_precision: Precision of the standard-normal prior expert.
        shared_reliability: Experts whose weight ``rho`` is that of another expert
            (e.g. ``{"gene": "rna"}``: the gene-level expert uses the RNA reliability).
    """

    def __init__(
        self,
        in_dims: Mapping[str, int],
        latent_dim: int,
        quality_dims: Mapping[str, int] | None = None,
        quality_signs: Mapping[str, Sequence[float]] | None = None,
        use_reliability: bool = True,
        fixed_reliability: Iterable[str] = (),
        rule: FusionRule = "poe",
        expert_hidden: int = 0,
        dropout: float = 0.0,
        prior_precision: float = 1.0,
        shared_reliability: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if rule not in ("poe", "mean"):
            raise ValueError(f"Unknown fusion rule {rule!r}")
        self.modalities = list(in_dims)
        self.latent_dim = int(latent_dim)
        self.rule: FusionRule = rule
        self.use_reliability = bool(use_reliability)
        self.fixed_reliability = set(fixed_reliability)
        unknown = self.fixed_reliability - set(self.modalities)
        if unknown:
            raise ValueError(f"fixed_reliability names unknown modalities: {sorted(unknown)}")
        self.prior_precision = float(prior_precision)
        quality_dims = dict(quality_dims or {})
        quality_signs = dict(quality_signs or {})
        self.quality_dims = {m: int(quality_dims.get(m, 0)) for m in self.modalities}
        self.shared_reliability = {
            m: src
            for m, src in dict(shared_reliability or {}).items()
            if m in self.modalities and src in self.modalities and src != m
        }
        self.experts = nn.ModuleDict(
            {m: GaussianExpert(d, latent_dim, expert_hidden, dropout) for m, d in in_dims.items()}
        )
        self.reliability = nn.ModuleDict(
            {
                m: MonotoneReliability(self.quality_dims[m], signs=quality_signs.get(m))
                for m in self.modalities
                if m not in self.shared_reliability
            }
        )

    def reliability_weights(
        self, quality: Mapping[str, Tensor] | None, batch_size: int, device: torch.device
    ) -> dict[str, Tensor]:
        """Compute ``rho_m`` ``(B,)`` for every modality from its quality inputs."""
        quality = quality or {}
        rho: dict[str, Tensor] = {}
        for m in self.modalities:
            if m in self.shared_reliability:
                continue
            if not self.use_reliability or m in self.fixed_reliability:
                rho[m] = torch.ones(batch_size, device=device)
                continue
            q = quality.get(m)
            dim = self.quality_dims[m]
            if dim and q is None:
                q = torch.zeros(batch_size, dim, device=device)
            if dim and q is not None and q.shape[-1] != dim:
                raise ValueError(f"quality[{m!r}] has {q.shape[-1]} columns, expected {dim}")
            rho[m] = self.reliability[m](q if dim else None, batch_size).to(device)
        for m, src in self.shared_reliability.items():
            rho[m] = rho[src]
        return rho

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        present: Mapping[str, Tensor],
        quality: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Fuse the experts.

        Args:
            embeddings: Modality -> ``(B, d_m)`` embedding (rows of absent modalities ignored).
            present: Modality -> ``(B,)`` boolean availability.
            quality: Modality -> ``(B, Q_m)`` quality inputs.

        Returns:
            ``mu`` and ``precision`` ``(B, D)``, and per-modality ``rho``, ``expert_mu`` and
            ``expert_precision`` dictionaries (zero for absent experts) and the unmasked expert
            outputs ``expert_mu_raw`` and ``expert_precision_raw``.
        """
        names = [m for m in self.modalities if m in embeddings]
        if not names:
            raise ValueError("At least one modality embedding is required")
        ref = embeddings[names[0]]
        b, device = ref.shape[0], ref.device
        rho = self.reliability_weights(quality, b, device)
        num = torch.zeros(b, self.latent_dim, device=device)
        prec = torch.full((b, self.latent_dim), self.prior_precision, device=device)
        count = torch.zeros(b, 1, device=device)
        expert_mu: dict[str, Tensor] = {}
        expert_prec: dict[str, Tensor] = {}
        raw_mu: dict[str, Tensor] = {}
        raw_prec: dict[str, Tensor] = {}
        for m in names:
            mu_m, lam_m = self.experts[m](embeddings[m])
            raw_mu[m], raw_prec[m] = mu_m, lam_m
            avail = present[m].to(device=device, dtype=torch.bool).unsqueeze(1)
            mu_m = torch.where(avail, mu_m, torch.zeros_like(mu_m))
            lam_m = torch.where(avail, lam_m, torch.zeros_like(lam_m))
            expert_mu[m], expert_prec[m] = mu_m, lam_m
            if self.rule == "poe":
                weight = rho[m].unsqueeze(1) * lam_m
                num = num + weight * mu_m
                prec = prec + weight
            else:
                num = num + mu_m
                count = count + avail.float()
        if self.rule == "poe":
            mu = num / prec
        else:
            mu = num / count.clamp_min(1.0)
            prec = self.prior_precision + count.expand(-1, self.latent_dim)
        return {
            "mu": mu,
            "precision": prec,
            "rho": rho,
            "expert_mu": expert_mu,
            "expert_precision": expert_prec,
            "expert_mu_raw": raw_mu,
            "expert_precision_raw": raw_prec,
        }
