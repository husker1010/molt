"""Task heads on the fused latent representation and the gradient-reversal domain classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from molt.models.common import make_mlp

__all__ = [
    "BinaryHead",
    "DiscreteTimeSurvivalHead",
    "DomainClassifier",
    "GradientReversal",
    "grad_reverse",
    "survival_from_hazards",
]


class GradientReversal(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by ``-alpha`` in the backward pass."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return -ctx.alpha * grad, None


def grad_reverse(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Apply the gradient-reversal layer with strength ``alpha``."""
    return GradientReversal.apply(x, alpha)


def survival_from_hazards(hazards: Tensor) -> Tensor:
    """Survival ``S(tau_k) = prod_{l <= k} (1 - h_l)`` from conditional hazards ``(B, K)``."""
    return torch.cumprod(1.0 - hazards, dim=-1)


class BinaryHead(nn.Module):
    """MLP producing one logit per sample.

    Args:
        in_dim: Latent dimension.
        hidden: Hidden width (0 for a linear head).
        dropout: Dropout before the output layer.
    """

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, [hidden] if hidden else [], 1, dropout=dropout)

    def forward(self, z: Tensor) -> Tensor:
        """Return logits ``(B,)``."""
        return self.net(z).squeeze(-1)


class DiscreteTimeSurvivalHead(nn.Module):
    """Conditional hazards over ``K`` time bins.

    The head outputs logits ``eta_k`` with ``h_k = sigmoid(eta_k)``; survival is the cumulative
    product of ``1 - h_k`` and is therefore non-increasing over the bins.

    Args:
        in_dim: Latent dimension.
        n_bins: Number of time bins ``K``.
        hidden: Hidden width (0 for a linear head).
        dropout: Dropout before the output layer.
        init_hazard: Initial hazard per bin (sets the output bias).
    """

    def __init__(
        self,
        in_dim: int,
        n_bins: int,
        hidden: int = 64,
        dropout: float = 0.2,
        init_hazard: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.net = make_mlp(in_dim, [hidden] if hidden else [], self.n_bins, dropout=dropout)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.bias.fill_(float(torch.logit(torch.tensor(init_hazard))))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return hazard logits, hazards and survival, each ``(B, K)``."""
        logits = self.net(z).float()
        hazards = torch.sigmoid(logits)
        survival = torch.exp(torch.cumsum(-F.softplus(logits), dim=-1))
        return logits, hazards, survival


class DomainClassifier(nn.Module):
    """Domain (site / plate / cohort) classifier behind a gradient-reversal layer.

    Args:
        in_dim: Latent dimension.
        n_domains: Number of domain classes.
        hidden: Hidden width.
        dropout: Dropout in the hidden layer.
        alpha: Gradient-reversal strength (mutable via :attr:`alpha`).
    """

    def __init__(
        self,
        in_dim: int,
        n_domains: int,
        hidden: int = 64,
        dropout: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.net = make_mlp(in_dim, [hidden], n_domains, dropout=dropout)

    def forward(self, z: Tensor) -> Tensor:
        """Return domain logits ``(B, n_domains)`` with reversed gradients flowing into ``z``."""
        return self.net(grad_reverse(z, self.alpha))
