"""Shared neural-network building blocks used by the MOLT encoders and heads."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AttentionPool",
    "GEGLUFeedForward",
    "load_checkpoint_payload",
    "make_mlp",
    "masked_softmax",
    "sanitize",
]


def sanitize(x: Tensor, value: float = 0.0) -> Tensor:
    """Replace NaN and infinite entries of ``x`` by ``value`` (gradient-safe)."""
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


def masked_softmax(scores: Tensor, mask: Tensor | None, dim: int = -1) -> Tensor:
    """Softmax over ``dim`` restricted to entries where ``mask`` is ``True``.

    Rows without any valid entry return all zeros instead of NaN.

    Args:
        scores: Unnormalised scores.
        mask: Boolean tensor broadcastable to ``scores`` (``True`` = keep), or ``None``.
        dim: Dimension to normalise over.
    """
    if mask is None:
        return torch.softmax(scores, dim=dim)
    mask = mask.to(torch.bool)
    filled = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(filled, dim=dim)
    weights = weights * mask.to(weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights / denom


def make_mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    *,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.GELU,
    norm: bool = True,
) -> nn.Sequential:
    """Build ``Linear -> [LayerNorm] -> activation -> Dropout`` blocks followed by a linear output.

    Args:
        in_dim: Input feature dimension.
        hidden: Hidden layer widths (may be empty for a single linear map).
        out_dim: Output dimension.
        dropout: Dropout probability after every hidden activation.
        activation: Activation module class.
        norm: Insert ``LayerNorm`` after every hidden linear layer.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        if norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GEGLUFeedForward(nn.Module):
    """Gated feed-forward block ``W_o (GELU(W_a x) * W_b x)`` used inside transformer layers.

    Args:
        dim: Token dimension.
        mult: Hidden expansion factor.
        dropout: Dropout probability applied to the gated hidden units and the output.
    """

    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = round(dim * mult)
        self.proj_in = nn.Linear(dim, 2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward transform to ``x`` of shape ``(..., dim)``."""
        a, b = self.proj_in(x).chunk(2, dim=-1)
        return self.drop(self.proj_out(self.drop(F.gelu(a) * b)))


class AttentionPool(nn.Module):
    """Attention pooling over a set of instance embeddings.

    Scores are ``w^T tanh(V u)``; with ``gated=True`` they become
    ``w^T (tanh(V u) * sigmoid(U u))``. Weights are a softmax over the valid instances and the
    pooled representation is the weighted sum of the instance embeddings.

    Args:
        dim: Instance embedding dimension.
        hidden: Width of the attention scoring network.
        gated: Use the sigmoid-gated variant of the scoring network.
        dropout: Dropout applied to the scoring network's hidden units.
    """

    def __init__(
        self, dim: int, hidden: int = 128, gated: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.proj_v = nn.Linear(dim, hidden)
        self.proj_u = nn.Linear(dim, hidden) if gated else None
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Pool ``x`` of shape ``(B, N, dim)``.

        Args:
            x: Instance embeddings.
            mask: Optional ``(B, N)`` boolean validity mask (``True`` = valid instance).

        Returns:
            Pooled embeddings ``(B, dim)`` and attention weights ``(B, N)``.
        """
        h = torch.tanh(self.proj_v(x))
        if self.proj_u is not None:
            h = h * torch.sigmoid(self.proj_u(x))
        scores = self.score(self.drop(h)).squeeze(-1)
        weights = masked_softmax(scores, mask, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, x)
        return pooled, weights


def load_checkpoint_payload(
    path: str | os.PathLike[str], map_location: str | torch.device
) -> dict[str, Any]:
    """Read a MOLT checkpoint payload (weights-only first, full unpickling as a fallback)."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:  # noqa: BLE001 - metadata with custom objects needs full unpickling
        payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "molt-checkpoint":
        raise ValueError(f"{path} is not a MOLT checkpoint")
    return payload
