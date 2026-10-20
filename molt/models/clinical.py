"""Gated tabular transformer encoder for structured clinical variables.

Each patient is represented as a sequence of tokens:

* one token per continuous variable, ``m_f * (x_f w_f + b_f) + (1 - m_f) * e_f^miss`` plus a
  column embedding, where ``m_f`` is the observed-mask gate;
* one token per categorical variable from a per-variable embedding table whose index 0 is the
  missing token and index 1 the out-of-vocabulary (OOV) token;
* optional missingness-indicator tokens (one per indicator, a learned embedding for each of the
  two indicator states) for the continuous and categorical variables and for any extra
  study-level indicators such as systematic non-collection flags;
* a learned ``[CLS]`` summary token whose final state is the clinical embedding.

Tokens pass through pre-norm self-attention layers with gated (GEGLU) feed-forward blocks.
During training, stochastic feature masking replaces observed categories by the missing or the
OOV token with equal probability so that both reserved embeddings are trained.

Partially specified categories (see :mod:`molt.training.clinical_hierarchy`) are handled in
one of two ways: with a *coarsening table* an observed fine level is replaced during training,
with probability ``coarsen_p``, by one of its coarse ancestors (and the continuous variables
the coarse record does not carry are hidden), so the coarse embedding is trained on the
mixture of its children; with a *mixing matrix* the embedding of a coarse level is the fixed
training-frequency-weighted mean of the embeddings of its compatible fine levels.

With ``linear_skip`` a linear map of the flat record (standardized continuous values with their
missingness flags, one-hot categories including the reserved missing/OOV indices, and the extra
indicators) followed by layer normalisation is added to the ``[CLS]`` embedding, so that
effects that are close to additive are available from the first epoch while the attention
layers model interactions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from molt.models.common import GEGLUFeedForward, sanitize

__all__ = ["MISSING_INDEX", "OOV_INDEX", "ClinicalEncoder", "TabularTransformerLayer"]

MISSING_INDEX = 0
"""Categorical vocabulary index reserved for a missing value."""
OOV_INDEX = 1
"""Categorical vocabulary index reserved for an out-of-vocabulary value."""


class TabularTransformerLayer(nn.Module):
    """Pre-norm transformer layer: multi-head self-attention followed by a GEGLU block.

    Args:
        dim: Token dimension.
        n_heads: Number of attention heads.
        dropout: Residual dropout.
        attn_dropout: Attention-weight dropout.
        ff_mult: Hidden expansion of the feed-forward block.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ff_mult: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = GEGLUFeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: Tensor, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        """Transform tokens ``x`` of shape ``(B, T, dim)``.

        Returns:
            The updated tokens and, if ``need_weights``, head-averaged attention ``(B, T, T)``.
        """
        h = self.norm1(x)
        attn_out, weights = self.attn(h, h, h, need_weights=need_weights)
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm2(x))
        return x, weights


class ClinicalEncoder(nn.Module):
    """Encode continuous and categorical clinical variables into a single embedding.

    Args:
        n_num: Number of continuous variables ``Fn``.
        cat_cardinalities: Vocabulary size of each categorical variable, including the reserved
            missing (0) and OOV (1) indices.
        dim: Token and output dimension.
        n_layers: Number of transformer layers.
        n_heads: Attention heads per layer.
        dropout: Residual and feed-forward dropout.
        attn_dropout: Attention-weight dropout.
        n_extra_indicators: Number of additional binary indicators supplied per patient
            (e.g. systematic non-collection flags); each becomes one token.
        indicator_tokens: Add one missingness-indicator token per continuous and categorical
            variable.
        cat_mask_p: Probability of masking each observed category during training.
        num_mask_p: Probability of masking each observed continuous value during training.
        linear_skip: Add the linear encoding of the flat record (see the module docstring).
        cat_mixing: Optional row-stochastic ``(total, total)`` map over the concatenated
            category indices; row ``i`` gives the embedding weights of index ``i`` (identity
            except for mixed coarse levels).
        coarsen_table: Optional ``(total, A)`` coarse ancestors of every flat category index
            (``-1`` padded) for stochastic coarsening during training.
        coarsen_hide: Optional ``(total, Fn)`` continuous variables hidden when a value is
            coarsened to the flat index of the row.
        coarsen_p: Coarsening probability per observed category with ancestors.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: Sequence[int],
        dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        n_extra_indicators: int = 0,
        indicator_tokens: bool = True,
        cat_mask_p: float = 0.10,
        num_mask_p: float = 0.0,
        linear_skip: bool = False,
        cat_mixing: np.ndarray | None = None,
        coarsen_table: np.ndarray | None = None,
        coarsen_hide: np.ndarray | None = None,
        coarsen_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_num = int(n_num)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        if any(c < 2 for c in self.cat_cardinalities):
            raise ValueError("Every categorical cardinality must include the 2 reserved indices")
        self.n_cat = len(self.cat_cardinalities)
        self.n_extra_indicators = int(n_extra_indicators)
        self.indicator_tokens = bool(indicator_tokens)
        self.cat_mask_p = float(cat_mask_p)
        self.num_mask_p = float(num_mask_p)
        self.dim = int(dim)

        n_ind = (self.n_num + self.n_cat if self.indicator_tokens else 0) + self.n_extra_indicators
        self.n_indicators = n_ind
        self.n_tokens = self.n_num + self.n_cat + n_ind

        # Continuous variables: per-feature affine value token, missing embedding, column embedding.
        self.num_weight = nn.Parameter(torch.randn(self.n_num, dim) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(self.n_num, dim))
        self.num_missing = nn.Parameter(torch.randn(self.n_num, dim) * 0.02)
        # Categorical variables: one embedding table per feature (single flat table + offsets).
        offsets = [0]
        for card in self.cat_cardinalities[:-1]:
            offsets.append(offsets[-1] + card)
        self.register_buffer(
            "cat_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "cat_cards",
            torch.tensor(self.cat_cardinalities or [2], dtype=torch.long),
            persistent=False,
        )
        total_cat = max(1, sum(self.cat_cardinalities))
        self.cat_mixing: Tensor | None
        self.coarsen_table: Tensor | None
        self.coarsen_hide: Tensor | None
        mixing = (
            None
            if cat_mixing is None
            else torch.as_tensor(np.asarray(cat_mixing), dtype=torch.float32)
        )
        if mixing is not None and mixing.shape != (total_cat, total_cat):
            raise ValueError(f"cat_mixing must be ({total_cat}, {total_cat})")
        self.register_buffer("cat_mixing", mixing, persistent=False)
        table = (
            None
            if coarsen_table is None
            else torch.as_tensor(np.asarray(coarsen_table), dtype=torch.long)
        )
        if table is not None and (table.ndim != 2 or table.shape[0] != total_cat):
            raise ValueError(f"coarsen_table must have {total_cat} rows")
        self.register_buffer("coarsen_table", table, persistent=False)
        hide = None
        if table is not None:
            hide = torch.zeros(total_cat, max(self.n_num, 1), dtype=torch.bool)
            if coarsen_hide is not None:
                given = torch.as_tensor(np.asarray(coarsen_hide), dtype=torch.bool)
                hide[: given.shape[0], : given.shape[1]] = given
        self.register_buffer("coarsen_hide", hide, persistent=False)
        self.coarsen_p = float(coarsen_p) if table is not None else 0.0
        self.cat_embedding = nn.Embedding(total_cat, dim)
        nn.init.normal_(self.cat_embedding.weight, std=0.02)
        # Indicator tokens: a learned embedding for each of the two states of every indicator.
        self.indicator_embedding = nn.Parameter(torch.randn(max(n_ind, 1), 2, dim) * 0.02)
        # Column (feature-identity) embeddings for all non-CLS tokens.
        self.column_embedding = nn.Parameter(torch.randn(max(self.n_tokens, 1), dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.input_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TabularTransformerLayer(dim, n_heads, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.skip: nn.Module | None = None
        if linear_skip:
            flat = 2 * self.n_num + total_cat + self.n_extra_indicators
            self.skip = nn.Sequential(nn.Linear(max(flat, 1), dim), nn.LayerNorm(dim))

    @property
    def out_dim(self) -> int:
        """Dimension of the clinical embedding."""
        return self.dim

    def mask_features(
        self, num_mask: Tensor, cat: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Apply stochastic feature masking.

        Each observed category (index >= 2) is replaced, with probability ``cat_mask_p``, by the
        missing or the OOV index with equal probability; each observed continuous value is
        hidden with probability ``num_mask_p``.

        Args:
            num_mask: ``(B, Fn)`` boolean observed mask.
            cat: ``(B, Fc)`` category indices.
            generator: Optional random generator.

        Returns:
            The masked observed mask and category indices.
        """
        if self.n_cat and self.coarsen_p > 0 and self.coarsen_table is not None:
            num_mask, cat = self.coarsen(num_mask, cat, generator)
        if self.n_cat and self.cat_mask_p > 0:
            observed = cat > OOV_INDEX
            hit = torch.rand(cat.shape, device=cat.device, generator=generator) < self.cat_mask_p
            to_oov = torch.rand(cat.shape, device=cat.device, generator=generator) < 0.5
            replacement = torch.where(
                to_oov, torch.full_like(cat, OOV_INDEX), torch.full_like(cat, MISSING_INDEX)
            )
            cat = torch.where(observed & hit, replacement, cat)
        if self.n_num and self.num_mask_p > 0:
            hit = torch.rand(num_mask.shape, device=num_mask.device, generator=generator)
            num_mask = num_mask & ~(hit < self.num_mask_p)
        return num_mask, cat

    def coarsen(
        self, num_mask: Tensor, cat: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Stochastic category coarsening (training only; see the module docstring).

        Each category with declared ancestors is replaced, with probability ``coarsen_p``, by
        one of its ancestors chosen uniformly; continuous variables declared for that ancestor
        are hidden.
        """
        flat = cat + self.cat_offsets
        anc = self.coarsen_table[flat]
        n_anc = (anc >= 0).sum(-1)
        u = torch.rand(cat.shape, device=cat.device, generator=generator)
        hit = (u < self.coarsen_p) & (n_anc > 0)
        pick = torch.rand(cat.shape, device=cat.device, generator=generator)
        choice = torch.clamp((pick * n_anc).long(), max=anc.shape[-1] - 1)
        choice = torch.minimum(choice, torch.clamp(n_anc - 1, min=0))
        coarse = anc.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        new_flat = torch.where(hit, coarse, flat)
        if self.n_num and self.coarsen_hide is not None:
            hide = (self.coarsen_hide[new_flat] & hit.unsqueeze(-1)).any(1)[:, : self.n_num]
            num_mask = num_mask & ~hide
        return num_mask, new_flat - self.cat_offsets

    def _category_tokens(self, cat: Tensor) -> Tensor:
        """Embeddings ``(B, Fc, dim)``; mixed coarse levels use their fixed mixture."""
        flat = cat + self.cat_offsets
        if self.cat_mixing is None:
            return self.cat_embedding(flat)
        weights = self.cat_mixing[flat].to(self.cat_embedding.weight.dtype)
        return weights @ self.cat_embedding.weight

    def forward(
        self,
        num: Tensor,
        num_mask: Tensor,
        cat: Tensor,
        extra_indicators: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Encode a batch of clinical records.

        Args:
            num: ``(B, Fn)`` standardised continuous values (NaN allowed where unobserved).
            num_mask: ``(B, Fn)`` boolean, ``True`` where the value is observed.
            cat: ``(B, Fc)`` vocabulary indices (0 = missing, 1 = OOV). Indices outside the
                vocabulary map to OOV and negative indices to missing.
            extra_indicators: Optional ``(B, n_extra_indicators)`` binary indicators.
            return_attention: Also return the last layer's ``[CLS]`` attention over the tokens.

        Returns:
            Clinical embedding ``(B, dim)`` and optionally attention ``(B, n_tokens)``.
        """
        batch = num.shape[0] if self.n_num else cat.shape[0]
        device = self.cls_token.device
        num = num.to(device=device, dtype=self.num_weight.dtype)
        num_mask = num_mask.to(device=device, dtype=torch.bool) & torch.isfinite(num)
        num = sanitize(num)
        cat = cat.to(device=device, dtype=torch.long)
        if self.n_cat:
            cat = torch.where(cat < 0, torch.full_like(cat, MISSING_INDEX), cat)
            cat = torch.where(cat >= self.cat_cards, torch.full_like(cat, OOV_INDEX), cat)
        if self.training:
            num_mask, cat = self.mask_features(num_mask, cat)

        tokens: list[Tensor] = []
        if self.n_num:
            gate = num_mask.unsqueeze(-1).to(num.dtype)
            value_tok = num.unsqueeze(-1) * self.num_weight + self.num_bias
            tokens.append(gate * value_tok + (1.0 - gate) * self.num_missing)
        if self.n_cat:
            tokens.append(self._category_tokens(cat))
        indicators: list[Tensor] = []
        if self.indicator_tokens:
            if self.n_num:
                indicators.append((~num_mask).long())
            if self.n_cat:
                indicators.append((cat == MISSING_INDEX).long())
        if self.n_extra_indicators:
            if extra_indicators is None:
                extra = torch.zeros(batch, self.n_extra_indicators, dtype=torch.long, device=device)
            else:
                extra = (sanitize(extra_indicators.to(device=device).float()) > 0.5).long()
            indicators.append(extra)
        if indicators:
            ind = torch.cat(indicators, dim=1)
            idx = torch.arange(ind.shape[1], device=device)
            tokens.append(self.indicator_embedding[idx.unsqueeze(0), ind])
        if tokens:
            x = torch.cat(tokens, dim=1) + self.column_embedding[: self.n_tokens]
            x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)
        else:
            x = self.cls_token.expand(batch, -1, -1)
        x = self.input_drop(x)
        attn = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            x, weights = layer(x, need_weights=return_attention and last)
            if weights is not None:
                attn = weights[:, 0, 1:]
        out = self.norm(x[:, 0])
        if self.skip is not None:
            out = out + self.skip(self._flat_record(num, num_mask, cat, indicators))
        return out, attn

    def _flat_record(
        self, num: Tensor, num_mask: Tensor, cat: Tensor, indicators: list[Tensor]
    ) -> Tensor:
        """Flat linear-skip input: values, missingness, one-hot categories, extra indicators."""
        parts = []
        if self.n_num:
            gate = num_mask.to(num.dtype)
            parts += [num * gate, 1.0 - gate]
        if self.n_cat:
            total = self.cat_embedding.num_embeddings
            if self.cat_mixing is not None:
                one_hot = self.cat_mixing[cat + self.cat_offsets].sum(1).to(num.dtype)
            else:
                one_hot = torch.zeros(cat.shape[0], total, device=cat.device, dtype=num.dtype)
                one_hot.scatter_(1, cat + self.cat_offsets, 1.0)
            parts.append(one_hot)
        if self.n_extra_indicators:
            parts.append(indicators[-1].to(num.dtype))
        if not parts:
            return torch.zeros(num.shape[0], 1, device=num.device, dtype=num.dtype)
        return torch.cat(parts, dim=1)
