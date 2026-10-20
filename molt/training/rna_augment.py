"""Label-free, train-time perturbation of the transcriptomic inputs (shift-robust training).

Locked models meet expression profiles from other platforms and laboratories: after
per-cohort gene-wise standardization the remaining differences are gene-specific location and
scale residues, genes that are absent or unreliable on the target platform, and a partial
re-ordering of genes within a profile. :class:`RNAAugmenter` simulates these shifts on the
standardized gene inputs of the training participants of every mini-batch:

1. *mild rank corruption* - each profile is re-ordered by its values plus Gaussian noise
   (``rank_sd`` in gene SD units) and then receives its own sorted values in that order, so
   the within-sample distribution is unchanged and only the gene order is perturbed;
2. *per-gene random affine jitter* - ``x_g <- exp(scale_sd e1_g) x_g + shift_sd e2_g``;
3. *random gene masking* - a fraction ``f ~ U(mask_min, mask_max)`` of the genes of a profile
   is hidden (treated as not measured).

The pathway activities of the pathway expert are recomputed from the perturbed genes with the
fold's incidence matrix (mean over the observed genes of every pathway) and standardized with
the fold's activity moments; the gene panel of the gene-level expert is perturbed the same way
(independent draws). The stacked linear scores that take these blocks as inputs are shifted
by the linear effect of the perturbation, ``delta s = (Z(x') - Z(x))' beta / sd(s)``, with the
fold's refitted coefficients (``Z`` = the scorer's training standardization, missing = mean).
The masked-gene targets of the transcriptomic decoder keep the unperturbed genes (a denoising
objective). Validation, test
and external participants are never perturbed.

Configuration (``data.rna.augment``)::

    augment: {enabled: true, p: 1.0, scale_sd: 0.1, shift_sd: 0.1,
              mask_min: 0.05, mask_max: 0.20, rank_sd: 0.1}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = ["RNAAugmentConfig", "RNAAugmenter"]


@dataclass(frozen=True)
class RNAAugmentConfig:
    """Settings of :class:`RNAAugmenter` (see the module docstring)."""

    enabled: bool = False
    p: float = 1.0
    scale_sd: float = 0.1
    shift_sd: float = 0.1
    mask_min: float = 0.05
    mask_max: float = 0.20
    rank_sd: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RNAAugmentConfig:
        """Build from a mapping, ignoring unknown keys and ``None`` values."""
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in dict(data or {}).items() if k in names and v is not None}
        cfg = cls(**values)
        if not 0.0 <= cfg.mask_min <= cfg.mask_max < 1.0:
            raise ValueError("data.rna.augment needs 0 <= mask_min <= mask_max < 1")
        return cfg


def _tensor(values: np.ndarray | None) -> Tensor | None:
    return None if values is None else torch.as_tensor(np.asarray(values, np.float32))


@dataclass
class _StackedMap:
    key: str
    block: str
    column: int
    coef: Tensor
    center: Tensor
    scale: Tensor
    score_sd: float


class RNAAugmenter:
    """Perturb the transcriptomic inputs of training mini-batches.

    Args:
        config: Augmentation settings.
        genes: ``(n, G)`` standardized pathway genes of the dataset rows (NaN = missing).
        incidence: ``(G, P)`` gene-pathway incidence.
        pathway_mean, pathway_sd: Activity standardization (``None`` = unscaled activities).
        panel: ``(n, Gp)`` gene panel of the gene-level expert (``None`` when disabled).
        stacked: Linear maps of the stacked scorers with transcriptomic input blocks.
    """

    def __init__(
        self,
        config: RNAAugmentConfig,
        genes: np.ndarray | None,
        incidence: np.ndarray | None,
        pathway_mean: np.ndarray | None,
        pathway_sd: np.ndarray | None,
        panel: np.ndarray | None,
        stacked: list[_StackedMap] | None = None,
    ) -> None:
        self.config = config
        t = _tensor
        self.genes = t(genes)
        self.incidence = t(incidence)
        self.pathway_mean, self.pathway_sd = t(pathway_mean), t(pathway_sd)
        self.panel_available = panel is not None
        self.stacked = list(stacked or [])

    @classmethod
    def build(cls, config: RNAAugmentConfig, pre: Any, feats: Any) -> RNAAugmenter | None:
        """Augmenter for the rows of ``feats`` using the fitted preprocessor ``pre``."""
        if not config.enabled or not np.any(feats.rna_present):
            return None
        genes = feats.rna_genes if pre.n_genes and pre.incidence is not None else None
        incidence = pre.incidence if genes is not None else None
        maps: list[_StackedMap] = []
        stacking = getattr(pre, "stacking", None)
        if stacking is not None:
            blocks = stacking.options.get("components") == "blocks"
            for name, fitted in stacking.fitted.items():
                start = 0
                for b, (block, width) in enumerate(zip(fitted.spec.inputs, fitted.widths)):
                    sl = slice(start, start + width)
                    start += width
                    if block not in ("genes", "pathways"):
                        continue
                    if block == "pathways" and genes is None:
                        continue
                    column = b if blocks else 0
                    maps.append(
                        _StackedMap(
                            key=f"stacked_{name}",
                            block=block,
                            column=column,
                            coef=torch.as_tensor(fitted.fit.coef[sl], dtype=torch.float32),
                            center=torch.as_tensor(fitted.fit.center[sl], dtype=torch.float32),
                            scale=torch.as_tensor(fitted.fit.scale[sl], dtype=torch.float32),
                            score_sd=float(fitted.score_sd[column]),
                        )
                    )
        return cls(
            config,
            genes,
            incidence,
            pre.pathway_mean if genes is not None else None,
            pre.pathway_sd if genes is not None else None,
            getattr(feats, "gene_panel", None),
            maps,
        )

    # ------------------------------------------------------------------ perturbation
    def perturb(self, x: Tensor) -> Tensor:
        """Rank corruption, affine jitter and masking of standardized profiles ``(n, G)``."""
        c = self.config
        finite = torch.isfinite(x)
        out = x
        if c.rank_sd > 0:
            filled = torch.where(finite, x, torch.full_like(x, float("inf")))
            noisy = torch.where(finite, x + c.rank_sd * torch.randn_like(x), filled)
            ranks = torch.argsort(torch.argsort(noisy, dim=1), dim=1)
            out = torch.gather(torch.sort(filled, dim=1).values, 1, ranks)
            out = torch.where(finite, out, x)
        if c.scale_sd > 0 or c.shift_sd > 0:
            scale = torch.exp(c.scale_sd * torch.randn_like(out))
            out = scale * out + c.shift_sd * torch.randn_like(out)
        if c.mask_max > 0:
            frac = c.mask_min + (c.mask_max - c.mask_min) * torch.rand(out.shape[0], 1)
            out = torch.where(torch.rand_like(out) < frac, torch.full_like(out, float("nan")), out)
        return torch.where(finite, out, torch.full_like(out, float("nan")))

    def activities(self, genes: Tensor) -> Tensor:
        """Standardized pathway activities of standardized genes (missing pathways = 0)."""
        mask = torch.isfinite(genes)
        filled = torch.where(mask, genes, torch.zeros_like(genes))
        counts = mask.float() @ self.incidence
        act = (filled @ self.incidence) / counts.clamp(min=1.0)
        if self.pathway_mean is not None:
            act = (act - self.pathway_mean) / self.pathway_sd
        return torch.where(counts > 0, act, torch.zeros_like(act))

    def _shift_scores(self, batch: dict[str, Any], rows: Tensor, block: str, old, new) -> None:
        for m in self.stacked:
            if m.block != block or batch.get(m.key) is None:
                continue
            z_old = torch.nan_to_num((old - m.center) / m.scale, nan=0.0, posinf=0.0, neginf=0.0)
            z_new = torch.nan_to_num((new - m.center) / m.scale, nan=0.0, posinf=0.0, neginf=0.0)
            delta = ((z_new - z_old) @ m.coef) / m.score_sd
            scores = batch[m.key]
            scores[rows, m.column] = scores[rows, m.column] + delta

    def __call__(self, batch: dict[str, Any], index: Tensor) -> dict[str, Any]:
        """Perturb the batch in place (rows with an expression profile, probability ``p``)."""
        present = batch.get("rna_present")
        if present is None or not bool(present.any()):
            return batch
        chosen = present & (torch.rand(present.shape[0]) < self.config.p)
        rows = torch.nonzero(chosen).squeeze(1)
        if rows.numel() == 0:
            return batch
        if self.genes is not None and batch.get("rna") is not None:
            genes = self.genes.index_select(0, index.index_select(0, rows))
            old, new = batch["rna"][rows].clone(), self.activities(self.perturb(genes))
            batch["rna"][rows] = new
            self._shift_scores(batch, rows, "pathways", old, new)
        if self.panel_available and batch.get("rna_panel") is not None:
            old = batch["rna_panel"][rows].clone()
            new = self.perturb(old)
            batch["rna_panel"][rows] = new
            self._shift_scores(batch, rows, "genes", old, new)
        return batch
