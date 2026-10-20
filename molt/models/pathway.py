"""Pathway graph attention encoder for tumour transcriptomes.

The transcriptome enters as ``P`` pathway activities ``p = D^{-1} A^T x`` (``A`` the gene-to-pathway
incidence, ``D`` the number of retained genes per pathway, ``x`` standardised gene expression).
Each pathway is a graph node whose input is its activity, a missing-activity flag and a learned
node-identity embedding. Multi-head graph attention layers (GATv2-style scoring) exchange
information between pathways:

* ``graph_mode="prior"`` restricts attention to the edges of a fixed adjacency prior
  (self-loops are always added);
* ``graph_mode="dense"`` lets every node attend to every other node and reports the attention mass
  placed outside the prior, which the training objective can penalise;
* ``graph_mode="mlp"`` is a graph-free multilayer perceptron over the activity vector.

Attention pooling over the nodes yields the transcriptomic embedding ``h_rna``. With
``activity_skip`` (the default) the output of the graph-free network (a two-layer perceptron
over the activity vector and its missingness flags, followed by layer normalisation) is added
to the pooled node representation, so the graph layers refine a direct encoding of the
activities. Without this path the pooled embedding varies little between samples at
initialisation (node tokens differ mainly through their identity embeddings, which are learned
slowly); on real cohorts the graph encoder then stays on a plateau where it carries almost no
sample-specific information while faster experts (clinical variables) fit the training data.
A gene-level decoder predicts standardised gene expression from ``h_rna`` and provides a
masked-gene prediction (and denoising) objective for self-supervised training.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from molt.models.common import AttentionPool, make_mlp, masked_softmax, sanitize

__all__ = ["GraphMode", "PathwayEncoder", "PathwayGraphAttention", "masked_mse"]

GraphMode = Literal["prior", "dense", "mlp"]


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error over entries where ``mask`` is true and ``target`` is finite.

    Returns a zero that stays attached to the graph when no entry is valid.
    """
    valid = torch.isfinite(target)
    if mask is not None:
        valid = valid & mask.to(torch.bool)
    if not bool(valid.any()):
        return pred.sum() * 0.0
    diff = pred[valid] - target[valid]
    return diff.pow(2).mean()


class PathwayGraphAttention(nn.Module):
    """One multi-head graph attention layer over a dense node set.

    Scores follow GATv2, ``e_pq = a^T LeakyReLU(W_d h_p + W_s h_q)``, normalised over the
    neighbours ``q`` of node ``p`` given by ``mask``. Messages ``W_s h_q`` are aggregated with the
    attention weights, passed through ELU and added residually.

    Args:
        dim: Node feature dimension (divisible by ``n_heads``).
        n_heads: Number of attention heads.
        dropout: Residual dropout.
        attn_dropout: Dropout on attention coefficients.
        negative_slope: LeakyReLU slope used in the scoring function.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.lin_src = nn.Linear(dim, dim, bias=False)
        self.lin_dst = nn.Linear(dim, dim, bias=False)
        self.att = nn.Parameter(torch.randn(n_heads, self.head_dim) * (self.head_dim**-0.5))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.negative_slope = negative_slope
        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """Update node features.

        Args:
            h: ``(B, P, dim)`` node features.
            mask: ``(P, P)`` boolean neighbourhood (row = receiving node), or ``None`` for dense.

        Returns:
            Updated features ``(B, P, dim)`` and attention coefficients ``(B, H, P, P)``.
        """
        b, p, _ = h.shape
        src = self.lin_src(h).view(b, p, self.n_heads, self.head_dim)
        dst = self.lin_dst(h).view(b, p, self.n_heads, self.head_dim)
        pair = F.leaky_relu(dst.unsqueeze(2) + src.unsqueeze(1), self.negative_slope)
        scores = torch.einsum("bpqhd,hd->bhpq", pair, self.att)
        alpha = masked_softmax(scores, None if mask is None else mask.view(1, 1, p, p), dim=-1)
        msg = torch.einsum("bhpq,bqhd->bphd", self.attn_drop(alpha), src).reshape(b, p, -1)
        out = self.norm(h + self.drop(F.elu(msg + self.bias)))
        return out, alpha


class PathwayEncoder(nn.Module):
    """Encode pathway activities into a transcriptomic embedding ``h_rna``.

    Args:
        n_pathways: Number of pathway nodes ``P``.
        adjacency: ``(P, P)`` 0/1 adjacency prior; self-loops are added. ``None`` means a fully
            connected prior.
        dim: Node and output dimension.
        n_layers: Number of graph attention layers.
        n_heads: Attention heads per layer.
        node_id_dim: Size of the learned node-identity embedding.
        graph_mode: ``"prior"``, ``"dense"`` or ``"mlp"`` (see module docstring).
        dropout: Residual dropout.
        attn_dropout: Attention-coefficient dropout.
        pool_hidden: Width of the node attention-pooling network.
        n_genes: Number of genes predicted by the gene decoder (0 disables it).
        incidence: Optional ``(G, P)`` gene-to-pathway incidence used to recompute pathway
            activities from masked gene profiles for the masked-gene objective.
        gene_mask_ratio: Fraction of genes hidden in the masked-gene objective.
        gene_noise_std: Standard deviation of Gaussian noise added to visible genes (denoising).
        denoise_weight: Weight of the prediction error on the visible genes relative to the
            masked genes.
        activity_skip: Add the graph-free encoding of ``[activities, missing flags]`` to the
            pooled graph embedding (``prior`` and ``dense`` modes).
    """

    def __init__(
        self,
        n_pathways: int,
        adjacency: np.ndarray | Tensor | None = None,
        dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        node_id_dim: int = 16,
        graph_mode: GraphMode = "prior",
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pool_hidden: int = 64,
        n_genes: int = 0,
        incidence: np.ndarray | Tensor | None = None,
        gene_mask_ratio: float = 0.15,
        gene_noise_std: float = 0.1,
        denoise_weight: float = 0.5,
        activity_skip: bool = True,
    ) -> None:
        super().__init__()
        if graph_mode not in ("prior", "dense", "mlp"):
            raise ValueError(f"Unknown graph_mode {graph_mode!r}")
        self.n_pathways = int(n_pathways)
        self.dim = int(dim)
        self.graph_mode: GraphMode = graph_mode
        prior = self._prepare_adjacency(adjacency, self.n_pathways)
        self.register_buffer("prior_mask", prior)

        self.node_id = nn.Parameter(torch.randn(self.n_pathways, node_id_dim) * 0.1)
        if graph_mode == "mlp":
            self.mlp = make_mlp(2 * self.n_pathways, [2 * dim, dim], dim, dropout=dropout)
            self.out_norm = nn.LayerNorm(dim)
        else:
            self.node_in = nn.Sequential(
                nn.Linear(2 + node_id_dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
            )
            self.layers = nn.ModuleList(
                PathwayGraphAttention(dim, n_heads, dropout=dropout, attn_dropout=attn_dropout)
                for _ in range(n_layers)
            )
            self.pool = AttentionPool(dim, hidden=pool_hidden)
        self.skip: nn.Module | None = None
        if activity_skip and graph_mode != "mlp":
            self.skip = nn.Sequential(
                make_mlp(2 * self.n_pathways, [2 * dim, dim], dim, dropout=dropout),
                nn.LayerNorm(dim),
            )

        self.n_genes = int(n_genes)
        self.decoder = make_mlp(dim, [4 * dim], self.n_genes, dropout=dropout) if n_genes else None
        if incidence is not None:
            inc = torch.as_tensor(np.asarray(incidence, dtype=np.float32))
            if inc.shape != (self.n_genes, self.n_pathways):
                raise ValueError(
                    f"incidence must have shape (n_genes, n_pathways)=({self.n_genes}, "
                    f"{self.n_pathways}), got {tuple(inc.shape)}"
                )
            self.register_buffer("incidence", (inc > 0).float())
        else:
            self.register_buffer("incidence", None)
        self.gene_mask_ratio = float(gene_mask_ratio)
        self.gene_noise_std = float(gene_noise_std)
        self.denoise_weight = float(denoise_weight)

    @staticmethod
    def _prepare_adjacency(adjacency: np.ndarray | Tensor | None, n: int) -> Tensor:
        if adjacency is None:
            return torch.ones(n, n, dtype=torch.bool)
        adj = torch.as_tensor(np.asarray(adjacency, dtype=np.float32))
        if adj.shape != (n, n):
            raise ValueError(f"adjacency must have shape ({n}, {n}), got {tuple(adj.shape)}")
        adj = (adj > 0) | (adj.T > 0)
        return adj | torch.eye(n, dtype=torch.bool)

    @property
    def out_dim(self) -> int:
        """Dimension of the transcriptomic embedding."""
        return self.dim

    def forward(self, activities: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode pathway activities.

        Args:
            activities: ``(B, P)`` pathway activities; NaN marks an unavailable pathway.

        Returns:
            ``h_rna`` ``(B, dim)``, node attention-pooling weights ``(B, P)`` and the mean
            attention mass outside the adjacency prior (a scalar, zero unless ``graph_mode`` is
            ``"dense"``).
        """
        x = activities.to(self.node_id.dtype)
        missing = ~torch.isfinite(x)
        x = sanitize(x)
        b = x.shape[0]
        penalty = x.new_zeros(())
        if self.graph_mode == "mlp":
            flat = torch.cat([x, missing.to(x.dtype)], dim=1)
            h = self.out_norm(self.mlp(flat))
            uniform = torch.full((b, self.n_pathways), 1.0 / self.n_pathways, device=x.device)
            return h, uniform, penalty
        node_feat = torch.cat(
            [
                x.unsqueeze(-1),
                missing.to(x.dtype).unsqueeze(-1),
                self.node_id.unsqueeze(0).expand(b, -1, -1),
            ],
            dim=-1,
        )
        h = self.node_in(node_feat)
        mask = self.prior_mask if self.graph_mode == "prior" else None
        outside = (~self.prior_mask).to(h.dtype)
        penalties = []
        for layer in self.layers:
            h, alpha = layer(h, mask)
            if self.graph_mode == "dense":
                penalties.append((alpha * outside).sum(-1).mean())
        if penalties:
            penalty = torch.stack(penalties).mean()
        pooled, weights = self.pool(h)
        if self.skip is not None:
            pooled = pooled + self.skip(torch.cat([x, missing.to(x.dtype)], dim=1))
        return pooled, weights, penalty

    def activities_from_genes(self, genes: Tensor, keep: Tensor | None = None) -> Tensor:
        """Compute pathway activities ``D^{-1} A^T x`` from (partially hidden) gene profiles.

        Args:
            genes: ``(B, G)`` standardised expression; NaN entries are treated as unavailable.
            keep: Optional ``(B, G)`` boolean mask of visible genes.

        Returns:
            ``(B, P)`` activities, NaN for pathways without any visible gene.
        """
        if self.incidence is None:
            raise RuntimeError("activities_from_genes requires an incidence matrix")
        visible = torch.isfinite(genes)
        if keep is not None:
            visible = visible & keep
        vis = visible.to(self.incidence.dtype)
        values = sanitize(genes).to(self.incidence.dtype) * vis
        total = values @ self.incidence
        count = vis @ self.incidence
        act = total / count.clamp_min(1.0)
        return torch.where(count > 0, act, torch.full_like(act, float("nan")))

    def decode(self, h: Tensor) -> Tensor:
        """Predict standardised gene expression ``(B, G)`` from ``h_rna``."""
        if self.decoder is None:
            raise RuntimeError("The gene decoder is disabled (n_genes=0)")
        return self.decoder(h)

    def masked_gene_loss(
        self, genes: Tensor, activities: Tensor | None = None, h: Tensor | None = None
    ) -> Tensor:
        """Masked gene-level prediction loss.

        With an incidence matrix, a random ``gene_mask_ratio`` of the genes is hidden, the
        visible genes are corrupted with Gaussian noise, pathway activities are recomputed from
        the corrupted profile and encoded, and the decoder must predict the clean profile; the
        loss is the MSE on hidden genes plus ``denoise_weight`` times the MSE on visible genes.
        Without an incidence matrix the decoder predicts all observed genes from the
        embedding of the supplied ``activities`` (or of the precomputed ``h``).

        Args:
            genes: ``(B, G)`` standardised expression target (NaN = not measured).
            activities: ``(B, P)`` activities used when no incidence matrix is available.
            h: Optional precomputed embedding of ``activities``.

        Returns:
            Scalar loss.
        """
        if self.decoder is None:
            raise RuntimeError("The gene decoder is disabled (n_genes=0)")
        genes = genes.to(self.node_id.dtype)
        if self.incidence is not None:
            hidden = torch.rand(genes.shape, device=genes.device) < self.gene_mask_ratio
            noisy = genes + self.gene_noise_std * torch.randn_like(genes)
            corrupted = self.activities_from_genes(noisy, keep=~hidden)
            h_masked, _, _ = self(corrupted)
            pred = self.decode(h_masked)
            loss = masked_mse(pred, genes, hidden)
            if self.denoise_weight > 0:
                loss = loss + self.denoise_weight * masked_mse(pred, genes, ~hidden)
            return loss
        if h is None:
            if activities is None:
                raise ValueError("activities or h is required without an incidence matrix")
            h, _, _ = self(activities)
        return masked_mse(self.decode(h), genes)
