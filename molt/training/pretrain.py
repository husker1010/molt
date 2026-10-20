"""Self-supervised initialization of the gene-level expert.

:func:`pretrain_gene_encoder` trains the hidden layer of a
:class:`~molt.models.experts.GeneEncoder` as the encoder of a denoising autoencoder on
standardized gene-panel profiles of training participants (no labels are used): a random
fraction of the genes of every profile is hidden (set to the training mean, i.e. 0), Gaussian
noise is added to the visible genes, and a linear decoder predicts the clean profile.
The fitted encoder weights stay in the module and initialise supervised training; the decoder
is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from molt.models.experts import GeneEncoder

__all__ = ["PretrainResult", "pretrain_gene_encoder"]


@dataclass
class PretrainResult:
    """Summary of a pretraining run."""

    n_profiles: int
    epochs: int
    first_loss: float
    last_loss: float


def pretrain_gene_encoder(
    encoder: GeneEncoder,
    profiles: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mask_ratio: float = 0.25,
    noise_std: float = 0.1,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> PretrainResult | None:
    """Fit the encoder layer of ``encoder`` as a denoising autoencoder on ``profiles``.

    Args:
        encoder: Gene-level expert encoder with a hidden layer (``hidden > 0``).
        profiles: ``(n, G)`` standardized panel profiles; rows without any finite value are
            ignored and NaN entries are excluded from the prediction error.
        epochs: Passes over the profiles.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        mask_ratio: Fraction of genes hidden per profile.
        noise_std: Standard deviation of the noise added to visible genes.
        batch_size: Mini-batch size.
        device: Torch device.
        seed: Random seed.

    Returns:
        A :class:`PretrainResult`, or ``None`` when the encoder has no hidden layer or there
        are fewer than two profiles.
    """
    if encoder.encoder is None or epochs <= 0:
        return None
    x = np.asarray(profiles, dtype=np.float32)
    x = x[np.isfinite(x).any(axis=1)] if x.size else x
    if len(x) < 2:
        return None
    gen = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    device = torch.device(device)
    target = torch.from_numpy(x).to(device)
    observed = torch.isfinite(target)
    target = torch.nan_to_num(target)
    net = encoder.encoder.to(device)
    decoder = nn.Linear(encoder.hidden, encoder.n_genes).to(device)
    params = list(net.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    was_training = net.training
    net.train()
    first = last = float("nan")
    n = len(target)
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=gen).to(device)
        total, count = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            clean, obs = target[idx], observed[idx]
            hidden = torch.rand(clean.shape, generator=gen).to(device) < mask_ratio
            noisy = clean + noise_std * torch.randn(clean.shape, generator=gen).to(device)
            corrupted = torch.where(hidden, torch.zeros_like(clean), noisy)
            recon = decoder(net(corrupted))
            err = (recon - clean).pow(2) * obs.float()
            loss = err.sum() / obs.float().sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        last = total / max(count, 1)
        if epoch == 0:
            first = last
    net.train(was_training)
    return PretrainResult(n_profiles=n, epochs=int(epochs), first_loss=first, last_loss=last)
