"""Monte Carlo dropout inference helpers for MOLT-compatible networks."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["mc_dropout", "mc_dropout_predict", "model_device"]


@contextlib.contextmanager
def mc_dropout(model: nn.Module) -> Iterator[nn.Module]:
    """Evaluate ``model`` with dropout layers active (Monte Carlo dropout).

    All modules are put in evaluation mode except ``nn.Dropout`` variants, which stay
    stochastic; the original training flags are restored on exit.
    """
    flags = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        yield model
    finally:
        for module, flag in flags.items():
            module.train(flag)


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    batch: Mapping[str, Any],
    n_samples: int = 20,
    *,
    omit_modalities: Sequence[str] = (),
) -> dict[str, Tensor]:
    """Monte Carlo dropout predictions.

    CT backbone features are computed once and reused across samples.

    Args:
        model: A :class:`MOLT` (or compatible) model.
        batch: Input batch.
        n_samples: Number of stochastic forward passes.
        omit_modalities: Modalities omitted for all samples.

    Returns:
        ``p_egfr_mean``/``p_egfr_std`` ``(B,)``, ``p_egfr_samples`` ``(S, B)``,
        ``survival_mean``/``survival_std`` ``(B, K)`` and ``survival_samples`` ``(S, B, K)``.
    """
    batch = dict(batch)
    imaging = getattr(model, "imaging", None)
    if (
        imaging is not None
        and batch.get("ct_patches") is not None
        and batch.get("ct_features") is None
    ):
        was_training = imaging.training
        imaging.eval()
        batch["ct_features"] = imaging.extract_features(batch["ct_patches"].to(model_device(model)))
        imaging.train(was_training)
    probs, survs = [], []
    with mc_dropout(model):
        for _ in range(int(n_samples)):
            out = model(batch, omit_modalities=omit_modalities)
            probs.append(out["p_egfr"].float())
            survs.append(out["survival"].float())
    p = torch.stack(probs)
    s = torch.stack(survs)
    return {
        "p_egfr_mean": p.mean(0),
        "p_egfr_std": p.std(0, unbiased=False),
        "p_egfr_samples": p,
        "survival_mean": s.mean(0),
        "survival_std": s.std(0, unbiased=False),
        "survival_samples": s,
    }


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter of ``model``."""
    return next(model.parameters()).device
