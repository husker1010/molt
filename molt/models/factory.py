"""Model construction by name and checkpoint loading for all MOLT network classes."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from molt.models.common import load_checkpoint_payload as _load_payload
from molt.models.molt import MOLT, MOLTConfig

__all__ = ["MODEL_REGISTRY", "build_model", "load_checkpoint"]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "molt": MOLT,
}
"""Model names accepted by :func:`build_model`."""

_CLASS_NAMES = {cls.__name__: cls for cls in MODEL_REGISTRY.values()}


def build_model(name: str, config: MOLTConfig, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model (``"molt"``)."""
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc
    return cls(config, **kwargs)


def load_checkpoint(
    path: str | os.PathLike[str], map_location: str | torch.device = "cpu", strict: bool = True
) -> nn.Module:
    """Load any model saved with ``save_checkpoint``, dispatching on the stored class name."""
    payload = _load_payload(path, map_location)
    name = payload.get("model_class", "MOLT")
    if name not in _CLASS_NAMES:
        raise ValueError(f"Checkpoint {path} stores unknown model class {name!r}")
    return _CLASS_NAMES[name].from_checkpoint(path, map_location=map_location, strict=strict)
