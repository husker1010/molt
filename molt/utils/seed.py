"""Reproducibility helpers: global seeding and deterministic execution flags."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["DEFAULT_SEED", "seed_everything", "make_rng", "derive_seed"]

DEFAULT_SEED = 20240613


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and optionally enable determinism.

    Args:
        seed: Seed value.
        deterministic: Request deterministic cuDNN/cuBLAS kernels. Operations without a
            deterministic implementation emit a warning instead of raising.

    Returns:
        The seed that was applied.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def derive_seed(base: int, *keys: int | str) -> int:
    """Derive a stable child seed from ``base`` and a sequence of keys (e.g. fold indices)."""
    from molt.utils.hashing import stable_hash

    return int(stable_hash([int(base), *map(str, keys)])[:8], 16)


def make_rng(seed: int = DEFAULT_SEED, *keys: int | str) -> np.random.Generator:
    """Return a NumPy ``Generator`` seeded with :func:`derive_seed` of ``seed`` and ``keys``."""
    return np.random.default_rng(derive_seed(seed, *keys) if keys else seed)
