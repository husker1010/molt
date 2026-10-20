"""Deterministic patch sampling from preprocessed CT volumes and label-preserving augmentation.

:func:`sample_patches` draws ``n`` cubic patches from one ``.npz`` volume: half centred on
lung voxels drawn uniformly from the lung mask and half centred on the top-ranked candidate
regions (cycling through the candidates when there are fewer than requested, and falling
back to lung voxels when there are none). The random stream is derived from
``(seed, participant_id, epoch)``, so the same call always returns the same patches.

The augmentation functions keep the label intact: axis flips, small in-plane rotations,
intensity scale/shift jitter, Gaussian noise (acquisition noise) and Gaussian blur
(images made with a softer convolution kernel). :func:`augment_patch` chains them with the
probabilities of :class:`AugmentConfig`.

Example:
    >>> patches = sample_patches(volume_path, n=64,
    ...                          seed=20240613)
    >>> patches.shape
    (64, 64, 64, 64)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from molt.utils.seed import DEFAULT_SEED, derive_seed

__all__ = [
    "AugmentConfig",
    "add_noise",
    "augment_patch",
    "extract_patch",
    "intensity_jitter",
    "kernel_blur",
    "load_npz",
    "normalize_hu",
    "patch_centres",
    "random_flip",
    "random_rotate",
    "sample_patches",
]

AIR_HU = -1024.0


@dataclass
class AugmentConfig:
    """Probabilities and magnitudes of the patch augmentations.

    Attributes:
        flip_p: Probability of flipping each axis.
        rotate_p: Probability of an in-plane rotation.
        max_rotation_deg: Largest absolute rotation angle.
        jitter_p: Probability of intensity jitter.
        scale: Largest relative HU scale change.
        shift_hu: Largest HU offset.
        noise_p: Probability of additive Gaussian noise.
        max_noise_hu: Largest noise standard deviation (HU).
        blur_p: Probability of Gaussian blur.
        blur_sigma: Range of the blur standard deviation (voxels).
    """

    flip_p: float = 0.5
    rotate_p: float = 0.5
    max_rotation_deg: float = 10.0
    jitter_p: float = 0.5
    scale: float = 0.03
    shift_hu: float = 20.0
    noise_p: float = 0.3
    max_noise_hu: float = 15.0
    blur_p: float = 0.2
    blur_sigma: tuple[float, float] = (0.3, 1.0)


def load_npz(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Load a preprocessed ``.npz`` (or pass through an already loaded mapping)."""
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        with np.load(source, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        data.setdefault("participant_id", Path(source).stem)
    meta = data.get("meta", {})
    if isinstance(meta, np.ndarray):
        meta = meta.item() if meta.ndim == 0 else str(meta)
    if isinstance(meta, (str, bytes)):
        meta = json.loads(meta)
    data["meta"] = dict(meta or {})
    return data


def _candidate_zyx(candidates: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    cand = np.asarray(candidates, dtype=np.float32).reshape(-1, 4)
    if cand.size == 0:
        return np.zeros((0, 3), np.float32)
    order = np.argsort(-cand[:, 3], kind="stable")
    xyz = cand[order, :3]
    axes = str(meta.get("array_axes", "zyx")).lower()
    return xyz[:, ::-1] if axes == "zyx" else xyz


def patch_centres(
    lung_mask: np.ndarray,
    candidates: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    n_lung: int | None = None,
    meta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Patch centres ``(n, 3)`` in array index order: lung samples, then candidates."""
    n_lung = n // 2 if n_lung is None else int(n_lung)
    n_cand = n - n_lung
    flat = np.flatnonzero(np.asarray(lung_mask).ravel() > 0)
    shape = np.asarray(lung_mask.shape)

    def lung_points(k: int) -> np.ndarray:
        if k <= 0:
            return np.zeros((0, 3), np.int64)
        if flat.size:
            picks = flat[rng.integers(0, flat.size, size=k)]
            return np.stack(np.unravel_index(picks, lung_mask.shape), 1)
        return rng.integers(shape // 4, np.maximum(3 * shape // 4, shape // 4 + 1), size=(k, 3))

    cand = np.round(_candidate_zyx(candidates, meta or {})).astype(np.int64)
    if len(cand) and n_cand > 0:
        cand_pts = cand[np.arange(n_cand) % len(cand)]
    else:
        cand_pts = lung_points(n_cand)
    return np.concatenate([lung_points(n_lung), cand_pts], 0)


def extract_patch(
    volume: np.ndarray, centre: Sequence[int], size: int = 64, fill: float = AIR_HU
) -> np.ndarray:
    """Cubic patch of edge ``size`` centred at ``centre``; out-of-volume voxels get ``fill``."""
    out = np.full((size,) * 3, fill, dtype=np.float32)
    src, dst = [], []
    for axis, c in enumerate(centre):
        lo = int(c) - size // 2
        s_lo, s_hi = max(lo, 0), min(lo + size, volume.shape[axis])
        if s_hi <= s_lo:
            return out
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_hi - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def sample_patches(
    npz: str | os.PathLike[str] | Mapping[str, Any],
    n: int = 64,
    seed: int = DEFAULT_SEED,
    *,
    participant_id: str | None = None,
    epoch: int = 0,
    patch_size: int = 64,
    n_lung: int | None = None,
    augment: bool | AugmentConfig = False,
    return_centres: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` patches (HU, float32) of edge ``patch_size`` from one volume.

    Args:
        npz: Path of a preprocessed ``.npz`` or its loaded contents.
        n: Number of patches.
        seed: Study seed.
        participant_id: Participant identifier used in the random stream (defaults to the
            file stem or ``meta["participant_id"]``).
        epoch: Epoch index (different epochs draw different patches).
        patch_size: Patch edge in voxels.
        n_lung: Number of lung-field patches (default ``n // 2``); the rest are candidates.
        augment: Apply :func:`augment_patch` (``True`` or an :class:`AugmentConfig`).
        return_centres: Also return the ``(n, 3)`` patch centres.

    Returns:
        Array ``(n, patch_size, patch_size, patch_size)`` (and the centres if requested).
    """
    data = load_npz(npz)
    meta = data["meta"]
    pid = participant_id or meta.get("participant_id") or str(data.get("participant_id", ""))
    rng = np.random.default_rng(derive_seed(seed, pid, int(epoch)))
    volume = np.asarray(data["volume"])
    mask = np.asarray(data.get("lung_mask", np.zeros(volume.shape, np.uint8)))
    cands = np.asarray(data.get("candidates", np.zeros((0, 4), np.float32)))
    centres = patch_centres(mask, cands, n, rng, n_lung=n_lung, meta=meta)
    patches = np.stack([extract_patch(volume, c, patch_size) for c in centres], 0)
    if augment:
        cfg = augment if isinstance(augment, AugmentConfig) else AugmentConfig()
        patches = np.stack([augment_patch(p, rng, cfg) for p in patches], 0)
    return (patches, centres) if return_centres else patches


def normalize_hu(patches: np.ndarray, window: tuple[float, float] = (-1024.0, 600.0)) -> np.ndarray:
    """Clip to ``window`` and rescale linearly to ``[0, 1]``."""
    lo, hi = window
    return ((np.clip(patches, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def random_flip(patch: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Flip each axis independently with probability ``p``."""
    axes = tuple(a for a in range(patch.ndim) if rng.random() < p)
    return np.flip(patch, axis=axes).copy() if axes else patch


def random_rotate(
    patch: np.ndarray, rng: np.random.Generator, max_deg: float = 10.0, fill: float = AIR_HU
) -> np.ndarray:
    """Rotate in the axial plane by an angle drawn from ``[-max_deg, max_deg]``."""
    angle = float(rng.uniform(-max_deg, max_deg))
    return ndimage.rotate(
        patch, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=fill
    ).astype(np.float32)


def intensity_jitter(
    patch: np.ndarray, rng: np.random.Generator, scale: float = 0.03, shift_hu: float = 20.0
) -> np.ndarray:
    """Affine HU jitter ``x * (1 + a) + b`` around air, with ``|a| <= scale``, ``|b| <= shift_hu``."""
    a = float(rng.uniform(-scale, scale))
    b = float(rng.uniform(-shift_hu, shift_hu))
    return ((patch - AIR_HU) * (1.0 + a) + AIR_HU + b).astype(np.float32)


def add_noise(
    patch: np.ndarray, rng: np.random.Generator, max_sigma_hu: float = 15.0
) -> np.ndarray:
    """Additive Gaussian noise with a standard deviation drawn from ``[0, max_sigma_hu]``."""
    sigma = float(rng.uniform(0.0, max_sigma_hu))
    return (patch + rng.normal(0.0, sigma, size=patch.shape)).astype(np.float32)


def kernel_blur(
    patch: np.ndarray, rng: np.random.Generator, sigma: tuple[float, float] = (0.3, 1.0)
) -> np.ndarray:
    """Gaussian blur emulating a softer convolution kernel."""
    s = float(rng.uniform(*sigma))
    return ndimage.gaussian_filter(patch, s, mode="nearest").astype(np.float32)


def augment_patch(
    patch: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig | None = None
) -> np.ndarray:
    """Apply flips, rotation, intensity jitter, noise and blur with the configured probabilities."""
    cfg = cfg or AugmentConfig()
    out = random_flip(np.asarray(patch, dtype=np.float32), rng, cfg.flip_p)
    if rng.random() < cfg.rotate_p:
        out = random_rotate(out, rng, cfg.max_rotation_deg)
    if rng.random() < cfg.blur_p:
        out = kernel_blur(out, rng, cfg.blur_sigma)
    if rng.random() < cfg.jitter_p:
        out = intensity_jitter(out, rng, cfg.scale, cfg.shift_hu)
    if rng.random() < cfg.noise_p:
        out = add_noise(out, rng, cfg.max_noise_hu)
    return out
