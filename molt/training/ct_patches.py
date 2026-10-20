"""Patch sampling and label-preserving augmentation for preprocessed CT volumes.

Every selected CT is stored as ``<participant_id>.npz`` with an int16 HU ``volume`` at 1 mm
isotropic spacing, a ``lung_mask`` and ranked ``candidates`` (``x, y, z, score`` in voxel
coordinates). :class:`CTPatchSampler` turns one volume into a fixed-size set of cubic
patches: ``n_lung`` centred on random lung-field voxels and ``n_candidates`` centred on the
top-ranked candidate regions (with positional jitter during training). The random stream is
seeded by the study seed, the participant identifier and the epoch, so a given epoch always
sees the same patches. Intensities are clipped to the model-input HU window and rescaled
linearly to ``[0, 1]``.

Training augmentation is label-preserving: axis flips, in-plane 90-degree rotations, small
intensity scale/shift, Gaussian noise (acquisition noise) and occasional Gaussian smoothing
(softer reconstruction kernels).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from molt.utils.seed import derive_seed

__all__ = ["CTPatchConfig", "CTPatchSampler", "extract_patch"]


@dataclass
class CTPatchConfig:
    """Settings of :class:`CTPatchSampler`."""

    n_lung: int = 32
    n_candidates: int = 32
    patch_size: int = 64
    hu_window: tuple[float, float] = (-1024.0, 600.0)
    candidate_jitter: int = 4
    augment: bool = True
    flip_p: float = 0.5
    rotate_p: float = 0.5
    intensity_scale: float = 0.05
    intensity_shift: float = 0.02
    noise_std: float = 0.01
    smooth_p: float = 0.2
    cache_dir: str | None = None
    max_mask_points: int = 200_000
    memory_cache: int = 4

    @property
    def n_patches(self) -> int:
        return self.n_lung + self.n_candidates


def extract_patch(volume: np.ndarray, center: Sequence[int], size: int, fill: float) -> np.ndarray:
    """Cubic patch of edge ``size`` centred at ``center`` with out-of-bounds ``fill``."""
    out = np.full((size, size, size), fill, dtype=volume.dtype)
    half = size // 2
    src, dst = [], []
    for axis, c in enumerate(center):
        lo = int(c) - half
        hi = lo + size
        s_lo, s_hi = max(lo, 0), min(hi, volume.shape[axis])
        if s_hi <= s_lo:
            return out
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_lo - lo + (s_hi - s_lo)))
    out[tuple(dst)] = volume[tuple(src)]
    return out


class _Volume:
    """Loaded (or memory-mapped) CT arrays of one participant."""

    def __init__(self, volume: np.ndarray, mask_points: np.ndarray, candidates: np.ndarray) -> None:
        self.volume = volume
        self.mask_points = mask_points
        self.candidates = candidates


class CTPatchSampler:
    """Sample patch sets from preprocessed CT volumes.

    Args:
        config: Sampling and augmentation settings.
        seed: Study seed combined with participant id and epoch.
    """

    def __init__(self, config: CTPatchConfig | None = None, seed: int = 20240613) -> None:
        self.config = config or CTPatchConfig()
        self.seed = int(seed)
        self._cache: OrderedDict[str, _Volume] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    # ---------------------------------------------------------------- loading
    def _load(self, path: str) -> _Volume:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        vol = self._load_cached(path) if self.config.cache_dir else None
        if vol is None:
            vol = self._load_npz(path)
            if self.config.cache_dir:
                self._write_cache(path, vol)
        self._cache[path] = vol
        while len(self._cache) > max(self.config.memory_cache, 1):
            self._cache.popitem(last=False)
        return vol

    def _cache_stem(self, path: str) -> Path:
        root = Path(self.config.cache_dir or ".")
        return root / f"{Path(path).parent.name}__{Path(path).stem}"

    def _load_cached(self, path: str) -> _Volume | None:
        stem = self._cache_stem(path)
        files = [stem.with_name(stem.name + s) for s in ("_volume.npy", "_mask.npy", "_cand.npy")]
        if not all(f.exists() for f in files):
            return None
        if os.path.getmtime(files[0]) < os.path.getmtime(path):
            return None
        return _Volume(np.load(files[0], mmap_mode="r"), np.load(files[1]), np.load(files[2]))

    def _write_cache(self, path: str, vol: _Volume) -> None:
        stem = self._cache_stem(path)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, arr in (
            ("_mask.npy", vol.mask_points),
            ("_cand.npy", vol.candidates),
            ("_volume.npy", np.asarray(vol.volume)),
        ):
            target = stem.with_name(stem.name + suffix)
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, target)

    def _load_npz(self, path: str) -> _Volume:
        with np.load(path, allow_pickle=False) as data:
            volume = np.asarray(data["volume"])
            mask = np.asarray(data["lung_mask"]) if "lung_mask" in data.files else None
            cands = (
                np.asarray(data["candidates"], np.float32) if "candidates" in data.files else None
            )
            meta_raw = data["meta"] if "meta" in data.files else None
        meta: dict[str, Any] = {}
        if meta_raw is not None:
            try:
                meta = json.loads(
                    meta_raw.item() if isinstance(meta_raw, np.ndarray) else str(meta_raw)
                )
            except (ValueError, TypeError):
                meta = {}
        rng = np.random.default_rng(derive_seed(self.seed, "mask", Path(path).stem))
        if mask is not None and mask.any():
            points = np.argwhere(mask > 0).astype(np.int32)
            if len(points) > self.config.max_mask_points:
                points = points[rng.choice(len(points), self.config.max_mask_points, replace=False)]
        else:
            points = np.zeros((0, 3), np.int32)
        cand = np.zeros((0, 3), np.float32)
        if cands is not None and cands.size:
            cands = cands.reshape(-1, cands.shape[-1])
            order = np.argsort(-cands[:, 3]) if cands.shape[1] > 3 else np.arange(len(cands))
            cand = self._candidate_axes(cands[order, :3], volume.shape, meta)
        return _Volume(volume, points, cand)

    @staticmethod
    def _candidate_axes(xyz: np.ndarray, shape: Sequence[int], meta: dict[str, Any]) -> np.ndarray:
        """Map candidate ``x, y, z`` coordinates to array index order."""
        axes = str(meta.get("array_axes", meta.get("axes", ""))).lower()
        if axes == "zyx":
            return xyz[:, ::-1].copy()
        if axes == "xyz":
            return xyz.copy()
        inside = np.all((xyz >= 0) & (xyz < np.asarray(shape)), axis=1).mean() if len(xyz) else 1
        flipped = xyz[:, ::-1]
        inside_f = (
            np.all((flipped >= 0) & (flipped < np.asarray(shape)), axis=1).mean() if len(xyz) else 0
        )
        return flipped.copy() if inside_f > inside else xyz.copy()

    # ---------------------------------------------------------------- sampling
    def centers(self, vol: _Volume, rng: np.random.Generator, train: bool) -> np.ndarray:
        """Patch centres ``(n_patches, 3)``: lung-field voxels then ranked candidates."""
        cfg = self.config
        shape = np.asarray(vol.volume.shape)

        def lung(k: int) -> np.ndarray:
            if k <= 0:
                return np.zeros((0, 3), int)
            if len(vol.mask_points):
                return vol.mask_points[rng.integers(0, len(vol.mask_points), size=k)].astype(int)
            lo, hi = shape // 4, np.maximum(3 * shape // 4, shape // 4 + 1)
            return rng.integers(lo, hi, size=(k, 3))

        out = [lung(cfg.n_lung)]
        if cfg.n_candidates > 0:
            if len(vol.candidates):
                idx = np.arange(cfg.n_candidates) % len(vol.candidates)
                cand = np.round(vol.candidates[idx]).astype(int)
                if train and cfg.candidate_jitter > 0:
                    cand = cand + rng.integers(
                        -cfg.candidate_jitter, cfg.candidate_jitter + 1, cand.shape
                    )
                out.append(np.clip(cand, 0, shape - 1))
            else:
                out.append(lung(cfg.n_candidates))
        return np.concatenate(out, axis=0)

    def sample(
        self, path: str, participant_id: str, epoch: int = -1, *, train: bool = False
    ) -> np.ndarray:
        """Patch set ``(n_patches, 1, P, P, P)`` as float32 in ``[0, 1]``.

        Args:
            path: ``.npz`` volume path.
            participant_id: Participant identifier (enters the seed).
            epoch: Training epoch (enters the seed); evaluation uses a fixed epoch.
            train: Apply jitter and augmentation.
        """
        cfg = self.config
        vol = self._load(path)
        rng = np.random.default_rng(derive_seed(self.seed, "ct", participant_id, int(epoch)))
        centers = self.centers(vol, rng, train)
        lo, hi = cfg.hu_window
        size = cfg.patch_size
        patches = np.empty((len(centers), 1, size, size, size), np.float32)
        for i, c in enumerate(centers):
            patch = extract_patch(vol.volume, c, size, fill=lo).astype(np.float32)
            patches[i, 0] = (np.clip(patch, lo, hi) - lo) / (hi - lo)
        if train and cfg.augment:
            patches = self.augment(patches, rng)
        return patches

    def augment(self, patches: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Label-preserving geometric, intensity and acquisition-simulating augmentation."""
        cfg = self.config
        out = patches
        for axis in (2, 3, 4):
            if rng.random() < cfg.flip_p:
                out = np.flip(out, axis=axis)
        if rng.random() < cfg.rotate_p:
            out = np.rot90(out, k=int(rng.integers(1, 4)), axes=(3, 4))
        out = np.ascontiguousarray(out)
        if rng.random() < cfg.smooth_p:
            from scipy.ndimage import gaussian_filter

            sigma = float(rng.uniform(0.5, 1.0))
            out = gaussian_filter(out, sigma=(0, 0, sigma, sigma, sigma))
        scale = 1.0 + rng.uniform(-cfg.intensity_scale, cfg.intensity_scale)
        shift = rng.uniform(-cfg.intensity_shift, cfg.intensity_shift)
        out = out * scale + shift
        if cfg.noise_std > 0:
            sd = rng.uniform(0.0, cfg.noise_std)
            out = out + rng.standard_normal(out.shape, dtype=np.float32) * sd
        return np.clip(out, 0.0, 1.0).astype(np.float32)
