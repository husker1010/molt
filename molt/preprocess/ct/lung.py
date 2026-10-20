"""Lung segmentation (lungmask U-Net R231), mask correction and lung-level quality control.

:class:`LungSegmenter` runs the R231 model slice-wise on the GPU in batches and returns the
raw label map (``1`` = right lung, ``2`` = left lung). :func:`correct_lung_mask` removes
small fragments and sub-diaphragmatic gas that leaked into the mask (components that
separate from the lungs after a small opening, lie below the lung bases and contain gas
rather than parenchyma), then closes and hole-fills each lung. :func:`lung_qc` derives the
QC flags (no lung, implausible volume, truncation at the scan range or field-of-view border,
fragmented topology).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage

from molt.preprocess.ct.config import LungConfig

__all__ = ["LungSegmenter", "ball", "correct_lung_mask", "lung_qc"]


def ball(radius: float) -> np.ndarray:
    """Binary ball structuring element of the given radius (voxels)."""
    r = int(np.floor(radius))
    grid = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (grid**2).sum(0) <= radius**2 + 1e-6


class LungSegmenter:
    """Slice-wise lung segmentation with the lungmask R231 U-Net.

    Args:
        weights: Required local segmentation state supplied by the caller.
        device: ``"auto"``, ``"cuda"``, ``"cuda:1"``, ``"cpu"`` ...
        batch_size: Axial slices per forward pass.
    """

    def __init__(
        self, weights: str | None = None, device: str = "auto", batch_size: int = 32
    ) -> None:
        import torch
        from lungmask.resunet import UNet

        if not weights or not os.path.isfile(weights):
            raise ValueError("A local segmentation state must be supplied explicitly")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        n_classes = len(list(state.values())[-1])
        model = UNet(
            n_classes=n_classes,
            padding=True,
            depth=5,
            up_mode="upsample",
            batch_norm=True,
            residual=False,
        )
        model.load_state_dict(state)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = int(batch_size)

    def predict(self, volume: np.ndarray, gpu_lock: Any | None = None) -> np.ndarray:
        """Raw label map (``uint8``, same shape) of an LPS-ordered HU volume ``(z, y, x)``.

        Args:
            volume: HU volume.
            gpu_lock: Optional lock/semaphore held only during GPU inference (limits the
                number of processes using the GPU at the same time).
        """
        from contextlib import nullcontext

        from lungmask import utils

        slices, boxes = utils.preprocess(
            np.asarray(volume, dtype=np.float32), resolution=[256, 256]
        )
        slices = ((np.minimum(slices, 600.0) + 1024.0) / 1624.0).astype(np.float32)
        with gpu_lock if gpu_lock is not None else nullcontext():
            pred = self._infer(slices)
        out = np.zeros(volume.shape, dtype=np.uint8)
        for k in range(pred.shape[0]):
            if pred[k].any():
                out[k] = utils.reshape_mask(pred[k], boxes[k], volume.shape[1:]).astype(np.uint8)
        return out

    def _infer(self, slices: np.ndarray) -> np.ndarray:
        """Argmax labels of preprocessed slices; halves the batch on CUDA out-of-memory."""
        import torch

        batch_size = self.batch_size
        while True:
            try:
                preds = []
                with torch.inference_mode():
                    for start in range(0, len(slices), batch_size):
                        batch = torch.from_numpy(slices[start : start + batch_size, None]).to(
                            self.device
                        )
                        preds.append(self.model(batch).argmax(1).to(torch.uint8).cpu().numpy())
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(batch_size // 2, 1)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(preds, 0) if preds else np.zeros((0, 256, 256), np.uint8)


def _bbox(mask: np.ndarray, margin: int) -> tuple[slice, ...]:
    out = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        out.append(
            slice(max(int(hit[0]) - margin, 0), min(int(hit[-1]) + margin + 1, mask.shape[axis]))
        )
    return tuple(out)


def _large_components(
    mask: np.ndarray, cfg: LungConfig, voxel_ml: float
) -> tuple[int, list[float]]:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return 0, []
    sizes = np.bincount(labels.ravel())[1:] * voxel_ml
    total = sizes.sum()
    large = sizes[
        (sizes >= cfg.large_component_fraction * total) & (sizes >= cfg.large_component_min_ml)
    ]
    return len(large), sorted((round(float(s), 1) for s in sizes[sizes >= 1.0]), reverse=True)[:6]


def correct_lung_mask(
    raw: np.ndarray, volume: np.ndarray, cfg: LungConfig | None = None, spacing_mm: float = 1.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean a raw lung label map.

    1. components of the mask after an opening of radius ``leak_opening_mm`` are analysed;
       those that lie mostly below the lowest decile of the two main lungs and have a mean
       HU below ``leak_max_hu`` (gas) are removed as sub-diaphragmatic leakage, components
       centred outside the bounding box of the main lungs (grown by ``outside_margin_mm``)
       are removed as extra-thoracic false positives, and fragments below
       ``min_component_fraction`` of the mask are dropped;
    2. the retained components are mapped back to the unopened mask;
    3. each lung label is closed (radius ``closing_mm``) and hole-filled.

    Args:
        raw: Raw label map (``0`` background, ``1`` right, ``2`` left lung).
        volume: HU volume of the same shape.
        cfg: Lung settings.
        spacing_mm: Isotropic voxel size.

    Returns:
        The corrected label map and a dictionary of correction statistics.
    """
    cfg = cfg or LungConfig()
    voxel_ml = spacing_mm**3 / 1000.0
    info: dict[str, Any] = {"raw_volume_ml": round(float((raw > 0).sum()) * voxel_ml, 1)}
    out = np.zeros_like(raw, dtype=np.uint8)
    if not (raw > 0).any():
        info.update(
            raw_large_components=0,
            leak_removed_ml=0.0,
            outside_removed_ml=0.0,
            fragments_removed_ml=0.0,
        )
        return out, info
    box = _bbox(
        raw > 0, margin=int(np.ceil(max(cfg.closing_mm, cfg.leak_opening_mm) / spacing_mm)) + 2
    )
    lab = raw[box]
    hu = volume[box]
    mask = lab > 0
    info["raw_large_components"], info["raw_component_ml"] = _large_components(mask, cfg, voxel_ml)

    radius = cfg.leak_opening_mm / spacing_mm
    opened = ndimage.binary_opening(mask, structure=ball(radius)) if radius >= 1 else mask
    comp, n = ndimage.label(opened, structure=np.ones((3, 3, 3), bool))
    leak_ml = frag_ml = outside_ml = 0.0
    keep = np.zeros(n + 1, dtype=bool)
    if n:
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(np.ones_like(comp, dtype=np.float32), comp, idx)
        mean_hu = ndimage.mean(hu, comp, idx)
        centres = np.asarray(ndimage.center_of_mass(opened, comp, idx)).reshape(-1, 3)
        zc = centres[:, 0]
        # main components: for each lung label, the component holding most of its voxels
        main = sorted(
            {
                int(np.argmax(per_label))
                for per_label in (
                    ndimage.sum_labels((lab == label).astype(np.float32), comp, idx)
                    for label in (1, 2)
                )
                if per_label.max() >= 0.01 * sizes.sum()
            }
            or {int(np.argmax(sizes))}
        )
        main_mask = np.isin(comp, idx[main])
        zs = np.flatnonzero(main_mask.any(axis=(1, 2)))
        z_profile = main_mask.sum(axis=(1, 2)).astype(np.float64)
        cum = np.cumsum(z_profile) / max(z_profile.sum(), 1.0)
        z_low = int(np.searchsorted(cum, 0.10)) if len(zs) else 0
        total = float(sizes.sum())
        main_box = _bbox(main_mask, margin=0)
        grow = cfg.outside_margin_mm / spacing_mm
        lo = np.asarray([b.start for b in main_box]) - grow
        hi = np.asarray([b.stop for b in main_box]) + grow
        for i in range(n):
            if i in main:
                keep[i + 1] = True
                continue
            if zc[i] < z_low and mean_hu[i] < cfg.leak_max_hu:
                leak_ml += sizes[i] * voxel_ml
            elif np.any(centres[i] < lo) or np.any(centres[i] > hi):
                outside_ml += sizes[i] * voxel_ml
            elif sizes[i] < cfg.min_component_fraction * total:
                frag_ml += sizes[i] * voxel_ml
            else:
                keep[i + 1] = True
    kept = keep[comp]
    reach = ndimage.binary_dilation(kept, structure=ball(max(radius, 1.0) + 1.0))
    cleaned = mask & reach
    info["leak_removed_ml"] = round(leak_ml, 1)
    info["outside_removed_ml"] = round(outside_ml, 1)
    info["fragments_removed_ml"] = round(frag_ml, 1)

    final = np.zeros_like(lab, dtype=np.uint8)
    close_r = cfg.closing_mm / spacing_mm
    for label in (1, 2):
        part = cleaned & (lab == label)
        if not part.any():
            continue
        sub = _bbox(part, margin=0)
        piece = part[sub]
        if close_r >= 1:
            # zero padding keeps the closing extensive at the array border
            pad = int(np.ceil(close_r)) + 1
            closed = ndimage.binary_closing(np.pad(piece, pad), structure=ball(close_r))
            piece = closed[pad:-pad, pad:-pad, pad:-pad] | piece
        piece = ndimage.binary_fill_holes(piece)
        target = final[sub]
        target[piece & (target == 0)] = label
    out[box] = final
    return out, info


def lung_qc(
    lung: np.ndarray,
    fov_mask: np.ndarray | None,
    cfg: LungConfig | None = None,
    spacing_mm: float = 1.0,
) -> tuple[list[str], dict[str, Any]]:
    """QC flags and metrics of a corrected lung mask.

    Flags: ``no_lung``, ``lung_volume_low``/``lung_volume_high`` (outside
    ``[min_volume_l, max_volume_l]``), ``truncated_z`` (lung cut by the first/last slices),
    ``truncated_fov`` (lung touching the field-of-view border), ``topology`` (more than two
    large components); informative: ``touches_z_boundary``, ``touches_fov``, ``single_lung``.
    """
    cfg = cfg or LungConfig()
    voxel_l = spacing_mm**3 / 1e6
    mask = lung > 0
    n_vox = int(mask.sum())
    volume_l = n_vox * voxel_l
    metrics: dict[str, Any] = {"lung_volume_l": round(volume_l, 3)}
    flags: list[str] = []
    if volume_l < 0.05:
        return ["no_lung"], metrics
    metrics["right_volume_l"] = round(float((lung == 1).sum()) * voxel_l, 3)
    metrics["left_volume_l"] = round(float((lung == 2).sum()) * voxel_l, 3)
    if min(metrics["right_volume_l"], metrics["left_volume_l"]) < 0.1 * volume_l:
        flags.append("single_lung")
    if volume_l < cfg.min_volume_l:
        flags.append("lung_volume_low")
    if volume_l > cfg.max_volume_l:
        flags.append("lung_volume_high")

    area = mask.sum(axis=(1, 2)).astype(np.float64)
    edge = max(area[:2].max(), area[-2:].max())
    metrics["boundary_area_fraction"] = round(float(edge / max(area.max(), 1.0)), 3)
    if edge > 0:
        flags.append("touches_z_boundary")
    if metrics["boundary_area_fraction"] > cfg.truncation_area_fraction:
        flags.append("truncated_z")

    border = np.zeros(mask.shape[1:], dtype=bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True
    zs = np.flatnonzero(area > 0)
    sub = slice(int(zs[0]), int(zs[-1]) + 1)
    near = np.broadcast_to(border, mask[sub].shape).copy()
    if fov_mask is not None:
        outside = ~fov_mask[sub]
        if outside.any():
            plane = np.zeros((1, 5, 5), bool)
            plane[0, 1:4, 1:4] = True
            plane[0, 2, :] = plane[0, :, 2] = True
            near |= ndimage.binary_dilation(outside, structure=plane, iterations=1)
    touch = float((mask[sub] & near).sum()) / n_vox
    metrics["fov_touch_fraction"] = round(touch, 5)
    if touch > 0:
        flags.append("touches_fov")
    if touch > cfg.fov_touch_fraction:
        flags.append("truncated_fov")

    n_large, sizes = _large_components(mask[sub], cfg, voxel_l * 1000.0)
    metrics["large_components"] = n_large
    metrics["component_ml"] = sizes
    if n_large > 2:
        flags.append("topology")
    return flags, metrics


def hard_failures(flags: Sequence[str], hard: Sequence[str]) -> list[str]:
    """Subset of ``flags`` that fail QC."""
    return [f for f in flags if f in set(hard)]
