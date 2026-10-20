"""Candidate abnormality regions inside the lung field and a compact radiomics summary.

The recipe of :func:`detect_candidates` is fixed (no learned parameters):

1. the search region is the lung mask, closed per lung (to include juxta-pleural lesions)
   and hole-filled in 3-D (to include lesions enclosed by lung);
2. dense tissue (HU above ``dense_hu``) inside the region is opened with a small ball to
   detach thin vessels and split into connected components;
3. tubular or branching components (vessels) are suppressed using a local shape ratio
   (largest inscribed radius divided by the radius of the sphere of equal volume) and the
   principal-axis elongation; thick cores of large vessel-like components are re-examined
   so that lesions abutting vessels are not lost;
4. the remaining components are ranked by ``volume x (mean HU - mean HU of the surrounding
   parenchymal shell)`` and the centres of the top ``n_candidates`` are returned.

:func:`radiomics_summary` returns three features of the largest retained component
(volume, mean HU and HU entropy) for the radiomics comparator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from molt.preprocess.ct.config import CandidateConfig
from molt.preprocess.ct.lung import ball

__all__ = ["candidate_region", "detect_candidates", "hu_entropy", "radiomics_summary"]


def candidate_region(
    lung: np.ndarray, closing_mm: float = 6.0, spacing_mm: float = 1.0, fill: str = "3d"
) -> np.ndarray:
    """Lung search region: per-lung closing (at half resolution) plus 3-D hole filling.

    Holes are filled in 3-D only: cavities enclosed by lung (lesions excluded from the lung
    mask) are added, whereas structures that only look enclosed in an axial section, such as
    the diaphragm dome below the lung bases, stay outside the region.
    """
    region = lung > 0
    radius = closing_mm / (2.0 * spacing_mm)
    for label in np.unique(lung[lung > 0]):
        part = lung == label
        if radius >= 1:
            pad = math.ceil(radius) + 1
            small = np.pad(part[::2, ::2, ::2], pad)
            closed = ndimage.binary_closing(small, structure=ball(radius))
            closed = closed[pad:-pad, pad:-pad, pad:-pad]
            up = np.repeat(np.repeat(np.repeat(closed, 2, 0), 2, 1), 2, 2)
            region |= up[: part.shape[0], : part.shape[1], : part.shape[2]]
    if fill == "3d":
        return ndimage.binary_fill_holes(region)
    if fill != "none":
        raise ValueError(f"region fill must be '3d' or 'none', got {fill!r}")
    return region


def hu_entropy(
    values: np.ndarray, bin_hu: float = 25.0, lo: float = -1024.0, hi: float = 600.0
) -> float:
    """Shannon entropy (bits) of the HU histogram with fixed bins."""
    if values.size == 0:
        return float("nan")
    edges = np.arange(lo, hi + bin_hu, bin_hu)
    counts, _ = np.histogram(np.clip(values, lo, hi - 1e-3), bins=edges)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _component_stats(
    comp: np.ndarray,
    offset: np.ndarray,
    hu_crop: np.ndarray,
    edt_crop: np.ndarray,
    dense_crop: np.ndarray,
    region_crop: np.ndarray,
    cfg: CandidateConfig,
) -> dict[str, Any]:
    coords = np.argwhere(comp)
    volume = float(len(coords))
    centroid = coords.mean(0)
    if len(coords) > 2:
        evals = np.sort(np.linalg.eigvalsh(np.cov(coords.T.astype(np.float64))))[::-1]
    else:
        evals = np.ones(3)
    elongation = float(math.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6)))
    r_in = float(edt_crop[comp].max())
    r_eq = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    shape_ratio = r_in / max(r_eq, 1e-6)
    mean_hu = float(hu_crop[comp].mean())
    ring = ndimage.binary_dilation(comp, iterations=int(cfg.shell_mm))
    ring &= ~dense_crop & region_crop & ~comp
    shell_hu = float(hu_crop[ring].mean()) if ring.any() else -850.0
    contrast = mean_hu - shell_hu
    centre = np.round(centroid).astype(int)
    if not comp[tuple(np.clip(centre, 0, np.asarray(comp.shape) - 1))]:
        masked = np.where(comp, edt_crop, -1.0)
        centre = np.asarray(np.unravel_index(int(np.argmax(masked)), comp.shape))
    vessel = shape_ratio < cfg.vessel_shape_ratio or (
        elongation > cfg.vessel_elongation and r_in < cfg.vessel_max_radius_mm
    )
    zyx = centre + offset
    return {
        "x": float(zyx[2]),
        "y": float(zyx[1]),
        "z": float(zyx[0]),
        "volume_mm3": volume,
        "mean_hu": round(mean_hu, 1),
        "shell_hu": round(shell_hu, 1),
        "contrast_hu": round(contrast, 1),
        "inscribed_radius_mm": round(r_in, 2),
        "shape_ratio": round(shape_ratio, 3),
        "elongation": round(elongation, 2),
        "vessel_like": bool(vessel),
        "score": float(volume * max(contrast, 0.0) / 1000.0),
    }


def detect_candidates(
    volume: np.ndarray,
    lung: np.ndarray,
    cfg: CandidateConfig | None = None,
    spacing_mm: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    """Detect candidate abnormality regions.

    Args:
        volume: HU volume ``(z, y, x)`` at isotropic ``spacing_mm``.
        lung: Lung label map of the same shape.
        cfg: Candidate settings.
        spacing_mm: Voxel size.

    Returns:
        ``(candidates, details, radiomics)`` where ``candidates`` is a float32 ``(N, 4)``
        array of ``x, y, z, score`` (voxel coordinates, ``N <= n_candidates``, sorted by
        score), ``details`` holds per-candidate statistics and ``radiomics`` the summary of
        :func:`radiomics_summary`.
    """
    cfg = cfg or CandidateConfig()
    empty = np.zeros((0, 4), np.float32)
    if not (lung > 0).any():
        return empty, [], radiomics_summary(None, None, cfg)
    region = candidate_region(lung, cfg.region_closing_mm, spacing_mm, cfg.region_fill)
    dense = (volume > cfg.dense_hu) & region
    radius = cfg.opening_mm / spacing_mm
    opened = ndimage.binary_opening(dense, structure=ball(radius)) if radius >= 1 else dense
    labels, n = ndimage.label(opened)
    if n == 0:
        return empty, [], radiomics_summary(None, None, cfg)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    min_vox = cfg.min_volume_mm3 / spacing_mm**3
    ids = np.flatnonzero(sizes >= min_vox)
    ids = ids[np.argsort(-sizes[ids])][: cfg.max_components]
    edt = ndimage.distance_transform_edt(opened) * spacing_mm
    objects = ndimage.find_objects(labels)
    pad = int(cfg.shell_mm) + 1
    details: list[dict[str, Any]] = []
    masks: dict[int, tuple[tuple[slice, ...], np.ndarray]] = {}
    for lab in ids:
        sl = objects[lab - 1]
        sl = tuple(
            slice(max(s.start - pad, 0), min(s.stop + pad, dim)) for s, dim in zip(sl, labels.shape)
        )
        offset = np.asarray([s.start for s in sl])
        comp = labels[sl] == lab
        crops = (volume[sl], edt[sl], dense[sl], region[sl])
        stats = _component_stats(comp, offset, *crops, cfg)
        if stats["vessel_like"] and stats["volume_mm3"] > 2000.0:
            details.extend(_thick_cores(comp, offset, crops, cfg, spacing_mm))
        stats["label"] = int(lab)
        details.append(stats)
        masks[int(lab)] = (sl, comp)
    kept = sorted((d for d in details if not d["vessel_like"]), key=lambda d: -d["score"])
    top = kept[: cfg.n_candidates]
    cands = np.asarray([[d["x"], d["y"], d["z"], d["score"]] for d in top], np.float32).reshape(
        -1, 4
    )
    largest = max(kept, key=lambda d: d["volume_mm3"]) if kept else None
    radiomics = radiomics_summary(largest, _values_of(largest, masks, volume), cfg)
    return cands, top, radiomics


def _thick_cores(
    comp: np.ndarray,
    offset: np.ndarray,
    crops: tuple[np.ndarray, ...],
    cfg: CandidateConfig,
    spacing_mm: float,
) -> list[dict[str, Any]]:
    """Re-examine the thick parts (inscribed radius > vessel radius) of a vessel-like component."""
    hu_crop, edt_crop, dense_crop, region_crop = crops
    core = comp & (edt_crop > cfg.vessel_max_radius_mm)
    cores, n = ndimage.label(core)
    out = []
    grow = math.ceil(cfg.vessel_max_radius_mm / spacing_mm)
    for lab in range(1, n + 1):
        part = cores == lab
        if part.sum() * spacing_mm**3 < cfg.min_volume_mm3:
            continue
        blob = ndimage.binary_dilation(part, structure=ball(1.0), iterations=grow) & comp
        stats = _component_stats(blob, offset, hu_crop, edt_crop, dense_crop, region_crop, cfg)
        stats["from_core"] = True
        stats["label"] = -1
        out.append(stats)
    return out


def _values_of(
    largest: dict[str, Any] | None,
    masks: dict[int, tuple[tuple[slice, ...], np.ndarray]],
    volume: np.ndarray,
) -> np.ndarray | None:
    if largest is None:
        return None
    if largest.get("label", -1) in masks:
        sl, comp = masks[largest["label"]]
        return volume[sl][comp].astype(np.float32)
    r = max(round(largest["inscribed_radius_mm"]), 1)
    z, y, x = int(largest["z"]), int(largest["y"]), int(largest["x"])
    sl = tuple(slice(max(c - r, 0), c + r + 1) for c in (z, y, x))
    return volume[sl].astype(np.float32).ravel()


def radiomics_summary(
    largest: dict[str, Any] | None, values: np.ndarray | None, cfg: CandidateConfig | None = None
) -> dict[str, float]:
    """Volume (mm^3), mean HU and HU entropy (bits) of the largest candidate (NaN if none)."""
    cfg = cfg or CandidateConfig()
    if largest is None or values is None or values.size == 0:
        nan = float("nan")
        return {
            "largest_candidate_volume_mm3": nan,
            "largest_candidate_mean_hu": nan,
            "largest_candidate_hu_entropy": nan,
        }
    return {
        "largest_candidate_volume_mm3": float(largest["volume_mm3"]),
        "largest_candidate_mean_hu": round(float(values.mean()), 2),
        "largest_candidate_hu_entropy": round(hu_entropy(values, cfg.entropy_bin_hu), 4),
    }
