"""CT volume assembly: DICOM series -> 1 mm isotropic int16 HU volume in LPS axes.

Steps of :func:`load_isotropic_volume`:

1. read the slices with SimpleITK in the order given by their ``ImagePositionPatient``
   projected on the slice normal (rescale slope/intercept are applied, giving HU);
2. mark pixels outside the field of view (padding values) and set them to air;
3. derive the geometry from the DICOM headers: in-plane offsets between slices (gantry
   tilt or a change of field-of-view centre) are removed by shifting each slice onto the
   grid of the first slice, and irregular slice positions are interpolated onto a uniform
   grid;
4. reorient to LPS by axis permutation and flips only (no rotation of the voxel grid) and
   optionally restrict the native volume to a region of interest (the thorax);
5. resample to isotropic spacing with cubic B-spline interpolation (optionally after
   Gaussian anti-aliasing of downsampled axes), clip to the stored HU range and cast to int16.

Arrays use NumPy index order ``(z, y, x)``; the physical position of voxel ``(k, j, i)`` is
``origin + direction @ (i, j, k) * spacing`` with ``origin``/``spacing`` given in ``x, y, z``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from molt.preprocess.ct.config import VolumeConfig

__all__ = [
    "IsotropicVolume",
    "VolumeError",
    "crop_box",
    "load_isotropic_volume",
    "save_volume_npz",
]


class VolumeError(RuntimeError):
    """Raised when a series cannot be turned into a valid volume (``geometry`` failure)."""


@dataclass
class IsotropicVolume:
    """Resampled CT volume.

    Attributes:
        volume: int16 HU array ``(z, y, x)``.
        fov_mask: bool array, ``True`` inside the field of view.
        origin: Physical position (mm, LPS) of voxel ``(0, 0, 0)`` as ``x, y, z``.
        spacing: Voxel spacing ``x, y, z`` (mm).
        direction: Row-major 3x3 direction cosines of the array axes ``x, y, z``.
        flags: Informative flags raised while assembling the volume.
        info: Native geometry and processing details.
    """

    volume: np.ndarray
    fov_mask: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]
    flags: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def _read_pixels(files: Sequence[str]) -> np.ndarray:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in files])
    reader.SetOutputPixelType(sitk.sitkFloat32)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    if arr.ndim != 3 or arr.shape[0] != len(files):
        raise VolumeError(f"expected {len(files)} slices, read array of shape {arr.shape}")
    return arr


def _uniform_z(
    arr: np.ndarray, proj: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Linearly interpolate slices at positions ``proj`` onto a uniform grid."""
    step = float(np.median(np.diff(proj)))
    n = math.floor((proj[-1] - proj[0]) / step + 1e-6) + 1
    target = proj[0] + step * np.arange(n)
    idx = np.clip(np.searchsorted(proj, target, side="right") - 1, 0, len(proj) - 2)
    w = ((target - proj[idx]) / np.maximum(proj[idx + 1] - proj[idx], 1e-6)).astype(np.float32)
    w = np.clip(w, 0.0, 1.0)
    out = arr[idx] * (1.0 - w)[:, None, None] + arr[idx + 1] * w[:, None, None]
    new_offsets = offsets[idx] * (1.0 - w)[:, None] + offsets[idx + 1] * w[:, None]
    return out.astype(np.float32), target, new_offsets, step


def _shift_slices(
    arr: np.ndarray, outside: np.ndarray, shift_px: np.ndarray, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shift each axial slice by ``shift_px[k] = (rows, cols)`` (cubic spline, constant fill)."""
    from scipy import ndimage

    out = np.empty_like(arr)
    out_mask = np.empty_like(outside)
    for k in range(arr.shape[0]):
        if np.abs(shift_px[k]).max() <= 0.1:
            out[k], out_mask[k] = arr[k], outside[k]
            continue
        out[k] = ndimage.shift(arr[k], shift_px[k], order=3, mode="constant", cval=fill)
        out_mask[k] = ndimage.shift(
            outside[k].astype(np.uint8), shift_px[k], order=0, mode="constant", cval=1
        ).astype(bool)
    return out, out_mask


def load_isotropic_volume(
    files: Sequence[str],
    *,
    iop: Sequence[float],
    positions: Sequence[Sequence[float]],
    pixel_spacing: Sequence[float],
    cfg: VolumeConfig | None = None,
    roi_fn: (
        Callable[[np.ndarray, Sequence[float]], tuple[slice, slice, slice] | None] | None
    ) = None,
) -> IsotropicVolume:
    """Read a DICOM series and resample it to an isotropic LPS volume.

    Args:
        files: Slice files ordered along the slice normal.
        iop: ``ImageOrientationPatient`` (row and column direction cosines).
        positions: ``ImagePositionPatient`` of each file (same order as ``files``).
        pixel_spacing: DICOM ``PixelSpacing`` (row spacing, column spacing).
        cfg: Volume settings.
        roi_fn: Optional function ``(array_zyx, spacing_xyz) -> (z, y, x) slices`` applied to
            the LPS-permuted native array; only that region is resampled (used to restrict
            large fields of view to the thorax before interpolation).

    Returns:
        The :class:`IsotropicVolume`.

    Raises:
        VolumeError: Inconsistent geometry or an implausible physical extent.
    """
    import SimpleITK as sitk

    cfg = cfg or VolumeConfig()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(int(cfg.sitk_threads), 1))
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    flags: list[str] = []
    info: dict[str, Any] = {}
    if len(files) < 2 or len(files) != len(positions):
        raise VolumeError("need at least two slices with positions")
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:], dtype=np.float64)
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    pos = np.asarray(positions, dtype=np.float64)
    proj0 = pos @ normal
    if np.any(np.diff(proj0) <= 1e-4):
        raise VolumeError("slice positions are not strictly increasing along the normal")
    offsets0 = pos - pos[0] - np.outer(proj0 - proj0[0], normal)

    arr = _read_pixels(files)
    outside = arr < cfg.padding_hu
    arr[outside] = cfg.hu_clip[0]

    # In-plane offsets between slices (gantry tilt or changes of field-of-view centre) are
    # removed by shifting every slice onto the in-plane grid of the first slice.
    along0 = proj0 - proj0[0]
    shear = (offsets0 * along0[:, None]).sum(0) / max(float((along0**2).sum()), 1e-9)
    tilt = math.degrees(math.atan(float(np.linalg.norm(shear))))
    residual = float(np.abs(offsets0 - np.outer(along0, shear)).max())
    shift_px = np.stack([offsets0 @ col / pixel_spacing[0], offsets0 @ row / pixel_spacing[1]], 1)
    info["tilt_deg"] = round(tilt, 3)
    info["max_inplane_offset_mm"] = round(float(np.abs(offsets0).max()), 3)
    if tilt > cfg.max_tilt_deg:
        raise VolumeError(f"gantry tilt {tilt:.1f} deg exceeds {cfg.max_tilt_deg}")
    if np.abs(shift_px).max() > 0.1:
        arr, outside = _shift_slices(arr, outside, shift_px, fill=float(cfg.hu_clip[0]))
        flags.append(
            "gantry_tilt_corrected"
            if tilt > cfg.tilt_tolerance_deg and residual < 0.5
            else "inplane_offset_corrected"
        )
    gaps = np.diff(proj0)
    if gaps.max() - gaps.min() > 0.05 * max(float(np.median(gaps)), 1e-3):
        zero = np.zeros_like(offsets0)
        arr, _, _, step = _uniform_z(arr, proj0, zero)
        outside = _uniform_z(outside.astype(np.float32), proj0, zero)[0] > 0.5
        flags.append("irregular_spacing_interpolated")
    else:
        step = float(np.median(gaps))
    info.update(
        native_spacing=[float(pixel_spacing[1]), float(pixel_spacing[0]), step],
        native_shape=[int(v) for v in arr.shape[::-1]],
    )

    spacing_native = (float(pixel_spacing[1]), float(pixel_spacing[0]), step)
    direction = np.column_stack([row, col, normal])
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(spacing_native)
    image.SetOrigin(tuple(float(v) for v in pos[0]))
    image.SetDirection(tuple(direction.ravel()))
    fov = sitk.GetImageFromArray(outside.astype(np.uint8))
    fov.CopyInformation(image)
    del arr, outside

    oriented = sitk.DICOMOrient(image, "LPS")
    fov_oriented = sitk.DICOMOrient(fov, "LPS")
    info["native_orientation"] = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    out_dir = np.asarray(oriented.GetDirection()).reshape(3, 3)
    if np.min(np.abs(np.diag(out_dir))) < math.cos(math.radians(25.0)):
        raise VolumeError("orientation is not close to axial LPS after axis permutation")
    if roi_fn is not None:
        box = roi_fn(sitk.GetArrayViewFromImage(oriented), oriented.GetSpacing())
        if box is not None:
            zs, ys, xs = box
            oriented = oriented[xs, ys, zs]
            fov_oriented = fov_oriented[xs, ys, zs]
            info["roi_zyx"] = [[int(b.start), int(b.stop)] for b in box]

    if cfg.antialias:
        target = float(cfg.spacing_mm)
        variance = []
        for s in oriented.GetSpacing():
            factor = target / s
            sigma = s * (factor - 1.0) / 2.0 if factor >= 1.5 else 0.0
            variance.append(sigma**2)
        if max(variance) > 0:
            oriented = sitk.DiscreteGaussian(oriented, variance, 8, 0.01, True)
            info["antialias_sigma_mm"] = [round(math.sqrt(v), 3) for v in variance]

    # Output grid in the (permuted) image frame, covering all voxel centres of the input.
    size_in = oriented.GetSize()
    corners = np.array(
        [
            oriented.TransformIndexToPhysicalPoint((i, j, k))
            for i in (0, size_in[0] - 1)
            for j in (0, size_in[1] - 1)
            for k in (0, size_in[2] - 1)
        ]
    )
    ref = np.asarray(oriented.GetOrigin())
    local = (corners - ref) @ out_dir
    lo, hi = local.min(0), local.max(0)
    sp = float(cfg.spacing_mm)
    extent = hi - lo
    if np.any(extent > cfg.max_extent_mm):
        raise VolumeError(f"implausible physical extent {extent.round(1).tolist()} mm")
    size_out = [math.floor(e / sp + 1e-6) + 1 for e in extent]
    origin_out = ref + out_dir @ lo
    common = {
        "size": size_out,
        "transform": sitk.Transform(3, sitk.sitkIdentity),
        "outputOrigin": tuple(float(v) for v in origin_out),
        "outputSpacing": (sp, sp, sp),
        "outputDirection": tuple(out_dir.ravel()),
    }
    res = sitk.Resample(
        oriented,
        interpolator=sitk.sitkBSpline,
        defaultPixelValue=float(cfg.hu_clip[0]),
        outputPixelType=sitk.sitkFloat32,
        **common,
    )
    res_fov = sitk.Resample(
        fov_oriented,
        interpolator=sitk.sitkNearestNeighbor,
        defaultPixelValue=1,
        outputPixelType=sitk.sitkUInt8,
        **common,
    )
    vol = sitk.GetArrayFromImage(res)
    np.clip(vol, cfg.hu_clip[0], cfg.hu_clip[1], out=vol)
    volume = np.rint(vol).astype(np.int16)
    fov_mask = sitk.GetArrayFromImage(res_fov) == 0
    volume[~fov_mask] = int(cfg.hu_clip[0])
    if not np.allclose(out_dir, np.eye(3), atol=1e-4):
        flags.append("residual_rotation")
    return IsotropicVolume(
        volume=volume,
        fov_mask=fov_mask,
        origin=tuple(float(v) for v in origin_out),
        spacing=(sp, sp, sp),
        direction=tuple(float(v) for v in out_dir.ravel()),
        flags=flags,
        info=info,
    )


def crop_box(
    mask: np.ndarray, margin: int, shape: Sequence[int] | None = None
) -> tuple[slice, ...]:
    """Bounding box of ``mask`` grown by ``margin`` voxels and clipped to the array."""
    shape = tuple(shape or mask.shape)
    if not mask.any():
        return tuple(slice(0, s) for s in shape)
    out = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        out.append(slice(max(int(hit[0]) - margin, 0), min(int(hit[-1]) + margin + 1, shape[axis])))
    return tuple(out)


def save_volume_npz(
    path: str | os.PathLike[str],
    *,
    volume: np.ndarray,
    lung_mask: np.ndarray,
    candidates: np.ndarray,
    meta: Mapping[str, Any],
) -> Path:
    """Write the processed-volume ``.npz`` atomically (compressed).

    Keys: ``volume`` (int16), ``lung_mask`` (uint8), ``candidates`` (float32, ``N x 4``:
    ``x, y, z, score`` in voxel coordinates) and ``meta`` (JSON string).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        volume=np.ascontiguousarray(volume, dtype=np.int16),
        lung_mask=np.ascontiguousarray(lung_mask, dtype=np.uint8),
        candidates=np.asarray(candidates, dtype=np.float32).reshape(-1, 4),
        meta=np.asarray(json.dumps(meta, default=_json_default)),
    )
    os.replace(tmp, out)
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
