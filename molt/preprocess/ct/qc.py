"""Visual and tabular quality control of preprocessed CT volumes.

:func:`render_montage` writes a 2 x 2 PNG per scan: axial and coronal sections through the
lung centre plus axial and sagittal sections through the top-ranked candidate, each with
the lung-mask contour (right lung red, left lung blue) and the candidate centres near the
section (top candidate in orange, others in yellow). :func:`write_qc_summary` writes the
per-scan QC table as CSV.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["QC_SUMMARY_COLUMNS", "render_montage", "write_qc_summary"]

QC_SUMMARY_COLUMNS = [
    "participant_id",
    "series_uid",
    "selected",
    "qc_pass",
    "qc_flags",
    "lung_volume_l",
    "boundary_area_fraction",
    "fov_touch_fraction",
    "large_components",
    "leak_removed_ml",
    "n_candidates",
    "seconds",
    "npz_path",
    "png_path",
]


def _contours(ax: Any, mask2d: np.ndarray) -> None:
    # contour() draws in data coordinates (x = column, y = row), which follow the axis
    # orientation set by the preceding imshow() call.
    for label, color in ((1, "#e8463a"), (2, "#3a7be8")):
        part = mask2d == label
        if part.any() and not part.all():
            ax.contour(part.astype(np.float32), levels=[0.5], colors=[color], linewidths=0.8)


def render_montage(
    path: str | os.PathLike[str],
    volume: np.ndarray,
    lung: np.ndarray,
    candidates: np.ndarray,
    *,
    title: str = "",
    subtitle: str = "",
    window: tuple[float, float] = (-1350.0, 150.0),
    dpi: int = 90,
    near_mm: float = 8.0,
) -> Path:
    """Write the QC montage of one scan.

    Args:
        path: Output PNG path.
        volume: HU volume ``(z, y, x)`` (LPS: +x patient left, +y posterior, +z superior).
        lung: Lung label map of the same shape.
        candidates: ``(N, 4)`` array of ``x, y, z, score`` (voxel coordinates).
        title: First title line.
        subtitle: Second title line (flags, volumes).
        window: Display window in HU.
        dpi: Output resolution.
        near_mm: Candidates within this distance of a section are drawn on it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mask = lung > 0
    if mask.any():
        zc, yc, xc = (round(c) for c in np.argwhere(mask).mean(0))
    else:
        zc, yc, xc = (s // 2 for s in volume.shape)
    cand = np.asarray(candidates, dtype=np.float32).reshape(-1, 4)
    z_top = int(np.clip(round(cand[0, 2]), 0, volume.shape[0] - 1)) if len(cand) else zc
    x_top = int(np.clip(round(cand[0, 0]), 0, volume.shape[2] - 1)) if len(cand) else xc
    # (name, image, labels, imshow origin, candidate columns drawn as (h, v), normal column, pos)
    panels = [
        ("axial (lung centre)", volume[zc], lung[zc], "upper", (0, 1), 2, zc),
        ("coronal", volume[:, yc, :], lung[:, yc, :], "lower", (0, 2), 1, yc),
        (
            "sagittal (top candidate)",
            volume[:, :, x_top],
            lung[:, :, x_top],
            "lower",
            (1, 2),
            0,
            x_top,
        ),
        ("axial (top candidate)", volume[z_top], lung[z_top], "upper", (0, 1), 2, z_top),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))
    for ax, (name, img, lab, origin, (h, v), normal, pos) in zip(axes.ravel(), panels):
        ax.imshow(
            img, cmap="gray", vmin=window[0], vmax=window[1], origin=origin, interpolation="nearest"
        )
        _contours(ax, lab)
        coords = cand[:, :3]
        near = np.abs(coords[:, normal] - pos) <= near_mm if len(coords) else np.zeros(0, bool)
        for rank in np.flatnonzero(near)[:12]:
            color = "#ff8c00" if rank == 0 else "#ffd400"
            ax.add_patch(
                plt.Circle(
                    (coords[rank, h], coords[rank, v]), 7, fill=False, color=color, linewidth=1.2
                )
            )
            ax.text(coords[rank, h] + 8, coords[rank, v], str(rank + 1), color=color, fontsize=7)
        ax.set_title(f"{name} @ {pos}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{title}\n{subtitle}", fontsize=9)
    fig.tight_layout()
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp.png")
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    os.replace(tmp, out)
    return out


def write_qc_summary(rows: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]) -> Path:
    """Write the per-scan QC table (CSV) with :data:`QC_SUMMARY_COLUMNS` first."""
    df = pd.DataFrame(list(rows))
    for col in QC_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[QC_SUMMARY_COLUMNS + [c for c in df.columns if c not in QC_SUMMARY_COLUMNS]]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out
