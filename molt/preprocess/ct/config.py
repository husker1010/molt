"""Configuration for reusable CT preprocessing operations."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VolumeConfig:
    """Volume assembly and resampling.

    Attributes:
        spacing_mm: Isotropic output voxel size.
        hu_clip: HU range of the stored volume.
        padding_hu: Values below this (before clipping) are treated as outside the
            field of view.
        antialias: Smooth axes that are downsampled before B-spline interpolation.
        tilt_tolerance_deg: Gantry tilt (slice shear) above which the shear is corrected.
        max_tilt_deg: Gantry tilt above which the series fails geometry QC.
        crop_margin_mm: Margin kept around the lung bounding box when cropping.
        thorax_roi: Locate the lungs on every few native slices and resample only that region
            (plus ``roi_margin_mm``); avoids interpolating large fields of view.
        roi_margin_mm: Margin around the located lungs.
        roi_slice_step_mm: Approximate slice distance used to locate the lungs.
        max_extent_mm: Largest accepted physical extent along any axis.
        sitk_threads: SimpleITK threads per worker process.
    """

    spacing_mm: float = 1.0
    hu_clip: tuple[float, float] = (-1024.0, 600.0)
    padding_hu: float = -1500.0
    antialias: bool = True
    tilt_tolerance_deg: float = 0.5
    max_tilt_deg: float = 30.0
    crop_margin_mm: float = 40.0
    thorax_roi: bool = True
    roi_margin_mm: float = 50.0
    roi_slice_step_mm: float = 5.0
    max_extent_mm: float = 2000.0
    sitk_threads: int = 4


@dataclass
class LungConfig:
    """Lung segmentation, mask correction and lung QC.

    Attributes:
        weights: Optional local segmentation state supplied by the caller.
        batch_size: Axial slices per GPU batch.
        device: Torch device; ``"auto"`` selects CUDA when available.
        gpu_concurrency: Worker processes allowed to run GPU inference at the same time.
        min_volume_l: Lower bound of a plausible total lung volume (litres).
        max_volume_l: Upper bound of a plausible total lung volume (litres).
        min_component_fraction: Components smaller than this fraction of the mask are removed.
        large_component_fraction: Components above this fraction count as large (topology QC).
        large_component_min_ml: ... and above this absolute volume.
        leak_max_hu: Components with a mean HU below this value that lie below the lung bases
            are treated as sub-diaphragmatic gas.
        leak_opening_mm: Opening radius used to detach gas from the lung bases.
        outside_margin_mm: Components centred farther than this outside the bounding box of
            the two main lungs are removed.
        closing_mm: Closing radius of the final per-lung cleanup.
        truncation_area_fraction: Lung area in a boundary slice, relative to the largest
            axial lung area, above which the lung counts as truncated along z.
        fov_touch_fraction: Fraction of lung voxels adjacent to the field-of-view border
            above which the lung counts as truncated in-plane.
    """

    weights: str | None = None
    batch_size: int = 32
    device: str = "auto"
    gpu_concurrency: int = 4
    min_volume_l: float = 1.5
    max_volume_l: float = 9.0
    min_component_fraction: float = 0.005
    large_component_fraction: float = 0.05
    large_component_min_ml: float = 50.0
    leak_max_hu: float = -920.0
    leak_opening_mm: float = 2.0
    outside_margin_mm: float = 10.0
    closing_mm: float = 2.0
    truncation_area_fraction: float = 0.25
    fov_touch_fraction: float = 0.01


@dataclass
class CandidateConfig:
    """Candidate abnormality detection inside the lung field.

    Attributes:
        dense_hu: HU threshold of dense tissue.
        opening_mm: Radius of the opening applied to the dense-tissue mask.
        region_closing_mm: Closing radius used to include juxta-pleural lesions.
        region_fill: Hole filling of the search region: ``"3d"`` adds cavities enclosed by
            lung, ``"none"`` keeps the closed lung mask.
        min_volume_mm3: Smallest component considered.
        max_components: Largest components analysed (by volume).
        shell_mm: Thickness of the parenchymal shell used for the HU contrast.
        vessel_shape_ratio: Components whose inscribed radius divided by the
            equivalent-sphere radius is below this value are tubular/branching.
        vessel_elongation: Components with a principal-axis ratio above this value ...
        vessel_max_radius_mm: ... and an inscribed radius below this value are tubular.
        n_candidates: Number of candidate centres kept.
        entropy_bin_hu: Histogram bin width for the HU entropy feature.
    """

    dense_hu: float = -300.0
    opening_mm: float = 1.5
    region_closing_mm: float = 6.0
    region_fill: str = "3d"
    min_volume_mm3: float = 30.0
    max_components: int = 1500
    shell_mm: int = 3
    vessel_shape_ratio: float = 0.35
    vessel_elongation: float = 3.0
    vessel_max_radius_mm: float = 4.0
    n_candidates: int = 64
    entropy_bin_hu: float = 25.0


@dataclass
class QCConfig:
    """Quality-control outputs.

    Attributes:
        hard_flags: Flags that make a series fail QC (and trigger the next-ranked series).
        montage: Write a PNG montage for every selected scan.
        montage_dpi: Resolution of the montage.
    """

    hard_flags: tuple[str, ...] = (
        "processing_error",
        "geometry",
        "no_lung",
        "lung_volume_low",
        "lung_volume_high",
        "truncated_z",
        "truncated_fov",
        "topology",
    )
    montage: bool = True
    montage_dpi: int = 90
