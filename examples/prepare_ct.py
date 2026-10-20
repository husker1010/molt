"""Resample a DICOM series and package it with a caller-supplied aligned lung mask."""

from __future__ import annotations

import argparse
import numpy as np
from molt.preprocess.ct.dicom import scan_series
from molt.preprocess.ct.volume import load_isotropic_volume, save_volume_npz
from molt.preprocess.ct.config import VolumeConfig, LungConfig, CandidateConfig
from molt.preprocess.ct.lung import correct_lung_mask, lung_qc
from molt.preprocess.ct.candidates import detect_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-dir", required=True)
    parser.add_argument(
        "--lung-mask", required=True, help="NPY labels aligned to the resampled 1 mm LPS volume"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    headers = scan_series(args.series_dir)
    if headers.get("error"):
        raise ValueError(headers["error"])
    result = load_isotropic_volume(
        headers["files"],
        iop=headers["iop"],
        positions=headers["slice_positions"],
        pixel_spacing=headers["pixel_spacing"],
        cfg=VolumeConfig(thorax_roi=False),
    )
    raw = np.load(args.lung_mask, allow_pickle=False)
    if raw.shape != result.volume.shape or not np.isin(raw, [0, 1, 2]).all():
        raise ValueError("Mask must match the resampled volume and contain labels 0, 1 and 2")
    mask, correction = correct_lung_mask(raw, result.volume, LungConfig())
    flags, quality = lung_qc(mask, result.fov_mask, LungConfig())
    candidates, _, _ = detect_candidates(result.volume, mask, CandidateConfig())
    save_volume_npz(
        args.output,
        volume=result.volume,
        lung_mask=mask,
        candidates=candidates,
        meta={
            "spacing": result.spacing,
            "origin": result.origin,
            "direction": result.direction,
            "array_axes": "zyx",
            "qc_flags": flags,
            "quality": quality,
            "correction": correction,
        },
    )
    print(f"Saved {args.output}; QC flags: {flags}")


if __name__ == "__main__":
    main()
