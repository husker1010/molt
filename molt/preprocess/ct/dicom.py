"""DICOM header scanning and per-series geometry summaries.

:func:`scan_series` reads the headers of every file of one series (pixel data is not
loaded), orders the slices along the slice normal and summarises acquisition and
geometry attributes needed for eligibility rules and volume assembly.
:func:`scan_collection` applies it to all series of a TCIA collection laid out as
``<collection>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm``.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "HEADER_TAGS",
    "format_dicom_date",
    "parse_patient_age",
    "scan_collection",
    "scan_series",
    "series_directory",
]

HEADER_TAGS = [
    "SOPInstanceUID",
    "InstanceNumber",
    "AcquisitionNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "ImageType",
    "NumberOfFrames",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelSpacing",
    "SliceThickness",
    "Rows",
    "Columns",
    "Modality",
    "PatientID",
    "PatientAge",
    "PatientPosition",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "FrameOfReferenceUID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
    "ConvolutionKernel",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "XRayTubeCurrent",
    "Exposure",
    "CTDIvol",
    "ReconstructionDiameter",
    "PixelPaddingValue",
]
"""Header attributes read from every file (``specific_tags`` of :func:`pydicom.dcmread`)."""

_SERIES_TEXT = (
    "Modality",
    "PatientID",
    "PatientAge",
    "PatientPosition",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "FrameOfReferenceUID",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
    "ConvolutionKernel",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "Manufacturer",
    "ManufacturerModelName",
)


def _text(ds: Any, name: str) -> str | None:
    value = ds.get(name) if ds is not None else None
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        text = "\\".join(str(v).strip() for v in value if str(v).strip())
    else:
        text = str(value).strip()
    return text or None


def _float(ds: Any, name: str) -> float | None:
    value = ds.get(name) if ds is not None else None
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            value = value[0]
        out = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    return out if math.isfinite(out) else None


def _vector(ds: Any, name: str, size: int) -> np.ndarray | None:
    value = ds.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return arr if arr.size == size and np.all(np.isfinite(arr)) else None


def format_dicom_date(value: str | None) -> str | None:
    """``YYYYMMDD`` (or ``YYYY-MM-DD ...``) -> ``YYYY-MM-DD``; ``None`` if unparseable."""
    if not value:
        return None
    digits = "".join(ch for ch in str(value)[:10] if ch.isdigit())
    if len(digits) < 8:
        return None
    y, m, d = digits[:4], digits[4:6], digits[6:8]
    if not (1800 <= int(y) <= 2200 and 1 <= int(m) <= 12 and 1 <= int(d) <= 31):
        return None
    return f"{y}-{m}-{d}"


def parse_patient_age(value: str | None) -> tuple[float, float] | None:
    """Interval ``[lo, hi)`` in years implied by a DICOM ``PatientAge`` (``nnnD|W|M|Y``).

    Ages are reported in completed units, so ``"069Y"`` means an age in ``[69, 70)`` years.
    """
    if not value:
        return None
    text = str(value).strip().upper()
    unit = text[-1] if text and text[-1] in "DWMY" else "Y"
    digits = text[:-1] if text and text[-1] in "DWMY" else text
    try:
        n = float(digits)
    except ValueError:
        return None
    scale = {"D": 1 / 365.25, "W": 7 / 365.25, "M": 1 / 12, "Y": 1.0}[unit]
    if n <= 0 and unit == "Y":
        return None
    return n * scale, (n + 1) * scale


def series_directory(root: str | os.PathLike[str], row: Mapping[str, Any]) -> Path:
    """Directory of a series given a catalogue row (``PatientID``, study and series UIDs)."""
    return (
        Path(root)
        / str(row["PatientID"])
        / str(row["StudyInstanceUID"])
        / str(row["SeriesInstanceUID"])
    )


def _read_headers(files: Sequence[Path]) -> tuple[list[tuple[Path, Any]], list[str]]:
    import pydicom

    out, errors = [], []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HEADER_TAGS)
        except Exception as exc:  # noqa: BLE001 - unreadable files are reported, not raised
            errors.append(f"{path.name}: {type(exc).__name__}")
            continue
        out.append((path, ds))
    return out, errors


def _choose_slices(
    items: list[tuple[Path, Any]], flags: list[str]
) -> tuple[list[tuple[Path, Any, np.ndarray]], np.ndarray | None]:
    """Keep files of the dominant orientation and acquisition with one file per position."""
    oriented = []
    for path, ds in items:
        iop = _vector(ds, "ImageOrientationPatient", 6)
        ipp = _vector(ds, "ImagePositionPatient", 3)
        if iop is None or ipp is None:
            continue
        oriented.append((path, ds, ipp, tuple(np.round(iop, 3))))
    if not oriented:
        return [], None
    iop_counts = Counter(o[3] for o in oriented)
    if len(iop_counts) > 1:
        flags.append("mixed_orientation")
    main_iop = iop_counts.most_common(1)[0][0]
    keep = [o for o in oriented if o[3] == main_iop]
    iop = np.asarray(main_iop, dtype=np.float64)
    normal = np.cross(iop[:3], iop[3:])
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return [], None
    normal /= norm

    def pos_key(o: tuple) -> float:
        return round(float(np.dot(o[2], normal)), 2)

    n_unique = len({pos_key(o) for o in keep})
    if n_unique < len(keep):
        acq = Counter(_text(o[1], "AcquisitionNumber") for o in keep)
        if len(acq) > 1:
            groups = {a: [o for o in keep if _text(o[1], "AcquisitionNumber") == a] for a in acq}
            best = max(groups.values(), key=lambda g: (len({pos_key(o) for o in g}), len(g)))
            if len({pos_key(o) for o in best}) == len(best):
                flags.append("multiple_acquisitions")
                keep = best
        if len({pos_key(o) for o in keep}) < len(keep):
            flags.append("duplicate_positions")
            seen: dict[float, tuple] = {}
            for o in sorted(keep, key=lambda o: _float(o[1], "InstanceNumber") or 0.0):
                seen.setdefault(pos_key(o), o)
            keep = list(seen.values())
    keep.sort(key=lambda o: float(np.dot(o[2], normal)))
    return [(p, ds, ipp) for p, ds, ipp, _ in keep], iop


def scan_series(series_dir: str | os.PathLike[str], max_gap_ratio: float = 1.5) -> dict[str, Any]:
    """Summarise one DICOM series from its headers.

    Args:
        series_dir: Directory holding the ``*.dcm`` files of the series.
        max_gap_ratio: Largest slice gap, relative to the median gap, regarded as regular.

    Returns:
        A JSON-serialisable dictionary. ``files`` lists the selected slices ordered along the
        slice normal; ``header_flags`` lists anomalies; ``error`` is set when the series
        cannot be summarised.
    """
    series_dir = Path(series_dir)
    out: dict[str, Any] = {"series_dir": str(series_dir), "error": None, "header_flags": []}
    files = sorted(p for p in series_dir.glob("*.dcm") if p.is_file())
    out["n_files"] = len(files)
    if not files:
        out["error"] = "missing_files"
        return out
    items, errors = _read_headers(files)
    out["n_unreadable"] = len(errors)
    if not items:
        out["error"] = "unreadable"
        return out
    first = items[0][1]
    for name in _SERIES_TEXT:
        out[_snake(name)] = _text(first, name)
    out["study_date"] = format_dicom_date(_text(first, "StudyDate"))
    out["series_date"] = format_dicom_date(_text(first, "SeriesDate"))
    out["acquisition_date"] = format_dicom_date(_text(first, "AcquisitionDate"))
    out["series_number"] = _float(first, "SeriesNumber")
    for name in ("KVP", "XRayTubeCurrent", "Exposure", "CTDIvol", "ReconstructionDiameter"):
        out[_snake(name)] = _float(first, name)
    out["pixel_padding_value"] = _float(first, "PixelPaddingValue")
    image_types: set[str] = set()
    for _, ds in items:
        image_types.update(str(v).upper() for v in (ds.get("ImageType") or []))
    out["image_type"] = sorted(image_types)
    out["multiframe"] = any((_float(ds, "NumberOfFrames") or 1) > 1 for _, ds in items)
    out["has_rescale"] = all(
        _float(ds, "RescaleSlope") is not None and _float(ds, "RescaleIntercept") is not None
        for _, ds in items
    )
    slopes = {(_float(ds, "RescaleSlope"), _float(ds, "RescaleIntercept")) for _, ds in items}
    out["rescale_varies"] = len(slopes) > 1

    flags: list[str] = out["header_flags"]
    if errors:
        flags.append("unreadable_files")
    slices, iop = _choose_slices(items, flags)
    out["n_slices"] = len(slices)
    out["files"] = [str(p) for p, _, _ in slices]
    if iop is None or len(slices) < 2:
        out["error"] = out["error"] or ("no_geometry" if iop is None else None)
        out["iop"] = iop.tolist() if iop is not None else None
        _inplane(out, slices[0][1] if slices else first)
        return out
    normal = np.cross(iop[:3], iop[3:])
    normal /= np.linalg.norm(normal)
    positions = np.asarray([ipp for _, _, ipp in slices])
    proj = positions @ normal
    gaps = np.diff(proj)
    median_gap = float(np.median(gaps))
    out["iop"] = iop.tolist()
    out["normal"] = normal.tolist()
    out["normal_z"] = float(abs(normal[2]))
    out["slice_positions"] = np.round(positions, 4).tolist()
    out["first_position"] = positions[0].tolist()
    out["last_position"] = positions[-1].tolist()
    out["slice_spacing"] = median_gap
    out["max_gap"] = float(gaps.max())
    out["min_gap"] = float(gaps.min())
    out["spacing_irregular"] = bool(
        gaps.max() > max_gap_ratio * median_gap + 0.01 or gaps.min() < median_gap / max_gap_ratio
    )
    span = positions[-1] - positions[0]
    along = float(span @ normal)
    shear = span - along * normal
    out["tilt_deg"] = float(math.degrees(math.atan2(np.linalg.norm(shear), abs(along))))
    out["z_extent_mm"] = float(abs(along))
    _inplane(out, slices[0][1])
    return out


def _inplane(out: dict[str, Any], ds: Any) -> None:
    spacing = _vector(ds, "PixelSpacing", 2)
    out["pixel_spacing"] = spacing.tolist() if spacing is not None else None
    out["slice_thickness"] = _float(ds, "SliceThickness")
    out["rows"] = _float(ds, "Rows")
    out["columns"] = _float(ds, "Columns")


def _snake(name: str) -> str:
    special = {
        "KVP": "kvp",
        "CTDIvol": "ctdi_vol",
        "XRayTubeCurrent": "tube_current",
        "PatientID": "patient_id",
        "StudyInstanceUID": "study_uid",
        "SeriesInstanceUID": "series_uid",
        "FrameOfReferenceUID": "frame_of_reference_uid",
        "BodyPartExamined": "body_part",
        "ConvolutionKernel": "kernel",
        "ManufacturerModelName": "manufacturer_model",
    }
    if name in special:
        return special[name]
    chars = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            chars.append("_")
        chars.append(ch.lower())
    return "".join(chars)


def _scan_job(args: tuple[str, float]) -> dict[str, Any]:
    return scan_series(args[0], args[1])


def scan_collection(
    collection_dir: str | os.PathLike[str],
    catalogue: Iterable[Mapping[str, Any]],
    *,
    modalities: Sequence[str] = ("CT",),
    workers: int = 32,
    max_gap_ratio: float = 1.5,
) -> list[dict[str, Any]]:
    """Scan the headers of every catalogued series of the given modalities.

    Args:
        collection_dir: Collection root directory.
        catalogue: NBIA ``getSeries`` rows (``PatientID``, ``StudyInstanceUID``,
            ``SeriesInstanceUID``, ``Modality``, ...).
        modalities: Modalities to scan.
        workers: Worker processes.
        max_gap_ratio: Passed to :func:`scan_series`.

    Returns:
        One summary per catalogued series, in catalogue order. The catalogue row is kept
        under ``catalogue``.
    """
    rows = [r for r in catalogue if r.get("Modality") in modalities]
    jobs = [(str(series_directory(collection_dir, r)), max_gap_ratio) for r in rows]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_scan_job, jobs, chunksize=1))
    else:
        results = [_scan_job(j) for j in jobs]
    for row, res in zip(rows, results):
        res["catalogue"] = dict(row)
        res.setdefault("series_uid", row.get("SeriesInstanceUID"))
        res.setdefault("study_uid", row.get("StudyInstanceUID"))
        res.setdefault("patient_id", row.get("PatientID"))
        if res.get("series_uid") is None:
            res["series_uid"] = row.get("SeriesInstanceUID")
        if res.get("study_uid") is None:
            res["study_uid"] = row.get("StudyInstanceUID")
        if res.get("patient_id") is None:
            res["patient_id"] = row.get("PatientID")
    return results
