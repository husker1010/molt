"""Load and save processed cohort directories.

Layout of ``<cohorts_dir>/<cohort_id>/``::

    clinical.parquet        one row per participant
    labels.parquet          molecular status and outcomes
    expression.parquet      genes x participants (stored with a ``gene`` column)
    expression_meta.json    platform, units and provenance of the expression matrix
    ct_index.parquet        candidate CT series and selection outcome
    README.md               free-text provenance notes

CT volumes live separately under ``<ct_dir>/<cohort_id>/<participant_id>.npz``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molt.data import schema

__all__ = [
    "DEFAULT_COHORTS_DIR",
    "DEFAULT_CT_DIR",
    "Cohort",
    "CohortPaths",
    "cohort_paths",
    "list_cohorts",
    "load_clinical",
    "load_cohort",
    "load_ct_index",
    "load_ct_volume",
    "load_expression",
    "load_expression_meta",
    "load_labels",
    "save_clinical",
    "save_ct_index",
    "save_expression",
    "save_json",
    "save_labels",
    "write_parquet",
]

DEFAULT_COHORTS_DIR = None
DEFAULT_CT_DIR = None
_GENE_COL = "gene"


@dataclass(frozen=True)
class CohortPaths:
    """File locations of one processed cohort."""

    cohort_id: str
    root: Path
    ct_root: Path | None

    @property
    def clinical(self) -> Path:
        return self.root / "clinical.parquet"

    @property
    def labels(self) -> Path:
        return self.root / "labels.parquet"

    @property
    def expression(self) -> Path:
        return self.root / "expression.parquet"

    @property
    def expression_meta(self) -> Path:
        return self.root / "expression_meta.json"

    @property
    def ct_index(self) -> Path:
        return self.root / "ct_index.parquet"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    def ct_volume(self, participant_id: str) -> Path:
        """Path of the preprocessed CT volume of ``participant_id``."""
        if self.ct_root is None:
            raise ValueError("ct_dir must be supplied to resolve CT volumes")
        return self.ct_root / f"{participant_id}.npz"

    def available(self) -> dict[str, bool]:
        """Which component files exist on disk."""
        return {
            name: getattr(self, name).exists()
            for name in ("clinical", "labels", "expression", "expression_meta", "ct_index")
        }


def cohort_paths(
    cohort_id: str,
    cohorts_dir: str | os.PathLike[str] | None = None,
    ct_dir: str | os.PathLike[str] | None = None,
) -> CohortPaths:
    """Return :class:`CohortPaths` for ``cohort_id`` under explicitly supplied roots."""
    if cohorts_dir is None:
        raise ValueError("cohorts_dir must be supplied explicitly")
    root = Path(cohorts_dir).expanduser().resolve() / cohort_id
    ct_root = Path(ct_dir).expanduser().resolve() / cohort_id if ct_dir else None
    return CohortPaths(cohort_id, root, ct_root)


def list_cohorts(cohorts_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Cohort ids that have a ``clinical.parquet`` under ``cohorts_dir``."""
    if cohorts_dir is None:
        raise ValueError("cohorts_dir must be supplied explicitly")
    root = Path(cohorts_dir)
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/clinical.parquet"))


def _maybe_validate(report: schema.ValidationReport, validate: bool) -> None:
    if validate:
        report.raise_if_failed()


def load_clinical(path: str | os.PathLike[str], *, validate: bool = False) -> pd.DataFrame:
    """Read ``clinical.parquet``; optionally validate it against :data:`schema.CLINICAL_SPEC`."""
    df = pd.read_parquet(path)
    _maybe_validate(schema.validate_clinical(df), validate)
    return df


def load_labels(path: str | os.PathLike[str], *, validate: bool = False) -> pd.DataFrame:
    """Read ``labels.parquet``; optionally validate it against :data:`schema.LABELS_SPEC`."""
    df = pd.read_parquet(path)
    _maybe_validate(schema.validate_labels(df), validate)
    return df


def load_expression(
    path: str | os.PathLike[str],
    *,
    genes: list[str] | None = None,
    participants: list[str] | None = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Read a genes x participants expression matrix.

    Args:
        path: ``expression.parquet``.
        genes: Optional subset of gene symbols (rows); unknown symbols are skipped.
        participants: Optional subset of participant columns to read.
        validate: Validate the matrix with :func:`schema.validate_expression`.
    """
    columns = None if participants is None else [_GENE_COL, *participants]
    df = pd.read_parquet(path, columns=columns)
    if _GENE_COL in df.columns:
        df = df.set_index(_GENE_COL)
    df.index.name = _GENE_COL
    if genes is not None:
        df = df.loc[df.index.intersection(pd.Index(genes), sort=False)]
    _maybe_validate(schema.validate_expression(df), validate)
    return df


def load_expression_meta(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read ``expression_meta.json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ct_index(path: str | os.PathLike[str], *, validate: bool = False) -> pd.DataFrame:
    """Read ``ct_index.parquet``; optionally validate it."""
    df = pd.read_parquet(path)
    _maybe_validate(schema.validate_ct_index(df), validate)
    return df


@contextlib.contextmanager
def _atomic_target(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a unique temporary path next to ``path`` and move it into place on success.

    The temporary file lives in the destination directory so that :func:`os.replace` is an
    atomic rename: readers see either the previous file or the complete new one, never a
    partial write, and concurrent writers cannot collide on the temporary name. The
    temporary file is removed if writing fails.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.chmod(tmp, 0o644)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_parquet(df: pd.DataFrame, path: str | os.PathLike[str], *, index: bool = False) -> Path:
    """Write ``df`` to ``path`` atomically (temporary file in the same directory, then rename)."""
    with _atomic_target(path) as tmp:
        df.to_parquet(tmp, index=index)
    return Path(path)


def _write_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
    return write_parquet(df, path)


def save_json(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write ``payload`` as indented JSON (atomically)."""
    with _atomic_target(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return Path(path)


def save_clinical(df: pd.DataFrame, path: str | os.PathLike[str], *, validate: bool = True) -> Path:
    """Coerce to :data:`schema.CLINICAL_SPEC`, validate and write ``clinical.parquet``."""
    df = schema.coerce_table(df, schema.CLINICAL_SPEC)
    _maybe_validate(schema.validate_clinical(df), validate)
    return _write_parquet(df, path)


def save_labels(df: pd.DataFrame, path: str | os.PathLike[str], *, validate: bool = True) -> Path:
    """Coerce to :data:`schema.LABELS_SPEC`, validate and write ``labels.parquet``."""
    df = schema.coerce_table(df, schema.LABELS_SPEC)
    _maybe_validate(schema.validate_labels(df), validate)
    return _write_parquet(df, path)


def save_ct_index(df: pd.DataFrame, path: str | os.PathLike[str], *, validate: bool = True) -> Path:
    """Coerce to :data:`schema.CT_INDEX_SPEC`, validate and write ``ct_index.parquet``."""
    df = schema.coerce_table(df, schema.CT_INDEX_SPEC)
    _maybe_validate(schema.validate_ct_index(df), validate)
    return _write_parquet(df, path)


def save_expression(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    meta: Mapping[str, Any] | None = None,
    *,
    validate: bool = True,
) -> Path:
    """Write a genes x participants matrix (float32) and, if given, ``expression_meta.json``.

    ``n_genes`` and ``n_samples`` in ``meta`` are filled from the matrix when absent.
    """
    df = df.astype("float32")
    df.columns = [str(c) for c in df.columns]
    _maybe_validate(schema.validate_expression(df), validate)
    out = _write_parquet(df.rename_axis(_GENE_COL).reset_index(), path)
    if meta is not None:
        meta = {"n_genes": int(df.shape[0]), "n_samples": int(df.shape[1]), **dict(meta)}
        _maybe_validate(schema.validate_expression_meta(meta, df), validate)
        save_json(meta, out.with_name("expression_meta.json"))
    return out


@dataclass
class Cohort:
    """In-memory view of a processed cohort; absent components are ``None``."""

    cohort_id: str
    paths: CohortPaths
    clinical: pd.DataFrame
    labels: pd.DataFrame | None = None
    expression: pd.DataFrame | None = None
    expression_meta: dict[str, Any] = field(default_factory=dict)
    ct_index: pd.DataFrame | None = None

    @property
    def participants(self) -> list[str]:
        return self.clinical["participant_id"].astype(str).tolist()

    def table(self) -> pd.DataFrame:
        """Clinical table left-joined with labels on ``participant_id``."""
        if self.labels is None:
            return self.clinical.copy()
        return self.clinical.merge(self.labels, on="participant_id", how="left")

    def modality_mask(self) -> pd.DataFrame:
        """Per-participant availability of clinical, expression and CT data."""
        ids = pd.Index(self.participants, name="participant_id")
        expr = set(map(str, self.expression.columns)) if self.expression is not None else set()
        ct: set[str] = set()
        if self.ct_index is not None and "selected" in self.ct_index:
            ct = set(self.ct_index.loc[self.ct_index["selected"].astype(bool), "participant_id"])
        has_clin = (
            self.clinical["has_clinical_record"].astype(bool).to_numpy()
            if "has_clinical_record" in self.clinical
            else np.ones(len(ids), dtype=bool)
        )
        return pd.DataFrame(
            {
                "clinical": has_clin,
                "expression": ids.isin(expr),
                "ct": ids.isin(ct),
            },
            index=ids,
        )


def load_cohort(
    cohort_id: str,
    cohorts_dir: str | os.PathLike[str] | None = None,
    ct_dir: str | os.PathLike[str] | None = None,
    *,
    expression: bool = True,
    validate: bool = False,
) -> Cohort:
    """Load every available component of a processed cohort.

    Args:
        cohort_id: Cohort identifier (see :data:`schema.COHORT_IDS`).
        cohorts_dir: Required root of processed cohorts.
        ct_dir: Root of preprocessed CT volumes.
        expression: Load the expression matrix (can be large).
        validate: Validate each component and raise on schema violations.

    Raises:
        FileNotFoundError: If ``clinical.parquet`` does not exist.
    """
    paths = cohort_paths(cohort_id, cohorts_dir, ct_dir)
    if not paths.clinical.exists():
        raise FileNotFoundError(f"No clinical table for cohort {cohort_id!r} at {paths.clinical}")
    return Cohort(
        cohort_id=cohort_id,
        paths=paths,
        clinical=load_clinical(paths.clinical, validate=validate),
        labels=load_labels(paths.labels, validate=validate) if paths.labels.exists() else None,
        expression=(
            load_expression(paths.expression, validate=validate)
            if expression and paths.expression.exists()
            else None
        ),
        expression_meta=(
            load_expression_meta(paths.expression_meta) if paths.expression_meta.exists() else {}
        ),
        ct_index=(
            load_ct_index(paths.ct_index, validate=validate) if paths.ct_index.exists() else None
        ),
    )


def load_ct_volume(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a preprocessed CT ``.npz`` (``volume``, ``lung_mask``, ``candidates``, ``meta``)."""
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {k: data[k] for k in data.files}
    if "meta" in out:
        raw = out["meta"]
        out["meta"] = json.loads(raw.item() if isinstance(raw, np.ndarray) else str(raw))
    return out
