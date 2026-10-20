"""Column specifications and validators for processed cohort tables.

A processed cohort directory ``<processed>/cohorts/<cohort_id>/`` holds

* ``clinical.parquet`` - one row per participant (:data:`CLINICAL_SPEC`),
* ``labels.parquet`` - molecular and outcome labels (:data:`LABELS_SPEC`),
* ``expression.parquet`` - genes x participants matrix (:func:`validate_expression`),
* ``expression_meta.json`` - platform/units metadata (:data:`EXPRESSION_META_KEYS`),
* ``ct_index.parquet`` - one row per candidate CT series (:data:`CT_INDEX_SPEC`).

Missing categorical values may be stored either as nulls or as the literal string ``"NA"``.
Binary indicators are stored as floats with ``NaN`` for unknown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CLINICAL_SPEC",
    "COHORT_IDS",
    "CT_INDEX_SPEC",
    "EXPRESSION_META_KEYS",
    "LABELS_SPEC",
    "NA",
    "PLATFORMS",
    "ColumnSpec",
    "SchemaError",
    "TableSpec",
    "ValidationReport",
    "coerce_table",
    "empty_frame",
    "validate_clinical",
    "validate_ct_index",
    "validate_expression",
    "validate_expression_meta",
    "validate_labels",
    "validate_table",
]

NA = "NA"
"""Literal used for an explicitly unknown categorical value."""

COHORT_IDS = (
    "tcga_luad",
    "tcga_lusc",
    "nsclc_radiogenomics",
    "gse31210",
    "cptac_luad",
    "gse72094",
    "gse13213",
    "gse11969",
    "cptac_lscc",
    "oncosg_luad",
)

SEX = ("female", "male", NA)
SMOKING = (
    "never",
    "current",
    "former_le15",
    "former_gt15",
    "former_unspecified",
    "ever_unspecified",
    NA,
)
"""Smoking categories; ``ever_unspecified`` is a smoker whose current/former status is unknown."""
HISTOLOGY = ("LUAD", "LUSC", "NSCLC_other")
STAGE_GROUP = ("I", "II", "III", "IV", NA)
LATERALITY = ("left", "right", NA)
LOBE = ("upper", "middle", "lower", NA)
PLATFORMS = (
    "rnaseq_star_tpm",
    "rnaseq_rsem",
    "rnaseq_fpkm",
    "rnaseq_rpkm",
    "rnaseq_rsem_zscore",
    "affy_u133plus2",
    "affy_rsta",
    "agilent_4x44k",
    "agilent_whole_genome",
    "none",
)
PFI_ENDPOINTS = ("PFI_TCGA_CDR", "RFS", "RECURRENCE_OR_DEATH")
PT_PATTERN = r"T(X|0|is|[1-4](mi|[a-c])?)|NA"
PN_PATTERN = r"N(X|[0-3][a-c]?)|NA"
PM_PATTERN = r"M(X|0|1[a-c]?)|NA"

_KINDS = ("string", "float", "int", "bool", "binary")


class SchemaError(ValueError):
    """Raised when a table violates its schema and ``strict=True``."""


@dataclass(frozen=True)
class ColumnSpec:
    """Specification of a single column.

    Attributes:
        name: Column name.
        kind: ``string``, ``float``, ``int``, ``bool`` or ``binary`` (float in {0, 1, NaN}).
        nullable: Whether nulls/NaN are allowed.
        categories: Allowed values for string columns (nulls allowed if ``nullable``).
        pattern: Full-match regular expression for string values.
        min_value: Inclusive lower bound for numeric columns.
        max_value: Inclusive upper bound for numeric columns.
        description: Human-readable meaning of the column.
    """

    name: str
    kind: str
    nullable: bool = True
    categories: tuple[str, ...] | None = None
    pattern: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Unknown column kind {self.kind!r} for {self.name!r}")

    @property
    def pandas_dtype(self) -> str:
        """Storage dtype used by :func:`coerce_table` and :func:`empty_frame`."""
        return {
            "string": "object",
            "float": "float64",
            "binary": "float64",
            "int": "Int64",
            "bool": "bool",
        }[self.kind]


@dataclass(frozen=True)
class TableSpec:
    """Specification of a table: ordered columns plus an optional unique key."""

    name: str
    columns: tuple[ColumnSpec, ...]
    key: tuple[str, ...] = ()
    description: str = ""

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def __getitem__(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


@dataclass
class ValidationReport:
    """Outcome of a validation: hard ``errors`` and advisory ``warnings``."""

    table: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> ValidationReport:
        if self.errors:
            raise SchemaError(f"{self.table}: " + "; ".join(self.errors))
        return self

    def __bool__(self) -> bool:
        return self.ok


def _s(name: str, **kw: Any) -> ColumnSpec:
    return ColumnSpec(name, "string", **kw)


def _f(name: str, **kw: Any) -> ColumnSpec:
    return ColumnSpec(name, "float", **kw)


def _b(name: str, **kw: Any) -> ColumnSpec:
    return ColumnSpec(name, "binary", **kw)


def _bool(name: str, **kw: Any) -> ColumnSpec:
    return ColumnSpec(name, "bool", nullable=False, **kw)


CLINICAL_SPEC = TableSpec(
    name="clinical",
    key=("participant_id",),
    description="One row per participant with harmonised clinical covariates.",
    columns=(
        _s("participant_id", nullable=False, description="Participant identifier"),
        _s("cohort", nullable=False, categories=None),
        _s("site", description="Tissue source site or contributing centre"),
        _f("age", min_value=0, max_value=120, description="Age at diagnosis (years)"),
        _s("sex", categories=SEX),
        _s("smoking_status", categories=SMOKING),
        _f("pack_years", min_value=0),
        _f("smoking_years", min_value=0),
        _f("years_since_quit", min_value=0),
        _s("histology", nullable=False, categories=HISTOLOGY),
        _s("histology_subtype"),
        _s("stage_group", categories=STAGE_GROUP),
        _s("ajcc_stage", description="AJCC stage including substage, e.g. IIIA"),
        _s("pT", pattern=PT_PATTERN),
        _s("pN", pattern=PN_PATTERN),
        _s("pM", pattern=PM_PATTERN),
        _s("laterality", categories=LATERALITY),
        _s("lobe", categories=LOBE),
        _b("prior_malignancy"),
        _f("diagnosis_year", min_value=1950, max_value=2100),
        _b("neoadjuvant"),
        _b("adjuvant_recorded"),
        _s("platform", nullable=False, categories=PLATFORMS),
        _bool("has_clinical_record"),
    ),
)

LABELS_SPEC = TableSpec(
    name="labels",
    key=("participant_id",),
    description="Molecular status and time-to-event outcomes; times in months.",
    columns=(
        _s("participant_id", nullable=False),
        _b("egfr_status", description="1 = oncogenic EGFR exon 18-21 variant, 0 = wild type"),
        _s("egfr_variant"),
        _s("egfr_label_source"),
        _bool("egfr_ascertained"),
        _b("alk_status"),
        _s("alk_detail"),
        _bool("alk_ascertained"),
        _b("kras_status"),
        _f("pfi_time", min_value=0),
        _b("pfi_event"),
        _s("pfi_endpoint", categories=PFI_ENDPOINTS),
        _bool("pfi_available"),
        _f("os_time", min_value=0),
        _b("os_event"),
        _bool("os_available"),
        _f("dfi_time", min_value=0),
        _b("dfi_event"),
        _f("dss_time", min_value=0),
        _b("dss_event"),
    ),
)

CT_INDEX_SPEC = TableSpec(
    name="ct_index",
    key=("series_uid",),
    description="One row per candidate CT series with QC and selection outcome.",
    columns=(
        _s("participant_id", nullable=False),
        _s("cohort", nullable=False, categories=None),
        _s("patient_id_src", nullable=False),
        _s("study_uid", nullable=False),
        _s("series_uid", nullable=False),
        _s("study_date"),
        _s("modality"),
        _s("series_description"),
        _s("body_part"),
        ColumnSpec("n_slices", "int", min_value=0),
        _s("spacing_xyz", description="Voxel spacing 'x,y,z' in mm"),
        _s("kernel"),
        _b("contrast"),
        _s("manufacturer"),
        _bool("is_petct_component"),
        _f("scan_to_diagnosis_days", description="NaN if unknown"),
        _bool("timing_ok"),
        _bool("qc_pass"),
        _s("qc_flags"),
        _bool("selected"),
        _s("selection_reason"),
        _s("npz_path"),
    ),
)

EXPRESSION_META_KEYS = (
    "platform",
    "units",
    "n_genes",
    "n_samples",
    "sample_selection_rule",
    "source_files",
)


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna()


def _check_column(series: pd.Series, spec: ColumnSpec, max_examples: int = 5) -> list[str]:
    errors: list[str] = []
    missing = _is_missing(series)
    if not spec.nullable and missing.any():
        errors.append(f"column {spec.name!r} has {int(missing.sum())} null value(s)")
    present = series[~missing]
    if present.empty:
        return errors

    def bad(mask: pd.Series, what: str) -> None:
        if mask.any():
            examples = present[mask].astype(str).unique()[:max_examples].tolist()
            errors.append(f"column {spec.name!r}: {int(mask.sum())} {what} (e.g. {examples})")

    if spec.kind == "string":
        bad(~present.map(lambda v: isinstance(v, str)), "non-string value(s)")
        values = present.astype(str)
        if spec.categories is not None:
            bad(~values.isin(spec.categories), f"value(s) outside {list(spec.categories)}")
        if spec.pattern is not None:
            rx = re.compile(spec.pattern)
            bad(~values.map(lambda v: rx.fullmatch(v) is not None), "value(s) not matching pattern")
        return errors

    if spec.kind == "bool":
        if not pd.api.types.is_bool_dtype(series):
            bad(~present.map(lambda v: isinstance(v, (bool, np.bool_))), "non-boolean value(s)")
        return errors

    numeric = pd.to_numeric(present, errors="coerce")
    if pd.api.types.is_bool_dtype(present):
        numeric = present.astype(float)
    bad(numeric.isna(), "non-numeric value(s)")
    numeric = numeric.dropna()
    if spec.kind == "int":
        bad((numeric % 1 != 0).reindex(present.index, fill_value=False), "non-integer value(s)")
    if spec.kind == "binary":
        bad((~numeric.isin([0.0, 1.0])).reindex(present.index, fill_value=False), "non-binary")
    if spec.min_value is not None:
        bad((numeric < spec.min_value).reindex(present.index, fill_value=False), "below minimum")
    if spec.max_value is not None:
        bad((numeric > spec.max_value).reindex(present.index, fill_value=False), "above maximum")
    return errors


def validate_table(
    df: pd.DataFrame,
    spec: TableSpec,
    *,
    strict: bool = False,
    allow_extra: bool = True,
) -> ValidationReport:
    """Validate ``df`` against ``spec``.

    Args:
        df: Table to validate.
        spec: Expected schema.
        strict: Raise :class:`SchemaError` on failure instead of returning the report.
        allow_extra: Treat extra columns as warnings (``True``) or errors (``False``).
    """
    report = ValidationReport(spec.name)
    missing = [c for c in spec.names if c not in df.columns]
    if missing:
        report.errors.append(f"missing column(s) {missing}")
    extra = [c for c in df.columns if c not in spec.names]
    if extra:
        (report.warnings if allow_extra else report.errors).append(f"extra column(s) {extra}")
    if df.columns.duplicated().any():
        report.errors.append(f"duplicated column(s) {df.columns[df.columns.duplicated()].tolist()}")
    for col in spec.columns:
        if col.name in df.columns and not df.columns.duplicated().any():
            report.errors.extend(_check_column(df[col.name], col))
    if spec.key and all(k in df.columns for k in spec.key):
        dup = df.duplicated(list(spec.key), keep=False)
        if dup.any():
            report.errors.append(f"{int(dup.sum())} row(s) share a key {list(spec.key)}")
    if strict:
        report.raise_if_failed()
    return report


def validate_clinical(df: pd.DataFrame, *, strict: bool = False) -> ValidationReport:
    """Validate a ``clinical.parquet`` table, including stage/TNM consistency warnings."""
    report = validate_table(df, CLINICAL_SPEC)
    if "cohort" in df.columns and df["cohort"].nunique(dropna=True) > 1:
        report.warnings.append("more than one cohort id in a single clinical table")
    if {"stage_group", "pM"} <= set(df.columns):
        inconsistent = df["pM"].astype(str).str.startswith("M1") & df["stage_group"].isin(
            ["I", "II", "III"]
        )
        if inconsistent.any():
            report.warnings.append(f"{int(inconsistent.sum())} row(s) with M1 but stage < IV")
    if strict:
        report.raise_if_failed()
    return report


def validate_labels(df: pd.DataFrame, *, strict: bool = False) -> ValidationReport:
    """Validate a ``labels.parquet`` table, including availability/ascertainment consistency."""
    report = validate_table(df, LABELS_SPEC)
    cols = set(df.columns)

    def check(flag: str, required: Sequence[str]) -> None:
        if flag in cols and set(required) <= cols:
            avail = df[flag].astype(bool)
            incomplete = avail & df[list(required)].isna().any(axis=1)
            if incomplete.any():
                report.errors.append(
                    f"{int(incomplete.sum())} row(s) with {flag} but null {list(required)}"
                )

    check("pfi_available", ["pfi_time", "pfi_event", "pfi_endpoint"])
    check("os_available", ["os_time", "os_event"])
    for status, flag in (("egfr_status", "egfr_ascertained"), ("alk_status", "alk_ascertained")):
        if {status, flag} <= cols:
            orphan = df[status].notna() & ~df[flag].astype(bool)
            if orphan.any():
                report.errors.append(f"{int(orphan.sum())} row(s) with {status} but not {flag}")
    if strict:
        report.raise_if_failed()
    return report


def validate_expression(
    df: pd.DataFrame,
    *,
    participants: Iterable[str] | None = None,
    min_genes: int = 1,
    strict: bool = False,
) -> ValidationReport:
    """Validate a genes x participants expression matrix.

    Args:
        df: Matrix with gene symbols as index and participant ids as columns.
        participants: If given, every column must be one of these participant ids.
        min_genes: Minimum number of genes expected.
        strict: Raise :class:`SchemaError` on failure.
    """
    report = ValidationReport("expression")
    if df.shape[0] < min_genes:
        report.errors.append(f"only {df.shape[0]} gene(s), expected >= {min_genes}")
    if df.index.has_duplicates:
        report.errors.append(f"{int(df.index.duplicated().sum())} duplicated gene symbol(s)")
    if df.index.isna().any() or (df.index.astype(str).str.strip() == "").any():
        report.errors.append("empty gene symbol(s) in index")
    if df.columns.has_duplicates:
        report.errors.append(
            f"{int(df.columns.duplicated().sum())} duplicated participant column(s)"
        )
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        report.errors.append(f"{len(non_numeric)} non-numeric column(s), e.g. {non_numeric[:3]}")
    else:
        values = df.to_numpy(dtype=float, copy=False)
        if np.isinf(values).any():
            report.errors.append("infinite value(s) present")
        frac_nan = float(np.isnan(values).mean()) if values.size else 0.0
        if frac_nan > 0.2:
            report.warnings.append(f"{frac_nan:.1%} missing values")
        if values.size and np.nanmax(values) > 100:
            report.warnings.append("values above 100: matrix may not be log2-transformed")
    if participants is not None:
        unknown = sorted(set(map(str, df.columns)) - set(map(str, participants)))
        if unknown:
            report.errors.append(
                f"{len(unknown)} column(s) not in clinical table, e.g. {unknown[:3]}"
            )
    if strict:
        report.raise_if_failed()
    return report


def validate_expression_meta(
    meta: Mapping[str, Any], expression: pd.DataFrame | None = None, *, strict: bool = False
) -> ValidationReport:
    """Validate ``expression_meta.json`` and, optionally, its agreement with the matrix."""
    report = ValidationReport("expression_meta")
    missing = [k for k in EXPRESSION_META_KEYS if k not in meta]
    if missing:
        report.errors.append(f"missing key(s) {missing}")
    if "platform" in meta and meta["platform"] not in PLATFORMS:
        report.errors.append(f"unknown platform {meta['platform']!r}")
    if expression is not None:
        for key, actual in (("n_genes", expression.shape[0]), ("n_samples", expression.shape[1])):
            if key in meta and int(meta[key]) != actual:
                report.errors.append(f"{key}={meta[key]} but matrix has {actual}")
    if strict:
        report.raise_if_failed()
    return report


def validate_ct_index(df: pd.DataFrame, *, strict: bool = False) -> ValidationReport:
    """Validate ``ct_index.parquet``; at most one selected series per participant."""
    report = validate_table(df, CT_INDEX_SPEC)
    if {"participant_id", "selected"} <= set(df.columns):
        counts = df.loc[df["selected"].astype(bool), "participant_id"].value_counts()
        multi = counts[counts > 1]
        if not multi.empty:
            report.errors.append(f"{len(multi)} participant(s) with more than one selected series")
    if {"selected", "qc_pass"} <= set(df.columns):
        bad = df["selected"].astype(bool) & ~df["qc_pass"].astype(bool)
        if bad.any():
            report.warnings.append(f"{int(bad.sum())} selected series failed QC")
    if strict:
        report.raise_if_failed()
    return report


def empty_frame(spec: TableSpec) -> pd.DataFrame:
    """Return an empty data frame with the columns and storage dtypes of ``spec``."""
    return pd.DataFrame({c.name: pd.Series(dtype=c.pandas_dtype) for c in spec.columns})


def coerce_table(df: pd.DataFrame, spec: TableSpec, *, add_missing: bool = True) -> pd.DataFrame:
    """Cast columns to the storage dtypes of ``spec`` and order them as specified.

    Missing columns are added (null, or ``False`` for bool columns) when ``add_missing``;
    extra columns are kept after the specified ones.
    """
    out = df.copy()
    for col in spec.columns:
        if col.name not in out.columns:
            if not add_missing:
                continue
            out[col.name] = False if col.kind == "bool" else np.nan
        if col.kind in ("float", "binary"):
            out[col.name] = pd.to_numeric(out[col.name], errors="coerce").astype("float64")
        elif col.kind == "int":
            out[col.name] = pd.to_numeric(out[col.name], errors="coerce").round().astype("Int64")
        elif col.kind == "bool":
            out[col.name] = out[col.name].fillna(False).astype(bool)
        else:
            out[col.name] = out[col.name].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    ordered = [c for c in spec.names if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]
