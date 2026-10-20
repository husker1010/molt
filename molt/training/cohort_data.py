"""Load processed cohorts and pathway resources into one participant-level bundle.

:func:`load_bundle` reads one or more processed cohort directories (see
:mod:`molt.data.io`) and the pathway resources and returns a :class:`CohortBundle`:

* ``table`` - one row per participant (index ``participant_id``) with the clinical
  variables, labels, modality availability (``has_rna``, ``has_ct``), the path of the
  selected CT volume, the CT acquisition descriptors used as quality inputs and the CT
  subset (``ct_subset``; see :class:`CTSelection`: by default only scans with confirmed
  pretreatment timing and a passed quality control count as available);
* ``expression`` - per cohort, a genes x participants matrix restricted to pathway genes;
* ``rna_quality`` - per-participant sample-level RNA quality summaries computed on the
  full matrix before gene restriction (no labels involved);
* ``pathways`` - module definitions, the gene-to-module incidence matrix and the module
  adjacency prior.

All derived quantities in this module are label-free and fold-independent; every
quantity that depends on training participants is estimated by
:class:`molt.training.preprocess.Preprocessor`.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molt.data.io import cohort_paths, load_cohort
from molt.data.schema import LABELS_SPEC
from molt.data.units import expression_scale, is_relative, is_sequencing
from molt.utils.logging import get_logger

__all__ = [
    "CT_SUBSETS",
    "PATHWAY_PRIORS",
    "CTSelection",
    "CohortBundle",
    "PathwayResources",
    "analytic_sets",
    "ct_selection_frame",
    "ct_usable_counts",
    "detected_mask",
    "detection_threshold",
    "development_universe",
    "expression_scale",
    "is_relative",
    "is_sequencing",
    "load_bundle",
    "load_pathways",
    "pathway_prior_file",
    "sample_detection_thresholds",
]

log = get_logger(__name__)


@dataclass
class PathwayResources:
    """Pathway modules, incidence and adjacency aligned on a common module order.

    Attributes:
        module_ids: Module identifiers (length ``P``).
        module_names: Human-readable module names.
        genes: Gene symbols of the incidence rows (length ``G``).
        incidence: ``(G, P)`` 0/1 matrix.
        adjacency: ``(P, P)`` symmetric 0/1 matrix including self loops.
    """

    module_ids: list[str]
    module_names: list[str]
    genes: list[str]
    incidence: np.ndarray
    adjacency: np.ndarray

    @property
    def n_modules(self) -> int:
        return len(self.module_ids)


def _frame_with_index(df: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    for name in candidates:
        if name in df.columns:
            return df.set_index(name)
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        return df.set_index(first)
    return df


PATHWAY_PRIORS = ("strict", "curated_reactome", "full")
"""Module-adjacency priors listed in ``pathways/metadata.json`` (``full`` is the default)."""


def pathway_prior_file(pathways_dir: str | os.PathLike[str], prior: str | None = None) -> Path:
    """Adjacency file of a named prior (``strict``, ``curated_reactome`` or ``full``).

    The file name is read from the ``priors`` entry of ``metadata.json``; without metadata
    (or for ``prior=None``) the default ``adjacency.parquet`` is used.

    Raises:
        ValueError: For a prior that ``metadata.json`` does not list.
        FileNotFoundError: When the listed file does not exist.
    """
    root = Path(pathways_dir)
    name = prior or "full"
    meta_path = root / "metadata.json"
    priors: dict[str, Any] = {}
    if meta_path.exists():
        priors = dict(json.loads(meta_path.read_text(encoding="utf-8")).get("priors") or {})
    if not priors:
        if name != "full":
            raise ValueError(f"{meta_path} lists no pathway priors; cannot select {name!r}")
        return root / "adjacency.parquet"
    if name not in priors:
        raise ValueError(f"Unknown pathway prior {name!r}; expected one of {sorted(priors)}")
    path = root / str(priors[name].get("file", "adjacency.parquet"))
    if not path.exists():
        raise FileNotFoundError(f"Adjacency file of pathway prior {name!r} not found: {path}")
    return path


def load_pathways(
    pathways_dir: str | os.PathLike[str], prior: str | None = None
) -> PathwayResources:
    """Read ``modules.json``, ``incidence.parquet`` and the adjacency of a module prior.

    The incidence table may store genes as its index or in a ``gene`` column; the adjacency
    table may store module ids as index or in a ``module``/``id`` column. Self loops are
    added to the adjacency.

    Args:
        pathways_dir: Directory of the pathway resources.
        prior: Module-adjacency prior (``strict``, ``curated_reactome`` or ``full``; see
            :func:`pathway_prior_file`); ``None`` uses the default ``adjacency.parquet``.
            Training configurations select it with ``model.pathway_prior``.
    """
    root = Path(pathways_dir)
    modules = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    module_ids = [str(m["id"]) for m in modules]
    names = {str(m["id"]): str(m.get("name", m["id"])) for m in modules}
    incidence = _frame_with_index(pd.read_parquet(root / "incidence.parquet"), ("gene", "symbol"))
    incidence.columns = [str(c) for c in incidence.columns]
    missing = [m for m in module_ids if m not in incidence.columns]
    if missing:
        raise ValueError(f"incidence.parquet lacks module column(s) {missing[:5]}")
    incidence = incidence[module_ids].fillna(0).astype(np.float32)
    incidence.index = incidence.index.astype(str)
    incidence = incidence[incidence.sum(axis=1) > 0]
    adj_path = pathway_prior_file(root, prior) if prior else root / "adjacency.parquet"
    if adj_path.exists():
        adjacency = _frame_with_index(pd.read_parquet(adj_path), ("module", "id", "module_id"))
        adjacency.index = adjacency.index.astype(str)
        adjacency.columns = [str(c) for c in adjacency.columns]
        adj = adjacency.reindex(index=module_ids, columns=module_ids).fillna(0).to_numpy(float)
    else:
        adj = np.zeros((len(module_ids), len(module_ids)))
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return PathwayResources(
        module_ids=module_ids,
        module_names=[names[m] for m in module_ids],
        genes=incidence.index.tolist(),
        incidence=incidence.to_numpy(np.float32),
        adjacency=(adj > 0).astype(np.float32),
    )


RANK_DETECTION_QUANTILE = 0.20
"""Per-sample quantile above which a gene counts as detected on a relative scale."""


def detection_threshold(
    meta: Mapping[str, Any], values: np.ndarray, array_quantile: float = 0.25
) -> float:
    """Cohort-wide expression value regarded as detected (``>= 1`` TPM-equivalent).

    RNA-seq matrices store ``log2(x + 1)``, so 1 TPM corresponds to 1.0. Microarray
    intensities have no absolute scale; the cohort-wide ``array_quantile`` of all values
    is used as the detection floor. Relative units (z-scores, median-centred values, log
    ratios) use per-sample rank-based detection instead (:func:`sample_detection_thresholds`);
    for them the cohort-wide value is NaN.
    """
    if is_relative(meta):
        return float("nan")
    if is_sequencing(meta):
        return 1.0
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, array_quantile)) if finite.size else 0.0


def sample_detection_thresholds(
    expr: pd.DataFrame, quantile: float = RANK_DETECTION_QUANTILE
) -> pd.Series:
    """Per-sample ``quantile`` of the finite values of every column of ``expr``.

    On relative scales a gene counts as detected in a sample when its value lies above this
    within-sample threshold (rank-based detection).
    """
    values = expr.to_numpy(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        thr = np.nanquantile(values, quantile, axis=0) if values.size else np.zeros(0)
    return pd.Series(thr, index=expr.columns.astype(str), name="detection_threshold")


def detected_mask(
    values: np.ndarray, threshold: float | np.ndarray | None, *, rank_based: bool
) -> np.ndarray:
    """Boolean ``genes x samples`` detection mask (missing values are not detected).

    Args:
        values: ``(G, n)`` expression values.
        threshold: Cohort-wide threshold (absolute scales, ``value >= threshold``) or
            per-sample thresholds of shape ``(n,)`` (relative scales, ``value > threshold``).
        rank_based: Use the strict per-sample comparison.
    """
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    thr = np.asarray(1.0 if threshold is None else threshold, dtype=np.float64)
    hit = filled > thr if rank_based else filled >= thr
    return hit & finite


RNA_QUALITY_MISSING = {
    "rna_log_total": "rna_log_total_missing",
    "rna_mito_frac": "rna_mito_missing",
}
"""Quality summaries that are undefined on relative scales and their missingness indicators."""


def _rna_quality(
    expr: pd.DataFrame,
    meta: Mapping[str, Any],
    threshold: float | pd.Series,
    external: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per-sample quality summaries: log total signal, detected fraction, mito fraction, flag.

    ``units`` decides the computation (see :func:`expression_scale`): RNA-seq values are
    back-transformed to the linear scale for the library-size and mitochondrial summaries;
    array intensities give the mean log intensity; on relative scales (z-scores,
    median-centred values, log ratios) neither summary is defined, so both are NaN and
    flagged by ``rna_log_total_missing`` / ``rna_mito_missing``, and detection is rank-based.
    """
    values = expr.to_numpy(np.float64)
    finite = np.isfinite(values)
    scale = expression_scale(meta)
    n = values.shape[1]
    if scale == "sequencing":
        linear = np.where(finite, np.expm1(np.clip(values, 0, 30) * np.log(2.0)), 0.0)
        total = linear.sum(axis=0)
        log_total = np.log10(np.maximum(total, 1.0))
        mito = expr.index.astype(str).str.upper().str.startswith("MT-")
        mito_frac = linear[np.asarray(mito)].sum(axis=0) / np.maximum(total, 1.0)
    elif scale == "array":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_total = np.nanmean(np.where(finite, values, np.nan), axis=0)
        mito_frac = np.zeros(n)
    else:
        log_total = np.full(n, np.nan)
        mito_frac = np.full(n, np.nan)
    rank_based = scale == "relative"
    thr = (
        np.asarray(threshold.reindex(expr.columns.astype(str)), dtype=np.float64)
        if isinstance(threshold, pd.Series)
        else threshold
    )
    detected = detected_mask(values, thr, rank_based=rank_based).sum(axis=0) / np.maximum(
        finite.sum(axis=0), 1
    )
    out = pd.DataFrame(
        {
            "rna_log_total": log_total,
            "rna_detected_frac": detected,
            "rna_mito_frac": mito_frac,
            "rna_quality_flag": 0.0,
            "rna_log_total_missing": float(rank_based),
            "rna_mito_missing": float(rank_based),
        },
        index=pd.Index(expr.columns.astype(str), name="participant_id"),
    )
    if external is not None and not external.empty:
        ext = external.reindex(out.index)
        if "library_size" in ext:
            lib = pd.to_numeric(ext["library_size"], errors="coerce")
            out["rna_log_total"] = np.where(
                lib > 0, np.log10(lib.clip(lower=1)), out["rna_log_total"]
            )
        if "mito_fraction" in ext:
            v = pd.to_numeric(ext["mito_fraction"], errors="coerce")
            out["rna_mito_frac"] = v.fillna(out["rna_mito_frac"])
        if "detected_genes" in ext:
            v = pd.to_numeric(ext["detected_genes"], errors="coerce") / max(len(expr), 1)
            out["rna_detected_frac"] = v.fillna(out["rna_detected_frac"]).clip(0, 1)
        if "quality_flag" in ext:
            out["rna_quality_flag"] = pd.to_numeric(ext["quality_flag"], errors="coerce").fillna(0)
    return out


@dataclass
class CohortBundle:
    """Participant-level view of one or more processed cohorts.

    Attributes:
        cohorts: Cohort ids in load order.
        table: One row per participant (index ``participant_id``).
        expression: Cohort id -> genes x participants matrix (pathway genes, float32).
        expression_meta: Cohort id -> ``expression_meta.json`` content.
        detection: Cohort id -> detection threshold on the stored scale (NaN for cohorts
            with rank-based detection).
        rna_quality: Per-participant RNA quality summaries (index ``participant_id``).
        pathways: Pathway resources (``None`` when no pathway directory was given).
        detection_by_sample: Cohort id -> per-sample detection thresholds, for cohorts on a
            relative scale (z-scores, median-centred values, log ratios).
    """

    cohorts: list[str]
    table: pd.DataFrame
    expression: dict[str, pd.DataFrame] = field(default_factory=dict)
    expression_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    detection: dict[str, float] = field(default_factory=dict)
    rna_quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    pathways: PathwayResources | None = None
    detection_by_sample: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def participants(self) -> list[str]:
        return self.table.index.tolist()

    def detected(self, cohort_id: str, values: np.ndarray, ids: Sequence[str]) -> np.ndarray:
        """Detection mask ``(G, len(ids))`` of expression ``values`` of one cohort.

        Cohorts on a relative scale (see :func:`expression_scale`) use the per-sample
        thresholds of :attr:`detection_by_sample` (gene detected when above the
        within-sample 20th percentile); the others the cohort-wide :attr:`detection`.
        """
        per_sample = self.detection_by_sample.get(cohort_id)
        if per_sample is not None:
            thr = per_sample.reindex([str(i) for i in ids]).to_numpy(np.float64)
            return detected_mask(values, thr, rank_based=True)
        return detected_mask(values, self.detection.get(cohort_id, 1.0), rank_based=False)

    def subset(self, ids: Iterable[str]) -> pd.DataFrame:
        """Rows of :attr:`table` for ``ids`` (in the given order)."""
        return self.table.loc[list(ids)]


@dataclass(frozen=True)
class CTSelection:
    """Rules deciding which selected CT series enter the training table.

    The CT index (``ct_index.parquet``) holds one selected series per participant; this
    selection keeps it only when

    * ``require_qc``: the series passed the volume quality control (``qc_pass``), and
    * ``require_timing``: the scan is confirmed to precede treatment (``timing_ok``).

    With ``timing_unknown_subset`` the QC-passing scans whose timing could not be established
    (``timing_status == "unknown"``, for example TCGA-LUAD scans without a usable acquisition
    date) are admitted as well and labelled ``ct_subset = "timing_unknown"``; this is the
    sensitivity subset of the CT analyses. Scans known to fall outside the pretreatment window
    are never admitted while ``require_timing`` is set.

    Configuration keys: ``data.ct.require_timing``, ``data.ct.require_qc`` and
    ``data.ct.timing_unknown_subset``.
    """

    require_timing: bool = True
    require_qc: bool = True
    timing_unknown_subset: bool = False

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | CTSelection | None) -> CTSelection:
        """Build from a ``data.ct`` mapping (missing or ``None`` values keep the defaults)."""
        if isinstance(cfg, CTSelection):
            return cfg
        values = {
            name: bool(value)
            for name, value in dict(cfg or {}).items()
            if name in cls.__dataclass_fields__ and value is not None
        }
        return cls(**values)


CT_SUBSETS = ("primary", "timing_unknown", "timing_outside")
"""Named CT subsets: confirmed pretreatment timing, the timing-unknown sensitivity subset and
(only with ``require_timing: false``) scans outside the pretreatment window."""


def _bool_column(frame: pd.DataFrame, name: str, default: bool) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].astype("boolean").fillna(False).astype(bool)


def ct_selection_frame(
    ct_index: pd.DataFrame | None, selection: CTSelection | Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Selected series of every participant with the outcome of the CT selection rules.

    Returns one row per participant with a selected series (index ``participant_id``) and
    the boolean columns ``qc_ok``, ``timing_confirmed``, ``timing_unknown`` and ``usable``,
    plus ``ct_subset`` (``primary``, ``timing_unknown`` or ``None`` when not usable) next to
    the original index columns; ``timing_outside`` labels usable scans outside the
    pretreatment window, which only occur with ``require_timing: false``.
    """
    rules = CTSelection.from_mapping(selection)
    if ct_index is None or ct_index.empty or "selected" not in ct_index:
        cols = ["qc_ok", "timing_confirmed", "timing_unknown", "usable", "ct_subset"]
        return pd.DataFrame(columns=cols, index=pd.Index([], name="participant_id"))
    sel = ct_index[ct_index["selected"].astype("boolean").fillna(False).astype(bool)].copy()
    sel["participant_id"] = sel["participant_id"].astype(str)
    sel = sel.drop_duplicates("participant_id").set_index("participant_id")
    qc_ok = _bool_column(sel, "qc_pass", True)
    confirmed = _bool_column(sel, "timing_ok", False)
    status = (
        sel["timing_status"].astype(str).str.lower()
        if "timing_status" in sel
        else pd.Series("unknown", index=sel.index)
    )
    unknown = ~confirmed & status.isin(["unknown", "nan", "none", ""])
    qc_gate = qc_ok if rules.require_qc else pd.Series(True, index=sel.index)
    if rules.require_timing:
        timing_gate = confirmed | (unknown & rules.timing_unknown_subset)
    else:
        timing_gate = pd.Series(True, index=sel.index)
    usable = qc_gate & timing_gate
    subset = np.where(
        confirmed, "primary", np.where(unknown, "timing_unknown", "timing_outside")
    ).astype(object)
    subset[~usable.to_numpy()] = None
    sel["qc_ok"], sel["timing_confirmed"], sel["timing_unknown"] = qc_ok, confirmed, unknown
    sel["usable"], sel["ct_subset"] = usable, subset
    return sel


def _selected_ct(
    ct_index: pd.DataFrame | None,
    ct_root: Path | None,
    selection: CTSelection | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    cols = [
        "ct_path",
        "ct_qc_flags",
        "ct_spacing_z",
        "ct_contrast",
        "ct_petct",
        "ct_kernel",
        "ct_manufacturer",
        "ct_series_uid",
        "ct_subset",
        "ct_timing_confirmed",
        "ct_alt_path",
        "ct_alt_kernel",
    ]
    frame = ct_selection_frame(ct_index, selection)
    if frame.empty:
        return pd.DataFrame(columns=cols)
    sel = frame[frame["usable"].astype(bool)]

    def resolve(pid: str, raw: Any) -> str | None:
        candidates = []
        if isinstance(raw, (str, Path)) and str(raw):
            raw_path = Path(raw)
            if not raw_path.is_absolute():
                if ct_root is None:
                    raise ValueError("Relative CT paths require ct_dir")
                raw_path = ct_root / raw_path
            candidates.append(raw_path)
        if ct_root is not None:
            candidates.append(ct_root / f"{pid}.npz")
        for cand in candidates:
            if isinstance(cand, (str, Path)) and str(cand) and Path(cand).exists():
                return str(cand)
        return None

    def col(name: str, default: Any = None) -> pd.Series:
        return sel[name] if name in sel else pd.Series(default, index=sel.index, dtype=object)

    paths = [resolve(pid, raw) for pid, raw in col("npz_path").items()]
    alt = []
    for raw in col("paired_alternate_npz_path"):
        if isinstance(raw, str) and raw:
            candidate = Path(raw)
            if not candidate.is_absolute() and ct_root is not None:
                candidate = ct_root / candidate
            alt.append(str(candidate) if candidate.is_absolute() and candidate.exists() else None)
        else:
            alt.append(None)

    def spacing_z(value: Any) -> float:
        try:
            return float(str(value).replace(" ", "").split(",")[2])
        except (IndexError, ValueError):
            return float("nan")

    out = pd.DataFrame(
        {
            "ct_path": paths,
            "ct_qc_flags": col("qc_flags", "").fillna("").astype(str).to_numpy(),
            "ct_spacing_z": [spacing_z(v) for v in col("spacing_xyz")],
            "ct_contrast": pd.to_numeric(col("contrast"), errors="coerce").to_numpy(),
            "ct_petct": col("is_petct_component", False).fillna(False).astype(bool).to_numpy(),
            "ct_kernel": col("kernel").to_numpy(),
            "ct_manufacturer": col("manufacturer").to_numpy(),
            "ct_series_uid": col("series_uid").to_numpy(),
            "ct_subset": col("ct_subset").to_numpy(),
            "ct_timing_confirmed": col("timing_confirmed", False).astype(bool).to_numpy(),
            "ct_alt_path": alt,
            "ct_alt_kernel": col("paired_alternate_kernel").to_numpy(),
        },
        index=sel.index,
    )
    return out[out["ct_path"].notna()]


def ct_usable_counts(
    cohorts: Sequence[str],
    cohorts_dir: str | os.PathLike[str] | None = None,
    selection: CTSelection | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Per-cohort counts of selected, QC-passing, timing-confirmed and usable CT scans.

    Columns: ``selected``, ``qc_pass``, ``timing_confirmed``, ``qc_and_timing``,
    ``timing_unknown_qc`` (QC-passing scans with unknown timing, the sensitivity subset),
    ``usable`` (under ``selection``) and ``paired_alternate`` (usable scans with a processed
    alternate-kernel series).
    """
    rows = []
    for cohort_id in cohorts:
        path = cohort_paths(cohort_id, cohorts_dir).root / "ct_index.parquet"
        index = pd.read_parquet(path) if path.exists() else None
        frame = ct_selection_frame(index, selection)
        usable = frame["usable"].astype(bool) if len(frame) else pd.Series(dtype=bool)
        alt = (
            frame.get("paired_alternate_npz_path", pd.Series(None, index=frame.index))
            .notna()
            .astype(bool)
        )
        rows.append(
            {
                "cohort": cohort_id,
                "selected": len(frame),
                "qc_pass": int(frame["qc_ok"].sum()) if len(frame) else 0,
                "timing_confirmed": int(frame["timing_confirmed"].sum()) if len(frame) else 0,
                "qc_and_timing": (
                    int((frame["qc_ok"] & frame["timing_confirmed"]).sum()) if len(frame) else 0
                ),
                "timing_unknown_qc": (
                    int((frame["qc_ok"] & frame["timing_unknown"]).sum()) if len(frame) else 0
                ),
                "usable": int(usable.sum()),
                "paired_alternate": int((usable & alt).sum()) if len(frame) else 0,
            }
        )
    return pd.DataFrame(rows).set_index("cohort")


def _expression_columns(path: Path) -> list[str]:
    """Participant columns of an expression parquet file (read from the schema only)."""
    import pyarrow.parquet as pq

    return [c for c in pq.read_schema(path).names if c not in ("gene", "__index_level_0__")]


def load_bundle(
    cohorts: Sequence[str],
    *,
    cohorts_dir: str | os.PathLike[str] | None = None,
    ct_dir: str | os.PathLike[str] | None = None,
    pathways_dir: str | os.PathLike[str] | None = None,
    use_expression: bool = True,
    use_ct: bool = True,
    extra_genes: Sequence[str] = (),
    fill_ajcc_from_stage_group: bool = True,
    pathway_prior: str | None = None,
    ct_selection: CTSelection | Mapping[str, Any] | None = None,
) -> CohortBundle:
    """Load processed cohorts into a :class:`CohortBundle`.

    Args:
        cohorts: Cohort ids to load (participant ids must be unique across cohorts).
        cohorts_dir: Root of processed cohort directories.
        ct_dir: Root of preprocessed CT volumes.
        pathways_dir: Directory with ``modules.json``, ``incidence.parquet`` and
            ``adjacency.parquet``; required when expression is used.
        use_expression: Load expression matrices (availability flags are always filled).
        use_ct: Unused; CT availability (``has_ct``) always reflects the data on disk and the
            preprocessing decides whether CT enters the model. Kept for API stability.
        extra_genes: Genes kept in the expression matrices in addition to pathway genes.
        fill_ajcc_from_stage_group: Use ``stage_group`` where ``ajcc_stage`` is missing.
        pathway_prior: Module-adjacency prior passed to :func:`load_pathways`.
        ct_selection: CT selection rules (:class:`CTSelection` or a ``data.ct`` mapping);
            by default a selected scan needs confirmed pretreatment timing and a passed QC.
    """
    if not cohorts:
        raise ValueError("At least one cohort must be supplied")
    if use_expression and (pathways_dir is None or not Path(pathways_dir).is_dir()):
        raise ValueError("RNA requires an explicit pathways_dir containing pathway resources")
    pathways = (
        load_pathways(pathways_dir, pathway_prior)
        if pathways_dir and Path(pathways_dir).exists()
        else None
    )
    keep_genes = set(pathways.genes) | set(extra_genes) if pathways is not None else None
    tables, expressions, metas, detections, qualities = [], {}, {}, {}, []
    sample_thr: dict[str, pd.Series] = {}
    label_cols = [c.name for c in LABELS_SPEC.columns if c.name != "participant_id"]
    for cohort_id in cohorts:
        cohort = load_cohort(cohort_id, cohorts_dir, ct_dir, expression=use_expression)
        clinical = cohort.clinical.copy()
        clinical["participant_id"] = clinical["participant_id"].astype(str)
        table = clinical.set_index("participant_id")
        table["cohort"] = cohort_id
        labels = cohort.labels
        if labels is not None:
            labels = labels.copy()
            labels["participant_id"] = labels["participant_id"].astype(str)
            labels = labels.drop_duplicates("participant_id").set_index("participant_id")
            table = table.join(labels[[c for c in labels.columns if c not in table.columns]])
        for col in label_cols:
            if col not in table:
                table[col] = False if col.endswith(("_ascertained", "_available")) else np.nan
        for col in ("egfr_ascertained", "alk_ascertained", "pfi_available", "os_available"):
            table[col] = table[col].astype("boolean").fillna(False).astype(bool)

        has_rna = pd.Series(False, index=table.index)
        if use_expression and cohort.expression is not None and cohort.expression.shape[1] > 0:
            expr = cohort.expression
            expr.columns = expr.columns.astype(str)
            expr = expr.loc[:, expr.columns.isin(table.index)]
            meta = dict(cohort.expression_meta or {})
            values = expr.to_numpy(np.float32)
            thr = detection_threshold(meta, values)
            if is_relative(meta):
                sample_thr[cohort_id] = sample_detection_thresholds(expr)
            ext_q = None
            q_path = cohort.paths.root / "rna_quality.parquet"
            if q_path.exists():
                ext_q = pd.read_parquet(q_path)
                ext_q = ext_q.assign(participant_id=ext_q["participant_id"].astype(str))
                ext_q = ext_q.set_index("participant_id")
            qualities.append(_rna_quality(expr, meta, sample_thr.get(cohort_id, thr), ext_q))
            expr = expr.groupby(level=0).max() if expr.index.has_duplicates else expr
            if keep_genes is not None:
                expr = expr.loc[expr.index.isin(keep_genes)]
            expressions[cohort_id] = expr.astype(np.float32)
            metas[cohort_id] = meta
            detections[cohort_id] = thr
            has_rna = pd.Series(table.index.isin(expr.columns), index=table.index)
        elif cohort.paths.expression.exists():
            has_rna = pd.Series(
                table.index.isin(_expression_columns(cohort.paths.expression)), index=table.index
            )
        table["has_rna"] = has_rna.to_numpy()

        ct = _selected_ct(
            cohort.ct_index,
            cohort_paths(cohort_id, cohorts_dir, ct_dir).ct_root,
            ct_selection,
        )
        table = table.join(ct, how="left")
        table["has_ct"] = table["ct_path"].notna().to_numpy()
        if fill_ajcc_from_stage_group and {"ajcc_stage", "stage_group"} <= set(table.columns):
            missing = table["ajcc_stage"].isna() | table["ajcc_stage"].astype(str).isin(["NA", ""])
            table.loc[missing, "ajcc_stage"] = table.loc[missing, "stage_group"]
        tables.append(table)

    table = pd.concat(tables, axis=0)
    if table.index.has_duplicates:
        dup = table.index[table.index.duplicated()].unique().tolist()
        raise ValueError(f"Participant ids are not unique across cohorts, e.g. {dup[:5]}")
    table.index.name = "participant_id"
    if "has_clinical_record" in table:
        table["has_clinical_record"] = (
            table["has_clinical_record"].astype("boolean").fillna(False).astype(bool)
        )
    else:
        table["has_clinical_record"] = True
    rna_quality = pd.concat(qualities) if qualities else pd.DataFrame()
    log.info(
        "loaded %d participants from %s (RNA %d, CT %d)",
        len(table),
        ",".join(cohorts),
        int(table["has_rna"].sum()),
        int(table["has_ct"].sum()),
    )
    return CohortBundle(
        cohorts=list(cohorts),
        table=table,
        expression=expressions,
        expression_meta=metas,
        detection=detections,
        rna_quality=rna_quality,
        pathways=pathways,
        detection_by_sample=sample_thr,
    )


def analytic_sets(table: pd.DataFrame, *, require_modality: bool = True) -> dict[str, pd.Index]:
    """Participant ids of the analytic sets.

    * ``egfr``: EGFR-ascertained participants with a clinical record,
    * ``pfi`` / ``os``: participants with evaluable follow-up and, if ``require_modality``,
      at least one modality beyond the clinical record (RNA or a selected CT),
    * ``ct``: participants with a selected CT.
    """
    modality = table["has_rna"].astype(bool) | table["has_ct"].astype(bool)
    if not require_modality:
        modality = pd.Series(True, index=table.index)
    egfr = table["egfr_ascertained"].astype(bool) & table["has_clinical_record"].astype(bool)
    egfr &= pd.to_numeric(table["egfr_status"], errors="coerce").notna()
    pfi = table["pfi_available"].astype(bool) & modality
    pfi &= pd.to_numeric(table["pfi_time"], errors="coerce").notna()
    os_ = table["os_available"].astype(bool) & modality
    os_ &= pd.to_numeric(table["os_time"], errors="coerce").notna()
    return {
        "egfr": table.index[egfr],
        "pfi": table.index[pfi],
        "os": table.index[os_],
        "ct": table.index[table["has_ct"].astype(bool)],
    }


def development_universe(table: pd.DataFrame, *, require_modality: bool = True) -> pd.Index:
    """Participants in at least one analytic set or with a selected CT (fold universe)."""
    sets = analytic_sets(table, require_modality=require_modality)
    ids = sets["egfr"].union(sets["pfi"]).union(sets["os"]).union(sets["ct"])
    return table.index[table.index.isin(ids)]


def participant_regex(pattern: str | None) -> re.Pattern[str] | None:
    """Compile an optional participant-id pattern (helper for extra indicators)."""
    return re.compile(pattern) if pattern else None
