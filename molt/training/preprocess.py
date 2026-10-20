"""Fold-specific preprocessing of clinical, transcriptomic, imaging-quality and outcome data.

:class:`Preprocessor` is fitted on the training participants of one fold and transforms any
participants (validation, test or external) with the fitted parameters only:

* **Clinical** - continuous variables are winsorized at fixed physiological limits and
  standardized with training means and standard deviations; categorical variables use
  training vocabularies with a reserved missing token (0) and out-of-vocabulary token (1);
  systematic non-collection (a variable missing for every training participant of a site with
  at least ``systematic_min_site_size`` participants) and the absence of a clinical record are
  represented by indicators. Partially specified levels (``ever_unspecified`` smokers, stage
  groups without substage, unknown side) follow a declared hierarchy
  (:mod:`molt.training.clinical_hierarchy`: reserved coarse levels, stochastic coarsening or
  parent mixtures); structurally zero smoking quantities can be encoded as observed zeros.
* **Transcriptome** - genes are retained when detected (see below) in at least
  ``min_expr_fraction`` of the training samples, standardized gene-wise per cohort with
  training parameters, and aggregated to pathway activities ``p = D^-1 A^T x`` where ``D``
  counts the retained (and unmasked) genes of every pathway available for the sample.
  Optional target-gene masking removes genes (e.g. ``EGFR``) before ``D`` is recomputed.
  With ``sample_normalization="rank"`` each profile is replaced by the inverse-normal
  transform of its within-sample ranks before the gene-wise standardization. With
  ``gene_panel_size > 0`` a fold-filtered panel of the highest-variance genes
  (:class:`~molt.training.gene_panel.GenePanel`) is standardized as the input of the
  gene-level expert.
* **Stacked linear experts** - with ``stacking`` set, penalized linear scorers of the
  endpoint are fitted on the training participants
  (:class:`~molt.training.stacking.StackedScorers`); their scores are model inputs
  (``FeatureBundle.stacked``), cross-validated for the training participants.
* **Quality inputs** - RNA (log library size, detected-gene fraction, mitochondrial fraction,
  quality flag and, on relative scales such as z-scores where library size and mitochondrial
  fraction are undefined, their missingness indicators), CT (acquisition and segmentation
  flags) and clinical (missing fraction, systematic non-collection, no record). Gene
  detection follows the units of ``expression_meta.json``: ``>= 1`` TPM-equivalent for
  RNA-seq, a cohort-wide intensity quantile for arrays and a value above the per-sample 20th
  percentile for relative scales.
* **Outcomes** - EGFR labels of the EGFR analysis set; discrete-time survival bins of tunable
  width up to a final boundary; inverse probability of censoring weights at the primary
  horizon from a Kaplan-Meier fit of the training censoring distribution.

The fitted object holds only arrays and dictionaries and is picklable, so it can be stored
next to the model checkpoints of every fold.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from molt.calibration import KaplanMeierCensoring, ipcw_weights
from molt.training.clinical_hierarchy import (
    DEFAULT_COARSEN_HIDE,
    DEFAULT_HIERARCHY,
    CategoryHierarchy,
    apply_structural_zeros,
)
from molt.training.cohort_data import CohortBundle, is_sequencing
from molt.training.discrete_time import discretize, survival_bins
from molt.training.feature_bundle import FeatureBundle
from molt.training.gene_panel import GenePanel, GenePanelConfig, rank_inverse_normal
from molt.training.stacking import StackedScorers, parse_stacking

__all__ = [
    "MISSING",
    "OOV",
    "FeatureBundle",
    "PreprocessConfig",
    "Preprocessor",
    "discretize",
    "survival_bins",
]

MISSING = 0
OOV = 1
_NA_STRINGS = frozenset({"", "na", "nan", "none", "null", "unknown", "<na>"})

DEFAULT_NUMERIC = ("age", "pack_years", "smoking_years", "years_since_quit")
DEFAULT_CATEGORICAL = (
    "sex",
    "smoking_status",
    "histology_subtype",
    "ajcc_stage",
    "pT",
    "pN",
    "pM",
    "laterality",
    "lobe",
    "prior_malignancy",
)
DEFAULT_WINSOR = {
    "age": (18.0, 95.0),
    "pack_years": (0.0, 150.0),
    "smoking_years": (0.0, 80.0),
    "years_since_quit": (0.0, 70.0),
}
DEFAULT_CT_FLAGS = (
    "effusion",
    "collapse",
    "consolidation",
    "mask_corrected",
    "truncation",
    "low_lung_volume",
    "timing_uncertain",
)
RNA_QUALITY_COLUMNS = ("rna_log_total", "rna_detected_frac", "rna_mito_frac")
RNA_QUALITY_INDICATORS = ("rna_log_total_missing", "rna_mito_missing")
"""Missingness indicators of the RNA quality summaries that are undefined on relative scales
(z-scores, median-centred values, log ratios; see :func:`~molt.training.cohort_data.expression_scale`)."""


@dataclass
class PreprocessConfig:
    """Settings of :class:`Preprocessor` (see the module docstring)."""

    numeric: tuple[str, ...] = DEFAULT_NUMERIC
    categorical: tuple[str, ...] = DEFAULT_CATEGORICAL
    winsor_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_WINSOR)
    )
    min_category_count: int = 2
    partial_levels: str = "none"
    coarsen_p: float = 0.2
    partial_min_count: int = 10
    hierarchy: dict[str, Any] | None = None
    coarsen_hide: dict[str, Any] | None = None
    structural_zero: bool = False
    systematic_min_site_size: int = 5
    site_column: str = "site"
    use_rna: bool = True
    min_expr_fraction: float = 0.20
    mask_genes: tuple[str, ...] = ()
    exclude_genes: tuple[str, ...] = ("ALK",)
    standardize_pathways: bool = False
    external_standardization: str = "auto"
    sd_floor: float = 0.05
    z_clip: float = 8.0
    rna_quality_missing_indicators: bool = True
    sample_normalization: str = "none"
    gene_panel_size: int = 0
    gene_panel_min_mean: float | None = None
    gene_panel_forced: tuple[str, ...] = ()
    stacking: dict[str, Any] | None = None
    task: str = "egfr"
    use_ct: bool = True
    ct_flags: tuple[str, ...] = DEFAULT_CT_FLAGS
    survival_endpoint: str = "pfi"
    bin_width: float = 6.0
    max_time: float = 60.0
    horizon: float = 24.0
    require_modality_for_survival: bool = True
    domain_key: str = "site"
    domain_min_count: int = 5
    ipcw_min_prob: float = 0.05

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None = None, **overrides: Any
    ) -> PreprocessConfig:
        """Build from a (possibly nested-config) mapping, ignoring unknown keys."""
        merged = {**dict(data or {}), **overrides}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if merged.get(f.name) is None:
                continue
            value = merged[f.name]
            if f.name == "winsor_limits":
                value = {str(k): (float(v[0]), float(v[1])) for k, v in dict(value).items()}
            elif isinstance(value, list):
                value = tuple(value)
            kwargs[f.name] = value
        return cls(**kwargs)


def _normalize_category(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    text = str(value).strip()
    return None if text.lower() in _NA_STRINGS else text


def _nan_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations ignoring NaN (0 and 1 for all-missing columns)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, 0) if len(values) else np.full(values.shape[1], np.nan)
        sd = np.nanstd(values, 0) if len(values) else np.full(values.shape[1], np.nan)
    return np.nan_to_num(mean), np.nan_to_num(sd, nan=1.0) + 1e-6


def _column(table: pd.DataFrame, name: str) -> pd.Series:
    if name in table.columns:
        return table[name]
    return pd.Series(np.nan, index=table.index, dtype=object)


class Preprocessor:
    """Fit preprocessing parameters on training participants and transform any participants.

    Example:
        >>> pre = Preprocessor(PreprocessConfig(bin_width=6)).fit(bundle, train_ids)
        >>> train = pre.transform(bundle, train_ids)
        >>> test = pre.transform(bundle, test_ids)
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.fitted = False
        self._self_params: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_self_params"] = {}
        return state

    # ------------------------------------------------------------------ properties
    @property
    def n_num(self) -> int:
        return len(self.config.numeric)

    @property
    def n_cat(self) -> int:
        return len(self.config.categorical)

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 2 for c in self.config.categorical]

    @property
    def n_indicators(self) -> int:
        return len(self.systematic_vars) + 1

    @property
    def n_pathways(self) -> int:
        return int(self.incidence.shape[1]) if self.incidence is not None else 0

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_panel_genes(self) -> int:
        """Genes of the gene-level expert panel (0 when the panel is disabled)."""
        panel = getattr(self, "gene_panel", None)
        return len(panel) if panel is not None else 0

    @property
    def stacked_dims(self) -> dict[str, int]:
        """Input dimension of every stacked linear expert (empty when stacking is off)."""
        stacking = getattr(self, "stacking", None)
        return stacking.dims if stacking is not None else {}

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    @property
    def n_domains(self) -> int:
        return len(self.domain_vocab) + (1 if self.domain_other else 0)

    @property
    def quality_dims(self) -> dict[str, int]:
        return {
            "rna": len(RNA_QUALITY_COLUMNS) + 1 + len(self._rna_quality_indicators),
            "ct": len(self.config.ct_flags) + 2,
            "clinical": 3,
        }

    @property
    def quality_signs(self) -> dict[str, list[float]]:
        """Orientation of every quality input (+1 adverse when larger, -1 favourable)."""
        return {
            "rna": [-1.0, -1.0, 1.0, 1.0] + [1.0] * len(self._rna_quality_indicators),
            "ct": [1.0] * (len(self.config.ct_flags) + 2),
            "clinical": [1.0, 1.0, 1.0],
        }

    @property
    def _rna_quality_indicators(self) -> tuple[str, ...]:
        # Fixed at fit time; preprocessors fitted before the indicators existed have none.
        return RNA_QUALITY_INDICATORS if getattr(self, "rna_quality_indicators", False) else ()

    def horizon_bin(self, horizon: float | None = None) -> int:
        """Number of bins up to ``horizon`` (``S(horizon) = prod_{k < horizon_bin}(1 - h_k)``)."""
        h = self.config.horizon if horizon is None else horizon
        return int(np.searchsorted(self.bin_edges, h, side="left"))

    # ------------------------------------------------------------------ fit
    def fit(self, bundle: CohortBundle, train_ids: Sequence[str]) -> Preprocessor:
        """Estimate every fold-specific parameter from ``train_ids``."""
        cfg = self.config
        train_ids = [str(i) for i in train_ids]
        t = bundle.table.loc[train_ids]
        has_record = t["has_clinical_record"].astype(bool)

        # Clinical continuous variables.
        self.num_mean, self.num_sd = {}, {}
        for name in cfg.numeric:
            v = self._numeric(t, name)
            v = v[np.isfinite(v)]
            self.num_mean[name] = float(v.mean()) if v.size else 0.0
            sd = float(v.std()) if v.size > 1 else 1.0
            self.num_sd[name] = sd if sd > 1e-8 else 1.0

        # Categorical vocabularies (plus reserved coarse levels of the declared hierarchy).
        self.vocab: dict[str, dict[str, int]] = {}
        level_counts: dict[str, dict[str, int]] = {}
        for name in cfg.categorical:
            values = _column(t, name).map(_normalize_category).dropna()
            counts = values.value_counts()
            level_counts[name] = {str(k): int(v) for k, v in counts.items()}
            kept = sorted(counts.index[counts >= cfg.min_category_count].astype(str))
            self.vocab[name] = {v: i + 2 for i, v in enumerate(kept)}
        self.category_hierarchy: CategoryHierarchy | None = None
        if cfg.partial_levels != "none":
            declared = DEFAULT_HIERARCHY if cfg.hierarchy is None else cfg.hierarchy
            self.vocab = CategoryHierarchy.extend_vocab(self.vocab, declared)
            self.category_hierarchy = CategoryHierarchy.fit(
                self.vocab,
                level_counts,
                cfg.numeric,
                hierarchy=declared,
                hide=DEFAULT_COARSEN_HIDE if cfg.coarsen_hide is None else cfg.coarsen_hide,
                treatment=cfg.partial_levels,
                coarsen_p=cfg.coarsen_p,
                partial_min_count=cfg.partial_min_count,
                min_category_count=cfg.min_category_count,
            )

        # Systematic non-collection by site.
        site = _column(t, cfg.site_column).map(_normalize_category).to_numpy(object)
        self.systematic: dict[str, list[str]] = {}
        variables = list(cfg.numeric) + list(cfg.categorical)
        missing = self._missing_matrix(t, variables)
        for s_value in sorted({v for v in site if v is not None}):
            pos = np.flatnonzero((site == s_value) & has_record.to_numpy())
            if len(pos) < cfg.systematic_min_site_size:
                continue
            for j, name in enumerate(variables):
                if missing[pos, j].all():
                    self.systematic.setdefault(name, []).append(str(s_value))
        self.systematic_vars = [v for v in variables if v in self.systematic]

        # Domain vocabulary.
        dom = self._domain_values(t)
        counts = dom.dropna().value_counts()
        kept = sorted(counts.index[counts >= cfg.domain_min_count].astype(str))
        self.domain_vocab = {v: i for i, v in enumerate(kept)}
        self.domain_rare = sorted(counts.index[counts < cfg.domain_min_count].astype(str))
        self.domain_other = bool(self.domain_rare)

        # Transcriptome.
        self.rna_quality_indicators = bool(cfg.rna_quality_missing_indicators)
        self._fit_rna(bundle, t)
        self._fit_gene_panel(bundle, train_ids)

        # Outcomes.
        self.bin_edges = survival_bins(cfg.bin_width, cfg.max_time)
        time, event, present = self._survival_raw(t)
        self.censoring = KaplanMeierCensoring.fit(
            time[present], event[present], min_prob=cfg.ipcw_min_prob
        )
        y = self._egfr_labels(t)
        n_pos, n_neg = int(np.nansum(y == 1)), int(np.nansum(y == 0))
        self.pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        self.n_train = len(train_ids)
        self.fitted = True
        self.stacking: StackedScorers | None = None
        specs, options = parse_stacking(cfg.stacking)
        if specs:
            feats = self.transform(bundle, train_ids)
            self.stacking = StackedScorers(specs, options, seed=0)
            if self.category_hierarchy is not None:
                self.stacking.category_mixing = self.category_hierarchy.design_mixing(
                    self.cat_cardinalities
                )
            self.stacking.fit(feats, cfg.task, self.cat_cardinalities, self.n_bins)
        return self

    def _winsorize(self, name: str, values: np.ndarray) -> np.ndarray:
        lo, hi = self.config.winsor_limits.get(name, (-np.inf, np.inf))
        return np.clip(values, lo, hi)

    def _numeric(self, t: pd.DataFrame, name: str, winsorize: bool = True) -> np.ndarray:
        """Values of continuous variable ``name`` (structural zeros filled when enabled)."""
        v = pd.to_numeric(_column(t, name), errors="coerce").to_numpy(float)
        if getattr(self.config, "structural_zero", False):
            v = apply_structural_zeros(t, name, v)
        return self._winsorize(name, v) if winsorize else v

    def _missing_matrix(self, t: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
        cols = []
        for name in variables:
            if name in self.config.numeric:
                cols.append(~np.isfinite(self._numeric(t, name, winsorize=False)))
            else:
                cols.append(_column(t, name).map(_normalize_category).isna().to_numpy())
        return np.column_stack(cols) if cols else np.zeros((len(t), 0), dtype=bool)

    def _domain_values(self, t: pd.DataFrame) -> pd.Series:
        key = self.config.domain_key
        if key == "none":
            return pd.Series(None, index=t.index, dtype=object)
        site = _column(t, self.config.site_column).map(_normalize_category)
        cohort = t["cohort"].astype(str)
        if key == "site":
            return site
        if key == "cohort":
            return cohort
        if key == "cohort_site":
            return cohort + ":" + site.fillna("NA")
        raise ValueError(f"Unknown domain_key {key!r}")

    def _egfr_labels(self, t: pd.DataFrame) -> np.ndarray:
        y = pd.to_numeric(t["egfr_status"], errors="coerce").to_numpy(float)
        ok = (
            t["egfr_ascertained"].astype(bool).to_numpy()
            & t["has_clinical_record"].astype(bool).to_numpy()
        )
        return np.where(ok, y, np.nan)

    def _survival_raw(self, t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ep = self.config.survival_endpoint
        time = pd.to_numeric(_column(t, f"{ep}_time"), errors="coerce").to_numpy(float)
        event = pd.to_numeric(_column(t, f"{ep}_event"), errors="coerce").to_numpy(float)
        avail = _column(t, f"{ep}_available").astype("boolean").fillna(False).to_numpy(bool)
        present = avail & np.isfinite(time) & np.isfinite(event) & (time >= 0)
        if self.config.require_modality_for_survival:
            present &= t["has_rna"].astype(bool).to_numpy() | t["has_ct"].astype(bool).to_numpy()
        return time, event, present

    def _fit_rna(self, bundle: CohortBundle, t: pd.DataFrame) -> None:
        cfg = self.config
        self.genes: list[str] = []
        self.incidence: np.ndarray | None = None
        self.rna_params: dict[str, dict[str, Any]] = {}
        self.rna_quality_params: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pathway_mean: np.ndarray | None = None
        self.pathway_sd: np.ndarray | None = None
        self.pathway_ids: list[str] = []
        self.adjacency: np.ndarray | None = None
        pw = bundle.pathways
        if pw is None:
            return
        self.incidence = pw.incidence.copy()
        self.pathway_ids = list(pw.module_ids)
        self.adjacency = pw.adjacency.copy()
        if not cfg.use_rna:
            self.incidence = np.zeros((0, pw.n_modules), dtype=np.float32)
            return
        drop = {g.upper() for g in (*cfg.exclude_genes, *cfg.mask_genes)}
        candidates = [g for g in pw.genes if g.upper() not in drop]
        cand_index = pd.Index(candidates)
        detected = pd.Series(0.0, index=cand_index)
        observed = pd.Series(0.0, index=cand_index)
        train_rna: dict[str, list[str]] = {}
        for cohort_id, expr in bundle.expression.items():
            ids = [i for i in t.index[t["cohort"] == cohort_id] if i in expr.columns]
            if not ids:
                continue
            train_rna[cohort_id] = ids
            sub = expr.loc[expr.index.intersection(cand_index), ids]
            values = sub.to_numpy(np.float32)
            finite = np.isfinite(values)
            detected.loc[sub.index] += bundle.detected(cohort_id, values, ids).sum(1)
            observed.loc[sub.index] += finite.sum(1)
        frac = detected / observed.replace(0, np.nan)
        retained = [g for g in candidates if frac.get(g, 0.0) >= cfg.min_expr_fraction]
        self.genes = retained
        gene_pos = {g: i for i, g in enumerate(pw.genes)}
        self.incidence = (
            pw.incidence[[gene_pos[g] for g in retained]]
            if retained
            else np.zeros((0, pw.n_modules), dtype=np.float32)
        )
        for cohort_id, ids in train_rna.items():
            expr = bundle.expression[cohort_id]
            self.rna_params[cohort_id] = self._standardization(expr, ids)
            self.rna_params[cohort_id]["signature"] = self._signature(bundle, cohort_id)
            q = bundle.rna_quality.reindex(ids)[list(RNA_QUALITY_COLUMNS)].to_numpy(float)
            self.rna_quality_params[cohort_id] = _nan_moments(q)
        if cfg.standardize_pathways and self.genes:
            ids = [i for ids in train_rna.values() for i in ids]
            p, _, ok = self._pathways(bundle, ids)
            p = np.where(ok[:, None], p, np.nan)
            self.pathway_mean = np.nan_to_num(np.nanmean(p, 0))
            sd = np.nan_to_num(np.nanstd(p, 0), nan=1.0)
            self.pathway_sd = np.where(sd > 1e-6, sd, 1.0)

    def _fit_gene_panel(self, bundle: CohortBundle, train_ids: Sequence[str]) -> None:
        """Fit the fold-filtered gene panel of the gene-level expert (if enabled)."""
        cfg = self.config
        self.gene_panel: GenePanel | None = None
        if not cfg.use_rna or cfg.gene_panel_size <= 0 or not bundle.expression:
            return
        self.gene_panel = GenePanel(
            GenePanelConfig(
                size=cfg.gene_panel_size,
                min_mean=cfg.gene_panel_min_mean,
                forced=cfg.gene_panel_forced,
                excluded=(*cfg.exclude_genes, *cfg.mask_genes),
                sample_normalization=cfg.sample_normalization,
                sd_floor=cfg.sd_floor,
                z_clip=cfg.z_clip,
                external_standardization=cfg.external_standardization,
            )
        ).fit(bundle, train_ids)

    def _sample_normalize(self, values: np.ndarray) -> np.ndarray:
        """Optional per-sample rank normalization of a genes x samples block."""
        if getattr(self.config, "sample_normalization", "none") == "rank":
            return rank_inverse_normal(values, axis=0)
        return values

    @staticmethod
    def _signature(bundle: CohortBundle, cohort_id: str) -> str:
        meta = bundle.expression_meta.get(cohort_id, {})
        kind = "seq" if is_sequencing(meta) else "array"
        return f"{kind}:{meta.get('platform', 'unknown')}:{meta.get('units', '')}"

    def _standardization(self, expr: pd.DataFrame, ids: Sequence[str]) -> dict[str, Any]:
        genes = [g for g in self.genes if g in expr.index]
        values = self._sample_normalize(expr.loc[genes, list(ids)].to_numpy(np.float64))
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=1) if values.size else np.zeros(len(genes))
        sd = np.nanstd(values, axis=1) if values.size else np.ones(len(genes))
        sd = np.maximum(np.nan_to_num(sd, nan=1.0), self.config.sd_floor)
        return {"genes": genes, "mean": mean.astype(np.float32), "sd": sd.astype(np.float32)}

    def _params_for(self, bundle: CohortBundle, cohort_id: str) -> dict[str, Any]:
        if cohort_id in self.rna_params:
            return self.rna_params[cohort_id]
        mode = self.config.external_standardization
        if mode in ("auto", "train"):
            sig = self._signature(bundle, cohort_id)
            for params in self.rna_params.values():
                if params.get("signature") == sig or (mode == "train" and self.rna_params):
                    return params
        if cohort_id not in self._self_params:
            expr = bundle.expression[cohort_id]
            self._self_params[cohort_id] = self._standardization(expr, list(expr.columns))
        return self._self_params[cohort_id]

    def _gene_matrix(
        self, bundle: CohortBundle, ids: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardized expression ``(n, G)`` (NaN where unavailable) and RNA presence."""
        table = bundle.table.loc[list(ids)]
        out = np.full((len(ids), len(self.genes)), np.nan, dtype=np.float32)
        present = np.zeros(len(ids), dtype=bool)
        if not self.genes:
            return out, present
        gene_idx = {g: i for i, g in enumerate(self.genes)}
        cohorts = table["cohort"].astype(str).to_numpy()
        for cohort_id in np.unique(cohorts):
            expr = bundle.expression.get(cohort_id)
            if expr is None:
                continue
            rows = np.flatnonzero((cohorts == cohort_id) & np.isin(np.asarray(ids), expr.columns))
            if rows.size == 0:
                continue
            params = self._params_for(bundle, cohort_id)
            genes = [g for g in params["genes"] if g in expr.index]
            keep = np.isin(params["genes"], genes)
            mean, sd = params["mean"][keep], params["sd"][keep]
            values = self._sample_normalize(
                expr.loc[genes, [ids[r] for r in rows]].to_numpy(np.float32)
            ).T
            z = np.clip((values - mean) / sd, -self.config.z_clip, self.config.z_clip)
            out[np.ix_(rows, [gene_idx[g] for g in genes])] = z
            present[rows] = np.isfinite(z).any(axis=1)
        return out, present

    def _pathways(
        self, bundle: CohortBundle, ids: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, present = self._gene_matrix(bundle, ids)
        if self.incidence is None or self.incidence.shape[1] == 0:
            return np.zeros((len(ids), 0), np.float32), z, present
        mask = np.isfinite(z)
        zf = np.where(mask, z, 0.0).astype(np.float32)
        counts = mask.astype(np.float32) @ self.incidence
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (zf @ self.incidence) / counts
        p = np.where(counts > 0, p, np.nan).astype(np.float32)
        return p, z, present

    # ------------------------------------------------------------------ transform
    def transform(self, bundle: CohortBundle, ids: Sequence[str]) -> FeatureBundle:
        """Build model inputs and targets for ``ids`` with the fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        cfg = self.config
        ids = [str(i) for i in ids]
        t = bundle.table.loc[ids]
        n = len(ids)
        has_record = t["has_clinical_record"].astype(bool).to_numpy()

        num = np.zeros((n, self.n_num), np.float32)
        num_mask = np.zeros((n, self.n_num), bool)
        for j, name in enumerate(cfg.numeric):
            v = self._numeric(t, name)
            ok = np.isfinite(v) & has_record
            num[:, j] = np.where(
                ok, (np.nan_to_num(v) - self.num_mean[name]) / self.num_sd[name], 0.0
            )
            num_mask[:, j] = ok
        cat = np.zeros((n, self.n_cat), np.int64)
        for j, name in enumerate(cfg.categorical):
            values = _column(t, name).map(_normalize_category).to_numpy(object)
            if getattr(self, "category_hierarchy", None) is not None:
                # Missing values (None or NaN, depending on the storage dtype) -> missing token.
                values = np.array([None if pd.isna(v) else v for v in values], dtype=object)
            vocab = self.vocab[name]
            cat[:, j] = [MISSING if v is None else vocab.get(str(v), OOV) for v in values]
            cat[~has_record, j] = MISSING
        site = _column(t, cfg.site_column).map(_normalize_category).to_numpy(object)
        indicators = np.zeros((n, self.n_indicators), np.float32)
        for j, name in enumerate(self.systematic_vars):
            indicators[:, j] = np.isin(site.astype(str), self.systematic[name])
        indicators[:, -1] = ~has_record
        variables = list(cfg.numeric) + list(cfg.categorical)
        miss_frac = self._missing_matrix(t, variables).mean(axis=1) if variables else np.zeros(n)
        q_clin = np.column_stack(
            [
                np.where(has_record, miss_frac, 1.0),
                indicators[:, :-1].max(axis=1, initial=0.0),
                ~has_record,
            ]
        ).astype(np.float32)

        if self.n_pathways and self.genes:
            rna, genes, rna_present = self._pathways(bundle, ids)
            rna_present &= t["has_rna"].astype(bool).to_numpy()
            if self.pathway_mean is not None:
                rna = (rna - self.pathway_mean) / self.pathway_sd
            rna = np.where(rna_present[:, None], rna, 0.0).astype(np.float32)
            genes = np.where(rna_present[:, None], genes, np.nan).astype(np.float32)
        else:
            rna = np.zeros((n, self.n_pathways), np.float32)
            genes = None
            rna_present = np.zeros(n, bool)
        panel = None
        if getattr(self, "gene_panel", None) is not None:
            panel = self.gene_panel.transform(bundle, ids)
            panel = np.where(rna_present[:, None], panel, np.nan).astype(np.float32)
        q_rna = np.zeros((n, self.quality_dims["rna"]), np.float32)
        rna_flag = np.zeros(n, np.float32)
        n_q = len(RNA_QUALITY_COLUMNS)
        if len(bundle.rna_quality):
            rq = bundle.rna_quality.reindex(ids)
            raw = rq[list(RNA_QUALITY_COLUMNS)].to_numpy(float)
            cohorts = t["cohort"].astype(str).to_numpy()
            for cohort_id in np.unique(cohorts):
                rows = cohorts == cohort_id
                mean, sd = self.rna_quality_params.get(cohort_id, (None, None))
                if mean is None:
                    ref = bundle.rna_quality.loc[
                        bundle.rna_quality.index.isin(
                            bundle.table.index[bundle.table["cohort"] == cohort_id]
                        )
                    ][list(RNA_QUALITY_COLUMNS)].to_numpy(float)
                    mean, sd = _nan_moments(ref)
                q_rna[rows, :n_q] = np.nan_to_num(np.clip((raw[rows] - mean) / sd, -6, 6))
            rna_flag = np.nan_to_num(rq["rna_quality_flag"].to_numpy(float)).astype(np.float32)
            q_rna[:, n_q] = rna_flag
            for j, name in enumerate(self._rna_quality_indicators):
                col = rq[name] if name in rq.columns else pd.Series(0.0, index=rq.index)
                q_rna[:, n_q + 1 + j] = np.nan_to_num(pd.to_numeric(col, errors="coerce"))
            q_rna[~rna_present] = 0.0
            rna_flag = np.where(rna_present, rna_flag, 0.0).astype(np.float32)

        ct_present = t["has_ct"].astype(bool).to_numpy() & cfg.use_ct
        ct_path = [p if ok else None for p, ok in zip(_column(t, "ct_path").tolist(), ct_present)]
        q_ct = self._ct_quality(t, ct_present)

        domain = self._domain_index(self._domain_values(t).to_numpy(object))

        time, event, surv_present = self._survival_raw(t)
        surv_bin, surv_event, valid = discretize(time, event, self.bin_edges)
        surv_present = surv_present & valid
        status = np.full(n, np.nan, np.float32)
        status[(time <= cfg.horizon) & (event == 1) & surv_present] = 1.0
        status[(time > cfg.horizon) & surv_present] = 0.0
        w = ipcw_weights(np.where(surv_present, time, np.nan), event, cfg.horizon, self.censoring)

        out = FeatureBundle(
            ids=ids,
            cohort=t["cohort"].astype(str).to_numpy(object),
            clinical_num=num,
            clinical_num_mask=num_mask,
            clinical_cat=cat,
            clinical_indicators=indicators,
            clinical_present=has_record,
            rna=rna,
            rna_present=rna_present,
            rna_genes=genes,
            quality_rna=q_rna,
            quality_ct=q_ct,
            quality_clinical=q_clin,
            rna_flag=rna_flag,
            domain=domain,
            y_egfr=self._egfr_labels(t).astype(np.float32),
            surv_bin=surv_bin,
            surv_event=np.where(surv_present, surv_event, 0.0).astype(np.float32),
            surv_present=surv_present,
            surv_time=np.where(surv_present, time, np.nan).astype(np.float32),
            surv_status=status,
            ipcw=w.astype(np.float32),
            ct_present=ct_present,
            ct_path=ct_path,
            gene_panel=panel,
        )
        if getattr(self, "stacking", None) is not None:
            out.stacked = self.stacking.transform(out, self.cat_cardinalities)
        return out

    def _domain_index(self, values: np.ndarray) -> np.ndarray:
        """Training vocabulary index; rare training values share one index, unseen -> -1."""
        other = len(self.domain_vocab)
        rare = set(self.domain_rare)
        out = np.full(len(values), -1, dtype=np.int64)
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            key = str(v)
            if key in self.domain_vocab:
                out[i] = self.domain_vocab[key]
            elif key in rare:
                out[i] = other
        return out

    def _ct_quality(self, t: pd.DataFrame, ct_present: np.ndarray) -> np.ndarray:
        flags = self.config.ct_flags
        out = np.zeros((len(t), len(flags) + 2), np.float32)
        raw = _column(t, "ct_qc_flags").fillna("").astype(str).str.lower().tolist()
        thick = pd.to_numeric(_column(t, "ct_spacing_z"), errors="coerce").to_numpy(float)
        petct = _column(t, "ct_petct").astype("boolean").fillna(False).to_numpy(bool)
        for i, text in enumerate(raw):
            if not ct_present[i]:
                continue
            tokens = {
                tok
                for tok in text.replace("|", ";").replace(",", ";").replace(" ", ";").split(";")
                if tok
            }
            for j, flag in enumerate(flags):
                out[i, j] = float(flag in tokens)
            out[i, len(flags)] = float(bool(tokens - set(flags)) or petct[i])
            out[i, len(flags) + 1] = np.clip(np.nan_to_num(thick[i], nan=2.5) / 5.0, 0.0, 2.0)
        return out

    def summary(self) -> dict[str, Any]:
        """Compact description of the fitted parameters (for logging)."""
        return {
            "n_train": getattr(self, "n_train", 0),
            "n_genes": self.n_genes,
            "n_panel_genes": self.n_panel_genes,
            "stacked": self.stacking.summary() if getattr(self, "stacking", None) else {},
            "n_pathways": self.n_pathways,
            "cat_cardinalities": self.cat_cardinalities,
            "categories": (
                self.category_hierarchy.summary()
                if getattr(self, "category_hierarchy", None) is not None
                else {}
            ),
            "systematic_vars": self.systematic_vars,
            "n_domains": self.n_domains,
            "n_bins": self.n_bins,
            "pos_weight": self.pos_weight,
            "rna_cohorts": sorted(self.rna_params),
        }

    def with_config(self, **changes: Any) -> Preprocessor:
        """Unfitted copy with modified settings."""
        return Preprocessor(replace(self.config, **changes))
