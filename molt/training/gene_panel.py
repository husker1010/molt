"""Fold-filtered gene panel for the gene-level transcriptomic expert.

:class:`GenePanel` selects, on the training participants of one fold, the ``size`` genes
with the highest expression variance among genes whose mean expression reaches the cohort's
detection threshold (``log2(TPM + 1) >= 1`` for RNA-seq), optionally adds prespecified genes
(``forced``, e.g. ``EGFR``) and standardizes every gene with training means and standard
deviations. Variances are pooled within cohorts, so platform offsets between development
cohorts do not drive the selection. Participants of cohorts not seen in training are
standardized with the training parameters of a cohort on the same platform when one exists
and otherwise with their own cohort-wide parameters (label-free), matching the pathway
inputs of :class:`~molt.training.preprocess.Preprocessor`.

With ``sample_normalization="rank"`` every profile is first replaced by the inverse-normal
transform of its within-sample ranks (``Phi^-1((r - 0.5) / n)`` over the panel genes
measured in that sample), which removes sample-level scale and platform-specific intensity
distributions before the gene-wise standardization.

Helpers:

* :func:`rank_inverse_normal` - the per-sample rank transform;
* :func:`expression_gene_names` - gene symbols of a processed expression file (read from the
  file without loading the matrix), used to keep the full gene universe in a cohort bundle.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

from molt.training.cohort_data import CohortBundle, is_sequencing

__all__ = [
    "SAMPLE_NORMALIZATIONS",
    "GenePanel",
    "GenePanelConfig",
    "bundle_gene_universe",
    "expression_gene_names",
    "rank_inverse_normal",
]

SAMPLE_NORMALIZATIONS = ("none", "rank")
"""Accepted values of the per-sample normalization option."""


def rank_inverse_normal(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Per-sample inverse-normal rank transform along ``axis`` (the gene axis).

    Ties receive their average rank; NaN entries stay NaN and are excluded from ``n``.

    Args:
        values: Expression matrix.
        axis: Axis holding the genes of one sample (0 for genes x samples).

    Returns:
        ``Phi^-1((rank - 0.5) / n)`` with the shape of ``values`` (float32).
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    ranks = rankdata(x, axis=axis, nan_policy="omit")
    n = np.isfinite(x).sum(axis=axis, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (ranks - 0.5) / n
    out = ndtri(np.clip(q, 1e-6, 1.0 - 1e-6))
    return np.where(np.isfinite(x), out, np.nan).astype(np.float32)


def expression_gene_names(path: str | os.PathLike[str]) -> list[str]:
    """Gene symbols (row labels) of a processed ``expression.parquet`` file.

    Reads only the gene column (or the stored index) of the file.
    """
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    column = next((c for c in ("gene", "__index_level_0__") if c in names), None)
    if column is None:
        return pd.read_parquet(path).index.astype(str).tolist()
    return [str(g) for g in pq.read_table(path, columns=[column]).column(0).to_pylist()]


@dataclass
class GenePanelConfig:
    """Settings of :class:`GenePanel`.

    Attributes:
        size: Number of highest-variance genes (0 disables the panel).
        min_mean: Minimum training mean on the stored scale; ``None`` uses the cohort's
            detection threshold (1.0 = 1 TPM for ``log2(TPM + 1)`` RNA-seq; no mean filter on
            relative scales such as z-scores, where the threshold is undefined).
        forced: Genes added irrespective of the variance filter (when measured).
        excluded: Genes never used (e.g. target-gene masking).
        sample_normalization: ``"none"`` or ``"rank"`` (see the module docstring).
        sd_floor: Lower bound of the gene-wise standard deviation.
        z_clip: Absolute clip of standardized values.
        external_standardization: ``"auto"`` (same-platform training parameters, otherwise
            the cohort's own), ``"train"`` (always training parameters) or ``"self"``.
    """

    size: int = 0
    min_mean: float | None = None
    forced: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    sample_normalization: str = "none"
    sd_floor: float = 0.05
    z_clip: float = 8.0
    external_standardization: str = "auto"

    def __post_init__(self) -> None:
        if self.sample_normalization not in SAMPLE_NORMALIZATIONS:
            raise ValueError(
                f"sample_normalization must be one of {SAMPLE_NORMALIZATIONS}, "
                f"got {self.sample_normalization!r}"
            )
        self.forced = tuple(str(g) for g in self.forced)
        self.excluded = tuple(str(g) for g in self.excluded)


def _signature(bundle: CohortBundle, cohort_id: str) -> str:
    meta = bundle.expression_meta.get(cohort_id, {})
    kind = "seq" if is_sequencing(meta) else "array"
    return f"{kind}:{meta.get('platform', 'unknown')}:{meta.get('units', '')}"


class GenePanel:
    """Training-fold gene selection and standardization (see the module docstring).

    Example:
        >>> panel = GenePanel(GenePanelConfig(size=2000, forced=("EGFR",))).fit(bundle, train_ids)
        >>> x = panel.transform(bundle, test_ids)          # (n, len(panel.genes)), NaN = no RNA
    """

    def __init__(self, config: GenePanelConfig) -> None:
        self.config = config
        self.genes: list[str] = []
        self.params: dict[str, dict[str, Any]] = {}
        self._self_params: dict[str, dict[str, Any]] = {}
        self.fitted = False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_self_params"] = {}
        return state

    def __len__(self) -> int:
        return len(self.genes)

    # ------------------------------------------------------------------ fit
    def fit(self, bundle: CohortBundle, train_ids: Sequence[str]) -> GenePanel:
        """Select and standardize the panel on ``train_ids`` (participants without RNA ignored)."""
        cfg = self.config
        table = bundle.table.loc[[str(i) for i in train_ids]]
        excluded = {g.upper() for g in cfg.excluded}
        stats: list[pd.DataFrame] = []
        train_rna: dict[str, list[str]] = {}
        for cohort_id, expr in bundle.expression.items():
            ids = [i for i in table.index[table["cohort"] == cohort_id] if i in expr.columns]
            if len(ids) < 2:
                continue
            train_rna[cohort_id] = ids
            values = expr[ids].to_numpy(np.float64)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(values, axis=1)
                var = np.nanvar(values, axis=1)
            floor = cfg.min_mean if cfg.min_mean is not None else bundle.detection.get(cohort_id)
            floor = -np.inf if floor is None or not np.isfinite(floor) else float(floor)
            stats.append(
                pd.DataFrame(
                    {
                        "n": float(len(ids)),
                        "var": np.nan_to_num(var),
                        "ok": (np.nan_to_num(mean, nan=-np.inf) >= floor)
                        & (np.nan_to_num(var) > 1e-8),
                    },
                    index=expr.index.astype(str),
                )
            )
        self.genes = []
        if stats:
            merged = pd.concat(stats, keys=range(len(stats)), names=["block", "gene"])
            merged = merged.reset_index("block", drop=True)
            weighted = (merged["var"] * merged["n"]).groupby(level=0).sum()
            pooled = weighted / merged["n"].groupby(level=0).sum()
            ok = merged["ok"].groupby(level=0).all()
            keep = pooled[ok & ~pooled.index.str.upper().isin(excluded)]
            ranked = keep.sort_values(ascending=False, kind="stable").index.tolist()
            selected = ranked[: max(int(cfg.size), 0)]
            universe = set(pooled.index)
            for gene in cfg.forced:
                if gene.upper() not in excluded and gene in universe and gene not in selected:
                    selected.append(gene)
            self.genes = selected
        self.params = {}
        for cohort_id, ids in train_rna.items():
            self.params[cohort_id] = {
                **self._standardization(bundle.expression[cohort_id], ids),
                "signature": _signature(bundle, cohort_id),
            }
        self.n_train = sum(len(v) for v in train_rna.values())
        self.fitted = True
        return self

    def _values(self, expr: pd.DataFrame, genes: Sequence[str], ids: Sequence[str]) -> np.ndarray:
        """``(n, len(genes))`` raw (or rank-transformed) values; missing genes are NaN."""
        present = [g for g in genes if g in expr.index]
        out = np.full((len(ids), len(genes)), np.nan, dtype=np.float32)
        if not present or not len(ids):
            return out
        block = expr.loc[present, list(ids)].to_numpy(np.float32)
        if self.config.sample_normalization == "rank":
            block = rank_inverse_normal(block, axis=0)
        pos = {g: i for i, g in enumerate(genes)}
        out[:, [pos[g] for g in present]] = block.T
        return out

    def _standardization(self, expr: pd.DataFrame, ids: Sequence[str]) -> dict[str, Any]:
        values = self._values(expr, self.genes, ids).astype(np.float64)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(values, axis=0) if values.size else np.zeros(len(self.genes))
            sd = np.nanstd(values, axis=0) if values.size else np.ones(len(self.genes))
        mean = np.nan_to_num(mean)
        sd = np.maximum(np.nan_to_num(sd, nan=1.0), self.config.sd_floor)
        return {"mean": mean.astype(np.float32), "sd": sd.astype(np.float32)}

    def _params_for(self, bundle: CohortBundle, cohort_id: str) -> dict[str, Any]:
        if cohort_id in self.params:
            return self.params[cohort_id]
        mode = self.config.external_standardization
        if mode in ("auto", "train") and self.params:
            sig = _signature(bundle, cohort_id)
            for params in self.params.values():
                if params.get("signature") == sig or mode == "train":
                    return params
        if cohort_id not in self._self_params:
            expr = bundle.expression[cohort_id]
            self._self_params[cohort_id] = self._standardization(expr, list(expr.columns))
        return self._self_params[cohort_id]

    # ------------------------------------------------------------------ transform
    def transform(self, bundle: CohortBundle, ids: Sequence[str]) -> np.ndarray:
        """Standardized panel ``(n, len(genes))``; rows without an expression profile are NaN."""
        if not self.fitted:
            raise RuntimeError("GenePanel.transform called before fit")
        ids = [str(i) for i in ids]
        out = np.full((len(ids), len(self.genes)), np.nan, dtype=np.float32)
        if not self.genes or not ids:
            return out
        cohorts = bundle.table.loc[ids, "cohort"].astype(str).to_numpy()
        id_arr = np.asarray(ids)
        for cohort_id in np.unique(cohorts):
            expr = bundle.expression.get(cohort_id)
            if expr is None:
                continue
            rows = np.flatnonzero((cohorts == cohort_id) & np.isin(id_arr, expr.columns))
            if rows.size == 0:
                continue
            params = self._params_for(bundle, cohort_id)
            values = self._values(expr, self.genes, [ids[r] for r in rows])
            z = (values - params["mean"]) / params["sd"]
            out[rows] = np.clip(z, -self.config.z_clip, self.config.z_clip)
        return out

    def summary(self) -> dict[str, Any]:
        """Compact description of the fitted panel."""
        cfg = self.config
        return {
            "n_genes": len(self.genes),
            "forced_in_panel": [g for g in cfg.forced if g in self.genes],
            "sample_normalization": cfg.sample_normalization,
            "cohorts": sorted(self.params),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> GenePanel:
        """Unfitted panel from a configuration mapping (unknown keys ignored)."""
        names = set(GenePanelConfig.__dataclass_fields__)
        values = {k: v for k, v in dict(data or {}).items() if k in names and v is not None}
        for key in ("forced", "excluded"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(GenePanelConfig(**values))


def bundle_gene_universe(
    cohorts: Sequence[str], cohort_root: str | os.PathLike[str] | None
) -> list[str]:
    """Union of the gene symbols stored in the expression files of ``cohorts``."""
    from molt.data.io import cohort_paths

    genes: dict[str, None] = {}
    for cohort_id in cohorts:
        path = Path(cohort_paths(cohort_id, cohort_root).expression)
        if path.exists():
            genes.update(dict.fromkeys(expression_gene_names(path)))
    return list(genes)
