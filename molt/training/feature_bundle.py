"""Container of model inputs and targets produced by :class:`~molt.training.preprocess.Preprocessor`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

__all__ = ["FeatureBundle"]


@dataclass
class FeatureBundle:
    """Model inputs and targets for an ordered list of participants (NumPy arrays)."""

    ids: list[str]
    cohort: np.ndarray
    clinical_num: np.ndarray
    clinical_num_mask: np.ndarray
    clinical_cat: np.ndarray
    clinical_indicators: np.ndarray
    clinical_present: np.ndarray
    rna: np.ndarray
    rna_present: np.ndarray
    rna_genes: np.ndarray | None
    quality_rna: np.ndarray
    quality_ct: np.ndarray
    quality_clinical: np.ndarray
    rna_flag: np.ndarray
    domain: np.ndarray
    y_egfr: np.ndarray
    surv_bin: np.ndarray
    surv_event: np.ndarray
    surv_present: np.ndarray
    surv_time: np.ndarray
    surv_status: np.ndarray
    ipcw: np.ndarray
    ct_present: np.ndarray
    ct_path: list[str | None]
    gene_panel: np.ndarray | None = None
    stacked: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, index: Sequence[int] | np.ndarray) -> FeatureBundle:
        """Subset by positional index."""
        idx = np.asarray(index, dtype=int)
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                out[f.name] = None
            elif isinstance(value, dict):
                out[f.name] = {k: v[idx] for k, v in value.items()}
            elif isinstance(value, list):
                out[f.name] = [value[i] for i in idx]
            else:
                out[f.name] = value[idx]
        return FeatureBundle(**out)
