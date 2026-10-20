"""Training-fold parametric empirical-Bayes batch correction (ComBat) of expression profiles.

The correction follows Johnson, Li and Rabinovic (2007) without covariates: genes are
standardized with the pooled training mean and the pooled within-batch variance, the batch
location ``gamma`` and scale ``delta^2`` of every gene are estimated per batch and shrunk
towards gene-wide priors (normal prior for ``gamma``, inverse-gamma prior for ``delta^2``,
hyperparameters by the method of moments) by the usual fixed-point iteration, and profiles
are mapped back with ``x* = (z - gamma*) / delta* * sigma + mu``.

All parameters are estimated label-free on the *training* participants of a fold.
Participants of other splits whose batch occurs in the training participants receive that
batch's training parameters; participants of batches that are absent from the training set
(or have fewer than ``min_batch_size`` training profiles) are left uncorrected.
:func:`fold_bundle` applies the correction to a shallow copy of a
:class:`~molt.training.cohort_data.CohortBundle`, so every downstream step (pathway
activities, gene panel, stacked experts) of that fold sees the corrected matrix.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molt.utils.logging import get_logger

__all__ = ["ComBat", "fold_bundle", "load_batches"]

log = get_logger(__name__)


def _aprior(delta2: np.ndarray) -> float:
    m, s2 = float(np.mean(delta2)), float(np.var(delta2, ddof=1))
    return (2 * s2 + m**2) / s2


def _bprior(delta2: np.ndarray) -> float:
    m, s2 = float(np.mean(delta2)), float(np.var(delta2, ddof=1))
    return (m * s2 + m**3) / s2


@dataclass
class ComBat:
    """Parametric empirical-Bayes location/scale batch correction (genes x samples).

    Attributes:
        min_batch_size: Batches with fewer training profiles are not corrected.
        max_iter: Maximum fixed-point iterations of the empirical-Bayes estimates.
        tol: Relative convergence tolerance of the iteration.
    """

    min_batch_size: int = 2
    max_iter: int = 200
    tol: float = 1e-4
    mean_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    gamma_: dict[str, np.ndarray] = field(default_factory=dict)
    delta_: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, x: np.ndarray, batch: Sequence[Any]) -> ComBat:
        """Estimate the parameters on training profiles ``x`` (genes x samples)."""
        x = np.asarray(x, dtype=np.float64)
        b = np.asarray(batch, dtype=object).astype(str)
        labels, counts = np.unique(b, return_counts=True)
        usable = [lab for lab, c in zip(labels, counts) if c >= self.min_batch_size]
        self.mean_ = np.nanmean(x, axis=1)
        resid = np.zeros_like(x)
        for lab in labels:
            m = b == lab
            resid[:, m] = x[:, m] - np.nanmean(x[:, m], axis=1, keepdims=True)
        var = np.nansum(resid**2, axis=1) / max(x.shape[1] - labels.size, 1)
        self.sd_ = np.sqrt(np.maximum(var, 1e-12))
        z = (x - self.mean_[:, None]) / self.sd_[:, None]
        self.gamma_, self.delta_ = {}, {}
        if len(usable) < 2:
            log.warning("ComBat: fewer than two usable batches; no correction is applied")
            return self
        g_hat = np.column_stack([np.nanmean(z[:, b == lab], axis=1) for lab in usable])
        d_hat = np.column_stack([np.nanvar(z[:, b == lab], axis=1, ddof=1) for lab in usable])
        d_hat = np.maximum(d_hat, 1e-8)
        gamma_bar, tau2 = g_hat.mean(axis=0), g_hat.var(axis=0, ddof=1)
        for j, lab in enumerate(usable):
            zj = z[:, b == lab]
            n = zj.shape[1]
            a, bb = _aprior(d_hat[:, j]), _bprior(d_hat[:, j])
            g_old, d_old = g_hat[:, j].copy(), d_hat[:, j].copy()
            for _ in range(self.max_iter):
                g_new = (tau2[j] * n * g_hat[:, j] + d_old * gamma_bar[j]) / (tau2[j] * n + d_old)
                ss = np.nansum((zj - g_new[:, None]) ** 2, axis=1)
                d_new = (bb + 0.5 * ss) / (n / 2.0 + a - 1.0)
                change = max(
                    np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
                    np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)),
                )
                g_old, d_old = g_new, d_new
                if change < self.tol:
                    break
            self.gamma_[lab], self.delta_[lab] = g_old, np.maximum(d_old, 1e-8)
        return self

    def transform(self, x: np.ndarray, batch: Sequence[Any]) -> np.ndarray:
        """Correct profiles ``x`` (genes x samples) of fitted batches; others are unchanged."""
        if self.mean_ is None or self.sd_ is None:
            raise RuntimeError("ComBat must be fitted before transform")
        x = np.asarray(x, dtype=np.float64)
        out = x.copy()
        b = np.asarray(batch, dtype=object).astype(str)
        for lab, gamma in self.gamma_.items():
            m = b == lab
            if not m.any():
                continue
            z = (x[:, m] - self.mean_[:, None]) / self.sd_[:, None]
            z = (z - gamma[:, None]) / np.sqrt(self.delta_[lab])[:, None]
            out[:, m] = z * self.sd_[:, None] + self.mean_[:, None]
        return out


def load_batches(
    cohorts_dir: str | Path, cohorts: Sequence[str], column: str = "plate"
) -> pd.Series:
    """Batch label per participant from each cohort's ``rna_quality.parquet``."""
    parts = []
    for cohort in cohorts:
        path = Path(cohorts_dir) / cohort / "rna_quality.parquet"
        if not path.exists():
            continue
        q = pd.read_parquet(path, columns=["participant_id", column])
        parts.append(q.set_index("participant_id")[column].astype(str).rename(cohort))
    if not parts:
        return pd.Series(dtype=object)
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def fold_bundle(
    bundle: Any,
    train_ids: Sequence[str],
    batches: Mapping[str, str] | pd.Series,
    min_batch_size: int = 2,
) -> Any:
    """Shallow copy of ``bundle`` whose expression matrices are ComBat-corrected per cohort.

    Args:
        bundle: Cohort bundle (``expression``: cohort -> genes x participants frame).
        train_ids: Participants whose profiles estimate the correction.
        batches: Batch label per participant.
        min_batch_size: Minimum number of training profiles of a corrected batch.

    Returns:
        A copy of ``bundle`` with corrected ``expression``; every other attribute is shared.
    """
    batches = pd.Series(batches)
    out = copy.copy(bundle)
    out.expression = dict(bundle.expression)
    train = set(map(str, train_ids))
    for cohort, expr in bundle.expression.items():
        cols = [c for c in expr.columns if c in batches.index]
        tr = [c for c in cols if c in train]
        if len(tr) < 2 * min_batch_size:
            continue
        values = expr[cols].to_numpy(np.float64)
        tr_idx = [cols.index(c) for c in tr]
        # genes that vary in the training profiles (constant genes are left unchanged)
        finite = np.isfinite(values).all(axis=1) & (np.std(values[:, tr_idx], axis=1) > 1e-3)
        model = ComBat(min_batch_size=min_batch_size).fit(
            values[finite][:, tr_idx], batches.loc[tr].to_numpy()
        )
        corrected = values.copy()
        corrected[finite] = model.transform(values[finite], batches.loc[cols].to_numpy())
        new = expr.copy()
        new.loc[:, cols] = corrected.astype(expr.dtypes.iloc[0] if expr.shape[1] else np.float32)
        out.expression[cohort] = new
    return out
