"""Stacked linear experts: penalized linear scorers fitted inside every training fold.

A stacked expert is a convex, penalized linear model of the endpoint (elastic-net logistic
model for EGFR, penalized discrete-time hazard model for progression; see
:mod:`molt.training.convex`) fitted on the training participants of one fold. Its linear
predictor (or the contribution of each input block) is passed to the network as the input
of a dedicated Gaussian expert of the product-of-experts fusion, which learns its mean map
and precision together with the other experts. The network therefore learns the fusion,
calibration and context, while the high-dimensional linear signal is estimated by convex
optimisation.

Training rows receive their *cross-validated* score (from the K-fold search of the penalty,
at the chosen penalty), so that the network is trained on scores with the same
out-of-sample behaviour as the scores of validation, test and external participants, which
receive the score of the refitted model.

Expert names, their inputs and availability:

========================  ==========================================  =====================
expert                    inputs (blocks)                             available with
========================  ==========================================  =====================
``gene_lin``              ``genes`` (gene panel), optional others     RNA profile
``rna_lin``               ``pathways`` and/or ``clinical``            RNA profile
``clinical_lin``          ``clinical``                                clinical record
========================  ==========================================  =====================

Configuration (``data.stacking``)::

    stacking:
      folds: 5                   # K of the penalty search (cross-validated scores)
      n_lambdas: 12
      lambda_ratio: 0.01
      components: total          # total (one score) | blocks (one score per input block)
      experts:
        gene_lin: {inputs: [genes], alphas: [0.5, 0.9]}
        clinical_lin: {inputs: [clinical], alphas: [0.5]}
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from molt.training.convex import ConvexFit, cv_path
from molt.utils.logging import get_logger

__all__ = [
    "STACKED_INPUTS",
    "StackedExpertSpec",
    "StackedScorers",
    "clinical_design",
    "parse_stacking",
]

log = get_logger(__name__)

STACKED_INPUTS = ("genes", "pathways", "clinical")
"""Input blocks of a stacked scorer."""

_RNA_INPUTS = frozenset({"genes", "pathways"})


@dataclass(frozen=True)
class StackedExpertSpec:
    """Settings of one stacked expert (see the module docstring)."""

    name: str
    inputs: tuple[str, ...]
    alphas: tuple[float, ...] = (0.5,)

    @property
    def needs_rna(self) -> bool:
        return bool(_RNA_INPUTS & set(self.inputs))


def parse_stacking(cfg: Mapping[str, Any] | None) -> tuple[list[StackedExpertSpec], dict]:
    """Expert specifications and solver options from the ``data.stacking`` mapping."""
    from molt.models.experts import STACKED_EXPERTS

    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", True)):
        return [], {}
    specs = []
    for name, entry in dict(cfg.get("experts") or {}).items():
        if entry is None or entry is False:
            continue
        if name not in STACKED_EXPERTS:
            raise ValueError(f"Unknown stacked expert {name!r}; expected one of {STACKED_EXPERTS}")
        entry = dict(entry or {})
        inputs = tuple(entry.get("inputs") or ())
        bad = set(inputs) - set(STACKED_INPUTS)
        if not inputs or bad:
            raise ValueError(f"stacked expert {name!r} needs inputs from {STACKED_INPUTS}")
        if name != "clinical_lin" and not (_RNA_INPUTS & set(inputs)):
            raise ValueError(f"stacked expert {name!r} needs a transcriptomic input block")
        if name == "clinical_lin" and set(inputs) != {"clinical"}:
            raise ValueError("clinical_lin uses the clinical block only")
        alphas = tuple(float(a) for a in entry.get("alphas") or (0.5,))
        specs.append(StackedExpertSpec(str(name), inputs, alphas))
    options = {
        "folds": int(cfg.get("folds", 5)),
        "n_lambdas": int(cfg.get("n_lambdas", 12)),
        "lambda_ratio": float(cfg.get("lambda_ratio", 0.01)),
        "components": str(cfg.get("components", "total")),
    }
    if options["components"] not in ("total", "blocks"):
        raise ValueError("data.stacking.components must be 'total' or 'blocks'")
    return specs, options


def clinical_design(
    feats: Any, cardinalities: Sequence[int], mixing: np.ndarray | None = None
) -> np.ndarray:
    """Flat clinical design: standardized continuous values, missingness, one-hot categories
    (missing and out-of-vocabulary codes included) and the clinical indicators.

    ``mixing`` (``(total, total)`` over the concatenated category codes, see
    :meth:`~molt.training.clinical_hierarchy.CategoryHierarchy.design_mixing`) replaces the
    indicator row of a partially specified level by the training mix of its compatible levels.
    """
    num = np.asarray(feats.clinical_num, dtype=np.float64)
    miss = 1.0 - np.asarray(feats.clinical_num_mask, dtype=np.float64)
    cat = np.asarray(feats.clinical_cat, dtype=int)
    blocks = [num, miss]
    offset = 0
    for j, card in enumerate(cardinalities):
        codes = np.clip(cat[:, j], 0, int(card) - 1)
        if mixing is not None:
            onehot = np.asarray(mixing, dtype=np.float64)[offset + codes, offset : offset + card]
        else:
            onehot = np.zeros((len(cat), int(card)))
            onehot[np.arange(len(cat)), codes] = 1.0
        blocks.append(onehot)
        offset += int(card)
    blocks.append(np.asarray(feats.clinical_indicators, dtype=np.float64))
    return np.column_stack(blocks)


def _block(
    feats: Any, name: str, cardinalities: Sequence[int], mixing: np.ndarray | None = None
) -> np.ndarray | None:
    if name == "genes":
        panel = getattr(feats, "gene_panel", None)
        return None if panel is None else np.asarray(panel, dtype=np.float64)
    if name == "pathways":
        return np.asarray(feats.rna, dtype=np.float64)
    return clinical_design(feats, cardinalities, mixing)


@dataclass
class _Fitted:
    spec: StackedExpertSpec
    fit: ConvexFit
    widths: list[int]
    score_mean: np.ndarray
    score_sd: np.ndarray
    oof: dict[str, np.ndarray] = field(default_factory=dict)


class StackedScorers:
    """The stacked linear experts of one training fold.

    Args:
        specs: Expert specifications.
        options: Solver options (``folds``, ``n_lambdas``, ``lambda_ratio``, ``components``).
        seed: Seed of the cross-validation folds.
    """

    def __init__(
        self, specs: Sequence[StackedExpertSpec], options: Mapping[str, Any], seed: int = 0
    ) -> None:
        self.specs = list(specs)
        self.options = dict(options)
        self.seed = int(seed)
        self.fitted: dict[str, _Fitted] = {}
        self.category_mixing: np.ndarray | None = None

    @property
    def dims(self) -> dict[str, int]:
        """Input dimension of every fitted expert (1, or the number of blocks)."""
        blocks = self.options.get("components") == "blocks"
        return {m: (len(f.widths) if blocks else 1) for m, f in self.fitted.items()}

    def _design(
        self, feats: Any, spec: StackedExpertSpec, cards: Sequence[int]
    ) -> tuple[np.ndarray | None, list[int]]:
        mixing = getattr(self, "category_mixing", None)
        parts = [_block(feats, b, cards, mixing) for b in spec.inputs]
        if any(p is None for p in parts):
            return None, []
        return np.column_stack(parts), [p.shape[1] for p in parts]

    def _available(self, feats: Any, spec: StackedExpertSpec) -> np.ndarray:
        ok = np.asarray(feats.clinical_present, dtype=bool)
        if spec.needs_rna:
            ok = np.asarray(feats.rna_present, dtype=bool)
        return ok

    def fit(
        self, feats: Any, task: str, cardinalities: Sequence[int], n_bins: int
    ) -> StackedScorers:
        """Fit every expert on the labelled, available participants of ``feats``."""
        opts = self.options
        blocks = opts.get("components") == "blocks"
        for spec in self.specs:
            X, widths = self._design(feats, spec, cardinalities)
            if X is None:
                log.warning("stacked expert %s skipped: input block unavailable", spec.name)
                continue
            avail = self._available(feats, spec)
            if task == "egfr":
                y = np.asarray(feats.y_egfr, dtype=np.float64)
                rows = np.flatnonzero(avail & np.isfinite(y))
                if rows.size < 20 or len(np.unique(y[rows])) < 2:
                    continue
                fit = cv_path(
                    X[rows],
                    y[rows],
                    family="binary",
                    alphas=spec.alphas,
                    n_lambdas=opts["n_lambdas"],
                    lambda_ratio=opts["lambda_ratio"],
                    folds=opts["folds"],
                    seed=self.seed,
                    blocks=widths if blocks else None,
                )
            else:
                present = np.asarray(feats.surv_present, dtype=bool)
                rows = np.flatnonzero(avail & present)
                events = np.asarray(feats.surv_event, dtype=np.float64)[rows]
                if rows.size < 20 or events.sum() < 5:
                    continue
                fit = cv_path(
                    X[rows],
                    family="discrete_time",
                    bins=np.asarray(feats.surv_bin)[rows],
                    events=events,
                    n_bins=int(n_bins),
                    alphas=spec.alphas,
                    n_lambdas=opts["n_lambdas"],
                    lambda_ratio=opts["lambda_ratio"],
                    folds=opts["folds"],
                    seed=self.seed,
                    blocks=widths if blocks else None,
                )
            comps = np.asarray(fit.oof, dtype=np.float64)
            mean, sd = comps.mean(0), comps.std(0)
            sd = np.where(sd > 1e-8, sd, 1.0)
            ids = [feats.ids[i] for i in rows]
            self.fitted[spec.name] = _Fitted(spec, fit, widths, mean, sd, oof=dict(zip(ids, comps)))
            log.debug(
                "stacked %s: alpha %.2f lambda %.4g, %d non-zero of %d",
                spec.name,
                fit.alpha,
                fit.lam,
                fit.n_nonzero,
                X.shape[1],
            )
        return self

    def _components(self, fit: ConvexFit, X: np.ndarray, widths: list[int]) -> np.ndarray:
        Z = fit.standardize(X)
        if self.options.get("components") != "blocks":
            return (Z @ fit.coef)[:, None]
        edges = np.cumsum([0, *widths])
        return np.column_stack([Z[:, a:b] @ fit.coef[a:b] for a, b in itertools.pairwise(edges)])

    def transform(self, feats: Any, cardinalities: Sequence[int]) -> dict[str, np.ndarray]:
        """Standardized scores ``{expert: (n, d)}``; NaN where the expert is unavailable.

        Participants that were among the fitting rows receive their cross-validated score.
        """
        out: dict[str, np.ndarray] = {}
        for name, f in self.fitted.items():
            X, _ = self._design(feats, f.spec, cardinalities)
            d = len(f.score_mean)
            scores = np.full((len(feats.ids), d), np.nan, dtype=np.float64)
            if X is not None:
                scores = self._components(f.fit, X, f.widths)
                for i, pid in enumerate(feats.ids):
                    if pid in f.oof:
                        scores[i] = f.oof[pid]
            scores = (scores - f.score_mean) / f.score_sd
            scores[~self._available(feats, f.spec)] = np.nan
            out[name] = scores.astype(np.float32)
        return out

    def summary(self) -> dict[str, Any]:
        """Chosen penalties and sparsity of every expert."""
        return {
            m: {
                "inputs": list(f.spec.inputs),
                "alpha": f.fit.alpha,
                "lambda": f.fit.lam,
                "n_nonzero": f.fit.n_nonzero,
                "n_features": len(f.fit.coef),
                "n_train": len(f.oof),
            }
            for m, f in self.fitted.items()
        }
