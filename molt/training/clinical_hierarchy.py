"""Partially specified clinical categories: declared hierarchies, coarsening and mixtures.

Clinical records are specified at different resolutions across cohorts: a smoker of unknown
current/former status (``ever_unspecified``), a stage group without substage (``II`` instead
of ``IIA``/``IIB``), a ``T1`` tumour without the ``T1a``-``T1c`` split or an unknown tumour
side. A *hierarchy* declares, for every categorical variable, the partially specified
(coarse) levels and the fully specified levels compatible with them::

    smoking_status:
      ever_unspecified: [current, former_le15, former_gt15, former_unspecified]
      former_unspecified: [former_le15, former_gt15]
    ajcc_stage: {I: [IA, IA1, IA2, IA3, IB], II: [IIA, IIB], III: [IIIA, IIIB, IIIC]}
    laterality: {NA: [left, right]}          # NA = the missing token (unknown side)

Coarse levels are reserved vocabulary entries even when the training data never contain
them, so that an external participant recorded at the coarse level is never sent to the
out-of-vocabulary token. Two label-free treatments are available
(``PreprocessConfig.partial_levels``):

* ``coarsen`` - stochastic category coarsening during training: with probability ``p``
  (``coarsen_p``) an observed fine level is replaced by one of its declared coarse
  ancestors (chosen uniformly), so the coarse embedding is trained on the mixture of its
  children. The quantities that a coarse record does not carry are hidden together with the
  category (``coarsen_hide``; e.g. years since quitting and smoking duration when the
  current/former status is coarsened away).
* ``mixture`` - parent mixture: the embedding of a coarse level with fewer than
  ``partial_min_count`` training participants is the training-frequency-weighted mean of
  the embeddings of its compatible fine training levels (descendants, transitively), in
  training and at inference.

With a hierarchy treatment a missing value is always encoded by the missing token (index 0),
whatever the storage dtype of the column, so that "not recorded" has one representation in
every cohort (the ``NA`` coarse level of side and lobe is that token).

Fully specified levels without coarse ancestors (``never`` smoking in particular) are
never coarsened or mixed, so they are represented exactly in every cohort. With
``structural_zero`` the smoking quantities that are zero by definition (pack-years, smoking
duration and years since quitting of never-smokers; years since quitting of current
smokers) are encoded as observed zeros rather than as missing, so the missingness
indicators of these quantities mean "not recorded" and do not duplicate the never-smoking
information.

The stacked linear clinical scorer (:func:`molt.training.stacking.clinical_design`) uses the
parent mixture in both treatments: for a linear model the coarsened-training estimate of a
coarse level converges to the frequency-weighted mean effect of its children, which the
mixture gives in closed form.

:func:`partial_records` turns the records of evaluation participants into their coarsest
declared specification (smoking ever/never, stage group, side and lobe unknown, quantities
hidden as declared). The inner-fold loss on such records measures, without external data, how
well a model handles partially specified records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_COARSEN_HIDE",
    "DEFAULT_HIERARCHY",
    "MISSING_LEVEL",
    "PARTIAL_TREATMENTS",
    "STRUCTURAL_ZEROS",
    "CategoryHierarchy",
    "apply_structural_zeros",
    "coarsest_levels",
    "partial_records",
]

MISSING_LEVEL = "NA"
"""Name of the coarse level that coincides with the missing token (e.g. unknown side)."""

PARTIAL_TREATMENTS = ("none", "coarsen", "mixture")
"""Accepted values of ``PreprocessConfig.partial_levels``."""

DEFAULT_HIERARCHY: dict[str, dict[str, tuple[str, ...]]] = {
    "smoking_status": {
        "ever_unspecified": ("current", "former_le15", "former_gt15", "former_unspecified"),
        "former_unspecified": ("former_le15", "former_gt15"),
    },
    "ajcc_stage": {
        "I": ("IA", "IA1", "IA2", "IA3", "IB"),
        "II": ("IIA", "IIB"),
        "III": ("IIIA", "IIIB", "IIIC"),
        "IV": ("IVA", "IVB"),
    },
    "pT": {"T1": ("T1mi", "T1a", "T1b", "T1c"), "T2": ("T2a", "T2b")},
    "pM": {"M1": ("M1a", "M1b", "M1c")},
    "laterality": {MISSING_LEVEL: ("left", "right")},
    "lobe": {MISSING_LEVEL: ("upper", "middle", "lower")},
}
"""Default coarsening hierarchy of the clinical categorical variables."""

DEFAULT_COARSEN_HIDE: dict[str, dict[str, tuple[str, ...]]] = {
    "smoking_status": {
        "ever_unspecified": ("smoking_years", "years_since_quit"),
        "former_unspecified": ("smoking_years", "years_since_quit"),
    }
}
"""Continuous variables hidden when a category is coarsened to the named level."""

STRUCTURAL_ZEROS: dict[str, dict[str, tuple[str, ...]]] = {
    "smoking_status": {
        "never": ("pack_years", "smoking_years", "years_since_quit"),
        "current": ("years_since_quit",),
    }
}
"""Continuous variables that are zero by definition for a level of a categorical variable."""


def _normalize(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "na", "nan", "none", "null", "unknown", "<na>"} else text


def apply_structural_zeros(
    table: pd.DataFrame, name: str, values: np.ndarray, rules: Mapping[str, Mapping] | None = None
) -> np.ndarray:
    """Fill structurally zero, unrecorded values of continuous variable ``name`` with 0."""
    rules = STRUCTURAL_ZEROS if rules is None else rules
    out = np.asarray(values, dtype=float).copy()
    for variable, levels in rules.items():
        if variable not in table.columns:
            continue
        status = table[variable].map(_normalize).to_numpy(object)
        for level, names in levels.items():
            if name in names:
                hit = (status == level) & ~np.isfinite(out)
                out[hit] = 0.0
    return out


def _parse(mapping: Mapping[str, Any] | None) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        str(var): {str(level): tuple(str(c) for c in children) for level, children in m.items()}
        for var, m in dict(mapping or {}).items()
        if m
    }


@dataclass
class CategoryHierarchy:
    """Fitted hierarchy of the categorical clinical variables of one training fold.

    Attributes:
        treatment: ``none``, ``coarsen`` or ``mixture``.
        variables: Categorical variable names (encoder order).
        ancestors: Per variable, ``{fine index: [coarse ancestor indices]}`` (index 0 = the
            missing token for an ``NA`` coarse level).
        mixtures: Per variable, ``{coarse index: {fine index: weight}}`` for the coarse
            levels represented by their parent mixture.
        hide: Per variable, ``{coarse index: [continuous column indices]}`` hidden when a
            value is coarsened to that level.
        coarsen_p: Coarsening probability (``coarsen`` treatment).
        counts: Training counts of the partially specified levels (for reporting).
    """

    treatment: str = "none"
    variables: list[str] = field(default_factory=list)
    ancestors: list[dict[int, list[int]]] = field(default_factory=list)
    mixtures: list[dict[int, dict[int, float]]] = field(default_factory=list)
    hide: list[dict[int, list[int]]] = field(default_factory=list)
    coarsen_p: float = 0.0
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def extend_vocab(
        vocab: Mapping[str, dict[str, int]],
        hierarchy: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, int]]:
        """Add the declared coarse levels with a compatible training level to ``vocab``.

        ``hierarchy=None`` uses :data:`DEFAULT_HIERARCHY`.
        New levels are appended after the training levels, so existing indices are unchanged.
        """
        spec = _parse(DEFAULT_HIERARCHY if hierarchy is None else hierarchy)
        out = {name: dict(levels) for name, levels in vocab.items()}
        for name, levels in out.items():
            declared = spec.get(name, {})
            for coarse in declared:
                if coarse == MISSING_LEVEL or coarse in levels:
                    continue
                if _descendants(declared, coarse) & set(levels):
                    levels[coarse] = max(levels.values(), default=1) + 1
        return out

    @classmethod
    def fit(
        cls,
        vocab: Mapping[str, dict[str, int]],
        counts: Mapping[str, Mapping[str, int]],
        numeric: Sequence[str],
        *,
        hierarchy: Mapping[str, Any] | None,
        hide: Mapping[str, Any] | None = None,
        treatment: str = "coarsen",
        coarsen_p: float = 0.2,
        partial_min_count: int = 10,
        min_category_count: int = 2,
    ) -> CategoryHierarchy:
        """Fit the coarsening table and parent mixtures from training vocabularies and counts.

        Args:
            vocab: Per variable, level -> index (as extended by :meth:`extend_vocab`).
            counts: Per variable, training count of every observed level.
            numeric: Continuous variable names (column order of the encoder).
            hierarchy: Declared hierarchy ``{variable: {coarse: [fine levels]}}``
                (``None`` = :data:`DEFAULT_HIERARCHY`).
            hide: Continuous variables hidden per coarse level ``{variable: {coarse: [...]}}``
                (``None`` = :data:`DEFAULT_COARSEN_HIDE`).
            treatment: ``none``, ``coarsen`` or ``mixture``.
            coarsen_p: Coarsening probability of the ``coarsen`` treatment.
            partial_min_count: Coarse levels with fewer training participants are represented
                by their parent mixture (``mixture`` treatment and the stacked clinical scorer).
            min_category_count: Minimum count of a fine level to enter a mixture.
        """
        if treatment not in PARTIAL_TREATMENTS:
            raise ValueError(f"partial_levels must be one of {PARTIAL_TREATMENTS}")
        spec = _parse(DEFAULT_HIERARCHY if hierarchy is None else hierarchy)
        hide_spec = _parse(DEFAULT_COARSEN_HIDE if hide is None else hide)
        num_index = {n: i for i, n in enumerate(numeric)}
        out = cls(treatment=treatment, variables=list(vocab), coarsen_p=float(coarsen_p))
        for name, levels in vocab.items():
            declared = spec.get(name, {})
            c = {str(k): int(v) for k, v in dict(counts.get(name, {})).items()}
            anc: dict[int, list[int]] = {}
            mix: dict[int, dict[int, float]] = {}
            hid: dict[int, list[int]] = {}
            for coarse in declared:
                cidx = 0 if coarse == MISSING_LEVEL else levels.get(coarse)
                if cidx is None:
                    continue
                for fine in _descendants(declared, coarse):
                    if fine in levels and fine != coarse:
                        anc.setdefault(levels[fine], []).append(int(cidx))
                cols = [
                    num_index[v] for v in hide_spec.get(name, {}).get(coarse, ()) if v in num_index
                ]
                if cols and cidx:
                    hid[int(cidx)] = cols
                if coarse != MISSING_LEVEL and c.get(coarse, 0) < partial_min_count:
                    fine_levels = [
                        f
                        for f in _descendants(declared, coarse)
                        if f in levels
                        and c.get(f, 0) >= max(min_category_count, 1)
                        and not (f in declared and c.get(f, 0) < partial_min_count)
                    ]
                    total = float(sum(c[f] for f in fine_levels))
                    if total > 0:
                        mix[int(cidx)] = {levels[f]: c[f] / total for f in sorted(fine_levels)}
                if coarse != MISSING_LEVEL:
                    out.counts.setdefault(name, {})[coarse] = c.get(coarse, 0)
            out.ancestors.append({k: sorted(set(v)) for k, v in anc.items()})
            out.mixtures.append(mix)
            out.hide.append(hid)
        return out

    # ------------------------------------------------------------------ model inputs
    def _offsets(self, cardinalities: Sequence[int]) -> list[int]:
        return [0, *np.cumsum([int(c) for c in cardinalities])[:-1].tolist()]

    def mixing_matrix(self, cardinalities: Sequence[int]) -> np.ndarray | None:
        """Row-stochastic ``(total, total)`` map of flat category indices to embedding weights.

        Identity except for the rows of mixed coarse levels; ``None`` unless the treatment is
        ``mixture`` and at least one level is mixed.
        """
        if self.treatment != "mixture" or not any(self.mixtures):
            return None
        return self.design_mixing(cardinalities)

    def design_mixing(self, cardinalities: Sequence[int]) -> np.ndarray:
        """The mixing matrix of every mixed coarse level, irrespective of the treatment."""
        total = int(sum(cardinalities))
        mat = np.eye(total, dtype=np.float32)
        for j, off in enumerate(self._offsets(cardinalities)):
            for cidx, weights in (self.mixtures[j] if j < len(self.mixtures) else {}).items():
                row = np.zeros(total, dtype=np.float32)
                for fine, w in weights.items():
                    row[off + fine] = w
                mat[off + cidx] = row
        return mat

    def coarsen_tables(
        self, cardinalities: Sequence[int], n_num: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Ancestor table ``(total, A)`` of flat indices (-1 padded) and hidden-column mask
        ``(total, n_num)``; ``None`` unless the treatment is ``coarsen``."""
        if self.treatment != "coarsen" or self.coarsen_p <= 0 or not any(self.ancestors):
            return None
        total = int(sum(cardinalities))
        width = max((len(v) for a in self.ancestors for v in a.values()), default=1)
        table = np.full((total, width), -1, dtype=np.int64)
        hide = np.zeros((total, max(n_num, 1)), dtype=bool)
        for j, off in enumerate(self._offsets(cardinalities)):
            for fine, coarse in self.ancestors[j].items():
                table[off + fine, : len(coarse)] = [off + c for c in coarse]
            for cidx, cols in self.hide[j].items():
                hide[off + cidx, cols] = True
        return table, hide[:, :n_num]

    def top_ancestor(self, j: int, index: int) -> int | None:
        """The most general declared ancestor of a fine level (``None`` if it has none)."""
        anc = self.ancestors[j].get(int(index)) if j < len(self.ancestors) else None
        if not anc:
            return None
        # The most general ancestor is the one with the most fine levels below it.
        size = {a: sum(a in v for v in self.ancestors[j].values()) for a in anc}
        return max(anc, key=lambda a: (a == 0, size[a], -a))

    def summary(self) -> dict[str, Any]:
        """Compact description (treatment, mixed levels, coarsening targets)."""
        return {
            "treatment": self.treatment,
            "coarsen_p": self.coarsen_p,
            "mixed": {v: sorted(m) for v, m in zip(self.variables, self.mixtures) if m},
            "coarse_counts": self.counts,
        }


def _descendants(declared: Mapping[str, Sequence[str]], level: str) -> set[str]:
    """All levels below ``level`` (transitively through nested coarse levels)."""
    out: set[str] = set()
    stack = list(declared.get(level, ()))
    while stack:
        child = stack.pop()
        if child in out or child == level:
            continue
        out.add(child)
        stack.extend(declared.get(child, ()))
    return out


def coarsest_levels(hierarchy: Mapping[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Map every fine level to its most general declared ancestor, per variable."""
    spec = _parse(DEFAULT_HIERARCHY if hierarchy is None else hierarchy)
    out: dict[str, dict[str, str]] = {}
    for name, declared in spec.items():
        size = {c: len(_descendants(declared, c)) for c in declared}
        top: dict[str, str] = {}
        for coarse in sorted(declared, key=lambda c: (c == MISSING_LEVEL, size[c])):
            for fine in _descendants(declared, coarse):
                top[fine] = coarse
        out[name] = top
    return out


def partial_records(
    table: pd.DataFrame,
    ids: Sequence[str],
    hierarchy: Mapping[str, Any] | None = None,
    hide: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Copy of ``table`` with the records of ``ids`` reduced to their coarsest declared form.

    Every recorded level with declared ancestors is replaced by its most general ancestor
    (``NA`` = not recorded) and the continuous variables declared for that ancestor are
    removed: smokers become ``ever_unspecified`` without smoking duration and years since
    quitting, substages become stage groups, ``T1a`` becomes ``T1`` and side and lobe become
    unknown. The view does not depend on a fitted model, so every model (with or without a
    hierarchy treatment) can be scored on the same partially specified records.
    """
    hide_spec = _parse(DEFAULT_COARSEN_HIDE if hide is None else hide)
    out = table.copy()
    rows = out.index.isin([str(i) for i in ids])
    for name, top in coarsest_levels(hierarchy).items():
        if name not in out.columns:
            continue
        values = out[name].astype(object)
        normalized = values.map(_normalize)
        coarse = normalized.map(lambda v, top=top: top.get(v) if isinstance(v, str) else None)
        hit = rows & coarse.notna().to_numpy()
        new = coarse.where(coarse != MISSING_LEVEL, None)
        values = values.where(~hit, new)
        out[name] = values
        for level, names in hide_spec.get(name, {}).items():
            sel = hit & (coarse == level).to_numpy()
            for col in names:
                if col in out.columns:
                    out.loc[sel, col] = np.nan
    return out
