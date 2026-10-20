"""Scale of a processed expression matrix, as declared by ``expression_meta.json``.

``units`` governs (``platform`` alone is not enough): RNA-seq abundances on a ``log2(x + 1)``
scale, microarray log intensities and relative values without an absolute abundance scale
(z-scores, median-centred values, log ratios against a reference) need different detection
rules and quality summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["expression_scale", "is_relative", "is_sequencing"]

_SEQ_UNITS = ("tpm", "rsem", "fpkm", "count", "cpm")
_RELATIVE_UNIT_TERMS = (
    "zscore",
    "z-score",
    "z score",
    "median-centred",
    "median-centered",
    "median centred",
    "median centered",
    "ratio",
)


def expression_scale(meta: Mapping[str, Any]) -> str:
    """Scale of an expression matrix as stated by ``expression_meta.json`` (``units`` governs).

    Returns:
        ``relative`` for values without an absolute abundance scale (``zscore``,
        median-centred values, log ratios against a reference), ``sequencing`` for RNA-seq
        abundances on a ``log2(x + 1)`` scale and ``array`` for log intensities.
    """
    units = str(meta.get("units", "")).lower()
    platform = str(meta.get("platform", "")).lower()
    if any(term in units for term in _RELATIVE_UNIT_TERMS) or platform.endswith("zscore"):
        return "relative"
    if platform.startswith("rnaseq") or any(u in units for u in _SEQ_UNITS):
        return "sequencing"
    return "array"


def is_sequencing(meta: Mapping[str, Any]) -> bool:
    """Whether an expression matrix is RNA-seq on a ``log2(x + 1)`` scale."""
    return expression_scale(meta) == "sequencing"


def is_relative(meta: Mapping[str, Any]) -> bool:
    """Whether an expression matrix has no absolute abundance scale (see :func:`expression_scale`)."""
    return expression_scale(meta) == "relative"
