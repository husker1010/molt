"""Discrete-time survival bins shared by preprocessing, post-processing and stacked scorers."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["discretize", "survival_bins"]


def survival_bins(width: float, max_time: float) -> np.ndarray:
    """Bin edges ``[0, w, 2w, ..., max_time]`` (``max_time`` must be a multiple of ``w``)."""
    k = int(round(max_time / width))  # noqa: RUF046
    if not math.isclose(k * width, max_time):
        raise ValueError(f"max_time={max_time} is not a multiple of the bin width {width}")
    return np.arange(k + 1, dtype=np.float64) * width


def discretize(time: np.ndarray, event: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    """Observed bin, event indicator and validity for discrete-time survival.

    Events in ``[edges[j], edges[j+1])`` map to bin ``j``. A participant censored in the
    first half of bin ``j`` contributes survival through bin ``j - 1`` only (and nothing when
    ``j = 0``); follow-up beyond the last boundary is censored at the last bin.
    """
    width = np.diff(edges)
    k = len(width)
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(e)
    j = np.clip(np.searchsorted(edges, np.nan_to_num(t), side="right") - 1, 0, k - 1)
    beyond = finite & (t >= edges[-1])
    ev = (e == 1) & finite & ~beyond
    start = edges[j]
    first_half = (np.nan_to_num(t) - start) < 0.5 * width[j]
    bins = j.copy()
    cens_early = finite & ~ev & ~beyond & first_half
    bins[cens_early] -= 1
    bins[beyond] = k - 1
    valid = finite & (bins >= 0)
    bins = np.where(valid, bins, 0)
    return bins.astype(np.int64), ev.astype(np.float32), valid
