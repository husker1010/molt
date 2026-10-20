"""Resampling inference, multiplicity adjustment and random-effects meta-analysis.

The bootstrap helpers resample participants (optionally within fold-by-class strata) or
whole clusters such as tissue source sites, evaluate an arbitrary statistic on every
resample and return percentile intervals. Paired contrasts evaluate two prediction sets on
the *same* resample indices, so that differences, their intervals and bootstrap p-values
reflect the correlation between models. Survival metrics in :mod:`molt.metrics.survival`
estimate the censoring distribution from the data they receive, so censoring weights are
re-estimated inside every resample automatically.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import stats

from molt.utils.seed import DEFAULT_SEED

__all__ = [
    "BootstrapResult",
    "PairedBootstrapResult",
    "benjamini_hochberg",
    "bootstrap",
    "bootstrap_indices",
    "bootstrap_p_value",
    "cluster_bootstrap_indices",
    "hanley_mcneil_variance",
    "holm",
    "make_strata",
    "meta_analyze_auroc",
    "paired_bootstrap",
    "random_effects_meta",
    "site_aurocs",
]

DEFAULT_N_BOOT = 1000
DEFAULT_N_JOBS = 8
_SERIAL_BELOW = 64

Data = Union[Mapping[str, object], "pandas.DataFrame"]  # noqa: F821
StatFn = Callable[[dict], Mapping[str, float] | float]
Indices = np.ndarray | Sequence[np.ndarray]


# --------------------------------------------------------------------------- resampling


def make_strata(*keys: object) -> np.ndarray:
    """Integer stratum codes for the joint levels of one or more key vectors.

    Missing values form their own level. Typical use: ``make_strata(outer_fold, label)``
    for fold- and class-stratified resampling.
    """
    if not keys:
        raise ValueError("at least one key is required")
    cols = []
    for k in keys:
        arr = np.asarray(k, dtype=object)
        cols.append(np.array(["<NA>" if _is_missing(v) else str(v) for v in arr], dtype=object))
    n = cols[0].size
    if any(c.size != n for c in cols):
        raise ValueError("strata keys must have equal length")
    joined = np.array(["\x1f".join(row) for row in zip(*cols)], dtype=object)
    return np.unique(joined, return_inverse=True)[1].astype(np.int64)


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def bootstrap_indices(
    n: int, n_boot: int = DEFAULT_N_BOOT, strata: object | None = None, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Participant-level bootstrap indices, optionally resampled within strata.

    Within each stratum, as many participants are drawn with replacement as the stratum
    contains, so stratum sizes (for example positives per outer fold) are preserved.

    Args:
        n: Number of participants.
        n_boot: Number of resamples.
        strata: Optional stratum label per participant (see :func:`make_strata`).
        seed: Random seed.

    Returns:
        Integer array of shape ``(n_boot, n)``.
    """
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))
    codes = np.asarray(strata)
    if codes.size != n:
        raise ValueError("strata must have length n")
    codes = np.unique(codes, return_inverse=True)[1]
    out = np.empty((n_boot, n), dtype=np.int64)
    pos = 0
    for c in range(codes.max() + 1):
        members = np.flatnonzero(codes == c)
        k = members.size
        out[:, pos : pos + k] = members[rng.integers(0, k, size=(n_boot, k))]
        pos += k
    return out


def cluster_bootstrap_indices(
    clusters: object,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    strata: object | None = None,
) -> list[np.ndarray]:
    """Cluster (for example site) bootstrap: resample whole clusters with replacement.

    Args:
        clusters: Cluster label per participant.
        n_boot: Number of resamples.
        seed: Random seed.
        strata: Optional stratum per *participant*; clusters are resampled within the stratum
            of their first member (useful to keep large and pooled small sites separate).

    Returns:
        List of ``n_boot`` participant index arrays (lengths vary with cluster sizes).
    """
    rng = np.random.default_rng(seed)
    labels, codes = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    order = np.argsort(codes, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=labels.size))]
    members = [order[bounds[c] : bounds[c + 1]] for c in range(labels.size)]
    if strata is None:
        groups = [np.arange(labels.size)]
    else:
        s = np.asarray(strata)
        first = np.array([s[m[0]] for m in members])
        groups = [np.flatnonzero(first == g) for g in np.unique(first)]
    out = []
    for _ in range(n_boot):
        picks = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in groups])
        out.append(np.concatenate([members[c] for c in picks]))
    return out


# ----------------------------------------------------------------------------- evaluation


def _as_dict(data: Data) -> dict[str, np.ndarray]:
    if hasattr(data, "columns"):
        return {str(c): data[c].to_numpy() for c in data.columns}  # type: ignore[union-attr]
    return {k: np.asarray(v) for k, v in data.items()}  # type: ignore[union-attr]


def _call(stat_fn: StatFn, data: dict[str, np.ndarray], idx: np.ndarray | None) -> dict:
    sub = data if idx is None else {k: v[idx] for k, v in data.items()}
    out = stat_fn(sub)
    if isinstance(out, Mapping):
        return {k: float(v) for k, v in out.items()}
    return {"value": float(out)}


def _run_chunk(
    stat_fn: StatFn,
    datasets: Sequence[dict[str, np.ndarray]],
    idx_chunk: Sequence[np.ndarray],
    names: Sequence[str],
) -> np.ndarray:
    res = np.full((len(datasets), len(idx_chunk), len(names)), np.nan)
    for b, idx in enumerate(idx_chunk):
        for d, data in enumerate(datasets):
            try:
                vals = _call(stat_fn, data, idx)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                continue
            res[d, b] = [vals.get(nm, np.nan) for nm in names]
    return res


def _resolve_n_jobs(n_jobs: int | None, n_boot: int) -> int:
    if n_boot < _SERIAL_BELOW:
        return 1
    cpus = os.cpu_count() or 1
    if n_jobs is None:
        return min(DEFAULT_N_JOBS, cpus)
    if n_jobs < 0:
        return max(1, cpus + 1 + n_jobs)
    return max(1, int(n_jobs))


def _evaluate(
    stat_fn: StatFn,
    datasets: Sequence[dict[str, np.ndarray]],
    indices: Indices,
    names: Sequence[str],
    n_jobs: int | None,
) -> np.ndarray:
    idx_list = list(indices)
    jobs = _resolve_n_jobs(n_jobs, len(idx_list))
    if jobs == 1:
        res = _run_chunk(stat_fn, datasets, idx_list, names)
    else:
        from joblib import Parallel, delayed

        chunks = [c for c in np.array_split(np.arange(len(idx_list)), jobs * 2) if c.size]
        parts = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_run_chunk)(stat_fn, datasets, [idx_list[i] for i in c], names) for c in chunks
        )
        res = np.concatenate(parts, axis=1)
    return res


@dataclass
class BootstrapResult:
    """Point estimates and bootstrap distribution of one or more statistics.

    Attributes:
        names: Statistic names.
        estimate: Point estimates on the original data, shape ``(m,)``.
        samples: Bootstrap replicates, shape ``(n_boot, m)`` (NaN where undefined).
        alpha: Two-sided error level of the percentile intervals.
    """

    names: list[str]
    estimate: np.ndarray
    samples: np.ndarray
    alpha: float = 0.05

    def ci(self, name: str = "value") -> tuple[float, float]:
        """Percentile confidence interval of a statistic."""
        col = self.samples[:, self.names.index(name)]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return float("nan"), float("nan")
        lo, hi = np.percentile(col, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return float(lo), float(hi)

    def basic_ci(
        self, name: str = "value", lower: float | None = None, contain_estimate: bool = True
    ) -> tuple[float, float]:
        """Basic (bias-corrected, "reverse percentile") bootstrap interval of a statistic.

        ``[2 theta - q_{1 - alpha/2}, 2 theta - q_{alpha/2}]`` with the bootstrap quantiles
        ``q``. It corrects for the bias of statistics that the resampling inflates
        systematically, such as the expected calibration error. Bounds are clipped at
        ``lower`` (e.g. 0 for a non-negative statistic) and, with ``contain_estimate``,
        widened to include the point estimate ``theta``.
        """
        j = self.names.index(name)
        col = self.samples[:, j]
        col = col[np.isfinite(col)]
        theta = float(self.estimate[j])
        if col.size == 0 or not np.isfinite(theta):
            return float("nan"), float("nan")
        q_lo, q_hi = np.percentile(col, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        lo, hi = 2.0 * theta - float(q_hi), 2.0 * theta - float(q_lo)
        if lower is not None:
            lo, hi = max(lo, lower), max(hi, lower)
        if contain_estimate:
            lo, hi = min(lo, theta), max(hi, theta)
        return lo, hi

    def summary(self) -> dict[str, dict[str, float]]:
        """``name -> {value, ci_low, ci_high, se, n_valid}`` (percentile intervals)."""
        out = {}
        for j, name in enumerate(self.names):
            col = self.samples[:, j]
            valid = col[np.isfinite(col)]
            lo, hi = self.ci(name)
            out[name] = {
                "value": float(self.estimate[j]),
                "ci_low": lo,
                "ci_high": hi,
                "se": float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan"),
                "n_valid": int(valid.size),
            }
        return out


def _indices_for(
    n: int,
    n_boot: int,
    strata: object | None,
    clusters: object | None,
    seed: int,
    indices: Indices | None,
) -> Indices:
    if indices is not None:
        return indices
    if clusters is not None:
        return cluster_bootstrap_indices(clusters, n_boot, seed, strata)
    return bootstrap_indices(n, n_boot, strata, seed)


def bootstrap(
    stat_fn: StatFn,
    data: Data,
    n_boot: int = DEFAULT_N_BOOT,
    strata: object | None = None,
    clusters: object | None = None,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    indices: Indices | None = None,
) -> BootstrapResult:
    """Bootstrap one or more statistics of a participant-level data set.

    Args:
        stat_fn: Function of a dict of equally indexed arrays returning a float or a mapping
            ``name -> float``. It must be picklable when ``n_jobs > 1`` (lambdas are fine,
            joblib uses cloudpickle).
        data: DataFrame or mapping of equal-length arrays (one row per participant).
        n_boot: Number of resamples.
        strata: Stratum per participant for stratified resampling.
        clusters: Cluster per participant for a cluster bootstrap (takes precedence).
        seed: Random seed; equal seeds give identical resamples across calls.
        alpha: Two-sided error level.
        n_jobs: Worker processes (``None`` = up to 8, ``1`` = serial, negative = all but
            ``|n_jobs| - 1`` CPUs).
        indices: Pre-computed resample indices (overrides ``strata``/``clusters``).

    Returns:
        :class:`BootstrapResult`.
    """
    d = _as_dict(data)
    n = len(next(iter(d.values())))
    point = _call(stat_fn, d, None)
    names = list(point)
    idx = _indices_for(n, n_boot, strata, clusters, seed, indices)
    samples = _evaluate(stat_fn, [d], idx, names, n_jobs)[0]
    return BootstrapResult(names, np.array([point[k] for k in names]), samples, alpha)


def bootstrap_p_value(diff_samples: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for ``H0: difference = null`` from paired replicates.

    ``p = min(1, 2 * min((1 + #(d <= null)) / (B + 1), (1 + #(d >= null)) / (B + 1)))``.
    Identical predictions (all replicates equal to ``null``) give ``p = 1``.
    """
    d = np.asarray(diff_samples, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    b = d.size
    lower = (1 + np.sum(d <= null)) / (b + 1)
    upper = (1 + np.sum(d >= null)) / (b + 1)
    return float(min(1.0, 2 * min(lower, upper)))


@dataclass
class PairedBootstrapResult:
    """Paired bootstrap of two prediction sets on shared resample indices.

    Attributes:
        names: Statistic names.
        estimate_a: Point estimates for prediction set A.
        estimate_b: Point estimates for prediction set B.
        samples_a: Replicates for A, shape ``(n_boot, m)``.
        samples_b: Replicates for B on the same resamples.
        alpha: Two-sided error level.
    """

    names: list[str]
    estimate_a: np.ndarray
    estimate_b: np.ndarray
    samples_a: np.ndarray
    samples_b: np.ndarray
    alpha: float = 0.05

    def summary(self) -> dict[str, dict[str, float]]:
        """``name -> {value_a, value_b, diff, ci_low, ci_high, p_value, n_valid}`` (A minus B)."""
        out = {}
        q = [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)]
        for j, name in enumerate(self.names):
            d = self.samples_a[:, j] - self.samples_b[:, j]
            valid = d[np.isfinite(d)]
            lo, hi = np.percentile(valid, q) if valid.size else (np.nan, np.nan)
            out[name] = {
                "value_a": float(self.estimate_a[j]),
                "value_b": float(self.estimate_b[j]),
                "diff": float(self.estimate_a[j] - self.estimate_b[j]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_value": bootstrap_p_value(valid),
                "n_valid": int(valid.size),
            }
        return out


def paired_bootstrap(
    stat_fn: StatFn,
    data_a: Data,
    data_b: Data,
    n_boot: int = DEFAULT_N_BOOT,
    strata: object | None = None,
    clusters: object | None = None,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    n_jobs: int | None = None,
    indices: Indices | None = None,
) -> PairedBootstrapResult:
    """Paired contrast of a statistic between two aligned prediction sets.

    ``data_a`` and ``data_b`` must describe the same participants in the same row order
    (typically identical outcome columns with different prediction columns). Both are
    evaluated on every shared resample.

    Args:
        stat_fn: Statistic function (see :func:`bootstrap`).
        data_a: Data with the predictions of model A.
        data_b: Data with the predictions of model B.
        n_boot: Number of resamples.
        strata: Stratum per participant.
        clusters: Cluster per participant (cluster bootstrap).
        seed: Random seed.
        alpha: Two-sided error level.
        n_jobs: Worker processes (see :func:`bootstrap`).
        indices: Pre-computed resample indices.

    Returns:
        :class:`PairedBootstrapResult`.
    """
    da, db = _as_dict(data_a), _as_dict(data_b)
    n = len(next(iter(da.values())))
    if len(next(iter(db.values()))) != n:
        raise ValueError("paired data sets must have the same number of rows")
    pa, pb = _call(stat_fn, da, None), _call(stat_fn, db, None)
    names = list(pa)
    idx = _indices_for(n, n_boot, strata, clusters, seed, indices)
    res = _evaluate(stat_fn, [da, db], idx, names, n_jobs)
    return PairedBootstrapResult(
        names,
        np.array([pa[k] for k in names]),
        np.array([pb.get(k, np.nan) for k in names]),
        res[0],
        res[1],
        alpha,
    )


# --------------------------------------------------------------------------- multiplicity


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries are ignored and returned as NaN)."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (NaN entries ignored)."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    adj = p[order] * m / np.arange(1, m + 1)
    out[order] = np.minimum(1.0, np.minimum.accumulate(adj[::-1])[::-1])
    return out


# ------------------------------------------------------------------------ meta-analysis


def random_effects_meta(
    estimates: Sequence[float],
    variances: Sequence[float],
    method: str = "REML",
    hartung_knapp: bool = True,
    alpha: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Random-effects meta-analysis with REML (or DerSimonian-Laird) heterogeneity.

    Args:
        estimates: Study-level estimates (for example logit AUROC per site).
        variances: Their within-study variances.
        method: ``"REML"`` or ``"DL"``.
        hartung_knapp: Use the Hartung-Knapp(-Sidik-Jonkman) variance and a ``t_{k-1}``
            critical value for the pooled interval.
        alpha: Two-sided error level.
        max_iter: Maximum REML iterations.
        tol: REML convergence tolerance.

    Returns:
        Dict with ``mu``, ``se``, ``ci_low``, ``ci_high``, ``pi_low``, ``pi_high`` (prediction
        interval with ``t_{k-2}``; NaN for fewer than three studies), ``tau2``, ``i2``, ``q``,
        ``p_value`` and ``k``.
    """
    y = np.asarray(estimates, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = y.size
    nan = float("nan")
    if k == 0:
        return {
            "mu": nan,
            "se": nan,
            "ci_low": nan,
            "ci_high": nan,
            "pi_low": nan,
            "pi_high": nan,
            "tau2": nan,
            "i2": nan,
            "q": nan,
            "p_value": nan,
            "k": 0,
        }
    w = 1.0 / v
    mu_fe = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu_fe) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    if method.upper() == "REML" and k > 1:
        for _ in range(max_iter):
            wr = 1.0 / (v + tau2)
            mu = np.sum(wr * y) / np.sum(wr)
            new = np.sum(wr**2 * ((y - mu) ** 2 - v)) / np.sum(wr**2) + 1.0 / np.sum(wr)
            new = max(0.0, float(new))
            if abs(new - tau2) < tol * max(1.0, tau2):
                tau2 = new
                break
            tau2 = new
    elif method.upper() not in ("REML", "DL"):
        raise ValueError("method must be 'REML' or 'DL'")
    wr = 1.0 / (v + tau2)
    mu = float(np.sum(wr * y) / np.sum(wr))
    se = float(np.sqrt(1.0 / np.sum(wr)))
    if hartung_knapp and k > 1:
        se = float(np.sqrt(np.sum(wr * (y - mu) ** 2) / ((k - 1) * np.sum(wr))))
        crit = stats.t.ppf(1 - alpha / 2, k - 1)
        p_value = float(2 * stats.t.sf(abs(mu / se), k - 1)) if se > 0 else nan
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        p_value = float(2 * stats.norm.sf(abs(mu / se))) if se > 0 else nan
    if k >= 3:
        half = stats.t.ppf(1 - alpha / 2, k - 2) * np.sqrt(tau2 + se**2)
        pi = (mu - half, mu + half)
    else:
        pi = (nan, nan)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {
        "mu": mu,
        "se": se,
        "ci_low": float(mu - crit * se),
        "ci_high": float(mu + crit * se),
        "pi_low": float(pi[0]),
        "pi_high": float(pi[1]),
        "tau2": float(tau2),
        "i2": float(i2),
        "q": q,
        "p_value": p_value,
        "k": int(k),
    }


def hanley_mcneil_variance(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil (1982) approximation of the variance of an AUROC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    num = auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    return float(num / (n_pos * n_neg))


def site_aurocs(y: object, p: object, site: object, min_per_class: int = 5) -> list[dict]:
    """Per-site AUROC with DeLong variance for sites with enough observations per class.

    Args:
        y: Binary labels.
        p: Predicted scores.
        site: Site label per participant.
        min_per_class: Minimum number of positives and negatives for a site to be included
            (5 includes reporting tiers A and B, 10 tier A only).

    Returns:
        One dict per eligible site with ``site``, ``auroc``, ``variance``, ``n_pos``, ``n_neg``.
    """
    from molt.metrics.binary import auroc_variance_delong

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    site = np.asarray(site, dtype=object).astype(str)
    keep = np.isfinite(y) & np.isfinite(p)
    rows = []
    for s in np.unique(site[keep]):
        m = keep & (site == s)
        n_pos, n_neg = int((y[m] > 0.5).sum()), int((y[m] <= 0.5).sum())
        if min(n_pos, n_neg) < min_per_class:
            continue
        auc, var = auroc_variance_delong(y[m], p[m])
        rows.append({"site": s, "auroc": auc, "variance": var, "n_pos": n_pos, "n_neg": n_neg})
    return rows


def meta_analyze_auroc(
    aurocs: Sequence[float],
    variances: Sequence[float],
    n_pos: Sequence[int] | None = None,
    n_neg: Sequence[int] | None = None,
    alpha: float = 0.05,
    method: str = "REML",
    hartung_knapp: bool = True,
) -> dict[str, float]:
    """Random-effects pooling of per-site AUROCs on the logit scale.

    AUROCs of exactly 0 or 1 are shrunk by half an observation and degenerate variances are
    replaced by the Hanley-McNeil approximation when class counts are supplied. Results are
    back-transformed to the AUROC scale (``tau2`` stays on the logit scale).

    Returns:
        The output of :func:`random_effects_meta` with ``mu``, interval and prediction-interval
        bounds expressed as AUROCs, plus ``mu_logit``.
    """
    a = np.asarray(aurocs, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    if n_pos is not None and n_neg is not None:
        npos, nneg = np.asarray(n_pos, float), np.asarray(n_neg, float)
        eps = 0.5 / (npos * nneg)
        a = np.clip(a, eps, 1 - eps)
        hm = np.array(
            [hanley_mcneil_variance(ai, int(m), int(n)) for ai, m, n in zip(a, npos, nneg)]
        )
        v = np.where(np.isfinite(v) & (v > 0), v, hm)
    else:
        a = np.clip(a, 1e-4, 1 - 1e-4)
    y = np.log(a / (1 - a))
    v_logit = v / (a * (1 - a)) ** 2
    res = random_effects_meta(y, v_logit, method, hartung_knapp, alpha)
    expit = lambda x: float(1.0 / (1.0 + np.exp(-x))) if np.isfinite(x) else float("nan")
    res["mu_logit"] = res["mu"]
    for key in ("mu", "ci_low", "ci_high", "pi_low", "pi_high"):
        res[key] = expit(res[key])
    return res
