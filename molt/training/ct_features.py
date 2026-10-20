"""Frozen-backbone CT patch features, computed once and shared across folds.

Fine-tuning a 3-D transformer backbone inside every inner fold of a nested cross-validation
is rarely affordable. With ``data.ct.feature_cache: true`` the patch backbone (initialised
from the public checkpoint) is kept frozen: for every participant with a selected CT, the
backbone features of ``views`` patch sets are computed once - view 0 is the deterministic
evaluation patch set, views ``1..V-1`` are augmented training patch sets - and stored as
``float16`` arrays ``(V, n_patches, feature_dim)``. Training epochs cycle through the
augmented views; prediction uses view 0. Only the patch projection, the attention pooling and
the rest of the network are trained.

Features do not depend on any label or fold, so one cache serves every endpoint, fold,
configuration and external cohort that share the backbone and patch settings.

Two on-disk layouts are read:

* the cohort layout accepted by the feature-cache reader:
  ``<feature_dir>/<cohort>/<participant_id>.npz`` with ``features`` ``(V, N, F)`` float16,
  the patch ``centres`` ``(V, N, 3)`` (array index order ``z, y, x``), the cache ``key`` and a
  JSON ``meta`` entry; perturbation variants live in
  ``<feature_dir>/<cohort>/variants/<variant>/<participant_id>.npz`` (see
  :data:`FEATURE_VARIANTS`) and a ``manifest.json`` lists every file;
* the key layout ``<feature_dir>/<key>/<participant_id>.npy`` written on demand by training
  runs whose backbone or patch settings differ from the cohort cache.

A cohort-layout file is used only when its stored key equals the key of the run, so a cache
built with other settings is never mixed into a run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from molt.training.ct_patches import CTPatchSampler
from molt.utils.logging import get_logger

__all__ = [
    "FEATURE_VARIANTS",
    "CTFeatureStore",
    "build_feature_store",
    "feature_cache_key",
    "feature_file",
    "imaging_settings",
    "load_feature_file",
    "save_feature_file",
]

FEATURE_VARIANTS: dict[str, str] = {
    "base": "view 0: deterministic evaluation set (32 lung-field + 32 candidate patches); "
    "views 1-3: augmented training sets",
    "patches96": "deterministic evaluation set of 96 patches (48 lung-field + 48 candidate)",
    "tta": "4 label-preserving test-time augmentations of the view-0 patches (left-right flip, "
    "anterior-posterior flip, in-plane 90-degree rotation, cranio-caudal flip)",
    "noise_hu20": "view-0 patches with additive Gaussian noise (sigma 20 HU) before windowing",
    "alt_kernel": "view-0 patch centres applied to the paired alternate-kernel series of the "
    "same examination (centres mapped through the volume origins)",
}
"""Cached feature variants. Mean instead of attention pooling (``pool_mean``) needs no cache of
its own: it is an inference-time setting applied to the ``base`` features."""

log = get_logger(__name__)


class CTFeatureStore:
    """In-memory map ``participant_id -> (V, N, F)`` patch features."""

    def __init__(self, features: dict[str, np.ndarray], n_views: int) -> None:
        self.features = features
        self.n_views = int(n_views)

    def __contains__(self, participant_id: str) -> bool:
        return participant_id in self.features

    def get(self, participant_id: str, view: int = 0) -> np.ndarray:
        """Features ``(N, F)`` of one view as float32."""
        arr = self.features[participant_id]
        return np.asarray(arr[int(view) % arr.shape[0]], dtype=np.float32)

    def view_for_epoch(self, epoch: int, train: bool) -> int:
        """View index used at ``epoch`` (augmented views in training, view 0 otherwise)."""
        if not train or self.n_views <= 1:
            return 0
        return 1 + int(epoch) % (self.n_views - 1)


def _checkpoint_digest(path: str | None) -> str:
    if not path or not os.path.isfile(path):
        return "random"
    stat = os.stat(path)
    return f"{Path(path).name}:{stat.st_size}:{int(stat.st_mtime)}"


def feature_cache_key(model_cfg: Any, sampler: CTPatchSampler) -> str:
    """Stable key of the backbone and patch settings that determine the features."""
    settings = imaging_settings(model_cfg)
    fields = {
        name: getattr(settings, name, None)
        for name in ("ct_backbone", "swin_feature_size", "cnn_widths", "ct_hu_window")
    }
    fields["checkpoint"] = _checkpoint_digest(getattr(settings, "swin_checkpoint", None))
    payload = {"model": fields, "patches": asdict(sampler.config), "seed": sampler.seed}
    payload["patches"].pop("cache_dir", None)
    payload["patches"].pop("memory_cache", None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:16]


def imaging_settings(model: Any) -> SimpleNamespace:
    """Backbone settings of a model configuration or a ``model`` mapping, for cache keys.

    Mappings (for example the ``model`` section of a run configuration) are completed with the
    defaults of the MOLT configuration; patches are passed to the backbone already windowed, so
    ``ct_hu_window`` is ``None`` as in :func:`molt.training.model_factory.model_kwargs`.
    """
    if not isinstance(model, Mapping):
        return SimpleNamespace(
            **{
                name: getattr(model, name, None)
                for name in (
                    "ct_backbone",
                    "swin_feature_size",
                    "cnn_widths",
                    "ct_hu_window",
                    "swin_checkpoint",
                )
            }
        )
    defaults = {
        "ct_backbone": "cnn3d",
        "swin_feature_size": 48,
        "cnn_widths": (32, 64, 128, 256),
        "swin_checkpoint": None,
    }
    values = {k: model.get(k, v) if model.get(k) is not None else v for k, v in defaults.items()}
    values["cnn_widths"] = tuple(int(w) for w in values["cnn_widths"])
    values["ct_hu_window"] = None
    return SimpleNamespace(**values)


def feature_file(
    root: str | os.PathLike[str], cohort: str, participant_id: str, variant: str = "base"
) -> Path:
    """Path of a cached feature file in the cohort layout."""
    base = Path(root) / str(cohort)
    if variant == "base":
        return base / f"{participant_id}.npz"
    return base / "variants" / str(variant) / f"{participant_id}.npz"


def save_feature_file(
    path: str | os.PathLike[str],
    features: np.ndarray,
    *,
    key: str,
    centres: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``features`` ``(V, N, F)`` (float16) with centres, key and metadata atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "features": np.asarray(features, dtype=np.float16),
        "key": np.asarray(str(key)),
        "meta": np.asarray(json.dumps(dict(meta or {}), default=str)),
    }
    if centres is not None:
        arrays["centres"] = np.asarray(centres, dtype=np.int16)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_feature_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a cohort-layout feature file into ``features``, ``centres``, ``key`` and ``meta``."""
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {
            "features": np.asarray(data["features"]),
            "key": str(data["key"]) if "key" in data.files else "",
            "centres": np.asarray(data["centres"]) if "centres" in data.files else None,
        }
        raw = str(data["meta"]) if "meta" in data.files else "{}"
    try:
        out["meta"] = json.loads(raw)
    except ValueError:
        out["meta"] = {}
    return out


def _cohort_cached(
    root: Path | None, volume_path: str, pid: str, key: str | None, views: int
) -> np.ndarray | None:
    """Features of the cohort layout when the stored key matches and enough views exist."""
    if root is None or key is None:
        return None
    target = feature_file(root, Path(volume_path).parent.name, pid)
    if not _fresh(target, volume_path):
        return None
    try:
        cached = load_feature_file(target)
    except (OSError, ValueError, KeyError):
        return None
    arr = cached["features"]
    if cached["key"] != key or arr.ndim != 3 or arr.shape[0] < views:
        return None
    return arr[:views]


def _fresh(target: Path | None, volume_path: str) -> bool:
    """Whether a cached feature file exists and is not older than its CT volume."""
    if target is None or not target.exists():
        return False
    try:
        return os.path.getmtime(target) >= os.path.getmtime(volume_path)
    except OSError:
        return True


@torch.no_grad()
def build_feature_store(
    build_model: Callable[[], nn.Module],
    sampler: CTPatchSampler,
    items: Sequence[tuple[str, str]],
    *,
    views: int = 4,
    device: str | torch.device = "cpu",
    cache_dir: str | os.PathLike[str] | None = None,
    key: str | None = None,
    amp: bool = True,
) -> CTFeatureStore:
    """Compute (or load from the cache) backbone features of every ``(pid, npz_path)``.

    The cohort layout ``cache_dir/<cohort>/<pid>.npz`` (cohort = directory name of the CT
    volume) is used when its stored key equals ``key``; otherwise ``cache_dir/key/<pid>.npy``
    is read or written. A cached file older than its CT volume (for example after the volume
    was reprocessed) is recomputed.

    Args:
        build_model: Factory of a network with an ``imaging`` encoder exposing
            ``extract_features``.
        sampler: Patch sampler (view 0 = evaluation patches, others = augmented patches).
        items: Participants and their CT volume paths.
        views: Number of views per participant.
        device: Device used for feature extraction.
        cache_dir: Root of the on-disk cache (``None`` keeps features in memory only).
        key: Cache key (see :func:`feature_cache_key`).
        amp: bf16 autocast on CUDA.
    """
    device = torch.device(device)
    out_dir = Path(cache_dir) / key if cache_dir and key else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cache_dir) if cache_dir else None
    features: dict[str, np.ndarray] = {}
    todo = []
    n_cohort = 0
    for pid, path in items:
        from_cohort = _cohort_cached(root, path, pid, key, views)
        if from_cohort is not None:
            features[pid] = from_cohort
            n_cohort += 1
            continue
        target = out_dir / f"{pid}.npy" if out_dir is not None else None
        cached = np.load(target) if _fresh(target, path) else None
        if cached is not None and cached.shape[0] >= views:
            features[pid] = cached[:views]
        else:
            todo.append((pid, path, target))
    if n_cohort:
        log.info("read CT features of %d participant(s) from the cohort cache", n_cohort)
    if todo:
        model = build_model().to(device).eval()
        imaging = getattr(model, "imaging", None)
        if imaging is None or not hasattr(imaging, "extract_features"):
            raise ValueError("The model has no imaging encoder with extract_features")
        log.info(
            "computing CT backbone features for %d participant(s) x %d view(s)", len(todo), views
        )
        for pid, path, target in todo:
            per_view = []
            for v in range(views):
                patches = sampler.sample(path, pid, epoch=v - 1, train=v > 0)
                x = torch.from_numpy(patches).unsqueeze(0).to(device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
                ):
                    feats = imaging.extract_features(x)
                per_view.append(feats[0].float().cpu().numpy())
            arr = np.stack(per_view).astype(np.float16)
            features[pid] = arr
            if target is not None:
                tmp = target.with_name(target.name + f".{os.getpid()}.tmp.npy")
                np.save(tmp, arr)
                os.replace(tmp, target)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return CTFeatureStore(features, views)
