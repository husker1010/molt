"""PyTorch datasets and batch construction for MOLT.

:class:`MOLTDataset` wraps a :class:`~molt.training.preprocess.FeatureBundle` (and an
optional :class:`~molt.training.ct_patches.CTPatchSampler`) and yields participant-level
mini-batches in the dictionary format consumed by the model::

    {
      "clinical_num": (B, Fn) float, "clinical_num_mask": (B, Fn) bool,
      "clinical_cat": (B, Fc) long, "clinical_present": (B,) bool,
      "clinical_indicators": (B, Fi) float,
      "rna": (B, P) float, "rna_present": (B,) bool, "rna_genes": (B, G) float or absent,
      "rna_panel": (B, Gp) float or absent (gene-level expert input, NaN = not measured),
      "stacked_<expert>": (B, d) float per stacked linear expert (NaN = unavailable),
      "ct_patches": (B, N, 1, S, S, S) float or None, "ct_present": (B,) bool,
      "ct_features": (B, N, F) float (only with a precomputed feature store),
      "quality": {"rna": (B, Qr), "ct": (B, Qc), "clinical": (B, Qc2)},
      "domain": (B,) long, "y_egfr": (B,) float (NaN = unlabelled),
      "surv_bin": (B,) long, "surv_event": (B,) float, "surv_present": (B,) bool,
      "ipcw": (B,) float, "surv_status": (B,) float, "surv_time": (B,) float,
      "index": (B,) long,
    }

Tabular inputs are gathered by vectorised indexing in the collate function, so only CT
patch sampling runs per participant (optionally in DataLoader worker processes).
:class:`~molt.training.preprocess.Preprocessor` is re-exported here for convenience.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from molt.training.ct_features import CTFeatureStore
from molt.training.ct_patches import CTPatchConfig, CTPatchSampler
from molt.training.preprocess import FeatureBundle, PreprocessConfig, Preprocessor

__all__ = [
    "CTPatchConfig",
    "CTPatchSampler",
    "FeatureBundle",
    "MOLTDataset",
    "PreprocessConfig",
    "Preprocessor",
    "make_loader",
    "move_batch",
    "omit_modalities",
]


class MOLTDataset(Dataset):
    """Participant-level dataset over preprocessed features.

    Args:
        features: Model inputs and targets of the participants.
        ct_sampler: Patch sampler; ``None`` disables CT patches.
        train: Training mode (patch jitter/augmentation and epoch-dependent sampling).
        ct_store: Precomputed frozen-backbone patch features; when given, batches carry
            ``ct_features`` ``(B, N, F)`` instead of ``ct_patches``.
        augmenter: Optional train-time perturbation of the transcriptomic inputs
            (:class:`~molt.training.rna_augment.RNAAugmenter`), applied to training batches.
    """

    def __init__(
        self,
        features: FeatureBundle,
        ct_sampler: CTPatchSampler | None = None,
        *,
        train: bool = False,
        ct_store: CTFeatureStore | None = None,
        augmenter: Any | None = None,
    ) -> None:
        self.features = features
        self.augmenter = augmenter if train else None
        self.ct_sampler = None if ct_store is not None else ct_sampler
        self.ct_store = ct_store
        self.train = bool(train)
        self.epoch = 0
        self._tensors = self._to_tensors(features)

    @staticmethod
    def _to_tensors(f: FeatureBundle) -> dict[str, torch.Tensor | None]:
        def t(x: np.ndarray | None, dtype: torch.dtype) -> torch.Tensor | None:
            return None if x is None else torch.from_numpy(np.array(x, copy=True)).to(dtype)

        stacked = {
            f"stacked_{m}": t(v, torch.float32)
            for m, v in (getattr(f, "stacked", None) or {}).items()
        }
        return stacked | {
            "clinical_num": t(f.clinical_num, torch.float32),
            "clinical_num_mask": t(f.clinical_num_mask, torch.bool),
            "clinical_cat": t(f.clinical_cat, torch.long),
            "clinical_present": t(f.clinical_present, torch.bool),
            "clinical_indicators": t(f.clinical_indicators, torch.float32),
            "rna": t(f.rna, torch.float32),
            "rna_present": t(f.rna_present, torch.bool),
            "rna_genes": t(f.rna_genes, torch.float32),
            "rna_panel": t(getattr(f, "gene_panel", None), torch.float32),
            "ct_present": t(f.ct_present, torch.bool),
            "quality_rna": t(f.quality_rna, torch.float32),
            "quality_ct": t(f.quality_ct, torch.float32),
            "quality_clinical": t(f.quality_clinical, torch.float32),
            "domain": t(f.domain, torch.long),
            "y_egfr": t(f.y_egfr, torch.float32),
            "surv_bin": t(f.surv_bin, torch.long),
            "surv_event": t(f.surv_event, torch.float32),
            "surv_present": t(f.surv_present, torch.bool),
            "surv_status": t(f.surv_status, torch.float32),
            "surv_time": t(f.surv_time, torch.float32),
            "ipcw": t(f.ipcw, torch.float32),
        }

    @property
    def uses_ct(self) -> bool:
        """Whether CT patches are sampled from volumes (benefits from worker processes)."""
        return self.ct_sampler is not None and bool(np.any(self.features.ct_present))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that seeds CT patch sampling."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item: dict[str, Any] = {"index": int(i)}
        f = self.features
        if self.ct_store is not None:
            if f.ct_present[i] and f.ids[i] in self.ct_store:
                view = self.ct_store.view_for_epoch(self.epoch, self.train)
                item["ct_feat"] = self.ct_store.get(f.ids[i], view)
        elif self.ct_sampler is not None and f.ct_present[i] and f.ct_path[i]:
            epoch = self.epoch if self.train else -1
            item["ct"] = self.ct_sampler.sample(f.ct_path[i], f.ids[i], epoch, train=self.train)
        return item

    def collate(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Assemble a batch dictionary from dataset items."""
        idx = torch.as_tensor([it["index"] for it in items], dtype=torch.long)
        T = self._tensors
        batch: dict[str, Any] = {
            k: (v.index_select(0, idx) if v is not None else None)
            for k, v in T.items()
            if not k.startswith("quality_")
        }
        for key in ("rna_genes", "rna_panel"):
            if batch[key] is None:
                batch.pop(key)
        batch["quality"] = {
            "rna": T["quality_rna"].index_select(0, idx),
            "ct": T["quality_ct"].index_select(0, idx),
            "clinical": T["quality_clinical"].index_select(0, idx),
        }
        batch["index"] = idx
        if self.augmenter is not None:
            self.augmenter(batch, idx)
        ct_items = [it.get("ct") for it in items]
        feat_items = [it.get("ct_feat") for it in items]
        if any(c is not None for c in feat_items):
            ref = next(c for c in feat_items if c is not None)
            feats = np.zeros((len(items), *ref.shape), dtype=np.float32)
            present = torch.zeros(len(items), dtype=torch.bool)
            for b, c in enumerate(feat_items):
                if c is not None:
                    feats[b] = c
                    present[b] = True
            batch["ct_features"] = torch.from_numpy(feats)
            batch["ct_patches"] = None
            batch["ct_present"] = present
        elif any(c is not None for c in ct_items):
            ref = next(c for c in ct_items if c is not None)
            patches = np.zeros((len(items), *ref.shape), dtype=np.float32)
            present = torch.zeros(len(items), dtype=torch.bool)
            for b, c in enumerate(ct_items):
                if c is not None:
                    patches[b] = c
                    present[b] = True
            batch["ct_patches"] = torch.from_numpy(patches)
            batch["ct_present"] = present
        else:
            batch["ct_patches"] = None
            batch["ct_present"] = torch.zeros(len(items), dtype=torch.bool)
        return batch


def make_loader(
    dataset: MOLTDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader using the dataset's collate function.

    Worker processes are used only when CT patches are sampled; tabular batches are built
    in the main process.
    """
    workers = num_workers if dataset.uses_ct else 0
    generator = torch.Generator()
    generator.manual_seed(int(seed) % (2**63 - 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=dataset.collate,
        generator=generator,
        drop_last=drop_last and len(dataset) > batch_size,
        pin_memory=workers > 0 and torch.cuda.is_available(),
        persistent_workers=False,
    )


def move_batch(
    batch: Mapping[str, Any], device: torch.device | str, non_blocking: bool = True
) -> dict[str, Any]:
    """Move every tensor of a (nested) batch to ``device``."""

    def move(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=non_blocking)
        if isinstance(v, Mapping):
            return {k: move(x) for k, x in v.items()}
        return v

    return {k: move(v) for k, v in batch.items()}


def omit_modalities(batch: Mapping[str, Any], omit: Sequence[str]) -> dict[str, Any]:
    """Copy of ``batch`` with the given experts (``"rna"``, ``"ct"``, ``"clinical"``) absent.

    Used to obtain the light mode (transcriptomic expert omitted) and missing-modality stress
    tests from one fitted model.
    """
    out = dict(batch)
    for name in omit:
        key = f"{name}_present"
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = torch.zeros_like(out[key])
        if name == "ct":
            out["ct_patches"] = None
            out.pop("ct_features", None)
    return out


def iterate_batches(loader: DataLoader, device: torch.device | str) -> Iterator[dict[str, Any]]:
    """Yield batches from ``loader`` moved to ``device``."""
    for batch in loader:
        yield move_batch(batch, device)
