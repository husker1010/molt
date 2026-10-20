"""Optimisation of single networks and deep ensembles with uncertainty-aware prediction.

:class:`Trainer` fits one network with AdamW, a linear-warmup cosine learning-rate schedule,
gradient clipping, optional bf16 mixed precision and modality dropout, and stops early on a
validation criterion (restoring the best epoch): the endpoint loss (``monitor="endpoint"``,
default), the total loss (``"total"``) or the endpoint discrimination metric (``"metric"``:
AUROC for EGFR, Harrell's C for survival endpoints). The task heads can use a smaller learning
rate than the encoders (``head_lr_scale``), the gene-level expert can be initialised by a
denoising autoencoder on the training profiles (``pretrain_epochs``), and models exposing
``set_epoch`` are told the current epoch (modality-dropout curriculum).
:meth:`Trainer.fit_ensemble` trains ``M`` independently seeded members. :class:`Ensemble` predicts with every member (deterministic
pass) and, optionally, Monte Carlo dropout passes, returning point predictions and the
ensemble and MC-dropout spreads used for abstention.

Every epoch is logged through :class:`molt.utils.status.MetricsLogger` (train/val loss and a
validation discrimination metric) and reported to an optional progress callback.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from torch import nn

from molt.training.data import MOLTDataset, make_loader, move_batch, omit_modalities
from molt.utils.logging import get_logger
from molt.utils.status import MetricsLogger

__all__ = [
    "Ensemble",
    "EnsemblePrediction",
    "MemberResult",
    "TrainSettings",
    "Trainer",
    "harrell_c",
    "load_trainable_state",
    "trainable_state",
]

log = get_logger(__name__)

_ENDPOINT_KEYS = {
    "egfr": ("egfr", "loss_egfr", "focal"),
    "pfi": ("surv", "loss_surv", "survival", "nll"),
    "os": ("surv", "loss_surv", "survival", "nll"),
}


@dataclass
class TrainSettings:
    """Optimisation settings of one network (see :class:`Trainer`)."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    eval_batch_size: int = 64
    max_epochs: int = 60
    min_epochs: int = 3
    patience: int = 8
    warmup_epochs: float = 2.0
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    amp: bool = True
    members: int = 5
    mc_samples: int = 10
    modality_dropout: bool = True
    num_workers: int = 4
    monitor: str = "endpoint"
    ct_backbone_lr_scale: float = 0.1
    head_lr_scale: float = 1.0
    pretrain_epochs: int = 0
    pretrain_lr: float = 1e-3
    pretrain_mask: float = 0.25
    pretrain_noise: float = 0.1
    grl_schedule: bool = True
    seed: int = 20240613

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None, **overrides: Any) -> TrainSettings:
        merged = {**dict(data or {}), **overrides}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in names and v is not None})


@dataclass
class MemberResult:
    """Outcome of training one ensemble member."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val: float
    epochs_run: int
    seed: int
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index (higher risk = earlier event; ties count one half)."""
    time, event, risk = (np.asarray(a, dtype=np.float64) for a in (time, event, risk))
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[ok], event[ok], risk[ok]
    if event.sum() == 0 or len(time) < 2:
        return float("nan")
    cases = event == 1
    comparable = time[cases][:, None] < time[None, :]
    diff = risk[cases][:, None] - risk[None, :]
    conc = ((diff > 0) & comparable).sum() + 0.5 * ((diff == 0) & comparable).sum()
    total = comparable.sum()
    return float(conc / total) if total else float("nan")


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _floatify(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, Mapping):
        return {k: _floatify(v) for k, v in obj.items()}
    return obj


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _cosine_lambda(total_steps: int, warmup_steps: int, min_ratio: float) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the complete model state, including frozen parameters, to the CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_trainable_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load a state saved by :func:`trainable_state`; only frozen parameters may be absent."""
    result = model.load_state_dict(dict(state), strict=False)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    missing = set(result.missing_keys) - frozen
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"State dict mismatch: missing {sorted(missing)[:5]}, "
            f"unexpected {list(result.unexpected_keys)[:5]}"
        )


def _dropout_modules(model: nn.Module) -> list[nn.Module]:
    kinds = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    return [m for m in model.modules() if isinstance(m, kinds)]


class Trainer:
    """Train networks that honour the MOLT interface.

    Args:
        build_model: Zero-argument factory returning a freshly initialised network.
        loss_weights: Loss-weight object passed to ``model.compute_loss``.
        settings: Optimisation settings.
        task: ``egfr``, ``pfi`` or ``os`` (selects the monitored endpoint loss and metric).
        device: Torch device.
        metrics: Optional metrics stream.
        context: ``outer_fold``/``inner_fold``/``tag`` attached to every metric record.
        on_epoch: Callback ``f(member, epoch, max_epochs)`` after every epoch.
    """

    def __init__(
        self,
        build_model: Callable[[], nn.Module],
        loss_weights: Any,
        settings: TrainSettings,
        *,
        task: str = "egfr",
        device: str | torch.device = "cpu",
        metrics: MetricsLogger | None = None,
        context: Mapping[str, Any] | None = None,
        on_epoch: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self.build_model = build_model
        self.loss_weights = loss_weights
        self.settings = settings
        self.task = task
        self.device = torch.device(device)
        self.metrics = metrics
        self.context = dict(context or {})
        self.on_epoch = on_epoch
        self.use_amp = bool(settings.amp) and self.device.type == "cuda"

    # ------------------------------------------------------------------ helpers
    def _autocast(self) -> Any:
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        s = self.settings
        if hasattr(model, "param_groups") and callable(model.param_groups):
            extra = {}
            if _accepts(model.param_groups, "head_lr_scale"):
                extra["head_lr_scale"] = s.head_lr_scale
            groups = model.param_groups(
                weight_decay=s.weight_decay,
                lr=s.lr,
                ct_backbone_lr_scale=s.ct_backbone_lr_scale,
                **extra,
            )
        else:
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    (no_decay if p.ndim < 2 or "embedding" in name else decay).append(p)
            groups = [
                {"params": decay, "weight_decay": s.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
        return torch.optim.AdamW(groups, lr=s.lr, betas=(0.9, 0.999))

    def _log(self, metric: str, value: float, split: str, epoch: int | None, member: Any) -> None:
        if self.metrics is None:
            return
        tag = self.context.get("tag")
        self.metrics.log(
            metric,
            value,
            split=split,
            epoch=epoch,
            outer_fold=self.context.get("outer_fold"),
            inner_fold=self.context.get("inner_fold"),
            member=f"{tag}/m{member}" if tag is not None else member,
        )

    def _monitored(self, val: Mapping[str, float]) -> float:
        """Early-stopping criterion of a validation pass (lower is better)."""
        monitor = self.settings.monitor
        if monitor == "total":
            return float(val["total"])
        if monitor == "metric":
            value = val.get("auroc" if self.task == "egfr" else "c_index")
            return -float(value) if value is not None and np.isfinite(value) else math.inf
        if monitor != "endpoint":
            raise ValueError(f"Unknown train.monitor {monitor!r} (endpoint, total or metric)")
        return float(val["endpoint"])

    def _endpoint_loss(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.settings.monitor == "total":
            return losses["total"]
        for key in _ENDPOINT_KEYS.get(self.task, ()):
            if key in losses:
                return losses[key]
        return losses["total"]

    def _forward(
        self, model: nn.Module, batch: Mapping[str, Any], *, train: bool
    ) -> dict[str, Any]:
        with self._autocast():
            out = model(batch, dropout_modalities=train and self.settings.modality_dropout)
        return _floatify(out)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataset: MOLTDataset) -> dict[str, float]:
        """Validation loss (total and monitored endpoint) and discrimination."""
        model.eval()
        loader = make_loader(
            dataset,
            batch_size=self.settings.eval_batch_size,
            shuffle=False,
            num_workers=self.settings.num_workers,
        )
        sums: dict[str, float] = {}
        n_total = 0
        logits, risks, hz_idx = [], [], None
        for batch in loader:
            batch = move_batch(batch, self.device)
            out = self._forward(model, batch, train=False)
            losses = model.compute_loss(out, batch, self.loss_weights)
            b = int(batch["clinical_cat"].shape[0])
            n_total += b
            sums["total"] = sums.get("total", 0.0) + float(losses["total"]) * b
            sums["endpoint"] = sums.get("endpoint", 0.0) + float(self._endpoint_loss(losses)) * b
            logits.append(out["logit_egfr"].detach().cpu())
            surv = out["survival"].detach().cpu()
            hz_idx = getattr(getattr(model, "config", None), "primary_horizon_index", None)
            idx = surv.shape[1] - 1 if hz_idx is None else min(int(hz_idx), surv.shape[1] - 1)
            risks.append(1.0 - surv[:, idx])
        res = {k: v / max(n_total, 1) for k, v in sums.items()}
        f = dataset.features
        if self.task == "egfr":
            p = torch.sigmoid(torch.cat(logits)).numpy()
            res["auroc"] = _auroc(f.y_egfr.astype(float), p)
        else:
            r = torch.cat(risks).numpy()
            sp = f.surv_present
            res["c_index"] = harrell_c(f.surv_time[sp], f.surv_event[sp], r[sp])
        return res

    # ------------------------------------------------------------------ training
    def fit_member(
        self,
        train_ds: MOLTDataset,
        val_ds: MOLTDataset | None,
        *,
        member: int = 0,
        seed: int = 0,
    ) -> MemberResult:
        """Train one network; early stopping on ``val_ds`` when it is non-empty."""
        s = self.settings
        torch.manual_seed(seed % (2**63 - 1))
        np.random.seed(seed % (2**32 - 1))
        model = self.build_model().to(self.device)
        self._pretrain(model, train_ds, seed)
        opt = self._optimizer(model)
        loader = make_loader(
            train_ds,
            batch_size=s.batch_size,
            shuffle=True,
            num_workers=s.num_workers,
            seed=seed,
            drop_last=True,
        )
        steps_per_epoch = max(len(loader), 1)
        total = steps_per_epoch * s.max_epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, _cosine_lambda(total, int(s.warmup_epochs * steps_per_epoch), s.min_lr_ratio)
        )
        has_val = val_ds is not None and len(val_ds) > 0
        best_val, best_epoch, best_state, bad = math.inf, -1, None, 0
        history: list[dict[str, float]] = []
        start = time.time()
        step = 0
        epochs_run = 0
        for epoch in range(s.max_epochs):
            model.train()
            train_ds.set_epoch(epoch)
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)
            if s.grl_schedule and hasattr(model, "set_grl_alpha"):
                progress = epoch / max(s.max_epochs - 1, 1)
                model.set_grl_alpha(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            run_loss, run_n = 0.0, 0
            for batch in loader:
                batch = move_batch(batch, self.device)
                out = self._forward(model, batch, train=True)
                losses = model.compute_loss(out, batch, self.loss_weights)
                loss = losses["total"]
                if not torch.isfinite(loss):
                    log.warning("non-finite training loss at epoch %d; batch skipped", epoch)
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if s.grad_clip and s.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
                opt.step()
                sched.step()
                step += 1
                b = int(batch["clinical_cat"].shape[0])
                run_loss += float(loss.detach()) * b
                run_n += b
            epochs_run = epoch + 1
            record = {
                "epoch": epoch,
                "train_loss": run_loss / max(run_n, 1),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            self._log("loss", record["train_loss"], "train", epoch, member)
            if has_val:
                val = self.evaluate(model, val_ds)
                monitored = self._monitored(val)
                record.update({f"val_{k}": v for k, v in val.items()})
                self._log("loss", val["total"], "val", epoch, member)
                self._log("endpoint_loss", val["endpoint"], "val", epoch, member)
                metric = "auroc" if self.task == "egfr" else "c_index"
                self._log(metric, val.get(metric), "val", epoch, member)
                if monitored < best_val - 1e-6 or best_state is None:
                    best_val, best_epoch, bad = monitored, epoch, 0
                    best_state = trainable_state(model)
                else:
                    bad += 1
            history.append(record)
            if self.on_epoch is not None:
                self.on_epoch(member, epoch, s.max_epochs)
            if has_val and bad >= s.patience and epoch + 1 >= s.min_epochs:
                break
        if best_state is None:
            best_state = trainable_state(model)
            best_epoch, best_val = epochs_run - 1, float("nan")
        return MemberResult(
            best_state, best_epoch, float(best_val), epochs_run, seed, history, time.time() - start
        )

    def _pretrain(self, model: nn.Module, train_ds: MOLTDataset, seed: int) -> None:
        """Denoising-autoencoder initialization of the gene-level expert (if configured)."""
        s = self.settings
        encoder = getattr(model, "gene_encoder", None)
        panel = getattr(train_ds.features, "gene_panel", None)
        if s.pretrain_epochs <= 0 or encoder is None or panel is None:
            return
        from molt.training.pretrain import pretrain_gene_encoder

        rows = np.asarray(train_ds.features.rna_present, dtype=bool)
        result = pretrain_gene_encoder(
            encoder,
            panel[rows],
            epochs=s.pretrain_epochs,
            lr=s.pretrain_lr,
            mask_ratio=s.pretrain_mask,
            noise_std=s.pretrain_noise,
            device=self.device,
            seed=seed,
        )
        if result is not None:
            log.debug(
                "gene encoder pretrained on %d profiles: loss %.3f -> %.3f",
                result.n_profiles,
                result.first_loss,
                result.last_loss,
            )

    def fit_ensemble(
        self,
        train_ds: MOLTDataset,
        val_ds: MOLTDataset | None,
        seeds: Sequence[int],
    ) -> tuple[Ensemble, list[MemberResult]]:
        """Train one member per seed and return the ensemble and per-member results."""
        results = [
            self.fit_member(train_ds, val_ds, member=m, seed=sd) for m, sd in enumerate(seeds)
        ]
        ens = Ensemble(
            self.build_model,
            [r.state_dict for r in results],
            device=self.device,
            amp=self.settings.amp,
        )
        return ens, results


@dataclass
class EnsemblePrediction:
    """Predictions of every ensemble member for an ordered list of participants.

    Attributes:
        ids: Participant ids.
        p_members: ``(M, n)`` EGFR probabilities of the deterministic pass of each member.
        p_mc_sd: ``(M, n)`` MC-dropout standard deviation of each member's probability.
        surv_members: ``(M, n, K)`` survival curves at the bin boundaries.
        risk_mc_sd: ``(M, n)`` MC-dropout standard deviation of the horizon risk.
        rho: ``(n, 3)`` mean reliability weights (clinical, rna, ct).
        horizon_index: Survival column of the primary horizon.
    """

    ids: list[str]
    p_members: np.ndarray
    p_mc_sd: np.ndarray
    surv_members: np.ndarray
    risk_mc_sd: np.ndarray
    rho: np.ndarray
    horizon_index: int

    @property
    def p(self) -> np.ndarray:
        """Ensemble-mean EGFR probability."""
        return self.p_members.mean(0)

    @property
    def p_ensemble_sd(self) -> np.ndarray:
        return (
            self.p_members.std(0) if len(self.p_members) > 1 else np.zeros(self.p_members.shape[1])
        )

    @property
    def p_mc(self) -> np.ndarray:
        return np.sqrt((self.p_mc_sd**2).mean(0))

    @property
    def survival(self) -> np.ndarray:
        """Ensemble-mean survival curve ``(n, K)``."""
        return self.surv_members.mean(0)

    @property
    def risk(self) -> np.ndarray:
        """Ensemble-mean risk at the primary horizon."""
        return 1.0 - self.survival[:, self.horizon_index]

    @property
    def risk_ensemble_sd(self) -> np.ndarray:
        r = 1.0 - self.surv_members[:, :, self.horizon_index]
        return r.std(0) if len(r) > 1 else np.zeros(r.shape[1])

    @property
    def risk_mc(self) -> np.ndarray:
        return np.sqrt((self.risk_mc_sd**2).mean(0))

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "p_members": self.p_members,
            "p_mc_sd": self.p_mc_sd,
            "surv_members": self.surv_members,
            "risk_mc_sd": self.risk_mc_sd,
            "rho": self.rho,
        }

    @staticmethod
    def concat_members(preds: Sequence[EnsemblePrediction]) -> EnsemblePrediction:
        """Pool the members of several ensembles scored on the same participants."""
        first = preds[0]
        return EnsemblePrediction(
            ids=first.ids,
            p_members=np.concatenate([p.p_members for p in preds]),
            p_mc_sd=np.concatenate([p.p_mc_sd for p in preds]),
            surv_members=np.concatenate([p.surv_members for p in preds]),
            risk_mc_sd=np.concatenate([p.risk_mc_sd for p in preds]),
            rho=np.mean([p.rho for p in preds], axis=0),
            horizon_index=first.horizon_index,
        )


class Ensemble:
    """A deep ensemble of networks sharing one architecture.

    Args:
        build_model: Factory of the architecture.
        states: One ``state_dict`` per member.
        device: Device used for prediction.
        amp: Use bf16 autocast on CUDA.
    """

    def __init__(
        self,
        build_model: Callable[[], nn.Module],
        states: Sequence[Mapping[str, torch.Tensor]],
        *,
        device: str | torch.device = "cpu",
        amp: bool = True,
    ) -> None:
        self.build_model = build_model
        self.states = list(states)
        self.device = torch.device(device)
        self.amp = bool(amp) and self.device.type == "cuda"

    def __len__(self) -> int:
        return len(self.states)

    def _autocast(self) -> Any:
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _run(
        self, model: nn.Module, batch: Mapping[str, Any], omit: Sequence[str]
    ) -> dict[str, Any]:
        with self._autocast():
            if omit and _accepts(model.forward, "omit_modalities"):
                out = model(batch, dropout_modalities=False, omit_modalities=tuple(omit))
            else:
                out = model(
                    omit_modalities(batch, omit) if omit else batch, dropout_modalities=False
                )
        return _floatify(out)

    @torch.no_grad()
    def predict(
        self,
        dataset: MOLTDataset,
        *,
        omit: Sequence[str] = (),
        mc_samples: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> EnsemblePrediction:
        """Score ``dataset`` with every member.

        Args:
            dataset: Participants to score (evaluation mode, fixed CT patch sets).
            omit: Experts omitted for every participant (``("rna",)`` gives the light mode).
            mc_samples: Number of MC-dropout passes per member (0 disables).
            batch_size: Prediction batch size.
            num_workers: DataLoader workers (used only when CT patches are loaded).
        """
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        n = len(dataset)
        models = []
        for state in self.states:
            model = self.build_model().to(self.device)
            load_trainable_state(model, state)
            model.eval()
            models.append(model)
        m_count = len(models)
        h_idx = (
            getattr(getattr(models[0], "config", None), "primary_horizon_index", None)
            if models
            else None
        )
        chunks: list[list[list[torch.Tensor]]] = [[[] for _ in range(m_count)] for _ in range(5)]
        for batch in loader:
            batch = move_batch(batch, self.device)
            for m, model in enumerate(models):
                p, p_sd, surv, r_sd, rho = self._predict_batch(
                    model, batch, omit, mc_samples, h_idx
                )
                for slot, value in zip(chunks, (p, p_sd, surv, r_sd, rho)):
                    slot[m].append(value)
        stacked = [
            [torch.cat(slot[m]).numpy() for m in range(m_count)] if n else [] for slot in chunks
        ]
        k = stacked[2][0].shape[1] if m_count and n else 1
        return EnsemblePrediction(
            ids=list(dataset.features.ids),
            p_members=np.stack(stacked[0]) if m_count and n else np.zeros((m_count, n)),
            p_mc_sd=np.stack(stacked[1]) if m_count and n else np.zeros((m_count, n)),
            surv_members=np.stack(stacked[2]) if m_count and n else np.zeros((m_count, n, k)),
            risk_mc_sd=np.stack(stacked[3]) if m_count and n else np.zeros((m_count, n)),
            rho=np.mean(stacked[4], axis=0) if m_count and n else np.zeros((n, 3)),
            horizon_index=k - 1 if h_idx is None else min(int(h_idx), k - 1),
        )

    def _predict_batch(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        omit: Sequence[str],
        mc_samples: int,
        h_idx: int | None,
    ) -> tuple[torch.Tensor, ...]:
        out = self._run(model, batch, omit)
        p = torch.sigmoid(out["logit_egfr"]).cpu()
        surv = out["survival"].cpu()
        hi = surv.shape[1] - 1 if h_idx is None else min(int(h_idx), surv.shape[1] - 1)
        weights = out.get("expert_weights", {})
        rho = torch.stack(
            [weights.get(m, torch.zeros_like(p)).float().cpu() for m in ("clinical", "rna", "ct")],
            1,
        )
        drop = _dropout_modules(model)
        if mc_samples <= 0 or not drop:
            return p, torch.zeros_like(p), surv, torch.zeros_like(p), rho
        mc_batch = dict(batch)
        imaging = getattr(model, "imaging", None)
        if (
            imaging is not None
            and mc_batch.get("ct_patches") is not None
            and hasattr(imaging, "extract_features")
        ):
            with self._autocast():
                mc_batch["ct_features"] = imaging.extract_features(mc_batch["ct_patches"]).float()
        samples_p, samples_r = [], []
        try:
            for module in drop:
                module.train()
            for _ in range(mc_samples):
                o = self._run(model, mc_batch, omit)
                samples_p.append(torch.sigmoid(o["logit_egfr"]).cpu())
                samples_r.append(1.0 - o["survival"][:, hi].cpu())
        finally:
            for module in drop:
                module.eval()
        p_sd = torch.stack(samples_p).std(0, unbiased=False)
        r_sd = torch.stack(samples_r).std(0, unbiased=False)
        return p, p_sd, surv, r_sd, rho

    def save(self, directory: Any, *, extra: Mapping[str, Any] | None = None) -> None:
        """Write one ``member{m}.pt`` per member (state dict plus ``extra`` metadata)."""
        from pathlib import Path

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for m, state in enumerate(self.states):
            torch.save({"state_dict": dict(state), **dict(extra or {})}, out / f"member{m}.pt")

    @classmethod
    def load(
        cls,
        directory: Any,
        build_model: Callable[[], nn.Module],
        *,
        device: str | torch.device = "cpu",
        amp: bool = True,
    ) -> Ensemble:
        """Load members written by :meth:`save`."""
        from pathlib import Path

        files = sorted(Path(directory).glob("member*.pt"), key=lambda p: int(p.stem[6:]))
        states = [
            torch.load(f, map_location="cpu", weights_only=False)["state_dict"] for f in files
        ]
        return cls(build_model, states, device=device, amp=amp)
