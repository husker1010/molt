"""Training objectives: focal BCE, discrete-time survival likelihood, Brier terms and domain loss.

Optional terms: an IPCW-weighted pairwise ranking loss on the horizon risk, label smoothing of
the EGFR target, deep supervision of every expert (the endpoint loss applied to the outputs of
each single-expert posterior) and an L1 penalty on the gene-level expert.

All losses accept partially labelled batches: entries with NaN labels (or a false availability
mask) are excluded, and a loss without any valid entry returns a zero that remains attached to
the computation graph so that ``total.backward()`` is always defined.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from molt.models.heads import grad_reverse

__all__ = [
    "LossWeights",
    "brier_binary",
    "compute_multitask_loss",
    "discrete_time_nll",
    "domain_ce",
    "focal_bce",
    "grl",
    "info_nce",
    "ipcw_brier_survival",
    "pairwise_ranking_loss",
]

_EPS = 1e-6


def _zero(ref: Tensor) -> Tensor:
    return ref.sum() * 0.0


def focal_bce(
    logit: Tensor,
    y: Tensor,
    pos_weight: float | Tensor | None = None,
    gamma: float = 0.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Class-weighted focal binary cross-entropy averaged over labelled samples.

    ``l_i = -w_{y_i} (1 - p_{i,y_i})^gamma log p_{i,y_i}`` with ``w_1 = pos_weight`` and
    ``w_0 = 1``; ``gamma = 0`` gives weighted binary cross-entropy.

    Args:
        logit: ``(B,)`` logits.
        y: ``(B,)`` labels in ``{0, 1}`` (soft labels allowed), NaN when unlabelled.
        pos_weight: Weight of the positive class (e.g. training negatives / positives).
        gamma: Focusing parameter.
        mask: Optional ``(B,)`` boolean mask of samples to include.
    """
    y = y.to(logit.device)
    valid = torch.isfinite(y.float())
    if mask is not None:
        valid = valid & mask.to(device=logit.device, dtype=torch.bool)
    if not bool(valid.any()):
        return _zero(logit)
    z, t = logit[valid].float(), y[valid].float()
    log_p, log_q = F.logsigmoid(z), F.logsigmoid(-z)
    log_pt = t * log_p + (1.0 - t) * log_q
    pw = 1.0 if pos_weight is None else pos_weight
    if isinstance(pw, Tensor):
        pw = pw.to(z.device, z.dtype)
    weight = t * pw + (1.0 - t)
    loss = -weight * log_pt
    if gamma:
        loss = loss * (1.0 - log_pt.exp()).clamp_min(0.0).pow(gamma)
    return loss.mean()


def discrete_time_nll(
    hazards: Tensor,
    bin_index: Tensor,
    event: Tensor,
    mask: Tensor | None = None,
    logits: Tensor | None = None,
) -> Tensor:
    """Censored discrete-time negative log-likelihood averaged over participants.

    For observed bin ``k`` (0-based) and event indicator ``d``::

        l = -[ sum_{l<k} log(1 - h_l) + d log h_k + (1 - d) log(1 - h_k) ]

    i.e. an event contributes its hazard in bin ``k`` and survival through the earlier bins, and a
    participant censored in bin ``k`` contributes survival through bin ``k``. Bin indices beyond
    the last bin are treated as censored in the last bin; negative indices are excluded.

    Args:
        hazards: ``(B, K)`` conditional hazards in ``(0, 1)``.
        bin_index: ``(B,)`` observed bin index.
        event: ``(B,)`` event indicator (NaN excluded).
        mask: Optional ``(B,)`` availability mask.
        logits: Optional hazard logits ``(B, K)``; used for numerically stable log terms.
    """
    k = hazards.shape[1]
    device = hazards.device
    bin_index = bin_index.to(device).long()
    event = event.to(device).float()
    valid = (bin_index >= 0) & torch.isfinite(event)
    if mask is not None:
        valid = valid & mask.to(device=device, dtype=torch.bool)
    if not bool(valid.any()):
        return _zero(hazards if logits is None else logits)
    if logits is not None:
        eta = logits[valid].float()
        log_h, log_s = -F.softplus(-eta), -F.softplus(eta)
    else:
        h = hazards[valid].float().clamp(_EPS, 1.0 - _EPS)
        log_h, log_s = torch.log(h), torch.log1p(-h)
    b = bin_index[valid]
    d = event[valid]
    beyond = b >= k
    d = torch.where(beyond, torch.zeros_like(d), d)
    b = b.clamp(max=k - 1)
    steps = torch.arange(k, device=device).unsqueeze(0)
    before = (steps < b.unsqueeze(1)).float()
    at = (steps == b.unsqueeze(1)).float()
    ll = (before * log_s).sum(1) + (
        at * (d.unsqueeze(1) * log_h + (1 - d.unsqueeze(1)) * log_s)
    ).sum(1)
    return -ll.mean()


def brier_binary(prob: Tensor, y: Tensor, mask: Tensor | None = None) -> Tensor:
    """Brier score ``mean((p - y)^2)`` over labelled samples (NaN labels excluded)."""
    y = y.to(prob.device)
    valid = torch.isfinite(y.float())
    if mask is not None:
        valid = valid & mask.to(device=prob.device, dtype=torch.bool)
    if not bool(valid.any()):
        return _zero(prob)
    return (prob[valid].float() - y[valid].float()).pow(2).mean()


def ipcw_brier_survival(
    survival_at_horizon: Tensor,
    bin_index: Tensor,
    event: Tensor,
    ipcw: Tensor,
    horizon_index: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Inverse-probability-of-censoring-weighted Brier score at a horizon.

    The horizon ``H`` is the upper boundary of bin ``horizon_index`` (so that
    ``survival_at_horizon = S(H)``). Participants with an event in bins ``<= horizon_index`` have
    status 0, participants observed beyond the horizon have status 1, and participants censored
    before ``H`` receive zero weight. The weighted squared error is averaged over all valid
    participants, following the usual IPCW estimator.

    Args:
        survival_at_horizon: ``(B,)`` predicted ``S(H)``.
        bin_index: ``(B,)`` observed bin index.
        event: ``(B,)`` event indicator.
        ipcw: ``(B,)`` censoring weights at the horizon (``1/G(T_i)`` or ``1/G(H)``).
        horizon_index: 0-based index of the last bin before ``H``.
        mask: Optional ``(B,)`` availability mask.
    """
    device = survival_at_horizon.device
    bin_index = bin_index.to(device).long()
    event = event.to(device).float()
    ipcw = ipcw.to(device).float()
    valid = (bin_index >= 0) & torch.isfinite(event) & torch.isfinite(ipcw)
    if mask is not None:
        valid = valid & mask.to(device=device, dtype=torch.bool)
    if not bool(valid.any()):
        return _zero(survival_at_horizon)
    s = survival_at_horizon[valid].float()
    b, d, w = bin_index[valid], event[valid], ipcw[valid]
    event_before = (d > 0.5) & (b <= horizon_index)
    alive_after = b > horizon_index
    weight = w * (event_before | alive_after).float()
    return (weight * (s - alive_after.float()).pow(2)).mean()


def pairwise_ranking_loss(
    hazard_logits: Tensor,
    status: Tensor,
    time: Tensor,
    ipcw: Tensor,
    horizon_index: int,
    mask: Tensor | None = None,
    scale: float = 1.0,
) -> Tensor:
    """IPCW-weighted pairwise logistic ranking loss on the risk at the primary horizon.

    The score of a participant is the log cumulative hazard at the horizon,
    ``s = log sum_{k <= horizon_index} softplus(eta_k)``, a monotone function of the horizon
    risk ``1 - S(H)``. A pair ``(i, j)`` is comparable when ``i`` had the event before the
    horizon (``status_i = 1``) and ``j`` was still under observation after ``T_i``
    (``T_j > T_i``); it contributes ``w_i log(1 + exp(-(s_i - s_j) / scale))`` with Uno-type
    weights ``w_i = ipcw_i^2 = G(T_i-)^-2``. The loss is the weighted mean over comparable
    pairs of the batch (a zero attached to the graph when there are none).

    Args:
        hazard_logits: ``(B, K)`` hazard logits.
        status: ``(B,)`` horizon status (1 = event by the horizon, 0 = event-free beyond it,
            NaN otherwise).
        time: ``(B,)`` observed times (NaN when unavailable).
        ipcw: ``(B,)`` censoring weights (``1/G(T_i-)`` for cases).
        horizon_index: Last bin before the horizon.
        mask: Optional ``(B,)`` availability mask.
        scale: Temperature of the logistic surrogate.
    """
    device = hazard_logits.device
    status, time, ipcw = (x.to(device).float() for x in (status, time, ipcw))
    valid = torch.isfinite(time)
    if mask is not None:
        valid = valid & mask.to(device=device, dtype=torch.bool)
    case = valid & (status > 0.5) & torch.isfinite(ipcw)
    if not bool(case.any()):
        return _zero(hazard_logits)
    h = int(min(max(horizon_index, 0), hazard_logits.shape[1] - 1))
    cum = F.softplus(hazard_logits[:, : h + 1].float()).sum(1)
    score = torch.log(cum.clamp_min(1e-8))
    t = torch.nan_to_num(time, nan=-1.0)
    pairs = case.unsqueeze(1) & valid.unsqueeze(0) & (t.unsqueeze(0) > t.unsqueeze(1))
    if not bool(pairs.any()):
        return _zero(hazard_logits)
    weight = pairs.float() * torch.nan_to_num(ipcw, nan=0.0).pow(2).unsqueeze(1)
    diff = (score.unsqueeze(1) - score.unsqueeze(0)) / scale
    return (weight * F.softplus(-diff)).sum() / weight.sum().clamp_min(1e-8)


def _expert_aux_terms(
    aux: Mapping[str, Any], batch: Mapping[str, Any], weights: LossWeights, egfr_target: Any
) -> Tensor | None:
    """Mean endpoint loss of the single-expert outputs (deep supervision)."""
    terms: list[Tensor] = []
    present = aux.get("present", {})
    for m, logit in aux.get("logit_egfr", {}).items():
        mask = present.get(m)
        if mask is None or not bool(mask.any()):
            continue
        parts = []
        if weights.egfr and egfr_target is not None:
            parts.append(
                weights.egfr
                * focal_bce(
                    logit,
                    egfr_target,
                    pos_weight=weights.egfr_pos_weight,
                    gamma=weights.focal_gamma,
                    mask=mask,
                )
            )
        if weights.surv and batch.get("surv_bin") is not None:
            surv_mask = batch.get("surv_present")
            surv_mask = mask if surv_mask is None else surv_mask.to(mask.device) & mask
            parts.append(
                weights.surv
                * discrete_time_nll(
                    aux["hazards"][m],
                    batch["surv_bin"],
                    batch["surv_event"],
                    mask=surv_mask,
                    logits=aux["hazard_logits"][m],
                )
            )
        if parts:
            terms.append(torch.stack(parts).sum())
    return torch.stack(terms).mean() if terms else None


def grl(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Gradient-reversal layer (alias of :func:`molt.models.heads.grad_reverse`)."""
    return grad_reverse(x, alpha)


def domain_ce(logits: Tensor, domain: Tensor) -> Tensor:
    """Cross-entropy of a domain classifier; domains ``< 0`` or ``>= n_domains`` are ignored."""
    if logits.shape[-1] == 0:
        return _zero(logits)
    domain = domain.to(logits.device).long()
    valid = (domain >= 0) & (domain < logits.shape[-1])
    if not bool(valid.any()):
        return _zero(logits)
    return F.cross_entropy(logits[valid].float(), domain[valid])


def info_nce(z1: Tensor, z2: Tensor, temperature: float = 0.1) -> Tensor:
    """Symmetric InfoNCE (NT-Xent) loss between two views ``(M, D)`` of the same instances."""
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    logits = z1 @ z2.T / temperature
    target = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


@dataclass
class LossWeights:
    """Weights of the terms of the MOLT objective.

    ``total = egfr * l_EGFR + surv * l_surv + recon * l_recon + domain * l_domain
    + brier * (l_Brier,EGFR [if egfr > 0] + l_Brier,surv [if surv > 0]) + attn_prior * l_prior``.

    Attributes:
        egfr: Weight of the focal EGFR loss.
        surv: Weight of the discrete-time survival likelihood.
        recon: Weight of the self-supervised masked-modelling losses returned by the model.
        domain: Weight of the gradient-reversal domain cross-entropy.
        brier: Weight of the Brier terms of the active endpoints.
        attn_prior: Weight of the penalty on attention outside the pathway prior
            (dense graph mode only).
        egfr_pos_weight: Positive-class weight of the EGFR loss (``None`` = 1).
        focal_gamma: Focal-loss focusing parameter.
        horizon_index: Bin index of the Brier horizon (overrides the model configuration).
        label_smoothing: EGFR targets become ``y (1 - eps) + eps / 2``.
        expert_aux: Weight of the deep-supervision term (endpoint loss of every expert's
            single-expert output, averaged over experts; requires ``expert_aux_heads``).
        rank: Weight of the IPCW-weighted pairwise ranking loss on the horizon risk.
        rank_scale: Temperature of the ranking surrogate.
        gene_l1: Weight of the L1 penalty on the first linear map of the gene-level expert.
        rna_precision: Weight of the shrinkage penalty on the precision that the fusion gives
            the transcriptomic experts (mean over participants of
            ``sum_m rho_m mean_d lambda_m`` over the present transcriptomic experts), which
            pulls the fused posterior towards the non-transcriptomic experts unless the
            expression profile lowers the endpoint loss.
    """

    egfr: float = 1.0
    surv: float = 1.0
    recon: float = 0.1
    domain: float = 0.0
    brier: float = 0.0
    attn_prior: float = 0.01
    egfr_pos_weight: float | None = None
    focal_gamma: float = 0.0
    horizon_index: int | None = None
    label_smoothing: float = 0.0
    expert_aux: float = 0.0
    rank: float = 0.0
    rank_scale: float = 1.0
    gene_l1: float = 0.0
    rna_precision: float = 0.0

    @classmethod
    def for_task(cls, task: str, **overrides: Any) -> LossWeights:
        """Endpoint-specific weights: ``"egfr"``, ``"pfi"``/``"os"`` (survival) or ``"multi"``."""
        task = task.lower()
        if task == "egfr":
            base = cls(egfr=1.0, surv=0.0)
        elif task in ("pfi", "os", "surv", "survival"):
            base = cls(egfr=0.0, surv=1.0)
        elif task in ("multi", "multitask", "joint"):
            base = cls(egfr=1.0, surv=1.0)
        else:
            raise ValueError(f"Unknown task {task!r}")
        return replace(base, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return the weights as a plain dictionary."""
        return asdict(self)


def compute_multitask_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: LossWeights,
    horizon_index: int | None = None,
) -> dict[str, Tensor]:
    """Combine the supervised, Brier, masked-modelling, domain and prior terms.

    Args:
        outputs: Model outputs (``logit_egfr``, ``hazards``, ``survival`` and optionally
            ``hazard_logits``, ``recon_loss``, ``domain_logits``, ``attn_prior_penalty``).
        batch: Batch with ``y_egfr``, ``surv_bin``, ``surv_event``, ``surv_present`` and
            optionally ``ipcw`` and ``domain``.
        weights: Term weights.
        horizon_index: Default Brier horizon bin (``weights.horizon_index`` takes precedence).

    Returns:
        ``total`` (differentiable) and detached components ``egfr``, ``surv``, ``brier_egfr``,
        ``brier_surv``, ``recon``, ``domain`` and ``attn_prior``.
    """
    logit = outputs["logit_egfr"]
    device = logit.device
    zero = _zero(logit)
    terms: dict[str, Tensor] = {}

    y = batch.get("y_egfr")
    target = y
    if y is not None and weights.label_smoothing:
        target = y.float() * (1.0 - weights.label_smoothing) + 0.5 * weights.label_smoothing
    if y is not None:
        terms["egfr"] = focal_bce(
            logit, target, pos_weight=weights.egfr_pos_weight, gamma=weights.focal_gamma
        )
        terms["brier_egfr"] = brier_binary(torch.sigmoid(logit.float()), y)
    else:
        terms["egfr"] = terms["brier_egfr"] = zero

    hazards = outputs["hazards"]
    if batch.get("surv_bin") is not None and batch.get("surv_event") is not None:
        surv_mask = batch.get("surv_present")
        terms["surv"] = discrete_time_nll(
            hazards,
            batch["surv_bin"],
            batch["surv_event"],
            mask=surv_mask,
            logits=outputs.get("hazard_logits"),
        )
        h_idx = weights.horizon_index if weights.horizon_index is not None else horizon_index
        ipcw = batch.get("ipcw")
        if h_idx is not None and ipcw is not None:
            h_idx = int(min(max(h_idx, 0), hazards.shape[1] - 1))
            terms["brier_surv"] = ipcw_brier_survival(
                outputs["survival"][:, h_idx],
                batch["surv_bin"],
                batch["surv_event"],
                ipcw,
                h_idx,
                mask=surv_mask,
            )
        else:
            terms["brier_surv"] = zero
        status, s_time = batch.get("surv_status"), batch.get("surv_time")
        if weights.rank and h_idx is not None and status is not None and s_time is not None:
            terms["rank"] = pairwise_ranking_loss(
                outputs["hazard_logits"],
                status,
                s_time,
                batch.get("ipcw", torch.ones_like(s_time)),
                int(h_idx),
                mask=surv_mask,
                scale=weights.rank_scale,
            )
    else:
        terms["surv"] = terms["brier_surv"] = zero
    aux = outputs.get("aux")
    if weights.expert_aux and isinstance(aux, Mapping):
        aux_loss = _expert_aux_terms(aux, batch, weights, target)
        if aux_loss is not None:
            terms["expert_aux"] = aux_loss
    gene_l1 = outputs.get("gene_l1")
    if weights.gene_l1 and isinstance(gene_l1, Tensor):
        terms["gene_l1"] = gene_l1
    rna_prec = outputs.get("rna_precision")
    if weights.rna_precision and isinstance(rna_prec, Tensor):
        terms["rna_precision"] = rna_prec.float().mean()

    recon = outputs.get("recon_loss")
    terms["recon"] = recon if isinstance(recon, Tensor) else zero
    dom_logits = outputs.get("domain_logits")
    dom = batch.get("domain")
    if isinstance(dom_logits, Tensor) and dom is not None and dom_logits.shape[-1] > 0:
        terms["domain"] = domain_ce(dom_logits, dom)
    else:
        terms["domain"] = zero
    penalty = outputs.get("attn_prior_penalty")
    terms["attn_prior"] = penalty if isinstance(penalty, Tensor) else zero

    total = zero.to(device)
    if weights.egfr:
        total = total + weights.egfr * terms["egfr"]
        if weights.brier:
            total = total + weights.brier * terms["brier_egfr"]
    if weights.surv:
        total = total + weights.surv * terms["surv"]
        if weights.brier:
            total = total + weights.brier * terms["brier_surv"]
    if weights.recon:
        total = total + weights.recon * terms["recon"]
    if weights.domain:
        total = total + weights.domain * terms["domain"]
    if weights.attn_prior:
        total = total + weights.attn_prior * terms["attn_prior"]
    if weights.surv and "rank" in terms:
        total = total + weights.rank * terms["rank"]
    if "expert_aux" in terms:
        total = total + weights.expert_aux * terms["expert_aux"]
    if "gene_l1" in terms:
        total = total + weights.gene_l1 * terms["gene_l1"]
    if "rna_precision" in terms:
        total = total + weights.rna_precision * terms["rna_precision"]
    out = {"total": total}
    out.update({k: v.detach() for k, v in terms.items()})
    return out
