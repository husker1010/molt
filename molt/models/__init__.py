"""Neural network components of MOLT.

* :mod:`molt.models.clinical` - gated tabular transformer for clinical variables.
* :mod:`molt.models.pathway` - pathway graph attention encoder with a masked gene-level objective.
* :mod:`molt.models.imaging` - CT patch backbones (Swin-ViT, 3D CNN) with attention MIL pooling.
* :mod:`molt.models.fusion` - reliability-weighted Gaussian product-of-experts fusion.
* :mod:`molt.models.heads` - EGFR, discrete-time survival and domain heads.
* :mod:`molt.models.losses` - training objectives.
* :mod:`molt.models.molt` - the :class:`MOLT` network and its configuration.
"""

from __future__ import annotations

from molt.models.clinical import MISSING_INDEX, OOV_INDEX, ClinicalEncoder
from molt.models.factory import MODEL_REGISTRY, build_model, load_checkpoint
from molt.models.fusion import (
    MODALITIES,
    GaussianExpert,
    MonotoneReliability,
    ProductOfExpertsFusion,
    sample_modality_dropout,
)
from molt.models.heads import (
    BinaryHead,
    DiscreteTimeSurvivalHead,
    DomainClassifier,
    grad_reverse,
    survival_from_hazards,
)
from molt.models.imaging import (
    CNN3DBackbone,
    ImagingEncoder,
    SwinViTBackbone,
    load_swinvit_weights,
    remap_swinvit_key,
)
from molt.models.losses import (
    LossWeights,
    brier_binary,
    compute_multitask_loss,
    discrete_time_nll,
    domain_ce,
    focal_bce,
    grl,
    info_nce,
    ipcw_brier_survival,
)
from molt.models.molt import MOLT, MOLTConfig, mc_dropout, mc_dropout_predict
from molt.models.pathway import PathwayEncoder, masked_mse

__all__ = [
    "MISSING_INDEX",
    "MODALITIES",
    "MODEL_REGISTRY",
    "MOLT",
    "OOV_INDEX",
    "BinaryHead",
    "CNN3DBackbone",
    "ClinicalEncoder",
    "DiscreteTimeSurvivalHead",
    "DomainClassifier",
    "GaussianExpert",
    "ImagingEncoder",
    "LossWeights",
    "MOLTConfig",
    "MonotoneReliability",
    "PathwayEncoder",
    "ProductOfExpertsFusion",
    "SwinViTBackbone",
    "brier_binary",
    "build_model",
    "compute_multitask_loss",
    "discrete_time_nll",
    "domain_ce",
    "focal_bce",
    "grad_reverse",
    "grl",
    "info_nce",
    "ipcw_brier_survival",
    "load_checkpoint",
    "load_swinvit_weights",
    "masked_mse",
    "mc_dropout",
    "mc_dropout_predict",
    "remap_swinvit_key",
    "sample_modality_dropout",
    "survival_from_hazards",
]
