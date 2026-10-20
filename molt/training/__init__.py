"""Training, model selection and locked-ensemble prediction for MOLT.

Modules:

* :mod:`molt.training.cohort_data` - load processed cohorts and pathway resources into a
  participant-level :class:`~molt.training.cohort_data.CohortBundle`.
* :mod:`molt.training.preprocess` - fold-specific :class:`~molt.training.preprocess.Preprocessor`
  (clinical encoding, pathway activities, quality inputs, survival bins, IPCW weights).
* :mod:`molt.training.ct_patches` / :mod:`molt.training.ct_features` - CT patch sampling and
  augmentation; frozen-backbone patch features shared across folds.
* :mod:`molt.training.data` - :class:`~molt.training.data.MOLTDataset` and batch assembly.
* :mod:`molt.training.trainer` - :class:`~molt.training.trainer.Trainer` (AdamW, cosine
  schedule, early stopping, mixed precision, modality dropout) and deep
  :class:`~molt.training.trainer.Ensemble` prediction with MC dropout.
* :mod:`molt.training.nested_cv` - nested cross-validation with random search, cross-fitted
  recalibration, thresholds, abstention, conformal prediction and the locked ensemble.
* :mod:`molt.training.locked` - :func:`~molt.training.locked.predict_locked` for external cohorts.

Submodules are imported on demand, e.g. ``from molt.training.nested_cv import NestedCV``.
"""

from __future__ import annotations

__all__ = [
    "cohort_data",
    "ct_features",
    "ct_patches",
    "data",
    "evaluate",
    "locked",
    "model_factory",
    "nested_cv",
    "parallel",
    "postprocess",
    "preprocess",
    "settings",
    "trainer",
]
