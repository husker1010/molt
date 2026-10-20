"""Evaluation statistics for MOLT.

* :mod:`molt.metrics.binary` - discrimination, calibration, threshold and net-benefit
  metrics for binary outcomes, with tiered reporting helpers.
* :mod:`molt.metrics.survival` - IPCW time-dependent AUROC/AUPRC, concordance indices,
  Brier scores, horizon calibration, net benefit and competing-risk metrics.
* :mod:`molt.metrics.bootstrap` - stratified, paired and cluster bootstrap, multiplicity
  adjustment and random-effects meta-analysis (the generic resampler is
  :func:`molt.metrics.bootstrap.bootstrap`).
* :mod:`molt.metrics.report` - metric panels for prediction tables and persistence of the
  comparison table and results registry.
"""

from __future__ import annotations

from molt.metrics.binary import (
    auprc,
    auroc,
    brier,
    calibration_intercept_slope,
    expected_calibration_error,
    net_benefit,
    observed_expected,
    percentile_ranks,
    reporting_tier,
    threshold_metrics,
)
from molt.metrics.bootstrap import (
    benjamini_hochberg,
    holm,
    meta_analyze_auroc,
    paired_bootstrap,
    random_effects_meta,
)
from molt.metrics.report import (
    append_comparison,
    comparison_rows,
    evaluate_binary,
    evaluate_survival,
    fill_registry,
    panel_comparison_rows,
)
from molt.metrics.survival import (
    brier_score,
    harrell_c,
    integrated_brier_score,
    survival_calibration,
    time_dependent_auprc,
    time_dependent_auroc,
    uno_c,
    wolbers_concordance,
)

__all__ = [
    "append_comparison",
    "auprc",
    "auroc",
    "benjamini_hochberg",
    "brier",
    "brier_score",
    "calibration_intercept_slope",
    "comparison_rows",
    "evaluate_binary",
    "evaluate_survival",
    "expected_calibration_error",
    "fill_registry",
    "harrell_c",
    "holm",
    "integrated_brier_score",
    "meta_analyze_auroc",
    "net_benefit",
    "observed_expected",
    "paired_bootstrap",
    "panel_comparison_rows",
    "percentile_ranks",
    "random_effects_meta",
    "reporting_tier",
    "survival_calibration",
    "threshold_metrics",
    "time_dependent_auprc",
    "time_dependent_auroc",
    "uno_c",
    "wolbers_concordance",
]
