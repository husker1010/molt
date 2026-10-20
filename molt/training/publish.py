"""Comparison-table rows of training and locked-evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molt.metrics.report import append_comparison, panel_comparison_rows
from molt.training.settings import get
from molt.utils.logging import get_logger

__all__ = ["comparison_label", "publish_comparison"]

log = get_logger(__name__)


def comparison_label(cfg: Mapping[str, Any], model_name: str | None = None) -> str:
    """Model name used in the comparison table for runs of ``cfg``.

    ``experiment.comparison_label`` when set, otherwise the model name (default
    ``experiment.model``) suffixed by ``experiment.ablation`` for ablation runs.
    """
    explicit = get(cfg, "experiment.comparison_label")
    if explicit:
        return str(explicit)
    name = model_name or str(get(cfg, "experiment.model", "molt"))
    ablation = get(cfg, "experiment.ablation")
    return f"{name}_{ablation}" if ablation else name


def publish_comparison(
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    task: str,
    cohort: str,
    model: str,
    run_id: str,
) -> int:
    """Upsert the standard metrics of every mode panel of ``summary`` into the comparison table.

    Disabled by ``evaluation.comparison: false``; the table lives in ``paths.state_dir``
    (explicit caller-supplied directory). Returns the number of rows written (0 on failure).
    """
    if not bool(get(cfg, "evaluation.comparison", True)):
        return 0
    rows: list[dict[str, Any]] = []
    for mode, entry in summary.get("modes", {}).items():
        panel = entry.get("panel")
        if panel:
            rows += panel_comparison_rows(panel, task, cohort, model, str(mode), run_id)
    if not rows:
        return 0
    try:
        total = append_comparison(rows, path=get(cfg, "paths.state_dir"))
    except OSError as exc:
        log.warning("comparison table not updated: %s", exc)
        return 0
    log.info("comparison table: %d rows from %s (%d in table)", len(rows), run_id, total)
    return len(rows)
