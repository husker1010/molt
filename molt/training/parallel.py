"""Execute independent inner-fold jobs of a nested cross-validation in worker processes.

Within one outer fold, the ``configurations x inner folds`` ensembles are independent. With
``experiment.parallel_jobs: N > 1`` they are trained by ``N`` spawned worker processes that
share the GPU(s) given by the run device (``cuda:0`` or a comma-separated list such as
``cuda:0,cuda:1``, assigned round-robin to workers; the list may repeat devices and include
``cpu``, e.g. ``cuda:0,cuda:1,cpu,cpu``, to mix GPU and CPU workers). Each worker uses
``experiment.worker_threads`` intra-op CPU threads (default: a share of the CPU count). Every worker loads the cohort bundle
once, writes the same per-job artefacts as the sequential path and reports progress through a
small JSON file that the parent aggregates into ``status.json``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from molt.utils.logging import get_logger

__all__ = ["FileProgress", "parse_devices", "run_jobs_parallel"]

log = get_logger(__name__)

_STATE: dict[str, Any] = {}


def parse_devices(device: str) -> list[str]:
    """Split a device specification such as ``"cuda:0,cuda:1"``."""
    return [d.strip() for d in str(device).split(",") if d.strip()] or ["cpu"]


class FileProgress:
    """Progress sink of a worker job: completed work units and the latest message."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done = 0.0
        self.message = ""
        self._last = 0.0

    def advance(self, units: float, message: str | None = None, force: bool = False) -> None:
        self.done += units
        if message:
            self.message = message
        now = time.time()
        if force or now - self._last > 2.0:
            self._last = now
            self.flush()

    def flush(self) -> None:
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"units": self.done, "message": self.message}))
        os.replace(tmp, self.path)

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {"units": 0.0, "message": ""}


def _init_worker(spec: dict[str, Any]) -> None:
    import torch

    from molt.config import Config
    from molt.splits import NestedFolds
    from molt.training.nested_cv import NestedCV, ct_feature_store, load_bundle_from_config
    from molt.utils.status import MetricsLogger

    torch.set_num_threads(int(spec.get("threads", 4)))
    devices = spec["devices"]
    identity = mp.current_process()._identity
    device = devices[(identity[0] - 1) % len(devices)] if identity else devices[0]
    runner = NestedCV(
        Config(spec["cfg"]),
        task=spec["task"],
        run_id=spec["run_id"],
        cohorts=spec["cohorts"],
        device=device,
        runs_root=spec["runs_root"],
    )
    runner.bundle = load_bundle_from_config(runner.cfg, runner.cohorts, runner.settings)
    runner.metrics = MetricsLogger(runner.run_id, runs_root=runner.runs_root)
    runner.ct_store = None
    if spec.get("ct_feature_cache"):
        ids = NestedFolds.load(runner.run_dir / "folds.parquet").participants
        runner.ct_store = ct_feature_store(
            runner.bundle, ids, runner.settings, runner.backend, None, device
        )
    _STATE["runner"] = runner
    _STATE["folds"] = NestedFolds.load(runner.run_dir / "folds.parquet")
    _STATE["progress_dir"] = Path(spec["progress_dir"])


def _run_job(k: int, c: int, j: int) -> tuple[int, int, int]:
    runner = _STATE["runner"]
    progress = FileProgress(_STATE["progress_dir"] / f"o{k}_c{c}_i{j}.json")
    runner.progress = progress
    runner._fit_inner(k, c, j, _STATE["folds"])
    progress.flush()
    return k, c, j


def run_jobs_parallel(
    runner: Any,
    jobs: list[tuple[int, int, int]],
    n_workers: int,
    *,
    poll_seconds: float = 5.0,
) -> None:
    """Train ``jobs`` (``(outer, config, inner)`` triples) in ``n_workers`` processes.

    ``runner`` is the parent :class:`~molt.training.nested_cv.NestedCV`; its progress object is
    updated from the workers' progress files until every job has finished. The first failing
    job raises in the parent.
    """
    if not jobs:
        return
    progress_dir = runner.work_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "cfg": runner.cfg.to_dict(),
        "task": runner.task,
        "run_id": runner.run_id,
        "cohorts": runner.cohorts,
        "devices": parse_devices(runner.device_spec),
        "runs_root": str(runner.runs_root),
        "threads": int(
            getattr(runner, "worker_threads", 0)
            or max(1, (os.cpu_count() or 8) // (4 * max(n_workers, 1)))
        ),
        "ct_feature_cache": runner.ct_store is not None,
        "progress_dir": str(progress_dir),
    }
    files = [progress_dir / f"o{k}_c{c}_i{j}.json" for k, c, j in jobs]
    for f in files:
        f.unlink(missing_ok=True)
    base = runner.progress.done
    ctx = mp.get_context("spawn")
    workers = min(n_workers, len(jobs))
    log.info("training %d inner-fold job(s) with %d worker process(es)", len(jobs), workers)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(spec,)
    ) as pool:
        pending = {pool.submit(_run_job, *job) for job in jobs}
        finished = 0
        while pending:
            done, pending = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for fut in done:
                fut.result()
                finished += 1
            states = [FileProgress.read(f) for f in files if f.exists()]
            runner.progress.done = base + sum(float(s.get("units", 0.0)) for s in states)
            latest = next((s["message"] for s in reversed(states) if s.get("message")), "")
            runner.progress.advance(
                0.0, f"{finished}/{len(jobs)} inner jobs done ({workers} workers) · {latest}"
            )
