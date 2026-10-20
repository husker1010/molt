"""Run status and metric streams consumed by monitoring tools.

Each run owns a directory ``<runs_root>/<run_id>/`` containing

* ``status.json`` - a single JSON object describing the current state of the run, rewritten
  atomically on every update::

      {"run_id": "...", "task": "egfr|pfi|os", "model": "molt|elastic_net|...",
       "cohorts": ["tcga_luad"], "state": "queued|running|done|failed",
       "started_utc": "...", "updated_utc": "...", "progress": 0.42,
       "message": "outer fold 3/5 inner 2/3 epoch 17/60", "gpu": 0}

* ``metrics.jsonl`` - one JSON object per line::

      {"run_id", "outer_fold", "inner_fold", "member", "epoch", "split": "train|val|test",
       "metric": "loss|auroc|...", "value": 0.81, "utc": "..."}
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_RUNS_ROOT",
    "RUN_STATES",
    "METRIC_SPLITS",
    "RunStatus",
    "MetricsLogger",
    "utc_now",
    "resolve_runs_root",
    "read_metrics",
]

DEFAULT_RUNS_ROOT = None
RUN_STATES = ("queued", "running", "done", "failed")
METRIC_SPLITS = ("train", "val", "test")
_STATUS_KEYS = (
    "run_id",
    "task",
    "model",
    "cohorts",
    "state",
    "started_utc",
    "updated_utc",
    "progress",
    "message",
    "gpu",
)


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_runs_root(runs_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicitly supplied output root."""
    if not runs_root:
        raise ValueError("runs_root must be supplied explicitly or through paths.runs_root")
    return Path(runs_root).expanduser().resolve()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


class RunStatus:
    """Maintain ``status.json`` for a run.

    Example:
        >>> status = RunStatus("egfr_molt_s1", task="egfr", model="molt", cohorts=["tcga_luad"])
        >>> status.start("loading data")
        >>> status.update(progress=0.5, message="outer fold 3/5")
        >>> status.finish()
    """

    def __init__(
        self,
        run_id: str,
        *,
        task: str,
        model: str,
        cohorts: Sequence[str] = (),
        runs_root: str | os.PathLike[str] | None = None,
        gpu: int | None = None,
        state: str = "queued",
        message: str = "",
    ) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"state must be one of {RUN_STATES}, got {state!r}")
        self.run_dir = resolve_runs_root(runs_root) / run_id
        self.path = self.run_dir / "status.json"
        self._lock = threading.Lock()
        now = utc_now()
        self._data: dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "model": model,
            "cohorts": list(cohorts),
            "state": state,
            "started_utc": now,
            "updated_utc": now,
            "progress": 0.0,
            "message": message,
            "gpu": gpu,
        }
        self._write()

    @property
    def data(self) -> dict[str, Any]:
        """A copy of the current status payload."""
        return dict(self._data)

    @property
    def run_id(self) -> str:
        return str(self._data["run_id"])

    def _write(self) -> None:
        _atomic_write_json(self.path, {k: self._data.get(k) for k in _STATUS_KEYS})

    def update(
        self,
        *,
        progress: float | None = None,
        message: str | None = None,
        state: str | None = None,
        **extra: Any,
    ) -> None:
        """Update fields and rewrite ``status.json``.

        Args:
            progress: Fraction complete in ``[0, 1]`` (clipped).
            message: Human-readable progress message.
            state: One of :data:`RUN_STATES`.
            **extra: Other schema fields (``gpu``, ``cohorts``); unknown keys raise ``KeyError``.
        """
        with self._lock:
            if state is not None:
                if state not in RUN_STATES:
                    raise ValueError(f"state must be one of {RUN_STATES}, got {state!r}")
                self._data["state"] = state
            if progress is not None:
                self._data["progress"] = round(min(max(float(progress), 0.0), 1.0), 4)
            if message is not None:
                self._data["message"] = str(message)
            for key, value in extra.items():
                if key not in _STATUS_KEYS or key in ("run_id", "updated_utc"):
                    raise KeyError(f"Unknown or read-only status field {key!r}")
                self._data[key] = list(value) if key == "cohorts" else value
            self._data["updated_utc"] = utc_now()
            self._write()

    def start(self, message: str = "started") -> None:
        """Mark the run as running and reset ``started_utc``."""
        self._data["started_utc"] = utc_now()
        self.update(state="running", progress=0.0, message=message)

    def finish(self, message: str = "done") -> None:
        """Mark the run as successfully completed."""
        self.update(state="done", progress=1.0, message=message)

    def fail(self, message: str = "failed") -> None:
        """Mark the run as failed."""
        self.update(state="failed", message=message)

    def __enter__(self) -> "RunStatus":
        if self._data["state"] == "queued":
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.fail(f"{exc_type.__name__}: {exc}")
        elif self._data["state"] == "running":
            self.finish()

    @classmethod
    def read(cls, run_id: str, runs_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        """Read ``status.json`` of an existing run."""
        path = resolve_runs_root(runs_root) / run_id / "status.json"
        return json.loads(path.read_text(encoding="utf-8"))


class MetricsLogger:
    """Append metric records to ``metrics.jsonl``.

    Records are flushed on every call so that monitoring tools can tail the file.
    """

    def __init__(self, run_id: str, runs_root: str | os.PathLike[str] | None = None) -> None:
        self.run_id = run_id
        self.run_dir = resolve_runs_root(runs_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._lock = threading.Lock()

    def log(
        self,
        metric: str,
        value: float | None,
        *,
        split: str = "val",
        epoch: int | None = None,
        outer_fold: int | None = None,
        inner_fold: int | None = None,
        member: int | str | None = None,
    ) -> dict[str, Any]:
        """Append one metric record and return it.

        Args:
            metric: Metric name (``loss``, ``auroc``, ``c_index`` ...).
            value: Metric value; NaN/inf are stored as ``null``.
            split: ``train``, ``val`` or ``test``.
            epoch: Epoch index, if applicable.
            outer_fold: Outer cross-validation fold index.
            inner_fold: Inner cross-validation fold index.
            member: Ensemble member identifier.
        """
        if split not in METRIC_SPLITS:
            raise ValueError(f"split must be one of {METRIC_SPLITS}, got {split!r}")
        record = {
            "run_id": self.run_id,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "member": member,
            "epoch": epoch,
            "split": split,
            "metric": metric,
            "value": _clean_number(value),
            "utc": utc_now(),
        }
        line = json.dumps(record) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
        return record

    def log_many(self, metrics: Mapping[str, float | None], **context: Any) -> list[dict[str, Any]]:
        """Log several metrics sharing the same context (split, epoch, folds, member)."""
        return [self.log(name, value, **context) for name, value in metrics.items()]


def read_metrics(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Iterate over records of a ``metrics.jsonl`` file, skipping malformed lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
