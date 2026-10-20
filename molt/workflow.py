"""Public workflows with explicit input and output locations."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from molt.config import Config, load_config


def configuration(task: str = "egfr", overrides: Mapping[str, Any] | Sequence[str] = ()) -> Config:
    """Load an installed endpoint template with dotted-key overrides.

    ``os`` uses the survival template. Paths default to null and must be supplied
    before training. Relative paths passed through this API resolve against the
    caller's current directory.
    """
    if task not in ("egfr", "pfi", "os"):
        raise ValueError("task must be egfr, pfi or os")
    name = "egfr" if task == "egfr" else "pfi"
    path = Path(__file__).parent / "configs" / f"molt_{name}.yaml"
    return load_config(path, overrides, base_dir=Path.cwd())


def train(
    config: Config,
    *,
    task: str,
    run_id: str,
    cohorts: Sequence[str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train and select a nested ensemble, persisting it under paths.runs_root."""
    import torch
    from molt.training.nested_cv import NestedCV

    torch.set_num_threads(max(1, int(config.get_path("hardware.cpu_threads", 4))))
    return NestedCV(config, task=task, run_id=run_id, cohorts=cohorts, device=device).run()


def predict(
    run_dir: str | Path,
    *,
    cohort: str,
    cohorts_dir: str | Path,
    output: str | Path,
    pathways_dir: str | Path | None = None,
    ct_dir: str | Path | None = None,
    device: str = "cpu",
    mc_samples: int | None = None,
) -> Any:
    """Score a caller-supplied cohort and write a Parquet table.

    Labels are optional. The complete run directory may be relocated before
    inference. Only trusted, locally generated model runs should be loaded;
    fitted preprocessors use Python pickle serialization.
    """
    import torch
    from molt.training.locked import LockedEnsemble
    from molt.training.nested_cv import load_bundle_from_config
    from molt.data.io import write_parquet

    locked = LockedEnsemble(Path(run_dir).expanduser().resolve(), device=device)
    torch.set_num_threads(max(1, int(locked.cfg.get_path("hardware.cpu_threads", 4))))
    for key, value in {
        "cohorts_dir": cohorts_dir,
        "pathways_dir": pathways_dir,
        "ct_dir": ct_dir,
    }.items():
        locked.cfg.set_path(
            f"paths.{key}", str(Path(value).expanduser().resolve()) if value else None
        )
    if locked.settings.ct is not None:
        locked.settings.ct.cache_dir = None
    locked.settings.ct_feature_dir = None
    if mc_samples is not None:
        if mc_samples < 0:
            raise ValueError("mc_samples must be nonnegative")
        locked.settings.train.mc_samples = mc_samples
    bundle = load_bundle_from_config(locked.cfg, [cohort], locked.settings)
    frame = locked.predict(bundle)
    write_parquet(frame, Path(output).expanduser().resolve())
    return frame
