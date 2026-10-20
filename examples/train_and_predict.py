"""Train and predict using caller-supplied processed cohorts and explicit paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from molt.config import load_config
from molt.workflow import train, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["egfr", "pfi", "os"], required=True)
    parser.add_argument("--train-cohort", required=True)
    parser.add_argument("--predict-cohort", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, task=args.task, run_id=args.run_id, cohorts=[args.train_cohort], device=args.device)
    frame = predict(
        Path(cfg.paths.runs_root) / args.run_id,
        cohort=args.predict_cohort,
        cohorts_dir=cfg.paths.cohorts_dir,
        pathways_dir=cfg.paths.pathways_dir,
        ct_dir=cfg.paths.ct_dir,
        output=args.output,
        device=args.device,
    )
    print(f"Saved predictions for {frame.participant_id.nunique()} participants")


if __name__ == "__main__":
    main()
