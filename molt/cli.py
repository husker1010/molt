"""Command-line training, inference and configuration export."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from molt.config import load_config
from molt.workflow import configuration, predict, train


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser without importing the training stack."""
    parser = argparse.ArgumentParser(prog="molt", description=__doc__)
    parser.add_argument("--version", action="version", version="molt 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="export an installed endpoint configuration")
    init.add_argument("--task", choices=["egfr", "pfi", "os"], default="egfr")
    init.add_argument("--output", required=True)
    for name in ("train", "splits"):
        cmd = sub.add_parser(
            name, help="train an ensemble" if name == "train" else "create nested folds"
        )
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--task", choices=["egfr", "pfi", "os"], required=True)
        cmd.add_argument("--cohort", help="comma-separated cohort identifiers")
        cmd.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "train":
            cmd.add_argument("--run-id", required=True)
            cmd.add_argument("--device", default="cpu")
        else:
            cmd.add_argument("--rebuild", action="store_true")
    pred = sub.add_parser("predict", help="score a cohort with a saved ensemble")
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--cohort", required=True)
    pred.add_argument("--cohorts-dir", required=True)
    pred.add_argument("--pathways-dir")
    pred.add_argument("--ct-dir")
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--mc-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a public workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-config":
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            parser.error(f"Refusing to replace existing configuration: {output}")
        configuration(args.task).save(output)
        print(output)
    elif args.command == "predict":
        frame = predict(
            args.run_dir,
            cohort=args.cohort,
            cohorts_dir=args.cohorts_dir,
            pathways_dir=args.pathways_dir,
            ct_dir=args.ct_dir,
            output=args.output,
            device=args.device,
            mc_samples=args.mc_samples,
        )
        print(f"Wrote {len(frame)} prediction rows to {args.output}")
    else:
        config = load_config(args.config, args.set)
        import torch

        torch.set_num_threads(max(1, int(config.get_path("hardware.cpu_threads", 4))))
        cohorts = [x.strip() for x in args.cohort.split(",") if x.strip()] if args.cohort else None
        if args.command == "train":
            train(config, task=args.task, run_id=args.run_id, cohorts=cohorts, device=args.device)
        else:
            from molt.training.cohort_data import development_universe
            from molt.training.nested_cv import development_folds, load_bundle_from_config
            from molt.training.settings import run_settings

            cohorts = cohorts or config.get_path("cohorts.development", [])
            config.set_path("cv.rebuild", args.rebuild)
            settings = run_settings(config, args.task)
            bundle = load_bundle_from_config(config, cohorts, settings)
            folds = development_folds(
                config,
                bundle,
                development_universe(
                    bundle.table, require_modality=settings.preprocess.require_modality_for_survival
                ),
                cohorts,
                settings,
            )
            print(f"Created or verified folds for {len(folds.participants)} participants")
    return 0
