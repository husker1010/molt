"""Convert caller-supplied clinical, label and expression CSV files to cohort tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from molt.data.io import save_clinical, save_labels, save_expression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--expression")
    parser.add_argument("--expression-meta", help="JSON metadata describing the input scale")
    parser.add_argument(
        "--log2p1", action="store_true", help="convert nonnegative abundances to log2(x+1)"
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohorts-dir", required=True)
    args = parser.parse_args()
    if args.expression and not args.expression_meta:
        parser.error("--expression requires --expression-meta")
    if args.log2p1 and not args.expression:
        parser.error("--log2p1 requires --expression")
    out = Path(args.cohorts_dir) / args.cohort
    clinical = pd.read_csv(args.clinical, dtype={"participant_id": str})
    clinical["cohort"] = args.cohort
    if "has_clinical_record" not in clinical:
        clinical["has_clinical_record"] = True
    save_clinical(clinical, out / "clinical.parquet")
    if args.labels:
        labels = pd.read_csv(args.labels, dtype={"participant_id": str})
        if not set(labels.participant_id) <= set(clinical.participant_id):
            raise ValueError("Label participant IDs must be present in the clinical table")
        save_labels(labels, out / "labels.parquet")
    if args.expression:
        expression = pd.read_csv(args.expression, index_col=0)
        if not set(expression.columns) <= set(clinical.participant_id):
            raise ValueError("Expression participant IDs must be present in the clinical table")
        meta = json.loads(Path(args.expression_meta).read_text())
        if args.log2p1:
            if (expression.to_numpy() < 0).any():
                raise ValueError("log2(x+1) requires nonnegative input")
            expression = np.log2(expression + 1)
            meta["units"] = (
                "log2(TPM+1)"
                if "TPM" in str(meta.get("units", "")).upper()
                else "log2(abundance+1)"
            )
        meta.setdefault("sample_selection_rule", "One caller-selected sample per participant")
        meta.setdefault("source_files", [])
        save_expression(expression, out / "expression.parquet", meta)
    print(out.resolve())


if __name__ == "__main__":
    main()
