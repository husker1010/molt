# MOLT

MOLT combines clinical, transcriptomic and CT inputs to predict EGFR status and discrete-time
progression or survival risk. This standalone distribution includes the model architecture,
cohort readers and validators, fold-specific preprocessing, training and model selection,
ensemble inference, calibration, losses, metrics and complete configuration templates.

## Install

Python 3.10 or newer is required. From the extracted source directory:

```bash
python -m pip install .
# Add Swin-ViT and DICOM preprocessing support when needed:
python -m pip install '.[imaging]'
molt --help
```

Alternatively, install the accompanying wheel:

```bash
python -m pip install molt-0.1.0-py3-none-any.whl
```

The CPU-compatible CNN branch is included in the base installation. The endpoint templates
retain the Swin-ViT architecture, so install the imaging extra when using their CT branch,
or set `model.ct_backbone: cnn3d`. CT backbones start from random initialization by default
and are trained jointly (`data.ct.feature_cache: false`). No model states, data, acquisition
scripts or initialization-source information are included. Model states produced by training
are written only to the output root supplied by the caller.

## Inputs

Prepare one directory per cohort under an explicit `cohorts_dir`:

```text
<cohorts_dir>/<cohort>/
  clinical.parquet
  labels.parquet          # required for training; optional at inference
  expression.parquet      # optional modality; genes by participant IDs
  expression_meta.json    # required when expression is supplied
  ct_index.parquet        # optional modality
<pathways_dir>/
  modules.json
  incidence.parquet
  adjacency.parquet
<ct_dir>/<cohort>/<participant_id>.npz
```

Clinical variables are columns and participants are rows. RNA expression uses gene symbols as
rows and participant IDs as columns. RNA-seq input must already be on the log2(x+1) scale;
array and relative scales are described by metadata. Survival times are in months. CT arrays
are isotropic 1 mm HU volumes in `(z, y, x)` order with aligned lung masks and candidate centres.
See [input formats](docs/INPUT_FORMAT.md) for required fields and conversion examples.

## Complete command-line workflow

Export a fully expanded configuration, then set its paths and cohort names:

```bash
molt init-config --task egfr --output egfr.yaml
```

Edit `egfr.yaml` (replace these example locations with your own):

```yaml
paths:
  cohorts_dir: /absolute/input/cohorts
  pathways_dir: /absolute/input/pathways
  ct_dir: /absolute/input/ct
  runs_root: /absolute/output/runs
  splits_dir: /absolute/output/splits
  state_dir: null
cohorts:
  development: [development]
```

All input and output roots default to `null`. No data or output directory is inferred from
the source checkout. Paths in a YAML file resolve relative to that file. If a modality is
not used, set `data.rna.enabled: false` or `data.ct.enabled: false`; its input root may remain
null. For clinical-only survival training, also set `data.survival.require_modality: false`.

```bash
molt splits --config egfr.yaml --task egfr
molt train --config egfr.yaml --task egfr --run-id egfr-v1 --device cpu
molt predict --run-dir /absolute/output/runs/egfr-v1 \
  --cohort incoming --cohorts-dir /absolute/input/cohorts \
  --pathways-dir /absolute/input/pathways --ct-dir /absolute/input/ct \
  --output /absolute/output/predictions.parquet --device cpu
```

Use `--device cuda:0` for GPU execution. Use `--task pfi` or `--task os` in both
`init-config` and `train` for a survival endpoint. A full run performs nested cross-validation,
hyperparameter selection and deep ensemble fitting; it can be computationally expensive.
Overrides are repeatable, for example `--set train.members=1 --set train.max_epochs=10`.

The output run includes `config.yaml`, fold assignments, logs, predictions, evaluation
summaries, fitted preprocessors, complete model states and `locked/manifest.json`.
Copy the **whole run directory** when relocating a trained ensemble. Inference takes fresh
input paths and an explicit output file; it does not need the original training data.
Only load trusted runs because fitted preprocessors use pickle serialization.

## Python workflow

```python
from molt.workflow import configuration, train, predict

cfg = configuration("egfr", {
    "paths.cohorts_dir": "/absolute/input/cohorts",
    "paths.pathways_dir": "/absolute/input/pathways",
    "paths.ct_dir": "/absolute/input/ct",
    "paths.runs_root": "/absolute/output/runs",
    "paths.splits_dir": "/absolute/output/splits",
    "cohorts.development": ["development"],
    "model.ct_backbone": "cnn3d",
})
summary = train(cfg, task="egfr", run_id="egfr-v1", device="cpu")
predictions = predict(
    "/absolute/output/runs/egfr-v1", cohort="incoming",
    cohorts_dir="/absolute/input/cohorts",
    pathways_dir="/absolute/input/pathways", ct_dir="/absolute/input/ct",
    output="/absolute/output/predictions.parquet", device="cpu",
)
```

Runnable examples are in `examples/prepare_inputs.py`, `examples/prepare_ct.py` and
`examples/train_and_predict.py`; each exposes `--help`. Lower-level APIs include
`MOLT`, `MOLTConfig`, `Preprocessor`, `MOLTDataset`, `Trainer`, `Ensemble` and `LossWeights`.

## Parameters and evaluation

[Parameter reference](docs/PARAMETERS.md) documents every dataclass field and every template
key. Important groups are `data` (clinical/RNA/CT preparation), `model` (all encoder and fusion
settings), `loss`, `train`, `cv`, `search`, `calibration`, `recalibration` and `evaluation`.
Full and light (RNA omitted) prediction modes are enabled by default. Stacked linear experts
are components of MOLT's fusion model and remain included.

Binary evaluation provides AUROC, AUPRC, Brier score, calibration and operating-point metrics.
Survival evaluation includes Harrell/Uno concordance, time-dependent discrimination, IPCW
Brier scores and calibration. Bootstrap intervals, censoring utilities and conformal prediction
are retained. See [API and output guide](docs/API.md) for direct calls and prediction columns.
