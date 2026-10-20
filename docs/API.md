# API and outputs

## Entry points

- `molt.workflow.configuration(task, overrides)`: installed template plus dotted overrides.
- `molt.workflow.train(config, task=..., run_id=..., cohorts=..., device=...)`: complete nested
  model selection, training, calibration and saved ensemble.
- `molt.workflow.predict(run_dir, cohort=..., cohorts_dir=..., pathways_dir=..., ct_dir=...,
  output=..., device=..., mc_samples=...)`: prediction on a new cohort, including unlabelled data.
- `molt.models.MOLT` and `MOLTConfig`: direct tensor-level model API.
- `molt.training.preprocess.Preprocessor`: fitted clinical/RNA/quality/outcome preparation.
- `molt.training.data.MOLTDataset`, `make_loader`, `move_batch`: datasets and tensor batches.
- `molt.training.trainer.Trainer`, `TrainSettings`, `Ensemble`: lower-level optimization and prediction.
- `molt.models.losses.LossWeights`: all endpoint and auxiliary objective weights.

For the tensor API, clinical continuous input is `(B, n_num)`, categorical indices are
`(B, n_cat)`, pathway activities are `(B, P)`, panel expression is `(B, G)`, and CT patches
are `(B, N, 1, D, H, W)`. Use `MOLTDataset` to produce the correctly named batch fields,
availability masks, quality arrays and outcome targets. Direct `MOLT.forward` returns
`logit_egfr`, `hazards`, `survival`, expert weights and auxiliary outputs.

```python
import torch
from molt.models import MOLT, MOLTConfig, load_checkpoint

model = MOLT(MOLTConfig(n_num=2, n_cat=0, cat_cardinalities=[],
                        use_rna=False, use_ct=False))
batch = {"clinical_num": torch.zeros(3, 2),
         "clinical_cat": torch.zeros(3, 0, dtype=torch.long)}
model.eval()
with torch.no_grad():
    outputs = model(batch, dropout_modalities=False)
    probabilities = torch.sigmoid(outputs["logit_egfr"])
# checkpoint_path is an explicit destination supplied by the caller.
model.save_checkpoint(checkpoint_path)
restored = load_checkpoint(checkpoint_path)
```

A tensor-level checkpoint holds the model state and architecture. For raw cohort inference,
retain the high-level run directory too: preprocessing, calibration and ensemble membership
are part of that artifact. Low-level weights alone cannot reconstruct those transformations.

## Prediction columns

Each participant has one row per requested prediction mode. Common fields include:

- `participant_id`, `cohort`, `task`, `mode`: identity and endpoint/mode.
- `p_raw`, `p_cal`: raw and calibrated EGFR probability, or event risk at the primary horizon.
- `risk_12m`, `risk_24m`, `risk_36m`: calibrated event risks for survival endpoints.
- `surv_raw`, `surv_cal`, `bin_width`: raw/calibrated survival curves and interval width.
- `threshold`, `pred_pos`: development-derived EGFR operating point and decision.
- `uncertainty`, `unc_ensemble`, `unc_mc`: ensemble and MC-dropout summaries.
- `abstain`, `abstain_cutoff`, `rna_quality_flag`: uncertainty/quality decisions.
- `conformal_p0`, `conformal_p1`, `conformal_set`: EGFR conformal outputs.
- `rho_clinical`, `rho_rna`, `rho_ct`: modality reliability summaries.
- `has_clinical`, `has_rna`, `has_ct`: available modalities.

Fields that do not apply to an endpoint are null. Predictions for unlabelled participants
are returned even though `in_analytic_set` is false. Labels and survival outcomes are
copied into output only when available. `mc_samples=0` disables MC-dropout uncertainty.

## Evaluation

```python
import pandas as pd
from molt.metrics import evaluate_binary, evaluate_survival

predictions = pd.read_parquet(predictions_path)
full = predictions.loc[predictions["mode"] == "full"].copy()
binary_report = evaluate_binary(full, y_col="y_egfr", p_col="p_cal",
                                fold_col=None, n_boot=200, seed=7)
```

For survival, load a survival-endpoint prediction frame and use:

```python
survival_predictions = pd.read_parquet(survival_predictions_path)
full_survival = survival_predictions.loc[survival_predictions["mode"] == "full"]
survival_report = evaluate_survival(full_survival, time_col="time", event_col="event",
                                    risk_col="risk_24m", horizon=24, tau=36,
                                    fold_col=None, n_boot=200, seed=7)
```

The full function signature is in `molt/metrics/report.py`.
The high-level training workflow calls the matching endpoint evaluator automatically.
Small or single-class samples can yield undefined metrics or suppressed discrimination
estimates. These are reported as missing values, not interpreted as model performance.

## Release behavior

The full model implementation is required; there is no fallback placeholder network.
Preprocessing and model dimensions are retained with saved runs. Arbitrary cohort names are
accepted, and no machine-specific roots are compiled into the package. Config resources are
installed inside `molt/configs`, so the CLI works outside the source directory.

Named experiment recipes, baseline networks, paper assets, dataset acquisition tools,
verification scripts, input datasets and fitted artifacts are excluded. Endpoint templates
keep the model and optimization controls, including optional expert, graph, augmentation,
calibration and CT feature-cache choices. Stacked elastic-net scores remain because they are
inputs to MOLT's own fusion experts.
