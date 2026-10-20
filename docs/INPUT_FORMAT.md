# Input format

All filesystem roots are caller-supplied. Cohort identifiers are arbitrary nonempty directory
names. Participant IDs must be unique within each cohort and across cohorts combined for training.
No data files are included in this distribution.

## Clinical and label tables

Clinical Parquet has one row per participant. A complete schema is available as
`molt.data.schema.CLINICAL_SPEC`; `save_clinical` fills missing optional columns and validates.

Required non-null columns are `participant_id`, `cohort`, `histology`, `platform` and
`has_clinical_record`. Histology is `LUAD`, `LUSC` or `NSCLC_other`. Platform is a declared
expression platform from `PLATFORMS`, or `none` when no expression is present. The clinical
encoder's default continuous columns are `age`, `pack_years`, `smoking_years`,
`years_since_quit`; categorical columns are `sex`, `smoking_status`, `histology_subtype`,
`ajcc_stage`, `pT`, `pN`, `pM`, `laterality`, `lobe`, `prior_malignancy`. Optional `site`
is used for domain and collection-pattern indicators. Missing fields may be null; categorical
`NA` is also accepted. Vocabulary fitting and scaling use training participants only.

Labels are a separate one-row-per-participant Parquet table. For EGFR training, supply
`egfr_status` in `{0, 1}` and `egfr_ascertained: true` for known labels. For PFI, supply
`pfi_time`, `pfi_event`, `pfi_available`; for OS, use `os_time`, `os_event`, `os_available`.
Times are months and event flags are 1 for an event or 0 for censoring. Availability flags
must be true for evaluable rows. Other labels can be absent. Use null for unknown values.
Labels may be omitted entirely during prediction. See `LABELS_SPEC` for all supported fields.

To convert clinical and label CSV files:

```bash
python examples/prepare_inputs.py --clinical /absolute/input/clinical.csv \
  --labels /absolute/input/labels.csv --cohort development \
  --cohorts-dir /absolute/output/cohorts
```

The CSV clinical input must include `histology`, `platform` and `participant_id`; this example
adds `cohort` and defaults `has_clinical_record` to true. Bool columns should be encoded as
`true`/`false` or 1/0. If a label is known, set its availability flag explicitly.

## Expression and pathway resources

`expression.parquet` has a `gene` column and one numeric column per participant ID; the
Python API accepts a DataFrame indexed by gene symbols. Missing values may be NaN. Choose
one expression profile per participant. Duplicate participant columns are invalid.

`expression_meta.json` records `platform`, `units`, `sample_selection_rule`, `source_files`,
`n_genes` and `n_samples`. The save helper fills the two dimensions. `source_files` describes
caller-provided input provenance and may be an empty list. Example metadata:

```json
{
  "platform": "rnaseq_star_tpm",
  "units": "log2(TPM+1)",
  "sample_selection_rule": "One caller-selected sample per participant",
  "source_files": []
}
```

RNA-seq input must use log2(x+1) abundances. Array intensities and relative scales such as
`zscore` or log ratios are supported with matching metadata. The reader does not guess or
apply a log transform. For an abundance CSV with gene symbols in the first column:

```bash
python examples/prepare_inputs.py --clinical /absolute/input/clinical.csv \
  --labels /absolute/input/labels.csv --expression /absolute/input/expression.csv \
  --expression-meta /absolute/input/expression_meta.json --log2p1 \
  --cohort development --cohorts-dir /absolute/output/cohorts
```

Omit `--log2p1` for data already on the final scale. Do not apply it to z-scores or log ratios.

RNA preparation requires caller-supplied pathway definitions:

- `modules.json`: a list of objects with unique string `id` and optional `name`, `genes`.
- `incidence.parquet`: `gene` plus one 0/1 column per module ID.
- `adjacency.parquet`: `module` plus one 0/1 column per module ID.

Rows and columns are aligned by identifiers. Adjacency is symmetrized and self-loops are
added. Optional `metadata.json` may map named priors to alternative adjacency files; all
referenced files must be supplied by the caller. No pathway dataset is bundled.

## CT input and preparation

`ct_index.parquet` holds one row per series, with at most one `selected: true` row per
participant. A selected series normally requires `qc_pass: true` and `timing_ok: true`.
`timing_unknown_subset` can admit scans with unknown timing; known post-treatment scans are
excluded when timing is required. Relevant inputs include `participant_id`, `cohort`,
`series_uid`, `npz_path`, `spacing_xyz`, `kernel`, `manufacturer`, `contrast`, `qc_flags`.
The full validated schema also requires `patient_id_src`, `study_uid` and the boolean
`is_petct_component`; see `CT_INDEX_SPEC`.

`npz_path` may be an absolute path or a filename relative to `<ct_dir>/<cohort>/`. If no
explicit filename is present, `<participant_id>.npz` is checked in that directory.
The NPZ format contains:

- `volume`: int16 HU array `(Z, Y, X)` at 1 mm isotropic spacing in LPS orientation.
- `lung_mask`: uint8 array with the same shape, 0 background, 1 right lung, 2 left lung.
- `candidates`: float32 array `(N, 4)` of `(x, y, z, score)` in voxel coordinates.
- `meta`: a JSON scalar string describing spacing, origin, orientation and quality.

`molt.preprocess.ct` retains DICOM header reading, geometric resampling, mask cleanup,
candidate detection, quality control and patch sampling. A complete DICOM conversion example
accepts a caller-supplied lung mask already aligned to the resampled volume:

```bash
python examples/prepare_ct.py --series-dir /absolute/input/dicom-series \
  --lung-mask /absolute/input/aligned-mask.npy --output /absolute/output/participant.npz
```

For segmentation through the optional `LungSegmenter`, install the `segmentation` extra and
pass a compatible local state explicitly; no automatic retrieval occurs. A mask can instead
be supplied by any segmentation workflow that satisfies the format above. Quality flags must
be reviewed before setting the selected/QC/timing fields of the CT index.

## Programmatic loading

```python
from molt.data import load_cohort
from molt.training.cohort_data import load_bundle
from molt.training.preprocess import Preprocessor, PreprocessConfig

cohort = load_cohort("development", cohorts_dir="/absolute/input/cohorts",
                     ct_dir="/absolute/input/ct", validate=True)
bundle = load_bundle(["development"], cohorts_dir="/absolute/input/cohorts",
                     ct_dir="/absolute/input/ct", pathways_dir="/absolute/input/pathways")
# train_ids and validation_ids are disjoint participant-ID lists supplied by the caller.
preprocessor = Preprocessor(PreprocessConfig()).fit(bundle, train_ids)
training_features = preprocessor.transform(bundle, train_ids)
validation_features = preprocessor.transform(bundle, validation_ids)
```

The high-level training workflow creates and persists nested folds, fits separate
preprocessors inside them, and retains their vocabularies, selected genes, scaling and censoring
models for inference. `external_standardization: auto` may adapt RNA normalization to an
incoming cohort without labels; use `train` to reuse development statistics when strict
sample-independent prediction is needed (see the preprocessing implementation for behavior).
