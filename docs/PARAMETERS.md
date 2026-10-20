# Parameter reference

The endpoint YAML templates expose all applicable model, optimization and preprocessing controls.
Architecture dimensions derived from fitted data are configured by the preprocessor rather than hand-entered.
Python dataclass defaults below are API defaults; the endpoint templates may override them.
Null paths mean that the caller must supply a location when the corresponding operation is enabled.

## Configuration groups

- `paths`: explicit cohort, CT, pathway, run, split and optional metric-registry roots.
- `cohorts.development`: caller-defined cohort names; IDs must be unique across combined cohorts.
- `cv`: outer/inner fold counts, stratification labels, optional explicit `path` and `seed`, and `rebuild`.
- `search`: candidate grid, number of configurations, prediction-mode weights, metric contribution and optional fixed overrides per outer fold.
- `experiment`: run label, inference modes, outer fold subset, checkpoint retention and worker processes.
- `data`: clinical encoding, expression filtering/scaling, panel selection, stacked scores, CT sampling and survival bins.
- `model`: encoder, expert, fusion and head settings. `impl` is `molt` (`auto` is a compatibility alias).
- `loss`: endpoint weights, reconstruction, domain, Brier, attention, focal, ranking, sparsity and precision terms.
- `train`: optimizer, schedule, batch sizes, epochs, early stopping, ensemble size and MC-dropout samples.
- `calibration` and `recalibration`: sensitivity target, abstention coverage, conformal alpha, risk cutoffs and survival horizons.
- `evaluation`: bootstrap replicates and optional summary registry; `comparison: true` requires `paths.state_dir`.
- `hardware`: CPU thread cap; the selected `--device` or API `device` controls execution.

`data.rna.enabled` and `data.ct.enabled` control branch availability. Data-derived fields
include vocabulary sizes, pathway incidence/adjacency, quality dimensions and survival bins.
`loss.pos_weight: auto` estimates class weighting from each training fold; `loss.multitask`
activates both endpoint objectives. `loss.horizon_index` is derived from the configured horizon.
`data.rna.batch_correction` accepts `none` or `combat`; `batch_column` names a column in
`rna_quality.parquet`. Stacked scorers are MOLT components; `data.stacking.enabled: false` disables them.
`data.stacking.experts` maps expert names to input blocks and elastic-net `alphas`.
`data.stacking.components` is `total` or `blocks`. CT caching defaults to false; when enabled
it requires a fixed local Swin state provided explicitly through `model.swin_checkpoint`.
No initialization source is bundled or retrieved. Saved ensembles include the full backbone state.
Prediction rebuilds frozen features from that saved state without requiring the initialization file.

`model.pathway_prior` selects `strict`, `curated_reactome` or `full`; mappings can be supplied
in pathway `metadata.json`. The supplied adjacency is used when no alternative is declared.
Prediction modes are `full`, `light`, `no_ct`, `clinical`, `no_gene` and `no_lin`.

## Complete endpoint templates

### EGFR (OS uses the PFI template)

```yaml
# Complete endpoint template. Input and output roots must be supplied explicitly.
defaults:
- default.yaml
experiment:
  name: molt_egfr
  model: molt
  modes:
  - full
  - light
  outer_folds_subset: null
  keep_all_checkpoints: false
  keep_work: false
  parallel_jobs: 1
  worker_threads: 0
cv:
  outer_folds: 5
  inner_folds: 3
  stratify_on:
  - egfr_status
  - pfi_event
  - os_event
  extra_indicator: null
  rebuild: false
data:
  clinical:
    numeric:
    - age
    - pack_years
    - smoking_years
    - years_since_quit
    categorical:
    - sex
    - smoking_status
    - histology_subtype
    - ajcc_stage
    - pT
    - pN
    - pM
    - laterality
    - lobe
    - prior_malignancy
    winsor_limits:
      age:
      - 18
      - 95
      pack_years:
      - 0
      - 150
      smoking_years:
      - 0
      - 80
      years_since_quit:
      - 0
      - 70
    min_category_count: 2
    partial_levels: coarsen
    coarsen_p: 0.3
    partial_min_count: 10
    structural_zero: true
    hierarchy:
      smoking_status:
        ever_unspecified:
        - current
        - former_le15
        - former_gt15
        - former_unspecified
        former_unspecified:
        - former_le15
        - former_gt15
      ajcc_stage:
        I:
        - IA
        - IA1
        - IA2
        - IA3
        - IB
        II:
        - IIA
        - IIB
        III:
        - IIIA
        - IIIB
        - IIIC
        IV:
        - IVA
        - IVB
      pT:
        T1:
        - T1mi
        - T1a
        - T1b
        - T1c
        T2:
        - T2a
        - T2b
      pM:
        M1:
        - M1a
        - M1b
        - M1c
      laterality:
        NA:
        - left
        - right
      lobe:
        NA:
        - upper
        - middle
        - lower
    coarsen_hide:
      smoking_status:
        ever_unspecified:
        - smoking_years
        - years_since_quit
        former_unspecified:
        - smoking_years
        - years_since_quit
    systematic_min_site_size: 5
    site_column: site
  rna:
    enabled: true
    min_expr_fraction: 0.2
    mask_genes: []
    exclude_genes:
    - ALK
    standardize_pathways: true
    external_standardization: auto
    gene_targets: true
    sample_normalization: rank
    augment:
      enabled: true
      p: 1.0
      scale_sd: 0.05
      shift_sd: 0.05
      mask_min: 0.05
      mask_max: 0.1
      rank_sd: 0.05
    gene_panel:
      size: 2000
      min_mean: null
      forced:
      - EGFR
      - EGFR-AS1
    batch_correction: none
    batch_column: plate
    sd_floor: 0.05
    z_clip: 8.0
    quality_missing_indicators: true
  stacking:
    folds: 5
    n_lambdas: 12
    lambda_ratio: 0.01
    components: total
    experts:
      gene_lin:
        inputs:
        - genes
        alphas:
        - 0.5
        - 0.9
      clinical_lin:
        inputs:
        - clinical
        alphas:
        - 0.5
    enabled: true
  ct:
    enabled: true
    n_lung: 32
    n_candidates: 32
    patch_size: 64
    hu_window:
    - -1024
    - 600
    augment: true
    cache_dir: null
    feature_cache: false
    feature_views: 4
    feature_dir: null
    require_timing: true
    require_qc: true
    timing_unknown_subset: false
    candidate_jitter: 4
    flip_p: 0.5
    rotate_p: 0.5
    intensity_scale: 0.05
    intensity_shift: 0.02
    noise_std: 0.01
    smooth_p: 0.2
    max_mask_points: 200000
    memory_cache: 4
    flags:
    - effusion
    - collapse
    - consolidation
    - mask_corrected
    - truncation
    - low_lung_volume
    - timing_uncertain
  survival:
    endpoint: pfi
    bin_width_months: 6
    max_months: 60
    primary_horizon_months: 24
    require_modality: true
    ipcw_min_prob: 0.05
  domain:
    key: site
    min_count: 5
model:
  impl: molt
  latent_dim: 128
  ct_backbone: swin_vit
  swin_checkpoint: null
  dropout: 0.1
  head_dropout: 0.2
  modality_dropout_p: 0.2
  modality_dropout_by:
    clinical: 0.5
  modality_curriculum_epochs: 0
  modality_group_dropout:
    rna: 0.3
  pathway_graph_mode: prior
  pathway_prior: full
  pathway_skip: true
  clinical_skip: false
  gene_hidden: 0
  gene_input_dropout: 0.5
  gene_dropout: 0.2
  gene_weight_decay: null
  expert_aux_heads: shared
  fusion_rule: poe
  use_reliability: true
  rna_precision_experts: all
  clinical_dim: 64
  clinical_layers: 2
  clinical_heads: 4
  clinical_indicator_tokens: true
  cat_mask_p: 0.1
  num_mask_p: 0.0
  pathway_dim: 64
  pathway_layers: 2
  pathway_heads: 4
  node_id_dim: 16
  n_rna_aux: 0
  gene_mask_ratio: 0.15
  gene_noise_std: 0.1
  ct_embed_dim: 128
  mil_hidden: 128
  mil_gated: false
  ct_chunk_size: 64
  ct_grad_checkpoint: false
  ct_recon: false
  ct_recon_patches: 4
  cnn_widths:
  - 32
  - 64
  - 128
  - 256
  swin_feature_size: 48
  fixed_reliability: []
  expert_hidden: 0
  latent_sampling: false
  head_hidden: 64
  init_hazard: 0.05
  attn_dropout: 0.1
  modality_dropout_exclude: []
  grl_alpha: 1.0
  stacked_skip: false
  omit_experts: []
  gene_expert: true
loss:
  egfr: 1.0
  surv: 1.0
  multitask: false
  recon: 0.1
  domain: 0.0
  brier: 0.1
  attn_prior: 0.01
  focal_gamma: 1.0
  pos_weight: auto
  label_smoothing: 0.0
  expert_aux: 0.0
  rank: 0.0
  gene_l1: 0.1
  rna_precision: 0.0
train:
  batch_size: 32
  eval_batch_size: 64
  max_epochs: 60
  min_epochs: 5
  patience: 8
  lr: 0.0003
  weight_decay: 0.01
  warmup_epochs: 2
  grad_clip: 1.0
  amp: true
  members: 5
  mc_samples: 10
  modality_dropout: true
  num_workers: 0
  monitor: endpoint
  ct_backbone_lr_scale: 0.1
  head_lr_scale: 1.0
  pretrain_epochs: 0
  min_lr_ratio: 0.02
  pretrain_lr: 0.001
  pretrain_mask: 0.25
  pretrain_noise: 0.1
  grl_schedule: true
search:
  n_configs: 16
  selection_modes:
    full: 1.0
    light: 1.0
  selection_metric_weight: 0.0
  grid:
    train.lr:
    - 0.0003
    - 0.001
    train.weight_decay:
    - 0.0001
    - 0.01
    model.latent_dim:
    - 64
    - 128
    loss.focal_gamma:
    - 0
    - 1
    data.rna.gene_panel.size:
    - 1000
    - 2000
    - 4000
  per_outer_fold: null
calibration:
  sensitivity_target: 0.9
  coverage: 0.9
  conformal_alpha: 0.05
  risk_cutoffs:
  - 0.2
  - 0.4
  horizons_months:
  - 12
  - 24
  - 36
  abstain_on_quality_flag: true
recalibration:
  pfi_intercepts: horizon
evaluation:
  summary_bootstrap: 200
  comparison: false
hardware:
  cpu_threads: 4
```

### PFI (OS uses the PFI template)

```yaml
# Complete endpoint template. Input and output roots must be supplied explicitly.
defaults:
- default.yaml
experiment:
  name: molt_pfi
  model: molt
  modes:
  - full
  - light
  outer_folds_subset: null
  keep_all_checkpoints: false
  keep_work: false
  parallel_jobs: 1
  worker_threads: 0
cv:
  outer_folds: 5
  inner_folds: 3
  stratify_on:
  - egfr_status
  - pfi_event
  - os_event
  extra_indicator: null
  rebuild: false
data:
  clinical:
    numeric:
    - age
    - pack_years
    - smoking_years
    - years_since_quit
    categorical:
    - sex
    - smoking_status
    - histology_subtype
    - ajcc_stage
    - pT
    - pN
    - pM
    - laterality
    - lobe
    - prior_malignancy
    winsor_limits:
      age:
      - 18
      - 95
      pack_years:
      - 0
      - 150
      smoking_years:
      - 0
      - 80
      years_since_quit:
      - 0
      - 70
    min_category_count: 2
    partial_levels: coarsen
    coarsen_p: 0.3
    partial_min_count: 10
    structural_zero: true
    hierarchy:
      smoking_status:
        ever_unspecified:
        - current
        - former_le15
        - former_gt15
        - former_unspecified
        former_unspecified:
        - former_le15
        - former_gt15
      ajcc_stage:
        I:
        - IA
        - IA1
        - IA2
        - IA3
        - IB
        II:
        - IIA
        - IIB
        III:
        - IIIA
        - IIIB
        - IIIC
        IV:
        - IVA
        - IVB
      pT:
        T1:
        - T1mi
        - T1a
        - T1b
        - T1c
        T2:
        - T2a
        - T2b
      pM:
        M1:
        - M1a
        - M1b
        - M1c
      laterality:
        NA:
        - left
        - right
      lobe:
        NA:
        - upper
        - middle
        - lower
    coarsen_hide:
      smoking_status:
        ever_unspecified:
        - smoking_years
        - years_since_quit
        former_unspecified:
        - smoking_years
        - years_since_quit
    systematic_min_site_size: 5
    site_column: site
  rna:
    enabled: true
    min_expr_fraction: 0.2
    mask_genes: []
    exclude_genes:
    - ALK
    standardize_pathways: true
    external_standardization: auto
    gene_targets: true
    sample_normalization: rank
    augment:
      enabled: false
      p: 1.0
      scale_sd: 0.05
      shift_sd: 0.05
      mask_min: 0.05
      mask_max: 0.1
      rank_sd: 0.05
    gene_panel:
      size: 0
      min_mean: null
      forced: []
    batch_correction: none
    batch_column: plate
    sd_floor: 0.05
    z_clip: 8.0
    quality_missing_indicators: true
  stacking:
    folds: 5
    n_lambdas: 12
    lambda_ratio: 0.01
    components: blocks
    experts:
      rna_lin:
        inputs:
        - clinical
        - pathways
        alphas:
        - 0.5
      clinical_lin:
        inputs:
        - clinical
        alphas:
        - 0.5
    enabled: true
  ct:
    enabled: true
    n_lung: 32
    n_candidates: 32
    patch_size: 64
    hu_window:
    - -1024
    - 600
    augment: true
    cache_dir: null
    feature_cache: false
    feature_views: 4
    feature_dir: null
    require_timing: true
    require_qc: true
    timing_unknown_subset: false
    candidate_jitter: 4
    flip_p: 0.5
    rotate_p: 0.5
    intensity_scale: 0.05
    intensity_shift: 0.02
    noise_std: 0.01
    smooth_p: 0.2
    max_mask_points: 200000
    memory_cache: 4
    flags:
    - effusion
    - collapse
    - consolidation
    - mask_corrected
    - truncation
    - low_lung_volume
    - timing_uncertain
  survival:
    endpoint: pfi
    bin_width_months: 6
    max_months: 60
    primary_horizon_months: 24
    require_modality: true
    ipcw_min_prob: 0.05
  domain:
    key: site
    min_count: 5
model:
  impl: molt
  latent_dim: 32
  ct_backbone: swin_vit
  swin_checkpoint: null
  dropout: 0.1
  head_dropout: 0.2
  modality_dropout_p: 0.2
  modality_dropout_by:
    rna: 0.8
  modality_curriculum_epochs: 0
  pathway_graph_mode: prior
  pathway_prior: full
  pathway_skip: true
  clinical_skip: false
  gene_hidden: 0
  gene_input_dropout: 0.5
  gene_dropout: 0.2
  gene_weight_decay: null
  expert_aux_heads: shared
  fusion_rule: poe
  use_reliability: true
  rna_precision_experts: all
  clinical_dim: 64
  clinical_layers: 2
  clinical_heads: 4
  clinical_indicator_tokens: true
  cat_mask_p: 0.1
  num_mask_p: 0.0
  pathway_dim: 64
  pathway_layers: 2
  pathway_heads: 4
  node_id_dim: 16
  n_rna_aux: 0
  gene_mask_ratio: 0.15
  gene_noise_std: 0.1
  ct_embed_dim: 128
  mil_hidden: 128
  mil_gated: false
  ct_chunk_size: 64
  ct_grad_checkpoint: false
  ct_recon: false
  ct_recon_patches: 4
  cnn_widths:
  - 32
  - 64
  - 128
  - 256
  swin_feature_size: 48
  fixed_reliability: []
  expert_hidden: 0
  latent_sampling: false
  head_hidden: 64
  init_hazard: 0.05
  attn_dropout: 0.1
  modality_dropout_exclude: []
  grl_alpha: 1.0
  modality_group_dropout: null
  stacked_skip: false
  omit_experts: []
  gene_expert: true
loss:
  egfr: 1.0
  surv: 1.0
  multitask: false
  recon: 0.1
  domain: 0.0
  brier: 0.1
  attn_prior: 0.01
  focal_gamma: 1.0
  pos_weight: auto
  label_smoothing: 0.0
  expert_aux: 0.0
  rank: 0.0
  rank_scale: 1.0
  gene_l1: 0.0
  rna_precision: 0.0
train:
  batch_size: 32
  eval_batch_size: 64
  max_epochs: 60
  min_epochs: 10
  patience: 12
  lr: 0.0003
  weight_decay: 0.01
  warmup_epochs: 2
  grad_clip: 1.0
  amp: true
  members: 5
  mc_samples: 10
  modality_dropout: true
  num_workers: 0
  monitor: metric
  ct_backbone_lr_scale: 0.1
  head_lr_scale: 1.0
  pretrain_epochs: 0
  min_lr_ratio: 0.02
  pretrain_lr: 0.001
  pretrain_mask: 0.25
  pretrain_noise: 0.1
  grl_schedule: true
search:
  n_configs: 12
  selection_modes:
    full: 1.0
    light: 1.0
  selection_metric_weight: 0.0
  grid:
    train.lr:
    - 0.0003
    - 0.001
    train.weight_decay:
    - 0.0001
    - 0.01
    data.survival.bin_width_months:
    - 6
    - 12
    loss.rank:
    - 0.0
    - 0.5
  per_outer_fold: null
calibration:
  sensitivity_target: 0.9
  coverage: 0.9
  conformal_alpha: 0.05
  risk_cutoffs:
  - 0.2
  - 0.4
  horizons_months:
  - 12
  - 24
  - 36
  abstain_on_quality_flag: true
recalibration:
  pfi_intercepts: horizon
evaluation:
  summary_bootstrap: 200
  comparison: false
hardware:
  cpu_threads: 4
```

## molt.models.molt.MOLTConfig

Architecture and regularisation settings of :class:`MOLT`.

Attributes:
    n_num: Number of continuous clinical variables.
    n_cat: Number of categorical clinical variables.
    cat_cardinalities: Vocabulary size per categorical variable, including the reserved
        missing (0) and out-of-vocabulary (1) indices.
    n_pathways: Number of pathway nodes ``P``.
    pathway_adjacency: ``(P, P)`` 0/1 adjacency prior (self-loops are added if absent).
    latent_dim: Dimension ``D`` of the fused latent space.
    n_survival_bins: Number of discrete time bins ``K``.
    n_domains: Number of domain classes for the gradient-reversal classifier (0 disables).
    use_ct: Build the CT branch.
    ct_backbone: ``"swin_vit"`` or ``"cnn3d"``.
    swin_checkpoint: Path to a caller-supplied Swin-ViT state or ``None``.
    quality_dims: Number of quality inputs per modality (keys ``clinical``, ``rna``, ``ct``).
    quality_signs: Optional orientation of each quality input (+1 adverse, -1 favourable).
    use_rna: Build the transcriptomic branch.
    n_clinical_indicators: Extra binary clinical indicators (``batch["clinical_indicators"]``).
    clinical_dim: Clinical token dimension.
    clinical_layers: Clinical transformer layers.
    clinical_heads: Clinical attention heads.
    clinical_indicator_tokens: Add per-variable missingness-indicator tokens.
    clinical_skip: Add a linear encoding of the flat clinical record to the clinical
        embedding (see :class:`~molt.models.clinical.ClinicalEncoder`).
    cat_mask_p: Stochastic masking probability of observed categories during training.
    num_mask_p: Stochastic masking probability of observed continuous values.
    pathway_dim: Pathway node dimension.
    pathway_layers: Graph attention layers.
    pathway_heads: Graph attention heads.
    node_id_dim: Pathway node-identity embedding size.
    pathway_graph_mode: ``"prior"``, ``"dense"`` or ``"mlp"``.
    pathway_skip: Add the graph-free encoding of the pathway activities to the pooled
        graph embedding (see :class:`~molt.models.pathway.PathwayEncoder`).
    n_rna_aux: Number of auxiliary transcriptomic features (``batch["rna_aux"]``, e.g.
        selected gene-level values) added to the pathway embedding through an MLP.
    n_genes: Genes predicted by the transcriptomic decoder (0 disables the masked-gene objective).
    pathway_incidence: Optional ``(n_genes, P)`` incidence for the masked-gene objective.
    gene_mask_ratio: Fraction of genes hidden in the masked-gene objective.
    gene_noise_std: Noise added to visible genes in the masked-gene objective.
    ct_embed_dim: CT patch/scan embedding dimension.
    mil_hidden: Width of the MIL attention network.
    mil_gated: Use gated MIL attention.
    ct_hu_window: HU window mapped to ``[0, 1]`` before the backbone (``None`` = inputs are
        already normalised).
    ct_chunk_size: Patches per backbone call.
    ct_grad_checkpoint: Recompute backbone activations in the backward pass.
    ct_recon: Enable the masked-patch objective.
    ct_recon_patches: Patches per scan used for the masked-patch objective.
    cnn_widths: Channel widths of the ``cnn3d`` backbone.
    swin_feature_size: Swin-ViT feature size.
    fusion_rule: ``"poe"`` or ``"mean"``.
    use_reliability: Learn reliability weights (``False`` fixes all ``rho`` to 1).
    fixed_reliability: Modalities with ``rho`` fixed to 1.
    expert_hidden: Hidden width of the expert networks (0 = linear experts).
    latent_sampling: Sample ``z ~ N(mu, Lambda^-1)`` during training instead of using ``mu``.
    head_hidden: Hidden width of the task heads.
    init_hazard: Initial per-bin hazard of the survival head.
    primary_horizon_index: Survival column index of the primary horizon (Brier term, risk).
    dropout: Encoder dropout.
    attn_dropout: Attention dropout.
    head_dropout: Task-head dropout.
    modality_dropout_p: Expert drop probability when ``dropout_modalities=True``.
    modality_dropout_exclude: Modalities never dropped.
    grl_alpha: Gradient-reversal strength.
    n_panel_genes: Genes of the gene-level expert panel (``batch["rna_panel"]``; 0 disables
        the gene-level expert). The expert shares the RNA availability and reliability.
    gene_hidden: Hidden width of the gene-level expert (0 = linear in the genes).
    gene_input_dropout: Probability of hiding each panel gene during training.
    gene_dropout: Dropout after the hidden layer of the gene-level expert.
    gene_weight_decay: Weight decay of the gene-level expert (``None`` = global value).
    expert_aux_heads: ``"none"``, ``"shared"`` (fused heads applied to every expert's
        single-expert posterior) or ``"own"`` (expert-specific heads); the auxiliary
        outputs are trained with the endpoint loss (``LossWeights.expert_aux``).
    modality_dropout_by: Expert-specific drop probabilities (default
        ``modality_dropout_p``), e.g. ``{"clinical": 0.5}``.
    modality_curriculum_epochs: Length of the transcriptomic curriculum: in epoch ``e`` a
        participant with an expression profile is trained on its transcriptomic experts
        only with probability ``max(0, 1 - e / epochs)``.
    stacked_dims: Input dimension of every stacked linear expert
        (``batch["stacked_<name>"]``, see :mod:`molt.training.stacking`); each enters the
        product of experts with its own Gaussian expert and the reliability weight of
        its source modality.
    modality_group_dropout: Joint drop probability of whole modality groups
        (``{"rna": 0.3}`` drops every transcriptomic expert at once, as in light mode).
    stacked_skip: Also add the stacked scores linearly to the EGFR logit and the hazard
        logits (``sum_m rho_m w_m' s_m`` over present stacked experts, ``w`` learned from 0).
    cat_mixing, cat_coarsen, cat_coarsen_hide, cat_coarsen_p: Partially specified
        categories (mixture matrix, coarsening table, hidden continuous variables and
        probability; see :class:`~molt.models.clinical.ClinicalEncoder`).
    rna_precision_experts: Experts shrunk by ``loss.rna_precision``: ``all``
        transcriptomic experts or the ``neural`` ones (pathway and gene-level).
    omit_experts: Experts removed from the product of experts in training and inference
        (component ablations, e.g. ``("rna",)`` for the pathway expert alone).

Fields and API defaults:

- `n_num` (`int`): `required`.
- `n_cat` (`int`): `required`.
- `cat_cardinalities` (`list[int]`): `required`.
- `n_pathways` (`int`): `42`.
- `pathway_adjacency` (`np.ndarray | None`): `None`.
- `latent_dim` (`int`): `64`.
- `n_survival_bins` (`int`): `10`.
- `n_domains` (`int`): `0`.
- `use_ct` (`bool`): `True`.
- `ct_backbone` (`str`): `'cnn3d'`.
- `swin_checkpoint` (`str | None`): `None`.
- `quality_dims` (`dict[str, int]`): `{'clinical': 0, 'rna': 0, 'ct': 0}`.
- `quality_signs` (`dict[str, list[float]] | None`): `None`.
- `use_rna` (`bool`): `True`.
- `n_clinical_indicators` (`int`): `0`.
- `clinical_dim` (`int`): `64`.
- `clinical_layers` (`int`): `2`.
- `clinical_heads` (`int`): `4`.
- `clinical_indicator_tokens` (`bool`): `True`.
- `clinical_skip` (`bool`): `False`.
- `cat_mask_p` (`float`): `0.1`.
- `num_mask_p` (`float`): `0.0`.
- `pathway_dim` (`int`): `64`.
- `pathway_layers` (`int`): `2`.
- `pathway_heads` (`int`): `4`.
- `node_id_dim` (`int`): `16`.
- `pathway_graph_mode` (`str`): `'prior'`.
- `pathway_skip` (`bool`): `True`.
- `n_rna_aux` (`int`): `0`.
- `n_genes` (`int`): `0`.
- `pathway_incidence` (`np.ndarray | None`): `None`.
- `gene_mask_ratio` (`float`): `0.15`.
- `gene_noise_std` (`float`): `0.1`.
- `ct_embed_dim` (`int`): `128`.
- `mil_hidden` (`int`): `128`.
- `mil_gated` (`bool`): `False`.
- `ct_hu_window` (`tuple[float, float] | None`): `(-1024.0, 600.0)`.
- `ct_chunk_size` (`int`): `64`.
- `ct_grad_checkpoint` (`bool`): `False`.
- `ct_recon` (`bool`): `False`.
- `ct_recon_patches` (`int`): `4`.
- `cnn_widths` (`tuple[int, ...]`): `(32, 64, 128, 256)`.
- `swin_feature_size` (`int`): `48`.
- `fusion_rule` (`str`): `'poe'`.
- `use_reliability` (`bool`): `True`.
- `fixed_reliability` (`tuple[str, ...]`): `()`.
- `expert_hidden` (`int`): `0`.
- `latent_sampling` (`bool`): `False`.
- `head_hidden` (`int`): `64`.
- `init_hazard` (`float`): `0.05`.
- `primary_horizon_index` (`int | None`): `None`.
- `dropout` (`float`): `0.1`.
- `attn_dropout` (`float`): `0.1`.
- `head_dropout` (`float`): `0.2`.
- `modality_dropout_p` (`float`): `0.2`.
- `modality_dropout_exclude` (`tuple[str, ...]`): `()`.
- `grl_alpha` (`float`): `1.0`.
- `n_panel_genes` (`int`): `0`.
- `gene_hidden` (`int`): `0`.
- `gene_input_dropout` (`float`): `0.5`.
- `gene_dropout` (`float`): `0.2`.
- `gene_weight_decay` (`float | None`): `None`.
- `expert_aux_heads` (`str`): `'none'`.
- `modality_dropout_by` (`dict[str, float] | None`): `None`.
- `modality_curriculum_epochs` (`int`): `0`.
- `stacked_dims` (`dict[str, int] | None`): `None`.
- `modality_group_dropout` (`dict[str, float] | None`): `None`.
- `stacked_skip` (`bool`): `False`.
- `cat_mixing` (`Any`): `None`.
- `cat_coarsen` (`Any`): `None`.
- `cat_coarsen_hide` (`Any`): `None`.
- `cat_coarsen_p` (`float`): `0.0`.
- `rna_precision_experts` (`str`): `'all'`.
- `omit_experts` (`Sequence[str]`): `()`.

## molt.models.losses.LossWeights

Weights of the terms of the MOLT objective.

``total = egfr * l_EGFR + surv * l_surv + recon * l_recon + domain * l_domain
+ brier * (l_Brier,EGFR [if egfr > 0] + l_Brier,surv [if surv > 0]) + attn_prior * l_prior``.

Attributes:
    egfr: Weight of the focal EGFR loss.
    surv: Weight of the discrete-time survival likelihood.
    recon: Weight of the self-supervised masked-modelling losses returned by the model.
    domain: Weight of the gradient-reversal domain cross-entropy.
    brier: Weight of the Brier terms of the active endpoints.
    attn_prior: Weight of the penalty on attention outside the pathway prior
        (dense graph mode only).
    egfr_pos_weight: Positive-class weight of the EGFR loss (``None`` = 1).
    focal_gamma: Focal-loss focusing parameter.
    horizon_index: Bin index of the Brier horizon (overrides the model configuration).
    label_smoothing: EGFR targets become ``y (1 - eps) + eps / 2``.
    expert_aux: Weight of the deep-supervision term (endpoint loss of every expert's
        single-expert output, averaged over experts; requires ``expert_aux_heads``).
    rank: Weight of the IPCW-weighted pairwise ranking loss on the horizon risk.
    rank_scale: Temperature of the ranking surrogate.
    gene_l1: Weight of the L1 penalty on the first linear map of the gene-level expert.
    rna_precision: Weight of the shrinkage penalty on the precision that the fusion gives
        the transcriptomic experts (mean over participants of
        ``sum_m rho_m mean_d lambda_m`` over the present transcriptomic experts), which
        pulls the fused posterior towards the non-transcriptomic experts unless the
        expression profile lowers the endpoint loss.

Fields and API defaults:

- `egfr` (`float`): `1.0`.
- `surv` (`float`): `1.0`.
- `recon` (`float`): `0.1`.
- `domain` (`float`): `0.0`.
- `brier` (`float`): `0.0`.
- `attn_prior` (`float`): `0.01`.
- `egfr_pos_weight` (`float | None`): `None`.
- `focal_gamma` (`float`): `0.0`.
- `horizon_index` (`int | None`): `None`.
- `label_smoothing` (`float`): `0.0`.
- `expert_aux` (`float`): `0.0`.
- `rank` (`float`): `0.0`.
- `rank_scale` (`float`): `1.0`.
- `gene_l1` (`float`): `0.0`.
- `rna_precision` (`float`): `0.0`.

## molt.training.trainer.TrainSettings

Optimisation settings of one network (see :class:`Trainer`).

Fields and API defaults:

- `lr` (`float`): `0.0003`.
- `weight_decay` (`float`): `0.01`.
- `batch_size` (`int`): `32`.
- `eval_batch_size` (`int`): `64`.
- `max_epochs` (`int`): `60`.
- `min_epochs` (`int`): `3`.
- `patience` (`int`): `8`.
- `warmup_epochs` (`float`): `2.0`.
- `min_lr_ratio` (`float`): `0.02`.
- `grad_clip` (`float`): `1.0`.
- `amp` (`bool`): `True`.
- `members` (`int`): `5`.
- `mc_samples` (`int`): `10`.
- `modality_dropout` (`bool`): `True`.
- `num_workers` (`int`): `4`.
- `monitor` (`str`): `'endpoint'`.
- `ct_backbone_lr_scale` (`float`): `0.1`.
- `head_lr_scale` (`float`): `1.0`.
- `pretrain_epochs` (`int`): `0`.
- `pretrain_lr` (`float`): `0.001`.
- `pretrain_mask` (`float`): `0.25`.
- `pretrain_noise` (`float`): `0.1`.
- `grl_schedule` (`bool`): `True`.
- `seed` (`int`): `20240613`.

## molt.training.preprocess.PreprocessConfig

Settings of :class:`Preprocessor` (see the module docstring).

Fields and API defaults:

- `numeric` (`tuple[str, ...]`): `('age', 'pack_years', 'smoking_years', 'years_since_quit')`.
- `categorical` (`tuple[str, ...]`): `('sex', 'smoking_status', 'histology_subtype', 'ajcc_stage', 'pT', 'pN', 'pM', 'laterality', 'lobe', 'prior_malignancy')`.
- `winsor_limits` (`dict[str, tuple[float, float]]`): `{'age': (18.0, 95.0), 'pack_years': (0.0, 150.0), 'smoking_years': (0.0, 80.0), 'years_since_quit': (0.0, 70.0)}`.
- `min_category_count` (`int`): `2`.
- `partial_levels` (`str`): `'none'`.
- `coarsen_p` (`float`): `0.2`.
- `partial_min_count` (`int`): `10`.
- `hierarchy` (`dict[str, Any] | None`): `None`.
- `coarsen_hide` (`dict[str, Any] | None`): `None`.
- `structural_zero` (`bool`): `False`.
- `systematic_min_site_size` (`int`): `5`.
- `site_column` (`str`): `'site'`.
- `use_rna` (`bool`): `True`.
- `min_expr_fraction` (`float`): `0.2`.
- `mask_genes` (`tuple[str, ...]`): `()`.
- `exclude_genes` (`tuple[str, ...]`): `('ALK',)`.
- `standardize_pathways` (`bool`): `False`.
- `external_standardization` (`str`): `'auto'`.
- `sd_floor` (`float`): `0.05`.
- `z_clip` (`float`): `8.0`.
- `rna_quality_missing_indicators` (`bool`): `True`.
- `sample_normalization` (`str`): `'none'`.
- `gene_panel_size` (`int`): `0`.
- `gene_panel_min_mean` (`float | None`): `None`.
- `gene_panel_forced` (`tuple[str, ...]`): `()`.
- `stacking` (`dict[str, Any] | None`): `None`.
- `task` (`str`): `'egfr'`.
- `use_ct` (`bool`): `True`.
- `ct_flags` (`tuple[str, ...]`): `('effusion', 'collapse', 'consolidation', 'mask_corrected', 'truncation', 'low_lung_volume', 'timing_uncertain')`.
- `survival_endpoint` (`str`): `'pfi'`.
- `bin_width` (`float`): `6.0`.
- `max_time` (`float`): `60.0`.
- `horizon` (`float`): `24.0`.
- `require_modality_for_survival` (`bool`): `True`.
- `domain_key` (`str`): `'site'`.
- `domain_min_count` (`int`): `5`.
- `ipcw_min_prob` (`float`): `0.05`.

## molt.training.ct_patches.CTPatchConfig

Settings of :class:`CTPatchSampler`.

Fields and API defaults:

- `n_lung` (`int`): `32`.
- `n_candidates` (`int`): `32`.
- `patch_size` (`int`): `64`.
- `hu_window` (`tuple[float, float]`): `(-1024.0, 600.0)`.
- `candidate_jitter` (`int`): `4`.
- `augment` (`bool`): `True`.
- `flip_p` (`float`): `0.5`.
- `rotate_p` (`float`): `0.5`.
- `intensity_scale` (`float`): `0.05`.
- `intensity_shift` (`float`): `0.02`.
- `noise_std` (`float`): `0.01`.
- `smooth_p` (`float`): `0.2`.
- `cache_dir` (`str | None`): `None`.
- `max_mask_points` (`int`): `200000`.
- `memory_cache` (`int`): `4`.

## molt.training.rna_augment.RNAAugmentConfig

Settings of :class:`RNAAugmenter` (see the module docstring).

Fields and API defaults:

- `enabled` (`bool`): `False`.
- `p` (`float`): `1.0`.
- `scale_sd` (`float`): `0.1`.
- `shift_sd` (`float`): `0.1`.
- `mask_min` (`float`): `0.05`.
- `mask_max` (`float`): `0.2`.
- `rank_sd` (`float`): `0.1`.

## molt.training.gene_panel.GenePanelConfig

Settings of :class:`GenePanel`.

Attributes:
    size: Number of highest-variance genes (0 disables the panel).
    min_mean: Minimum training mean on the stored scale; ``None`` uses the cohort's
        detection threshold (1.0 = 1 TPM for ``log2(TPM + 1)`` RNA-seq; no mean filter on
        relative scales such as z-scores, where the threshold is undefined).
    forced: Genes added irrespective of the variance filter (when measured).
    excluded: Genes never used (e.g. target-gene masking).
    sample_normalization: ``"none"`` or ``"rank"`` (see the module docstring).
    sd_floor: Lower bound of the gene-wise standard deviation.
    z_clip: Absolute clip of standardized values.
    external_standardization: ``"auto"`` (same-platform training parameters, otherwise
        the cohort's own), ``"train"`` (always training parameters) or ``"self"``.

Fields and API defaults:

- `size` (`int`): `0`.
- `min_mean` (`float | None`): `None`.
- `forced` (`tuple[str, ...]`): `()`.
- `excluded` (`tuple[str, ...]`): `()`.
- `sample_normalization` (`str`): `'none'`.
- `sd_floor` (`float`): `0.05`.
- `z_clip` (`float`): `8.0`.
- `external_standardization` (`str`): `'auto'`.

## molt.preprocess.ct.config.VolumeConfig

Volume assembly and resampling.

Attributes:
    spacing_mm: Isotropic output voxel size.
    hu_clip: HU range of the stored volume.
    padding_hu: Values below this (before clipping) are treated as outside the
        field of view.
    antialias: Smooth axes that are downsampled before B-spline interpolation.
    tilt_tolerance_deg: Gantry tilt (slice shear) above which the shear is corrected.
    max_tilt_deg: Gantry tilt above which the series fails geometry QC.
    crop_margin_mm: Margin kept around the lung bounding box when cropping.
    thorax_roi: Locate the lungs on every few native slices and resample only that region
        (plus ``roi_margin_mm``); avoids interpolating large fields of view.
    roi_margin_mm: Margin around the located lungs.
    roi_slice_step_mm: Approximate slice distance used to locate the lungs.
    max_extent_mm: Largest accepted physical extent along any axis.
    sitk_threads: SimpleITK threads per worker process.

Fields and API defaults:

- `spacing_mm` (`float`): `1.0`.
- `hu_clip` (`tuple[float, float]`): `(-1024.0, 600.0)`.
- `padding_hu` (`float`): `-1500.0`.
- `antialias` (`bool`): `True`.
- `tilt_tolerance_deg` (`float`): `0.5`.
- `max_tilt_deg` (`float`): `30.0`.
- `crop_margin_mm` (`float`): `40.0`.
- `thorax_roi` (`bool`): `True`.
- `roi_margin_mm` (`float`): `50.0`.
- `roi_slice_step_mm` (`float`): `5.0`.
- `max_extent_mm` (`float`): `2000.0`.
- `sitk_threads` (`int`): `4`.

## molt.preprocess.ct.config.LungConfig

Lung segmentation, mask correction and lung QC.

Attributes:
    weights: Optional local segmentation state supplied by the caller.
    batch_size: Axial slices per GPU batch.
    device: Torch device; ``"auto"`` selects CUDA when available.
    gpu_concurrency: Worker processes allowed to run GPU inference at the same time.
    min_volume_l: Lower bound of a plausible total lung volume (litres).
    max_volume_l: Upper bound of a plausible total lung volume (litres).
    min_component_fraction: Components smaller than this fraction of the mask are removed.
    large_component_fraction: Components above this fraction count as large (topology QC).
    large_component_min_ml: ... and above this absolute volume.
    leak_max_hu: Components with a mean HU below this value that lie below the lung bases
        are treated as sub-diaphragmatic gas.
    leak_opening_mm: Opening radius used to detach gas from the lung bases.
    outside_margin_mm: Components centred farther than this outside the bounding box of
        the two main lungs are removed.
    closing_mm: Closing radius of the final per-lung cleanup.
    truncation_area_fraction: Lung area in a boundary slice, relative to the largest
        axial lung area, above which the lung counts as truncated along z.
    fov_touch_fraction: Fraction of lung voxels adjacent to the field-of-view border
        above which the lung counts as truncated in-plane.

Fields and API defaults:

- `weights` (`str | None`): `None`.
- `batch_size` (`int`): `32`.
- `device` (`str`): `'auto'`.
- `gpu_concurrency` (`int`): `4`.
- `min_volume_l` (`float`): `1.5`.
- `max_volume_l` (`float`): `9.0`.
- `min_component_fraction` (`float`): `0.005`.
- `large_component_fraction` (`float`): `0.05`.
- `large_component_min_ml` (`float`): `50.0`.
- `leak_max_hu` (`float`): `-920.0`.
- `leak_opening_mm` (`float`): `2.0`.
- `outside_margin_mm` (`float`): `10.0`.
- `closing_mm` (`float`): `2.0`.
- `truncation_area_fraction` (`float`): `0.25`.
- `fov_touch_fraction` (`float`): `0.01`.

## molt.preprocess.ct.config.CandidateConfig

Candidate abnormality detection inside the lung field.

Attributes:
    dense_hu: HU threshold of dense tissue.
    opening_mm: Radius of the opening applied to the dense-tissue mask.
    region_closing_mm: Closing radius used to include juxta-pleural lesions.
    region_fill: Hole filling of the search region: ``"3d"`` adds cavities enclosed by
        lung, ``"none"`` keeps the closed lung mask.
    min_volume_mm3: Smallest component considered.
    max_components: Largest components analysed (by volume).
    shell_mm: Thickness of the parenchymal shell used for the HU contrast.
    vessel_shape_ratio: Components whose inscribed radius divided by the
        equivalent-sphere radius is below this value are tubular/branching.
    vessel_elongation: Components with a principal-axis ratio above this value ...
    vessel_max_radius_mm: ... and an inscribed radius below this value are tubular.
    n_candidates: Number of candidate centres kept.
    entropy_bin_hu: Histogram bin width for the HU entropy feature.

Fields and API defaults:

- `dense_hu` (`float`): `-300.0`.
- `opening_mm` (`float`): `1.5`.
- `region_closing_mm` (`float`): `6.0`.
- `region_fill` (`str`): `'3d'`.
- `min_volume_mm3` (`float`): `30.0`.
- `max_components` (`int`): `1500`.
- `shell_mm` (`int`): `3`.
- `vessel_shape_ratio` (`float`): `0.35`.
- `vessel_elongation` (`float`): `3.0`.
- `vessel_max_radius_mm` (`float`): `4.0`.
- `n_candidates` (`int`): `64`.
- `entropy_bin_hu` (`float`): `25.0`.

## molt.preprocess.ct.config.QCConfig

Quality-control outputs.

Attributes:
    hard_flags: Flags that make a series fail QC (and trigger the next-ranked series).
    montage: Write a PNG montage for every selected scan.
    montage_dpi: Resolution of the montage.

Fields and API defaults:

- `hard_flags` (`tuple[str, ...]`): `('processing_error', 'geometry', 'no_lung', 'lung_volume_low', 'lung_volume_high', 'truncated_z', 'truncated_fov', 'topology')`.
- `montage` (`bool`): `True`.
- `montage_dpi` (`int`): `90`.

