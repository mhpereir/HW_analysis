# Product Stage 3: Event-feature PCA

## Contract

Stage 3 builds a reproducible PCA transform from the Stage-2 event-feature
table. It stops after writing PCA scores, loadings, feature matrices, scaling
parameters, and provenance metadata. It does not assign clusters.

Typical output:

```text
results/stage3_event_feature_pca/hw_event_feature_pca*.nc
```

Older generated artifacts may still exist under `results/event_features/`, but
new defaults use the stage-specific directory above.

## Producer

```text
scripts/event_features/build_stage3_event_feature_pca.py
```

## Consumes

- Stage-2 event-feature NetCDF table

## Dimensions

| Dimension | Meaning |
| --- | --- |
| `event` | Retained events after finite-value filtering. |
| `pc` | Principal component labels, such as `PC1`, `PC2`, `PC3`. |
| `feature` | PCA input feature names after derived-feature resolution. |
| `event_original` | Original event coordinate from the Stage-2 input table. |

## Core Variables

```text
pc_score(event, pc)
pc_loading(pc, feature)
explained_variance(pc)
explained_variance_ratio(pc)
cumulative_explained_variance_ratio(pc)
feature_matrix(event, feature)
feature_matrix_scaled(event, feature)
feature_center(feature)
feature_scale(feature)
valid_event_mask_original(event_original)
```

`pc_score(event, pc)` is the matrix consumed by clustering. `pc_loading(pc,
feature)` is the primary interpretation variable for PCA axes.

## Copied Event Metadata And Diagnostics

The product should copy retained one-dimensional event metadata when present:

```text
event_id(event)
start_time(event)
end_time(event)
peak_time(event)
duration(event)
tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)
```

It may also copy or derive diagnostic variables useful for interpretation:

```text
I_dTdt_pre(event)
I_adiabatic_pre(event)
I_diabatic_pre(event)
I_advection_pre(event)
I_lwa_a_pre_peak(event)
T_anom_mean_ant(event)
days_from_solstice(event)
log10_tas_excess_integral(event)
```

## Supported Derived PCA Inputs

The PCA builder supports raw Stage-2 variables and these derived feature names:

```text
f_adiabatic_pre = I_adiabatic_pre / budget_activity_pre
f_diabatic_pre  = I_diabatic_pre  / budget_activity_pre
f_advection_pre = I_advection_pre / budget_activity_pre

budget_activity_pre =
  abs(I_adiabatic_pre) + abs(I_diabatic_pre) + abs(I_advection_pre)

sqrt_I_lwa_a_pre_peak = sqrt(I_lwa_a_pre_peak)
cos_days_from_solstice = cos(days_from_solstice * 2*pi / 365)
log10_tas_excess_integral = log10(tas_excess_integral)
```

Invalid derived values, such as square roots of negative LWA exposure, are
treated as missing and can drop the event from the PCA matrix.

## Scaling

PCA is fit to standardized features, not raw mixed-unit variables. The current
scalers are:

```text
standard: (x - mean) / std
robust:   (x - median) / IQR
```

The fitted center and scale are saved in `feature_center(feature)` and
`feature_scale(feature)` for reproducibility.

## Required Metadata

Global attrs should include:

```text
pipeline_stage = "stage_3_event_feature_pca"
input_path = ...
source_feature_table = ...
pca_features = comma-separated feature list
scaler = "standard" or "robust"
pca_implementation = "sklearn.decomposition.PCA"
n_input_events = ...
n_valid_events = ...
n_dropped_events = ...
n_features = ...
n_components = ...
missing_event_policy = "drop_missing_events"
clustering_performed = 0
```

## Validation Expectations

The producer should fail clearly when the input has no `event` dimension, a
requested feature cannot be resolved, a required source variable is absent,
fewer than two valid events remain, fewer than two features are selected, a
selected feature has zero scale after filtering, `n_components` is too large, or
the output path exists without overwrite permission.

See [decision 002](../decisions/002_pca_feature_matrix.md).
