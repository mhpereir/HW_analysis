# Product Stage 4: Event-feature Clusters

## Contract

Stage 4 assigns event clusters in PCA score space and writes a derived cluster
product. The Stage-3 PCA source file must remain reusable and method-neutral.

Typical outputs:

```text
results/stage4_event_feature_clusters/hw_event_feature_clusters_<run>_<method>_k<n>_<pcs>.nc
```

The current implementation copies the PCA dataset into a derived output file and
adds cluster variables and summaries. This is acceptable because the source PCA
file is not modified destructively.

Older generated artifacts may still exist under `results/event_features/`, but
new defaults use the stage-specific directory above.

## Producer

```text
scripts/event_features/build_stage4_event_feature_clusters.py
```

Diagnostic scripts that consume this product are workflow consumers and are not
named as product-stage builders.

## Consumes

- Stage-3 event-feature PCA product
- `pc_score(event, pc)` for the selected PCs

## Dimensions

| Dimension | Meaning |
| --- | --- |
| `event` | Events retained in the PCA product. |
| `cluster` | Zero-based cluster IDs. |
| `tracked_variable` | Event-level variables retained for interpretation summaries. |

The product may also retain the Stage-3 `pc`, `feature`, and `event_original`
dimensions because the current implementation writes a PCA-derived dataset copy.

## Core Variables

```text
cluster_label(event)
cluster_probability(event, cluster)       # GMM only
tracked_variable_value(event, tracked_variable)
cluster_count(cluster)
cluster_variable_mean(cluster, tracked_variable)
cluster_variable_median(cluster, tracked_variable)
cluster_variable_std(cluster, tracked_variable)
cluster_variable_min(cluster, tracked_variable)
cluster_variable_max(cluster, tracked_variable)
cluster_variable_n_finite(cluster, tracked_variable)
```

## Required Metadata

Global attrs should include:

```text
pipeline_stage = "stage_4_event_feature_clusters"
source_pca_path = ...
cluster_method = "ward" | "kmeans" | "gmm"
cluster_n = integer
cluster_pcs = comma-separated PC names
cluster_tracked_variables = comma-separated variable names
random_state = integer
clustering_performed = 1
```

When valid for the label distribution, validation metrics should include:

```text
silhouette_score
davies_bouldin_score
calinski_harabasz_score
```

## Supported Methods

The current supported methods are:

```text
ward
kmeans
gmm
```

GMM writes posterior probabilities. Ward and K-means write deterministic labels
for the chosen inputs and method settings.

## Validation Expectations

The producer should fail clearly when the PCA product lacks `event`, `pc`, or
`pc_score`, requested PCs are absent, selected scores contain non-finite values,
`n_clusters < 2`, `n_clusters` exceeds the number of events, tracked variables
cannot be resolved, or an output exists without overwrite permission.

See [decision 003](../decisions/003_clustering_strategy.md).
