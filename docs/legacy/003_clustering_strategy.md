# Decision 003: Clustering Strategy

## Status

Accepted current default.

## Decision

Event clustering consumes Stage-3 `pc_score(event, pc)` and writes a separate
Stage-4 cluster product. Current defaults are:

```text
methods      = ward, kmeans, gmm
pcs          = PC1, PC2, PC3
n_clusters   = 3
random_state = 0
```

Tracked-variable summaries are written for selected PC scores, event features,
derived heat-budget fractions, LWA exposure, antecedent temperature, duration,
and event-severity metadata.

## Rationale

Keeping clustering downstream of PCA keeps the PCA transform reusable and
method-neutral. Writing a separate cluster product makes it possible to compare
methods, PC subsets, and cluster counts without overwriting the Stage-3 product.

## Consequences

- Cluster labels are method-dependent diagnostics, not intrinsic event types.
- GMM writes `cluster_probability(event, cluster)`; Ward and K-means do not.
- Validation metrics should be recorded when valid for the produced labels:
  `silhouette_score`, `davies_bouldin_score`, and `calinski_harabasz_score`.
- Cluster interpretation should use loadings, score plots, event metadata, and
  composites before assigning physical meaning.
