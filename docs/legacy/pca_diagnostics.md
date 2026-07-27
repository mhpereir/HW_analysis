# Workflow: PCA Diagnostics

## Purpose

PCA diagnostic workflows inspect the Stage-3 PCA product. They explain and
stress-test the PCA transform; they do not create a new product stage.

## Primary Input

- Stage-3 event-feature PCA product

## Current Scripts

```text
scripts/event_features/plot_pca_vector_loadings.py
scripts/event_features/plot_event_feature.py
scripts/event_features/event_feature_grid_plot.py
```

## Expected Diagnostics

- scree and cumulative explained-variance plots
- loading heatmaps or vector plots
- PC score scatter plots
- correlations between PC scores and copied event metadata or diagnostics
- feature-space plots for selected raw or derived event features

## Required Product Variables

Diagnostics should rely on documented PCA variables such as:

```text
pc_score(event, pc)
pc_loading(pc, feature)
explained_variance_ratio(pc)
cumulative_explained_variance_ratio(pc)
feature_matrix(event, feature)
feature_matrix_scaled(event, feature)
```

## Boundaries

Diagnostic scripts should not refit PCA unless explicitly running a sensitivity
workflow with a separate output path. They should not assign clusters.
