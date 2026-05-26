# Workflow: Cluster Interpretation

## Purpose

Cluster interpretation workflows consume the Stage-4 cluster product, and often
the Stage-1 time series, to understand whether clusters map onto distinct
physical pathways or event severities.

## Primary Inputs

- Stage-4 event-feature cluster product
- Stage-3 PCA product when deeper loading or score interpretation is needed
- Stage-1 harmonized regional time series for event-centered composites

## Expected Diagnostics

- cluster counts and tracked-variable summaries
- PC score scatter plots colored by `cluster_label`
- tracked-variable distributions by cluster
- cluster-conditioned event composites
- representative events from each cluster
- comparison of cluster labels against event severity metadata such as
  `tas_anom_peak` and `tas_excess_integral`

## Boundaries

Interpretation workflows should treat cluster labels as method-dependent
diagnostics. They should not modify the source PCA product, and they should not
treat PCA axes or clusters as physical mechanisms without checking loadings,
event metadata, and composites.
