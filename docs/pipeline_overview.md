# HW Analysis Pipeline Overview

This is the canonical architecture and index document for the heatwave analysis
pipeline. It should stay short and stable. Detailed variable contracts live in
`docs/products/`, script-oriented procedures live in `docs/workflows/`, and
changeable analysis choices live in `docs/decisions/`.

## Terminology

| Term | Meaning |
| --- | --- |
| Product stage | A durable saved artifact with a documented data contract. |
| Workflow | A procedure or script that consumes one or more products. |
| Diagnostic | A figure or table used to inspect a product or workflow result. |

## Design Principles

1. **Dataset-first rather than plot-first**: build reusable analysis products
   before making figures.
2. **Regional time series as the primary handoff**: the Stage-1 product is the
   central harmonized dataset consumed by downstream workflows.
3. **Explicit timestep handling**: native timestep and analysis timestep must be
   documented separately, with daily event IDs projected onto hourly diagnostics
   where needed.
4. **Reusable selector and event logic**: thresholds, masks, event IDs, duration
   filters, peaks, and event summaries should be built once and reused.
5. **Separation of concerns**: raw loading, harmonization, event features, PCA,
   clustering, composites, diagnostics, and plotting have distinct roles.
6. **Plotting consumes products**: plotting scripts should not reload raw data,
   rebuild event IDs, or hide analysis logic.

## Product-stage Map

```text
raw inputs
  ->
Product Stage 1: harmonized regional time series
  |-> Product Stage 2: baseline-day feature table
  `-> Product Stage 2: event-feature table
        ->
      Product Stage 3: event-feature PCA product
        ->
      Product Stage 4: event-feature cluster product
        ->
      cluster composites / PCA diagnostics / interpretation figures
```

| Product stage | Durable artifact | Producer | Main consumers |
| --- | --- | --- | --- |
| Stage 1 | `results/stage1/harmonized_regional_timeseries_*.nc` | `scripts/build_stage1_harmonized_timeseries.py` | event features, baseline features, composites, top-event plots |
| Stage 2 | event-feature table | `scripts/event_features/build_stage2_event_features.py` | PCA, feature plots, exploratory diagnostics |
| Stage 2 | baseline-day feature table | `scripts/event_features/build_stage2_baseline_features.py` | event/baseline comparisons, exploratory diagnostics |
| Stage 3 | event-feature PCA product | `scripts/event_features/build_stage3_event_feature_pca.py` | PCA diagnostics, clustering |
| Stage 4 | event-feature cluster product | `scripts/event_features/build_stage4_event_feature_clusters.py` | cluster interpretation, cluster composites |

Stage-4 cluster labels are method- and feature-dependent derived products; they should be interpreted through PCA loadings, event metadata, and cluster-conditioned composites rather than treated as physical mechanisms by default.

## Module Responsibility Map

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Paths, region definitions, seasons, source constants, and default settings. |
| `src/data_io.py` | Open raw/source datasets and handle source-specific file conventions. |
| `src/preprocess.py` | Low-level time, coordinate, unit, averaging, anomaly, and resampling utilities. |
| `src/harmonize.py` | Align sources into the Stage-1 regional time-series product. |
| `src/analysis_io.py` | Save/open internal products, especially the Stage-1 handoff, and validate metadata. |
| src/selectors.py | Filter event tables and build reusable event/time selection masks. |
| `src/events.py` | Convert masks into event IDs, peaks, durations, ranks, and event summaries. |
| `scripts/event_features/fixed_window_features.py` | Shared fixed-window reductions for event and baseline Stage-2 products. |
| `src/composites.py` | Build event-centered extracts, means, spreads, and top-event products. |
| `src/diagnostics.py` | Domain-specific derived diagnostics such as residual checks and heating-rate approximations. |
| `src/plotting.py` | Plot prepared products without raw loading or event generation. |

## File And Directory Conventions

```text
HW_analysis/
|-- docs/
|   |-- pipeline_overview.md
|   |-- products/
|   |-- workflows/
|   `-- decisions/
scripts/
|-- build_stage1_harmonized_timeseries.py
`-- event_features/
    |-- build_stage2_event_features.py
    |-- build_stage2_baseline_features.py
    |-- build_stage3_event_feature_pca.py
    |-- build_stage4_event_feature_clusters.py
    |-- plot_event_feature.py
    |-- plot_pca_vector_loadings.py
    `-- event_feature_grid_plot.py
|-- src/
|-- tests/
results/
|-- stage1/
|-- stage2_event_features/
|-- stage2_baseline_features/
|-- stage3_event_feature_pca/
`-- stage4_event_feature_clusters/
```

Product filenames should encode enough run context to distinguish region,
threshold variable, quantile, years, and pressure boundaries when applicable.
Scripts may have defaults, but product contracts should describe the required
dataset contents rather than one exact run filename.

## Product Specs

- [Stage 1: harmonized regional time series](products/stage1_harmonized_timeseries.md)
- [Stage 2: event features](products/stage2_event_features.md)
- [Stage 2: baseline-day features](products/stage2_baseline_features.md)
- [Stage 3: event-feature PCA](products/stage3_event_feature_pca.md)
- [Stage 4: event-feature clusters](products/stage4_event_feature_clusters.md)

## Workflow Docs

- [Composites](workflows/composites.md)
- [PCA diagnostics](workflows/pca_diagnostics.md)
- [Cluster interpretation](workflows/cluster_interpretation.md)

Diagnostic and plotting scripts are workflow consumers, so their names describe
the diagnostic they make rather than a product stage they produce.

## Decision Records

- [001: event-feature windows](decisions/001_event_feature_windows.md)
- [002: PCA feature matrix](decisions/002_pca_feature_matrix.md)
- [003: clustering strategy](decisions/003_clustering_strategy.md)
- [004: baseline season and window boundaries](decisions/004_baseline_season_windows.md)
- [005: Stage-1 event peak semantics](decisions/005_stage1_event_peak_semantics.md)
