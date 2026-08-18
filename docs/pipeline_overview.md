# HW Analysis Pipeline Overview

This is the canonical architecture and index document for the heatwave analysis
pipeline. It should stay short and stable. Detailed variable contracts live in
`docs/products/`, script-oriented procedures live in `docs/workflows/`, and
changeable analysis choices live in `docs/decisions/`.

Start at [docs/README.md](README.md) for the documentation reading order and
authority rules.

## Terminology

| Term | Meaning |
| --- | --- |
| Product stage | A durable saved artifact with a documented data contract. |
| Workflow | A procedure or script that consumes one or more products. |
| Diagnostic | A figure or table used to inspect a product or workflow result. |

## Design Principles

1. **Documentation-first compatibility**: record intended behavior,
   compatibility requirements, and validation before implementation. Code and
   tests must conform to the documented contract.
2. **Dataset-first rather than plot-first**: build reusable analysis products
   before making figures.
3. **Explicit reusable handoffs**: Stage 1 is the central regional handoff;
   spatial workflows use separately documented daily ERA5 and composite
   products.
4. **Explicit timestep handling**: native timestep and analysis timestep must be
   documented separately, with daily event IDs projected onto hourly diagnostics
   where needed.
5. **Reusable selector and event logic**: thresholds, masks, event IDs, duration
   filters, peaks, and event summaries should be built once and reused.
6. **Separation of concerns**: raw loading, harmonization, event features,
   composites, diagnostics, and plotting have distinct roles.
7. **Plotting consumes products**: plotting scripts should not reload raw data,
   rebuild event IDs, or hide reusable analysis logic.
8. **One shared figure contract**: active figures use `src/plot_style.py`
   directly or through `src/plotting.py`.
9. **Lightweight configured selections may be recomputed**: a deterministic
   event-table selector may run in memory for each figure when its settings are
   tracked, its logic is reusable under `src/`, and no new physical data are
   derived.

## Active Data-flow Map

### Regional products

```text
pre-calculated EHB + regional ERA5-family inputs + thresholds + LWA
  ->
Product Stage 1: harmonized regional time series
  |-> Stage 1 companion: regional calendar-hour climatology
  |-> Product Stage 2: baseline-day feature table
  `-> Product Stage 2: event-feature table
        |-> feature and event/baseline diagnostics
        `-> dynamical-sign spatial composites

Product Stage 1
  `-> temporal composites, top events, threshold, diurnal, and event summaries

Product Stage 2: event-feature table + tracked matching settings
  |-> in-memory I_dyn_pre sign matching -> matched-population diagnostics
  `-> matched event IDs + Stage 1 + regional hourly climatology
        -> matched face-advection climatological-anomaly composite figure
```

| Product stage | Durable artifact | Producer | Main consumers |
| --- | --- | --- | --- |
| Stage 1 | `results/stage1/harmonized_regional_timeseries_*.nc` | `scripts/build_stage1_harmonized_timeseries.py` | event features, baseline features, composites, top-event plots |
| Stage 1 companion | `results/stage1_climatology/regional_hourly_climatology_*.nc` | `scripts/build_stage1_hourly_climatology.py` | climatological-anomaly composites and face-advection diagnostics |
| Stage 2 | event-feature table | `scripts/event_features/build_stage2_event_features.py` | feature plots, event comparisons, exploratory diagnostics |
| Stage 2 | baseline-day feature table | `scripts/event_features/build_stage2_baseline_features.py` | event/baseline comparisons, exploratory diagnostics |

### Spatial products

```text
raw hourly ERA5 T2m + Z500
  -> annual daily T2m/Z500
  -> 366-day T2m/Z500 climatology

annual daily fields + climatology + Stage 2 event features with I_dyn_pre
  |-> lagged all-event positive/negative I_dyn_pre spatial composites
  |   `-> all-event sign-by-lag publication map
  `-> tracked matching settings + in-memory peak_anomaly_0p20 selection
      -> lagged matched positive/negative I_dyn_pre spatial composites
      -> matched sign-by-lag publication map
```

| Durable artifact | Producer | Main consumer |
| --- | --- | --- |
| `results/spatial_composites/daily/ERA5_daily_t2m_z500_<year>.nc` | `scripts/spatial_composites/build_era5_daily_spatial_data.sh` | climatology and composite builders |
| `results/spatial_composites/climatology/era5_daily_doy_climatology_*.nc` | `scripts/spatial_composites/build_era5_daily_doy_climatology.sh` | spatial composite builder |
| `results/spatial_composites/dyn_net_daily_spatial_composites_*.nc` | `scripts/spatial_composites/build_dyn_net_spatial_composites.py` | spatial composite plotter |
| `results/spatial_composites/matched_dyn_pre_daily_spatial_composites_*.nc` | `scripts/spatial_composites/build_matched_dyn_pre_spatial_composites.py` | matched spatial composite plotter |

Stages 3 and 4 are inactive legacy workflows. Their PCA and clustering
implementations are retained under `scripts/event_features/old/` for historical
reference and possible future reactivation, but they are not active pipeline
products or plotting dependencies.

## Module Responsibility Map

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Paths, region definitions, seasons, source constants, and default settings. |
| `src/data_io.py` | Open raw/source datasets and handle source-specific file conventions. |
| `src/preprocess.py` | Low-level time, coordinate, unit, averaging, anomaly, and resampling utilities. |
| `src/harmonize.py` | Align sources into the Stage-1 regional time-series product. |
| `src/analysis_io.py` | Save/open internal products, especially the Stage-1 handoff, and validate metadata. |
| `src/selectors.py` | Filter event tables, build reusable event/time masks, and perform deterministic standardized event matching. |
| `src/events.py` | Convert masks into event IDs, peaks, durations, ranks, and event summaries. |
| `scripts/event_features/fixed_window_features.py` | Shared fixed-window reductions for event and baseline Stage-2 products. |
| `src/composites.py` | Build event-centered extracts, means, spreads, and top-event products. |
| `src/climatology.py` | Build regional calendar-hour climatologies and apply timestamp-matched anomalies. |
| `src/diagnostics.py` | Domain-specific derived diagnostics such as residual checks and heating-rate approximations. |
| `src/plotting.py` | Plot prepared products without raw loading or event generation. |
| `src/plot_style.py` | Shared names, colors, dimensions, axis formatting, legends, and figure export. |
| `src/plot_paths.py` | Structured default output paths for Stage-1-based figures. |
| `scripts/spatial_composites/` | Prepare daily ERA5 fields, build lagged spatial products, and render maps. |

## File And Directory Conventions

```text
HW_analysis/
|-- docs/
|   |-- README.md
|   |-- pipeline_overview.md
|   |-- products/
|   |-- workflows/
|   |-- decisions/
|   `-- legacy/
|-- scripts/
|   |-- build_stage1_harmonized_timeseries.py
|   |-- build_stage1_hourly_climatology.py
|   |-- plot_composite_timeseries_all_clim_anom.py
|   |-- plot_composite_timeseries_split_clim_anom.py
|   |-- plot_advection_direction_exploration_clim_anom.py
|   |-- plot_advection_direction_exploration_matched_clim_anom.py
|   |-- event_features/
|   |   |-- build_stage2_event_features.py
|   |   |-- build_stage2_baseline_features.py
|   |   |-- plot_event_feature.py
|   |   |-- event_feature_grid_plot.py
|   |   `-- old/
|   `-- spatial_composites/
|       |-- build_era5_daily_spatial_data.sh
|       |-- build_era5_daily_doy_climatology.sh
|       |-- build_dyn_net_spatial_composites.py
|       |-- plot_dyn_net_spatial_composites.py
|       |-- build_matched_dyn_pre_spatial_composites.py
|       `-- plot_matched_dyn_pre_spatial_composites.py
|-- src/
|-- tests/
`-- results/
    |-- stage1/
    |-- stage1_climatology/
    |-- stage2_event_features/
    |-- stage2_baseline_features/
    `-- spatial_composites/
```

Product filenames should encode enough run context to distinguish region,
threshold variable, quantile, years, and pressure boundaries when applicable.
Scripts may have defaults, but product contracts should describe the required
dataset contents rather than one exact run filename.

## Product Specs

- [Stage 1: harmonized regional time series](products/stage1_harmonized_timeseries.md)
- [Stage 1 companion: regional hourly climatology](products/stage1_regional_hourly_climatology.md)
- [Stage 2: event features](products/stage2_event_features.md)
- [Stage 2: baseline-day features](products/stage2_baseline_features.md)

The Stage 3 and Stage 4 product documents are retained under
[legacy](legacy/README.md) as design records, not active product
specifications.

## Workflow Docs

- [Composites](workflows/composites.md)
- [Spatial composites](workflows/spatial_composites.md)
- [Plotting and shared style](workflows/plotting.md)

Diagnostic and plotting scripts are workflow consumers, so their names describe
the diagnostic they make rather than a product stage they produce.

## Decision Records

- [001: event-feature windows](decisions/001_event_feature_windows.md)
- [004: baseline season and window boundaries](decisions/004_baseline_season_windows.md)
- [005: Stage-1 event peak semantics](decisions/005_stage1_event_peak_semantics.md)
- [006: regional hourly climatology baseline](decisions/006_regional_hourly_climatology.md)
- [007: in-memory I_dyn sign matching](decisions/007_idyn_sign_matching.md)
- [008: retire PBL diagnostics from active products](decisions/008_retire_pbl_diagnostics.md)

PCA and clustering decisions are indexed under
[legacy documentation](legacy/README.md).
