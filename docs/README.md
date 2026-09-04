# HW_analysis Documentation

This is the starting point for understanding or changing `HW_analysis`.

## Authority and reading order

The documentation is the stable source of truth for intended behavior,
architecture, scientific contracts, and compatibility. Tests encode those
contracts as executable checks, and the code implements them.

When code or tests disagree with the documentation, preserve the documented
behavior by default and determine whether the implementation has regressed. An
intentional contract change must be planned and recorded in the relevant
canonical documents before implementation begins. Update the implementation
and tests to conform to the revised contract.

Read in this order:

1. [Pipeline overview](pipeline_overview.md) for the active architecture,
   product flow, and module boundaries.
2. The relevant contract under [products](products/) before changing Stage 1
   or Stage 2 data.
3. The relevant procedure under [workflows](workflows/) before changing
   composites or plotting.
4. The relevant record under [decisions](decisions/) before changing a
   scientific default.
5. [Legacy documentation](legacy/README.md) only when investigating historical
   Stage 3 or Stage 4 behavior.

`AGENTS.md` defines execution, validation, and contribution constraints. It
routes to these documents rather than defining a separate pipeline contract.

## Documentation-first change process

Before implementing new development:

1. Identify the affected product contracts, workflows, scientific decisions,
   and downstream consumers.
2. Record the proposed behavior in the relevant canonical documents. Add a
   decision record when a scientific default, product meaning, or architectural
   boundary changes.
3. State how existing products, scripts, figures, and production workflows
   remain compatible. Document any required migration explicitly.
4. Define the validation needed to demonstrate conformance.
5. Implement the documented plan, update or add tests, and verify that the
   documentation, tests, and code agree.

Do not allow implementation details to establish a new project contract
implicitly. Exploratory prototypes may precede a final design, but they must
remain isolated from the active production pipeline until their intended
behavior and compatibility requirements are documented.

## Active system at a glance

The repository has two active data paths.

### Regional analysis products

```text
pre-calculated EHB + regional ERA5-family inputs + thresholds + LWA
  -> Stage 1 harmonized regional time series
  -> Stage 2 event-feature and baseline-day tables
  -> temporal composites, event diagnostics, and feature comparisons
```

Stage 1 is the reusable hourly regional handoff. Stage 2 contains event-level
and baseline-day fixed-window features derived from Stage 1.

### Spatial composite products

```text
raw hourly ERA5 T2m and Z500
  -> annual daily T2m/Z500 files
  -> 366-day daily climatology

Stage 2 event features + annual daily fields + daily climatology
  |-> lagged all-event dynamical-sign spatial composite dataset
  |   `-> publication map figure
  `-> tracked I_dyn_pre matching settings
      -> lagged matched dynamical-sign spatial composite dataset
      -> matched-population publication map figure
```

The spatial path is an active workflow with its own durable intermediate and
composite datasets. It does not load spatial fields into Stage 1 or Stage 2.
See [Spatial composites](workflows/spatial_composites.md).

## Shared plotting layer

All active figures use `src/plot_style.py`, directly or through
`src/plotting.py`. Shared names, colors, sizes, fonts, line widths, numeric-axis
formatting, legends, and figure export belong in that module. Structured
Stage-1 plot output paths belong in `src/plot_paths.py`.

See [Plotting and shared style](workflows/plotting.md) before adding or changing
a figure.

## Documentation map

| Area | Canonical document |
| --- | --- |
| Active architecture and ownership | [Pipeline overview](pipeline_overview.md) |
| Stage 1 regional dataset | [Stage 1 product contract](products/stage1_harmonized_timeseries.md) |
| Stage 1 regional hourly climatology | [Regional hourly climatology contract](products/stage1_regional_hourly_climatology.md) |
| Stage 2 event features | [Event-feature contract](products/stage2_event_features.md) |
| Stage 2 baseline features | [Baseline-day contract](products/stage2_baseline_features.md) |
| Temporal composites | [Composite workflow](workflows/composites.md) |
| Spatial ERA5 preparation and composites | [Spatial composite workflow](workflows/spatial_composites.md) |
| Figure conventions and shared style | [Plotting workflow](workflows/plotting.md) |
| Scientific defaults | [Decision records](decisions/) |
| Inactive PCA and clustering material | [Legacy index](legacy/README.md) |

## Active entrypoint families

| Purpose | Entrypoints |
| --- | --- |
| Build Stage 1 | `scripts/build_stage1_harmonized_timeseries.py` |
| Build Stage 2 | `scripts/event_features/build_stage2_event_features.py`, `scripts/event_features/build_stage2_baseline_features.py` |
| Plot Stage 1 | `scripts/plot_*.py` |
| Plot Stage 1 region inventory | `scripts/region_vis/plot_stage1_regions.py` |
| Plot matched Stage-1 composites | `scripts/plot_advection_direction_exploration_matched_clim_anom.py` |
| Plot top-event face-advection values | `scripts/plot_advection_direction_exploration_top_events.py` |
| Plot top-event face-advection anomalies | `scripts/plot_advection_direction_exploration_top_events_clim_anom.py` |
| Plot Stage 2 | active plotting scripts in `scripts/event_features/` |
| Prepare spatial ERA5 | `scripts/spatial_composites/build_era5_daily_spatial_data.sh`, `scripts/spatial_composites/build_era5_daily_doy_climatology.sh` |
| Build and plot spatial composites | `scripts/spatial_composites/build_dyn_net_spatial_composites.py`, `scripts/spatial_composites/plot_dyn_net_spatial_composites.py` |
| Build and plot matched spatial composites | `scripts/spatial_composites/build_matched_dyn_pre_spatial_composites.py`, `scripts/spatial_composites/plot_matched_dyn_pre_spatial_composites.py` |
| Run on Venus | matching scripts under `schedulers/`; the self-contained region-inventory diagnostic keeps its scheduler beside the plotter under `scripts/region_vis/` |

Production inputs and results exist on Venus and remain outside Git. Local
`dev_env` is for tests, compilation checks, and small synthetic prototypes.
Production data work must use the tracked OpenPBS entrypoints.
