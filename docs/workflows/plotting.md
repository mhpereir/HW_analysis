# Workflow: Plotting and Shared Style

## Purpose

All active figures share one visual contract through `src/plot_style.py`.
Plotting scripts consume prepared Stage 1, Stage 2, composite, or spatial
products. They do not own ingestion, event detection, or reusable scientific
reductions.

## Shared style authority

`src/plot_style.py` owns:

- publication widths and aspect defaults;
- paper and legend font sizes;
- line, reference-line, and scatter styles;
- the 300 DPI export default;
- variable display names and colors;
- diagnostic and boundary-face colors;
- time-axis and numeric-axis formatting;
- standard axis, legend, zero-line, and layout helpers; and
- `save_figure()` for numeric formatting and consistent export.

Add a reusable visual choice there rather than defining a new palette, label
mapping, font configuration, or export convention in one script.

`src/plotting.py` is the shared renderer for Stage-1 temporal composites and
top-event figures. Its callers therefore use the shared style indirectly even
when they do not import `plot_style` themselves.

`src/plot_paths.py` owns structured default output directories for Stage-1
plots:

```text
results/plots_<plot_name>/
  region_<region>/
  boundary_<bottom>_<top>/
  time_range_<start>_<end>/
```

## Active figure families

| Input | Figure family | Entrypoints |
| --- | --- | --- |
| Stage 1 | all-event and split temporal composites | `scripts/plot_composite_timeseries_all.py`, `scripts/plot_composite_timeseries_split.py` |
| Stage 1 plus regional hourly climatology | all-event and split climatological-anomaly composites | `scripts/plot_composite_timeseries_all_clim_anom.py`, `scripts/plot_composite_timeseries_split_clim_anom.py` |
| Stage 1 plus regional hourly climatology | face-resolved advection climatological anomalies | `scripts/plot_advection_direction_exploration_clim_anom.py` |
| Stage 1 plus regional hourly climatology, Stage 2 event features, and matching settings | matched face-resolved advection climatological anomalies | `scripts/plot_advection_direction_exploration_matched_clim_anom.py` |
| Stage 1 | top-event traces | `scripts/plot_top_events.py` |
| Stage 1 | diurnal, threshold, and event-summary diagnostics | `scripts/plot_diurnal_cycle.py`, `scripts/plot_threshold_timeseries.py`, `scripts/plot_event_summary.py` |
| Stage 1 run inventory | Northern Hemisphere regional-domain overview | `scripts/region_vis/plot_stage1_regions.py` |
| Stage 2 event features | feature grids, splits, and combined comparisons | `event_feature_grid_plot.py`, `plot_event_feature.py`, `plot_event_feature_split.py`, `plot_event_feature_split_combined.py` under `scripts/event_features/` |
| Stage 2 event features | adiabatic, advection, and diabatic event diagnostics | `plot_adiabatic_advection_comparison.py`, `plot_adiabatic_diabatic_advection.py` under `scripts/event_features/` |
| Stage 2 event and baseline features | event-versus-baseline comparisons | `plot_adiabatic_advection_comparison_baseline.py`, `plot_adiabatic_diabatic_advection_baseline.py` under `scripts/event_features/` |
| Stage 2 event features plus tracked settings | matched positive/negative `I_dyn` diagnostics | `scripts/Idyn_matching_exploration/explore_idyn_matching.py` |
| Spatial composite product | sign-by-lag T2m/Z500 maps | `scripts/spatial_composites/plot_dyn_net_spatial_composites.py` |
| Matched spatial composite product | matched positive/negative `I_dyn_pre` sign-by-lag T2m/Z500 maps | `scripts/spatial_composites/plot_matched_dyn_pre_spatial_composites.py` |

Scripts under `scripts/event_features/old/` are legacy and are not active
figure entrypoints.

## Required behavior

- Use the non-interactive Matplotlib `Agg` backend for batch rendering.
- The regional-domain overview discovers unique regions from the Stage 1
  products in a supplied run directory, verifies their product marker and
  stored bounds against `src/config.py`, and draws unfilled, distinctly
  colored domain boundaries on one Northern Hemisphere map. It is a read-only
  inventory diagnostic: it creates no analysis product and does not change the
  Stage 1 contract or existing consumers. The default run is
  `bf232281_20260819`, but both input and output paths remain configurable.
  Because this diagnostic is intentionally self-contained, its Venus OpenPBS
  entrypoint is co-located under `scripts/region_vis/`.
- Open Stage 1 through `src.analysis_io.open_harmonized_timeseries()` when its
  product contract is required.
- Keep analysis and selection logic in reusable modules, not in visual styling
  code.
- Matching-aware figures must load a tracked static settings file and call the
  reusable matching implementation in `src/selectors.py`. They may recompute
  lightweight event indices in memory, but must not carry a private assignment
  implementation or mutate the Stage-2 input.
- The matched face-advection figure must leave the existing all-event figure
  unchanged. It applies a named settings specification to the Stage-2 event
  table, maps the returned event IDs to the authoritative Stage-1 event table,
  and validates matching Stage-1 and Stage-2 peak timestamps before building
  composites. Component identity remains color encoded. Positive `I_dyn_pre`
  uses solid lines and negative `I_dyn_pre` uses dashed lines.
- The matched spatial figure must consume a separate matched spatial-composite
  product built from the canonical Stage-2 table and tracked settings. It must
  not infer matched membership from an already averaged all-event spatial
  product. Its title and row labels identify `I_dyn_pre`, the named matching
  specification, its pooled-SD caliper, and the equal pair count. The existing
  all-event spatial product and figure remain unchanged.
- The production matching exploration runs through
  `schedulers/schedule_explore_idyn_matching.sh`. It publishes the four
  README-linked PNGs and a `matching_summary.json` containing the input and
  settings checksums plus the numerical diagnostics used to refresh the
  exploratory Markdown.
- Event, baseline, matching, and spatial-composite consumers must read the
  canonical Stage-2 `I_dyn_pre` variable. They must not reconstruct it from
  `I_adiabatic_pre` and `I_advection_pre`.
- Use `plot_style.VARIABLE_NAME_MAPPING` and shared color dictionaries for
  existing variables.
- Use `plot_style.publication_figsize()` and shared line-width constants.
- Use `plot_style.style_axis()` or `style_axes()` where appropriate.
- Use `plot_style.legend_kwargs()` for standard legends.
- Use shared numeric-axis helpers for reusable tick-spacing and formatting
  rules, including integer-only axes.
- Use `plot_style.save_figure()` for ordinary Matplotlib figures.
- Map exports may use a dedicated writer when Cartopy layout requires it, but
  must still use shared theme, size, and DPI settings.
- Add or update synthetic plot tests when changing visual semantics or the
  shared style API.
- Climatological-anomaly figures must identify the representation and baseline
  period in their title or metadata, include a zero reference where
  appropriate, and never overwrite their absolute-value counterparts.

### Extended temporal-composite layout

Extended all-event, split-event, climatological-anomaly, and top-event figures
use a 5x2 panel grid. The left column contains temperature and volume, `dTdt`,
advection, adiabatic heating, and diabatic heating. The right column contains:

1. anticyclonic and cyclonic LWA;
2. cloud cover;
3. longwave and shortwave radiative heating;
4. sensible and latent surface heating; and
5. soil moisture.

Cloud cover and soil moisture use separate axes. Absolute cloud cover is
bounded to the physical fraction range from zero to one, while a
climatological anomaly is not. PBL diagnostics and panels are inactive under
[decision 008](../decisions/008_retire_pbl_diagnostics.md).

### Climatological-anomaly composite labels and surface-flux signs

The all-event and split-event climatological-anomaly composite figures use the
symbol `Δ` in panel-axis labels instead of spelling out `anomaly`. Place `Δ`
immediately before the plotted variable name when the axis identifies a
variable, for example `ΔT_mean [K]` or `ΔLWA [m hPa]`. For a shared or
units-only tendency axis, place the symbol immediately before the units, for
example `Δ [K hr-1]`. Figure titles continue to identify the
climatological-anomaly representation explicitly.

ERA5 sensible and latent surface heat fluxes use the source convention that
positive values are directed toward the surface. In the all-event and
split-event climatological-anomaly composite figures, render
`sshf_heating_rate_approx` and `slhf_heating_rate_approx` with the opposite
sign, so positive plotted anomalies denote heat transfer into the atmosphere
and can be compared directly with the atmospheric diabatic-heating sign. Apply
the same sign reversal to the composite mean and its event-percentile bounds.

This is a display transform only. It must not mutate Stage 1, the climatology
companion, the assembled composite dataset, absolute-value figures, or
top-event figures. Synthetic plotting tests must verify the `Δ` labels, the
reversed anomaly traces and bounds, and retention of native signs in absolute
composites.

## Validation

Run the plotting and spatial plot tests locally:

```bash
mamba activate dev_env
python -m pytest -q \
  tests/test_plotting.py \
  tests/test_plot_*.py \
  tests/test_region_vis.py \
  tests/test_spatial_composite_plot.py
```

For a production figure change, also render a representative product through
PBS on Venus and inspect the saved artifact, labels, units, panel ordering,
legibility, and output path.
