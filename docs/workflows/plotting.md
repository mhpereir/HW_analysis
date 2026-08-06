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
| Stage 1 | top-event traces | `scripts/plot_top_events.py` |
| Stage 1 | diurnal, threshold, and event-summary diagnostics | `scripts/plot_diurnal_cycle.py`, `scripts/plot_threshold_timeseries.py`, `scripts/plot_event_summary.py` |
| Stage 2 event features | feature grids, splits, and combined comparisons | `event_feature_grid_plot.py`, `plot_event_feature.py`, `plot_event_feature_split.py`, `plot_event_feature_split_combined.py` under `scripts/event_features/` |
| Stage 2 event features | adiabatic, advection, and diabatic event diagnostics | `plot_adiabatic_advection_comparison.py`, `plot_adiabatic_diabatic_advection.py` under `scripts/event_features/` |
| Stage 2 event and baseline features | event-versus-baseline comparisons | `plot_adiabatic_advection_comparison_baseline.py`, `plot_adiabatic_diabatic_advection_baseline.py` under `scripts/event_features/` |
| Spatial composite product | sign-by-lag T2m/Z500 maps | `scripts/spatial_composites/plot_dyn_net_spatial_composites.py` |

Scripts under `scripts/event_features/old/` are legacy and are not active
figure entrypoints.

## Required behavior

- Use the non-interactive Matplotlib `Agg` backend for batch rendering.
- Open Stage 1 through `src.analysis_io.open_harmonized_timeseries()` when its
  product contract is required.
- Keep analysis and selection logic in reusable modules, not in visual styling
  code.
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

## Validation

Run the plotting and spatial plot tests locally:

```bash
mamba activate dev_env
python -m pytest -q \
  tests/test_plotting.py \
  tests/test_plot_*.py \
  tests/test_spatial_composite_plot.py
```

For a production figure change, also render a representative product through
PBS on Venus and inspect the saved artifact, labels, units, panel ordering,
legibility, and output path.
