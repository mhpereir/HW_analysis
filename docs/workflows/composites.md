# Workflow: Composites

## Purpose

Composite workflows consume prepared products to summarize event-centered
trajectories. They are workflows and diagnostics, not product stages.

## Primary Inputs

- Stage-1 harmonized regional time series
- event definitions and event-summary variables already present in Stage 1

## Current Scripts

```text
scripts/plot_composite_timeseries_all.py
scripts/plot_composite_timeseries_split.py
scripts/plot_top_events.py
scripts/plot_composite_timeseries_all_clim_anom.py
scripts/plot_composite_timeseries_split_clim_anom.py
scripts/plot_advection_direction_exploration_clim_anom.py
```

## Expected Behavior

- Open the Stage-1 product through the analysis-product IO layer when validation
  is needed.
- Use reusable event/composite helpers rather than rebuilding event IDs in the
  plotting layer.
- Align event-centered extracts on the documented event peak time.
- Render prepared composite data into figures without raw source loading.

Climatological-anomaly entrypoints additionally consume the documented
regional hourly climatology companion. They must match and subtract the
climatology at every source timestamp before extracting event windows. Event
means and event-percentile envelopes are then calculated from the anomalized
event samples. Subtracting a climatology only after the event reduction is not
permitted because it changes percentile semantics.

Event IDs, event tables, selection variables, and peak timestamps always come
from the absolute Stage-1 product. Climatology subtraction does not redefine
events. Absolute and climatological-anomaly figures use separate entrypoints
and output paths.

The Venus scheduler wrappers are likewise one operation per PBS job:

```text
schedulers/schedule_plot_composite_timeseries_all_clim_anom.sh
schedulers/schedule_plot_composite_timeseries_split_clim_anom.sh
schedulers/schedule_plot_advection_direction_exploration_clim_anom.sh
```

They require explicit Stage-1, climatology, and output paths plus a verified
runtime commit. They do not build prerequisites inside plotting jobs.

## Outputs

Outputs are figures and diagnostic tables under `results/plots_*`. They are not
durable pipeline product stages unless a later implementation explicitly writes
and documents a composite dataset contract.

## Boundaries

Composite plotting should not perform raw loading, threshold generation,
harmonization, feature extraction, or spatial-field loading.
