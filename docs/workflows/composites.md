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
scripts/plot_advection_direction_exploration_matched_clim_anom.py
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

The matched face-advection variant obtains matched membership from the
canonical Stage-2 event-feature table and tracked matching settings. It then
selects those event IDs from the absolute Stage-1 event table and verifies that
their Stage-1 and Stage-2 peak timestamps agree. Stage 1 remains authoritative
for event windows and alignment; Stage 2 supplies only the configured matched
membership. The positive and negative groups are composited separately with
the same event count and lag window.

The Venus scheduler wrappers are likewise one operation per PBS job:

```text
schedulers/schedule_plot_composite_timeseries_all_clim_anom.sh
schedulers/schedule_plot_composite_timeseries_split_clim_anom.sh
schedulers/schedule_plot_advection_direction_exploration_clim_anom.sh
schedulers/schedule_plot_advection_direction_exploration_matched_clim_anom.sh
```

They require explicit Stage-1, climatology, and output paths plus a verified
runtime commit. They do not build prerequisites inside plotting jobs.

The production split-composite schedulers render the same complete
split-variable matrix. The five numeric split variables are `duration`,
`tas_anom_peak`, `tas_excess_integral`, `tas_excess_peak`, and `tas_peak`.
The absolute scheduler defaults to a q0.90 split for those variables, while
the climatological-anomaly scheduler defaults to q0.75. Both schedulers also
render a `peak_time` year split beginning in 1982. The quantile and year are
runtime-configurable scheduler defaults; the complete variable matrix is the
production default for every region.

One split-scheduler invocation writes raw and 24-hour-smoothed figures for all
six variants. Every output is derived from one explicit base output path by
adding the split-variable token to the filename. The scheduler must preflight
all twelve derived paths before plotting and refuse partial overwrite. The
climatological-anomaly scheduler must continue to take event membership and
split thresholds from the absolute Stage-1 event table, while compositing the
timestamp-level anomaly values.

Both split schedulers accept the region, pressure boundaries, threshold
definition, analysis years, lag window, smoothing window, and output/log paths
as runtime configuration. This permits one exact deployed scheduler revision
to render isolated output matrices for each accepted Stage-1 region without
changing the tracked source between submissions.

## Outputs

Outputs are figures and diagnostic tables under `results/plots_*`. They are not
durable pipeline product stages unless a later implementation explicitly writes
and documents a composite dataset contract.

## Boundaries

Composite plotting should not perform raw loading, threshold generation,
harmonization, feature extraction, or spatial-field loading.
