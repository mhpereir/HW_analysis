# Product Stage 2: Baseline-day Features

## Contract

The baseline-day feature product converts each selected-source non-event
calendar day from Stage 1 into one row of fixed-window features centered on a
`reference_time`. It is a parallel Stage-2 product used to estimate the
non-event population; it is not an event-feature table.

Typical outputs:

```text
results/stage2_baseline_features/non_event_day_features_fixed_windows*.nc
results/stage2_baseline_features/non_event_day_features_fixed_windows*.csv
```

## Producer

```text
scripts/event_features/build_stage2_baseline_features.py
```

The producer uses the Stage-1 `event_id_source` metadata to define the
population. A baseline row is retained when that selected event-ID variable is
zero on the reference calendar day.

## Dimensions And Metadata

| Dimension | Meaning |
| --- | --- |
| `baseline_day` | One row per retained selected-source non-event calendar day. |

Core row metadata:

```text
reference_time(baseline_day)
event_adjacent(baseline_day)
```

`reference_time` is the first available Stage-1 timestamp for a calendar day.
`event_adjacent` is `1` when the selected event-ID source is nonzero anywhere
in the union of active fixed windows, and `0` otherwise. Both clean and
event-adjacent rows are retained.

## Fixed-window Features

The product uses the same inclusive fixed-window definitions and source
variables as the Stage-2 event-feature product. Default output includes:

```text
I_dTdt_pre(baseline_day)
I_advection_pre(baseline_day)
I_adiabatic_pre(baseline_day)
I_diabatic_pre(baseline_day)
I_lwa_a_pre_reference(baseline_day)
I_lwa_c_pre_reference(baseline_day)
T_anom_mean_ant(baseline_day)
days_from_solstice(baseline_day)

n_samples_heat_budget_pre(baseline_day)
n_samples_lwa_pre_reference(baseline_day)
n_samples_antecedent_state(baseline_day)
```

Extended mode adds the same land, cloud, PBL, and surface-energy features as
the event-feature product.

## Event-only Variables

Event-summary reductions remain event-only and are intentionally absent:

```text
event_id
start_time
end_time
duration
peak_time
peak_value
tas_peak
tas_anom_peak
tas_excess_peak
tas_excess_integral
lwa_a_peak
lwa_c_peak
```

Stage 1 already provides the time-indexed source variables needed for baseline
features. It should not add time-indexed variables named as event peaks.

## Required Metadata

Global attrs include:

```text
pipeline_stage = "stage_2_baseline_features"
feature_method = "fixed_windows_relative_to_reference_time"
event_id_source = ...
baseline_definition = ...
event_adjacency_definition = ...
n_calendar_days = ...
n_non_event_days = ...
n_selected_before_boundary = ...
dropped_boundary_days = ...
n_baseline_days = ...
n_event_adjacent_days = ...
n_clean_days = ...
```

The producer fails when the selected event-ID source is missing, contains
missing values, or is inconsistent within a calendar day.
