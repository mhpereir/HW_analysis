# Product Stage 2: Event Features

## Contract

Stage 2 converts each heatwave event from the Stage-1 product into one
event-level row of fixed-window features centered on `peak_time`.

Typical outputs:

```text
results/event_features/hw_event_features_fixed_windows*.nc
results/event_features/hw_event_features_fixed_windows*.csv
```

The NetCDF product is canonical. CSV output is an optional convenience export.

## Producer

```text
scripts/event_features/build_event_features.py
```

Configuration lives in:

```text
scripts/event_features/event_feature_config.py
```

## Consumes

- Stage-1 harmonized regional time series
- Stage-1 event-summary variables

The producer must not rebuild thresholds, event IDs, or the harmonized time
series, and must not modify the Stage-1 product.

## Dimensions

| Dimension | Meaning |
| --- | --- |
| `event` | One row per retained heatwave event. |

## Fixed Windows

Timestamp slices are inclusive. Current defaults are:

| Window | Lags relative to `peak_time` |
| --- | --- |
| `heat_budget_pre` | `(-96, 0)` hours |
| `lwa_pre_peak` | `(-96, 0)` hours |
| `antecedent_state` | `(-168, -24)` hours |
| `antecedent_change` | `(-168, 0)` hours |
| `near_peak` | `(-24, 24)` hours |
| `decay` | `(0, 72)` hours |

See [decision 001](../decisions/001_event_feature_windows.md).

## Default Variables

Default mode uses only the core Stage-1 variables and writes:

```text
event_id(event)
start_time(event)
end_time(event)
peak_time(event)
duration(event)
tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)

I_dTdt_pre(event)
I_advection_pre(event)
I_adiabatic_pre(event)
I_diabatic_pre(event)
I_lwa_a_pre_peak(event)
I_lwa_c_pre_peak(event)
T_anom_mean_ant(event)
days_from_solstice(event)

n_samples_heat_budget_pre(event)
n_samples_lwa_pre_peak(event)
n_samples_antecedent_state(event)
```

`T_anom_mean_ant` is based on `tas_region - tas_climatology` when a direct
`tas_anom` variable is not present. Heat-budget integrals use
`hourly_sum_assuming_1h_spacing` in the current product.

## Extended Variables

Extended mode may add:

```text
soil_moisture_mean_ant(event)
soil_moisture_change(event)
cloud_cover_mean_ant(event)
pbl_p_mean_ant(event)
I_nslr_pre(event)
I_nssr_pre(event)
I_sshf_pre(event)
I_slhf_pre(event)
I_surface_energy_pre(event)
n_samples_antecedent_change(event)
```

If extended mode is requested, missing extended variables should raise unless
the run explicitly allows missing extended variables. Surface-energy features
retain native Stage-1/source signs.

## Required Metadata

Global attrs should include:

```text
pipeline_stage = "stage_2_event_features"
feature_method = "fixed_windows_relative_to_peak_time"
integral_method = "hourly_sum_assuming_1h_spacing"
extended_variables_used = 0 or 1
adaptive_windows_used = 0
all_seasons = 0 or 1
season_months = comma-separated months or empty
require_full_event = 0 or 1
dropped_boundary_events = integer count
```

Each feature variable should carry `source_variable`, `window_name`,
`window_lag_hours`, `operation`, units where known, and
`window_endpoint_inclusion="inclusive"` for window-derived variables.

## Validation Expectations

The producer should fail clearly when required core variables are absent, no
events remain after selection, selected events have missing `peak_time`, or the
output path exists without overwrite permission.

The sample-count variables are part of the product contract because they expose
boundary events and missing hourly samples to PCA and clustering consumers.

## Non-goals

Stage 2 does not perform PCA, clustering, composites, new event detection, or
adaptive `dTdt > 0` growth-window calculations.
