# Product Stage 1: Harmonized Regional Time Series

## Contract

Stage 1 is the harmonized, analysis-ready regional time-series product. It is
the durable handoff between raw source loading/harmonization and all downstream
analysis.

```text
results/stage1/harmonized_regional_timeseries_*.nc
```

## Producer

```text
scripts/build_stage1_harmonized_timeseries.py
```

The producer assembles the product through the raw loading, preprocessing,
harmonization, selector, and event-summary modules. `src.analysis_io` owns the
stable save/open behavior and validates the product marker on read.

## Consumes

- raw ERA5 and locally stored ARCO/ERA5 inputs
- heatwave threshold products
- LWA and LWA threshold products
- Eulerian heat-budget diagnostics
- PBL pressure diagnostics
- optional surface radiation, turbulent flux, soil-moisture, and cloud-cover inputs

## Dimensions

| Dimension | Meaning |
| --- | --- |
| `time` | Harmonized analysis time axis, currently hourly for the primary product. |
| `event` | Heatwave event-summary axis. |

## Core Time-indexed Variables

Required and expected variables include:

```text
T_mean(time)
volume(time)
dTdt(time)
advection(time)
adiabatic(time)
diabatic(time)

tas_region(time)
tas_climatology(time)
hw_threshold(time)
hw_flag(time)
hw_event_id(time)

lwa_region(time)
lwa_threshold(time)
lwa_flag(time)
lwa_event_id(time)
lwa_a_region(time)
lwa_a_threshold(time)
lwa_a_flag(time)
lwa_a_event_id(time)
lwa_c_region(time)
lwa_c_threshold(time)
lwa_c_flag(time)
lwa_c_event_id(time)
```

When full diagnostics are available, the product may also include:

```text
soil_moisture(time)
cloud_cover(time)
pbl_p_mean(time)
pbl_p_p05(time)
pbl_p_p95(time)
nslr(time)
nssr(time)
sshf(time)
slhf(time)
nslr_heating_rate_approx(time)
nssr_heating_rate_approx(time)
sshf_heating_rate_approx(time)
slhf_heating_rate_approx(time)
surface_energy_heating_rate_approx(time)
```

Surface-energy source signs are preserved. Approximate heating-rate variables
use the pressure-coordinate control-volume approximation documented by the
variable metadata and diagnostics code.

## Event-summary Variables

The event axis stores one row per detected event. Common event-summary variables
include:

```text
event_id(event)
start_time(event)
end_time(event)
duration(event)
peak_time(event)
tas_peak(event)
tas_anom_peak(event)
tas_excess_peak(event)
tas_excess_integral(event)
lwa_a_peak(event)
lwa_c_peak(event)
```

## Required Metadata

The product must carry:

```text
pipeline_stage = "stage_1_harmonized_regional_timeseries"
analysis_time_resolution = "hourly"
time_axis = "time"
```

Run metadata should also record region, threshold variable, quantile, years,
heat-budget pressure boundaries when applicable, preprocessing choices, and
source paths or source identifiers where practical.

## Downstream Consumers

- `scripts/event_features/build_stage2_event_features.py`
- composite plotting workflows
- top-event plotting workflows
- diagnostic plots that inspect the harmonized time series

Downstream consumers should open this product through
`src.analysis_io.open_harmonized_timeseries()` when they require the Stage-1
contract validation.
