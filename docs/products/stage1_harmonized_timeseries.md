# Product Stage 1: Harmonized Regional Time Series

## Contract

Stage 1 is the harmonized, analysis-ready regional time-series product. It is
the durable handoff between raw source loading/harmonization and all downstream
analysis.

New production products use `stage1_contract_version = 2`. Version 2 includes
the normalized signed EHB boundary-face tendencies directly in the standard
producer. They are not appended later by a plotting or exploration workflow.
Readers may continue to open older unversioned products for compatibility, but
consumers that require face tendencies or regional climatology must explicitly
require contract version 2.

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

advection_west(time)
advection_east(time)
advection_south(time)
advection_north(time)
advection_top(time)

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

Cloud cover is loaded from the global hourly ERA5 grid and reduced to the
configured region with the same cosine-latitude weighted regional-mean
procedure used by other gridded surface diagnostics. PBL diagnostics are not
part of the active Stage-1 contract; see
[decision 008](../decisions/008_retire_pbl_diagnostics.md).

Production builds use
`schedulers/schedule_build_stage1_harmonized_timeseries.sh`. The scheduler
requires an explicit region and output path, enables the complete diagnostic
set, and selects the canonical `global-hourly-grid` cloud-cover layout
explicitly. Production output paths must be run-specific so a rebuild can be
validated before any existing Stage-1 product is promoted or replaced.

Surface-energy source signs are preserved. Approximate heating-rate variables
use the pressure-coordinate control-volume approximation documented by the
variable metadata and diagnostics code.

The face variables are normalized from the corresponding signed EHB
`flux_contribution_<face>` variables using:

```text
advection_<face> = flux_contribution_<face> / domain_volume * 3600
```

Positive values warm the domain and negative values cool it. Their sum must
reconstruct `advection` within the recorded numerical tolerance. A fixed
pressure lower boundary may additionally contain `advection_bottom`; the
surface-lower-boundary production case has no bottom-face term.

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

These variables are reductions over detected event intervals. In particular,
`tas_peak` and `tas_anom_peak` are not defined for non-event days and must not
be interpreted as time-indexed diagnostics. Baseline-day workflows derive
fixed-window features directly from the core time-indexed Stage-1 variables.

## Required Metadata

The product must carry:

```text
pipeline_stage = "stage_1_harmonized_regional_timeseries"
stage1_contract_version = 2
analysis_time_resolution = "hourly"
time_axis = "time"
```

Run metadata should also record region, threshold variable, quantile, years,
heat-budget pressure boundaries when applicable, preprocessing choices, and
source paths or source identifiers where practical.

For full-diagnostic builds, metadata must identify
`cloud_cover_source_layout = "global-hourly-grid"` and the global cloud-cover
source root. The temporary legacy-regional compatibility layout is not valid
for a production rebuild.

## Downstream Consumers

- `scripts/event_features/build_stage2_event_features.py`
- `scripts/event_features/build_stage2_baseline_features.py`
- composite plotting workflows
- top-event plotting workflows
- diagnostic plots that inspect the harmonized time series

Downstream consumers should open this product through
`src.analysis_io.open_harmonized_timeseries()` when they require the Stage-1
contract validation.

The Stage-1 writer validates the product and atomically publishes the completed
NetCDF file from a temporary sibling path. An interrupted write must not leave
a partial file at the final product path.

The regional climatology is a separate compact companion product. Stage 1 is
not expanded with climatology or anomaly variables. See
[Regional hourly climatology](stage1_regional_hourly_climatology.md).
