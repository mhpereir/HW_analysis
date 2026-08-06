# Stage 1 Companion: Regional Hourly Climatology

## Purpose

The regional hourly climatology is a compact reusable companion to a canonical
Stage-1 harmonized regional time series. It supports timestamp-matched anomaly
calculations without adding ad-hoc variables or dimensions to Stage 1.

```text
results/stage1_climatology/regional_hourly_climatology_*.nc
```

## Producer

```text
scripts/build_stage1_hourly_climatology.py
```

The Venus entrypoint is
`schedulers/schedule_build_stage1_hourly_climatology.sh`.

## Input contract

The producer consumes one Stage-1 contract-version-2 product. The production
input must use canonical global cloud cover and contain the normalized signed
advection-face variables. Climatologies assembled from differently provenanced
Stage-1 files must not be merged.

## Calendar matching

The climatology groups observations by calendar month, calendar day, and UTC
hour. It must not group directly by the native `dayofyear` value because dates
after February 29 have different day-of-year numbers in leap and non-leap
years.

The `climatology_time` coordinate uses reference year 2000. The reference year
is only an encoding of month, day, and hour; it is not a climatology year or a
forecast timestamp. The product contains exactly the calendar-hour keys found
in its Stage-1 input. For the current May-October 1940-2024 product, that is
4,414 keys with 85 source years per complete key.

## Variables

For every selected source variable `<name>`, the product contains:

```text
<name>(climatology_time)        arithmetic climatological mean
<name>_std(climatology_time)    interannual sample standard deviation
<name>_count(climatology_time)  number of finite source values
```

The required source variables cover both temporal-composite and face-advection
figures:

```text
T_mean
volume
dTdt
advection
adiabatic
diabatic
lwa_a_region
lwa_c_region
pbl_p_mean
pbl_p_p05
pbl_p_p95
nslr_heating_rate_approx
nssr_heating_rate_approx
sshf_heating_rate_approx
slhf_heating_rate_approx
soil_moisture
cloud_cover
advection_west
advection_east
advection_south
advection_north
advection_top
```

For a non-surface lower boundary, `advection_bottom` is included when that
signed face exists in the source Stage-1 product. The current surface-to-700
hPa production case has no bottom-face variable.

Daily-native LWA variables are already projected to the hourly Stage-1 axis.
Every calendar-hour key still contains at most one value per source year, so
the stored count represents years rather than treating within-day replicas as
independent samples.

Grouped zonal, meridional, horizontal, vertical, and all-face terms are derived
from the five face means. They are not stored redundantly.

## Scientific method

The initial baseline is the fixed 1940-2024 arithmetic mean over all finite
observations, including heatwave dates. No calendar smoothing or event
exclusion is applied. A non-event baseline or evolving climate-normal baseline
is a separate future scientific choice.

Derived Stage-1 heating rates and face tendencies are climatologized after
their documented normalization. A ratio of source climatologies must not be
substituted for a climatology of a derived ratio or normalized tendency.

## Anomaly application

For every selected Stage-1 timestamp:

```text
anomaly(time) = value(time) - climatology(month, day, hour_utc)
```

Anomalies are computed before event stacking, event means, event percentiles,
and display smoothing. An ephemeral anomaly view may retain the source
variable names for compatibility with shared composite and rendering code, but
it must mark the dataset and variables explicitly as climatological anomalies.

For the plotted PBL spatial envelope, `pbl_p_mean`, `pbl_p_p05`, and
`pbl_p_p95` all subtract the matched `pbl_p_mean` climatology. This preserves
the ordering and contemporaneous spatial width of the percentile band while
placing it around the anomalous regional mean. The independently calculated
PBL-percentile climatologies remain available for audit and future diagnostics.

## Required metadata

```text
pipeline_stage = "stage_1_regional_hourly_climatology"
climatology_reference_year = 2000
climatology_matching = "calendar month, day, and UTC hour"
climatology_method = "arithmetic mean over source years"
climatology_event_exclusion = "none"
```

The product also records source Stage-1 path, source contract version, region,
pressure boundaries, climatology start and end years, source time coverage,
selected variables, and missing-value policy.

The writer validates the product and atomically publishes the completed
NetCDF file. An interrupted write must not leave a partial file at the final
product path.

## Validation

- `climatology_time` is unique, strictly increasing, and contains every source
  month-day-hour key exactly once.
- Every source timestamp maps to one climatology key.
- No source year contains duplicate values for a calendar-hour key.
- Counts are positive; the current complete production input has count 85 for
  every required variable and key.
- Mean source anomaly at every key is zero within floating-point tolerance.
- Heat-budget and face-reconstruction identities remain valid after anomaly
  subtraction.
- Units and signs are preserved and anomaly metadata is explicit.
- Existing absolute products and figures are not overwritten.
