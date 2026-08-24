# Workflow: PBL Top-Pressure Justification

## Purpose

This standalone workflow supports the paper's use of 700 hPa as the upper
boundary of the Eulerian heat-budget domain. It does not restore planetary
boundary-layer diagnostics to Stage 1, Stage 2, regional climatologies, or the
standard temporal-composite figures.

The diagnostic uses PBL-top pressure rather than converting pressure to an
approximate geometric height. Pressure is the exact vertical coordinate of the
700 hPa heat-budget boundary. Lower PBL-top pressure means a deeper and higher
boundary layer.

## Inputs and event population

The workflow consumes:

- the accepted PNW Bartusek TAS q90 Stage-1 product for 1940-2024, used only
  for its event table and event IDs;
- the accepted annual hourly `pbl_p(time, lat, lon)` files for
  `pnw_bartusek`; and
- the configured `pnw_bartusek` bounds from `src.config.REGIONS`.

The event population must match the production all-event temporal composite:
select TAS q90 heatwaves whose complete inclusive event interval is in June,
July, or August. Peak times and event IDs remain authoritative in Stage 1.
Heatwave days for the map are every distinct UTC calendar day belonging to one
of those selected events. The workflow records this UTC-day basis explicitly
and does not redefine the event contract.

## Compact diagnostic product

The builder writes one immutable NetCDF product before plotting. It contains:

- `pbl_top_pressure_area_mean(lag_hour)`, the event-mean peak-aligned
  cosine-latitude area mean;
- `pbl_top_pressure_spatial_p05(lag_hour)` and
  `pbl_top_pressure_spatial_p95(lag_hour)`, the event means of the hourly
  cosine-latitude area-weighted spatial percentiles;
- `mean_daily_min_pbl_top_pressure(lat, lon)`, the mean across selected
  heatwave days of each day's minimum hourly PBL-top pressure at every grid
  cell; and
- `event_sample_count(lag_hour)`, plus scalar event-day and event counts.

The default peak-aligned window is seven days before through seven days after
the Stage-1 event peak. Source values remain in Pa in the product. Plotting
converts them to hPa.

The map quantity is the average daily maximum PBL height expressed in pressure
coordinates: the maximum height within a day is the minimum PBL-top pressure
within that day. The workflow must not describe this field as a geometric
height in metres.

Required global provenance includes the Stage-1 source path and SHA-256, PBL
source root and annual-file inventory, region and bounds, selected years,
season selection, event and heatwave-day counts, lag window, pressure
interpretation, and source commit.

## Figure contract

The two-panel paper figure contains:

1. a peak-aligned time series in hPa with the area mean as a line, the spatial
   5th-95th percentile range as a shaded envelope, an inverted pressure axis,
   a vertical peak marker, and a horizontal 700 hPa reference; and
2. a map of `mean_daily_min_pbl_top_pressure` in hPa with coastlines, borders,
   geographic labels, and the configured PNW Bartusek domain outlined by a red
   rectangle.

Pressure ticks are displayed directly in hPa without scientific-notation
scaling. The time-series legend and map color bar occupy dedicated strips below
their panels so neither obscures data, and the color-bar label must remain
fully visible in the saved artifact.

The map extent extends 2.5 degrees beyond every configured domain edge. The PBL
field is shown only where source data exist. The surrounding margin provides
geographic context and must not be filled by interpolation or extrapolation.

The figure uses `src.plot_style`, panel labels, a shared publication font and
line-weight contract, and an isolated output path. Existing figures are never
overwritten.

## Validation

Acceptance requires:

- exact 1940-2024 annual PBL file coverage with unique monotonic hourly time;
- the documented full-JJA event selection and exact selected event IDs;
- complete hourly samples for every event lag and every selected heatwave day;
- finite product values and `0 < pbl_p <= 120000 Pa`;
- ordered spatial percentiles at every lag;
- a map grid matching the PBL source and configured domain bounds;
- atomic NetCDF publication with no partial sibling;
- a commit-pinned Venus OpenPBS build and plot run with clean logs; and
- original-resolution visual inspection of the time-series labels, 700 hPa
  reference, pressure orientation, color bar, geographic context, and red
  domain rectangle.

Passing scheduler and artifact checks alone is not figure acceptance.
