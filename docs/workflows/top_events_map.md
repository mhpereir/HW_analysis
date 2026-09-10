# Workflow: Top-event maps

## Purpose and compatibility

`top_events_map` produces one map per ranked heatwave event: three-day mean
ERA5 2 m temperature anomalies in color and 500 hPa geopotential-height
anomalies in contours. The map defaults to both `pnw_hotz` and `pnw_bartusek`
outlines for a PNW event, using the authoritative bounds in `src/config.py`.

This is a separate consumer of the existing Stage-2 event table and the
[spatial workflow's](spatial_composites.md) daily ERA5 and climatology files.
It does not change Stage 1, Stage 2, event definitions, temporal `top_events`,
or sign/matched composites. Existing products need no migration. Its scripts,
README, and PBS entrypoint live together under `scripts/top_events_map/`.
Reusable loading, product I/O, calculation, and rendering retain their `src/`
ownership.

## Scientific choices

- Rank descending by `tas_peak`, consistent with temporal `top_events`.
  Default `top_n=1`; optional `peak_year` filters the population before ranking.
  Other existing event-level ranking metrics can be selected explicitly.
- Read `peak_time` from Stage 2, retaining the distinction between the event
  anchor and independent anomaly maxima in [decision 005](../decisions/005_stage1_event_peak_semantics.md).
- Normalize the anchor to its UTC calendar day and use exactly days -1, 0,
  and +1. A non-midnight peak still selects those three complete UTC days.
  This is not a rolling 72-hour interval centered on the precise peak hour.
- Use the existing arithmetic daily-mean `t2m`, not daily maximum temperature
  (TX). Average all three days equally without skipping missing values.
- Match climatology by month and day, including February 29. Require the
  existing 366-key daily climatology. Never align by numeric day-of-year.
- Subtract the mean of the three matched climatological days from the mean
  event field. Convert ERA5 geopotential to height with `z / config.G_M_S2`.
  T2m anomalies are in K (numerically equal to degrees C); Z500 anomalies are m.
- Require identical finite grids, complete dates, valid 500 hPa metadata,
  Kelvin T2m, and geopotential units. Crop before loading fields and read only
  requested days, including adjacent annual files at a year boundary.
- The climatology baseline years are required build arguments and must agree
  with any baseline-year metadata in the input. Record them explicitly, and
  never infer baseline years from the climatology's representative timestamps.

## Product and entrypoints

`build_top_events_map.py` writes a reusable NetCDF with
`pipeline_stage="top_events_map"`, `top_events_map_contract_version=1`.

Dimensions are `event`, `window_day=(-1,0,1)`, `latitude`, and `longitude`.
The six `(event, latitude, longitude)` fields are `t2m_event_mean`,
`t2m_climatology_mean`, `t2m_anomaly`, `z500_event_mean`,
`z500_climatology_mean`, and `z500_anomaly`. Audit variables include
`event_id`, `peak_time`, `selection_rank`, `rank_value`, and
`sample_date(event, window_day)`. Preserve source event labels.

Metadata records the event region and configured bounds, rank metric and
population, climatology period and matching method, window and conversion,
map bounds, event-table SHA-256, source paths and file sizes/modification times,
and the producing Git commit. The default map extent is the spatial workflow's
North American domain (170 W to 40 W, 10 N to 80 N), with explicit bounds
available for other regions.

`plot_top_events_map.py` consumes only that prepared product and writes a
single-panel Plate Carree map PNG per event. The rectangular geographic domain
keeps the displayed field inside the prepared grid. Filenames include region,
rank, event ID, and peak date. All figures from one product share a symmetric zero-centered
temperature scale. Z500 contours use m, negative dashed and nonnegative solid,
with a default 50 m interval. Titles identify the averaging dates and event
region; captions identify the climatology baseline and peak anchor. Region
colors, theme, sizes, line widths, and export DPI use `src/plot_style.py`.
Natural Earth map assets must already be available in the Venus environment.

Each CLI refuses existing output paths. The PBS wrapper requires a new run
directory and publishes it only after the prepared product and all figures
validate, retaining an immutable manifest with inputs, configuration, exact
commit, job identity, and output checksums. It uses a clean commit-verified
checkout, `dev_env`, one CPU, 8 GB, and 20 minutes. These resources are an
initial smoke-test request, to be checked against observed use.

## Validation

Local synthetic tests must independently verify ranking and year selection,
three-day arithmetic, exact month/day matching across leap and year boundaries,
geopotential conversion, missing-date/nonfinite/unit/grid rejection, metadata
and NetCDF round-trip, and refusal to overwrite outputs. Plot tests verify one
map plus colorbar, both configured outlines, dates, units, contour signs, and
shared style without requiring network access. Exercise the two CLIs together
on tiny synthetic files, check PBS syntax and failure gates, and run the full
repository test suite before handoff.

After authorized Git publication/deployment, use the tracked PBS entrypoint
for a 2021 PNW smoke run. Independently compare the stored fields with direct
three-day sums from the annual files and matching climatology, confirm event
selection, finite arrays, provenance and checksums, and inspect the exported
PNG at original resolution. A local synthetic render is not a production
2021 heatwave figure.

`validate_top_events_map.py` performs that independent source comparison on a
PBS worker before publication. It independently ranks the source event table,
checks IDs and peak dates, selects source cells by the saved grid coordinates,
matches climatology by month/day, and directly sums three daily samples. All
six fields must agree with absolute tolerance `1e-10` and zero relative
tolerance. It does not call the builder's field loader or reduction. Source
file identities and the event-table checksum must still match the product.
The immutable `source_validation.json` records maximum absolute errors and
the validated events, input identities, product hash, and validator commit.
