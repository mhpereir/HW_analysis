# Workflow: Spatial Composites

## Purpose

The spatial-composite workflow shows daily evolution of ERA5 2 m temperature
and 500 hPa geopotential-height anomalies around heatwave peak dates. Events
are separated by the sign of their integrated net dynamical contribution.

This is an active workflow with three durable dataset layers:

1. annual daily ERA5 T2m/Z500 files;
2. a 366-day T2m/Z500 climatology; and
3. lagged all-event or matched dynamical-sign spatial composite products.

## Data flow

```text
hourly native-grid ERA5 T2m + Z500
  -> scripts/spatial_composites/build_era5_daily_spatial_data.sh
  -> results/spatial_composites/daily/ERA5_daily_t2m_z500_<year>.nc
  -> scripts/spatial_composites/build_era5_daily_doy_climatology.sh
  -> results/spatial_composites/climatology/
       era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc

Stage 2 event-feature table + annual daily files + climatology
  |-> scripts/spatial_composites/build_dyn_net_spatial_composites.py
  |   -> results/spatial_composites/
  |        dyn_net_daily_spatial_composites_*.nc
  |   -> scripts/spatial_composites/plot_dyn_net_spatial_composites.py
  |   -> results/spatial_composites/*.png
  `-> tracked matching settings
      -> scripts/spatial_composites/build_matched_dyn_pre_spatial_composites.py
      -> results/spatial_composites/
           matched_dyn_pre_daily_spatial_composites_*.nc
      -> scripts/spatial_composites/plot_matched_dyn_pre_spatial_composites.py
      -> results/spatial_composites/matched_dyn_pre_*.png
```

## Step 1: Annual daily ERA5 fields

`build_era5_daily_spatial_data.sh` consumes annual hourly T2m and Z500 files
from configurable roots. The Venus defaults point to the existing native-grid
ERA5 holdings. For each year it:

- rechunks both hourly inputs with `nccopy`;
- computes UTC calendar-day arithmetic means with CDO;
- selects Z at 500 hPa;
- merges `t2m` and `z` into one compressed NetCDF file; and
- validates variables, grid size, level, dates, and midnight timestamps.

Dry-run mode validates paths and prints commands without requiring CDO or
`nccopy`. Production runs require both tools and write only after validation.

The tracked PBS array wrapper is
`schedulers/submit_era5_daily_spatial_array.sh`. Its annual worker is
`schedulers/schedule_build_era5_daily_spatial_data.sh`.

## Step 2: Daily climatology

`build_era5_daily_doy_climatology.sh` merges the requested annual daily files
and applies CDO `ydaymean`. The output must contain:

- `t2m` and `z`;
- Z at 500 hPa;
- the expected global grid;
- midnight daily timestamps; and
- 365 or 366 day-of-year values, with the active 1940-2024 product containing
  366 unique month-day keys.

The Venus entrypoint is
`schedulers/schedule_build_era5_daily_doy_climatology.sh`.

## Step 3: Dynamical-sign composite product

`build_dyn_net_spatial_composites.py` consumes:

- a Stage 2 event-feature dataset containing `event_id`, `peak_time`,
  and `I_dyn_pre`;
- annual daily T2m/Z500 files; and
- the daily climatology.

The builder reads `I_dyn_pre` directly and retains the existing `I_dyn_net`
audit-variable names in the spatial product for compatibility. Zero-valued
events are excluded. Remaining events are assigned to `positive` or `negative`
`dyn_sign` groups. Event peak timestamps are normalized to UTC calendar dates.
The default lag window is `-3` through `+3` days, with lag zero representing the
peak date.

The composite uses equal event weight within each sign and lag. Actual fields
are matched by timestamp. Climatology fields are matched by month and day.
Geopotential is converted to height with `z / G_M_S2`.

Default spatial bounds are 10 to 80 degrees north and longitude -170 to -40
degrees. Bounds, lags, input paths, and output path are configurable CLI
arguments.

### Product contract

The output has dimensions:

```text
dyn_sign = positive, negative
lag
latitude
longitude
event
```

Core gridded variables are:

```text
t2m_event_mean(dyn_sign, lag, latitude, longitude)
t2m_climatology_mean(dyn_sign, lag, latitude, longitude)
t2m_anomaly(dyn_sign, lag, latitude, longitude)
z500_event_mean(dyn_sign, lag, latitude, longitude)
z500_climatology_mean(dyn_sign, lag, latitude, longitude)
z500_anomaly(dyn_sign, lag, latitude, longitude)
```

Audit variables include:

```text
event_count(dyn_sign)
I_dyn_pre_mean(dyn_sign)
I_dyn_net_mean(dyn_sign)
event_id(event)
peak_time(event)
I_dyn_pre(event)
I_dyn_net(event)
event_dyn_sign(event)
```

`I_dyn_pre_mean` and `I_dyn_pre` are canonical. The corresponding
`I_dyn_net` names are retained as compatibility aliases in spatial products.

The product marker is:

```text
pipeline_stage = "daily_dyn_net_spatial_composites"
```

Metadata also records source paths, lags, bounds, weighting, climatology
matching, geopotential conversion, and the number of excluded zero-valued
events.

## Matched dynamical-sign composite product

`build_matched_dyn_pre_spatial_composites.py` consumes the same Stage-2 event
features, annual daily fields, and daily climatology, plus the tracked A2.8
matching settings. It loads the named specification, calls
`src.selectors.match_events_by_metric_sign()`, and selects membership before
any spatial averaging. The production default is `peak_anomaly_0p20`, which
matches on `tas_anom_peak` using a 0.20 pooled-standard-deviation caliper.

The matched builder reuses the all-event field reduction, grid validation,
climatology matching, and equal-event weighting. It writes a separate product
with:

```text
pipeline_stage = "daily_matched_idyn_spatial_composites"
```

Both sign groups must contain the same nonzero number of events. In addition to
the common spatial and event audit variables, the product records
`matched_pair_id(event)` and `matched_pair_distance(event)`. Global attributes
record the specification, family, variables, caliper, reference sign, pair
count, source sign counts, unmatched sign counts, matching method, Stage-2 path
and SHA-256, and settings path and SHA-256. The product is an
analysis-specific spatial aggregate. It does not persist matched membership as
a new Stage-2 table.

This product cannot be reconstructed from the all-event spatial composite,
because that product no longer contains per-event spatial fields. It must be
built from the canonical Stage-2 table and the unchanged annual daily inputs.

## Plot

`plot_dyn_net_spatial_composites.py` validates the composite contract and
creates a sign-by-lag Lambert conformal map. It uses:

- the default display lags `-2`, `0`, and `+2`, labeled `t-2`, `t (peak)`,
  and `t+2`;
- filled `t2m_anomaly` colors centered on zero;
- `z500_anomaly` contours;
- one shared horizontal colorbar;
- 50 m coastlines and borders;
- the `pnw_bartusek` region outline; and
- shared sizes, fonts, and DPI from `src/plot_style.py`.

The plot-lag selection is configurable without changing the composite dataset.
Every requested lag must be present in the input. The underlying default
composite product remains the complete `-3` through `+3` lag window, so this
display change requires no dataset rebuild or migration.

Cartopy Natural Earth data must be available in the production environment.
Use `schedulers/schedule_plot_dyn_net_spatial_composites.sh` to regenerate only
the figure from an existing validated composite product.

`plot_matched_dyn_pre_spatial_composites.py` renders the same map variables,
projection, three default display lags, contours, region outline, and shared
color scale from the separate matched product. Its title reports the matching
label, pooled-SD caliper, and pair count. Its two row labels identify positive
and negative `I_dyn_pre` explicitly. It writes a distinct PNG and does not
replace the all-event map.

The commit-pinned production entrypoint is
`schedulers/schedule_matched_dyn_pre_spatial_composites.sh`. It stages the
matched NetCDF and PNG together and publishes them only after both validate.

## Validation

Local synthetic coverage:

```bash
mamba activate dev_env
python -m pytest -q \
  tests/test_spatial_composite_builder.py \
  tests/test_matched_spatial_composite_builder.py \
  tests/test_spatial_composite_plot.py \
  tests/test_matched_spatial_composite_plot.py \
  tests/test_spatial_composite_shell_scripts.py
```

For production changes, also run the relevant PBS smoke workflow and validate
input coverage, event counts, exact grids, finite anomalies, metadata, output
files, the expected six map panels, and the rendered map. Matched production
validation additionally requires the accepted Stage-2 and settings checksums,
the named `peak_anomaly_0p20` specification, equal 97-event sign groups, pair
audit consistency, and no unmatched-output replacement.

## Boundaries

- Do not load full spatial fields into Stage 1 or Stage 2.
- Do not rebuild event definitions in the spatial workflow.
- Do not compute hourly ERA5 daily means inside the Python composite builder.
- Do not treat the spatial composite as a replacement for the Stage 1 regional
  time-series product.
- Do not run CDO processing or composite construction on the Venus login node.
