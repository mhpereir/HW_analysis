# Top-event maps

Make one map per top-ranked event, with the mean T2m anomaly in color and Z500
height-anomaly contours over the peak UTC day and its two neighboring days.
For either PNW event population the default map outlines both `pnw_hotz` and
`pnw_bartusek`.

The [workflow contract](../../docs/workflows/top_events_map.md) defines event
selection, dates, baseline, units, provenance, and acceptance checks. Source
and output paths are explicit so this workflow can use a clean isolated code
checkout and the shared Venus data holdings.

## Inputs

- A canonical Stage-2 event-feature NetCDF for the event region. Ranking
  defaults to `tas_peak`; `--rank-metric tas_anom_peak` selects by anomaly.
- Existing annual daily T2m/geopotential files named
  `ERA5_daily_t2m_z500_<year>.nc`. These contain daily means, not TX.
- The spatial workflow's 366-key climatology with `t2m` and `z` at 500 hPa.
  Its baseline years must be stated explicitly. The current PNW example uses
  1940-2024, not a 1991-2020 baseline.

Full spatial data stay out of Stage 1 and Stage 2. If an input is missing,
prepare it through the existing tracked spatial-composite PBS workflow. The
new builder reads only requested dates and map cells; it never downloads data
or computes hourly-to-daily means.

## Venus example: the June 2021 event

Run these commands inside an authorized PBS allocation, using `dev_env`.
The wrapper below runs both steps automatically. Set `HWA_ARTIFACT_ROOT` to
the external artifact root before using these commands.

```bash
python scripts/top_events_map/build_top_events_map.py \
  --event-features-path ${HWA_ARTIFACT_ROOT}/stage2_event_features/runs/bf232281_20260820/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc \
  --region pnw_bartusek \
  --daily-dir ${HWA_ARTIFACT_ROOT}/spatial_composites/daily \
  --climatology-path ${HWA_ARTIFACT_ROOT}/spatial_composites/climatology/era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc \
  --climatology-start-year 1940 --climatology-end-year 2024 \
  --peak-year 2021 --top-n 1 \
  --output-path /absolute/new-run/top_events_map.nc

python scripts/top_events_map/plot_top_events_map.py \
  --input-path /absolute/new-run/top_events_map.nc \
  --output-dir /absolute/new-run/figures \
  --outline-regions pnw_hotz pnw_bartusek \
  --height-contour-interval 50
```

The inspected PNW tables select event 1127 for Bartusek and 1217 for Hotz,
both anchored on 29 June 2021. Both therefore average 28-30 June 2021. To rank
across the full table, omit `--peak-year`. For a Hotz event population, change
both `--region` and the event-table path. A top-N request produces N separate
single-panel maps (or all finite-ranked events if fewer than N are available).

Older Stage-2 files lack a `region` attribute. In that case `--region` is an
explicit provenance declaration, recorded as such. If region metadata exists,
it must agree with the argument. The raw spatial maps for two populations
with the same peak day and baseline are identical; their regional event
identities and filenames differ.

## Dedicated PBS entrypoint

`schedule_top_events_map.sh` requests one CPU, 8 GB and 20 minutes. It requires
`PROJECT_ROOT`, `EXPECTED_COMMIT`, `EVENT_FEATURES_PATH`, `REGION`, `DAILY_DIR`,
`CLIMATOLOGY_PATH`, and `RUN_DIR` for a new product. Alternatively, provide
`PREPARED_MAP_PATH` and `SOURCE_VALIDATION_PATH` with `RUN_DIR` to replot an
accepted product. All prepared paths must be beneath `HWA_ARTIFACT_ROOT`.
Logs use `HWA_LOG_ROOT` or an explicit `LOG_DIR` beneath it.
`RUN_DIR` must be new. Optional settings are `TOP_N`, `RANK_METRIC`, `PEAK_YEAR`,
`CLIMATOLOGY_START_YEAR`, `CLIMATOLOGY_END_YEAR`, `MAP_EXTENT` (space-separated
west/east/south/north), `OUTLINE_REGIONS` (space-separated), `TEMPERATURE_LIMIT`,
`HEIGHT_CONTOUR_INTERVAL`, and `PLOT_RANKS` (space-separated stored ranks).
Defaults match the workflow contract.

Before submission, record the task/run through command center and deploy the
exact reviewed commit with authorization. Submit the tracked script from that
clean checkout with the required environment variables passed explicitly via
`qsub -v`. Do not use `qsub -V`.

The job writes into a fresh staging directory, validates the NetCDF and PNGs,
independently compares all six fields to direct daily-source sums, records
`source_validation.json` and `manifest.json`, then publishes the run directory
without replacement.
Failed staging directories are retained for diagnosis. Its independent log is
`LOG_DIR/<PBS_JOBID>_top_events_map.log`. A completed run contains:

```text
<RUN_DIR>/
  top_events_map.nc
  figures/top_events_map_<region>_rank01_event<id>_<YYYYMMDD>.png
  source_validation.json
  manifest.json
```

Scientific acceptance of a new product additionally requires a direct
comparison with the three source daily fields and their matched climatology,
and original-resolution inspection of the figure. A replot retains the exact
product and successful source audit, verifies their hash linkage and records
the new plotting commit separately. It does not recalculate fields or change
the historical event population. Local synthetic tests do not accept a
production run.

## Local checks

```bash
mamba activate dev_env
python -m pytest -q tests/test_top_events_map.py
bash -n scripts/top_events_map/schedule_top_events_map.sh
python -m pytest -q
```

## Reproduce a saved run with single-column typography

The map is 6 by 4.5 inches with 18 pt labels, 14 pt legends and 10.5 pt contour labels.
Titles and footers are omitted, and line widths are halved. The horizontal
colorbar is approximately 0.242 inches below the map (half the prior gap).
The PNW Hotz et al. / PNW Bartusek et al. legend has two columns in one row
inside the lower map. PNG export preserves its size at 300 DPI.

For a Bartusek-only version, pass `--outline-regions pnw_bartusek` or set
`OUTLINE_REGIONS=pnw_bartusek` for PBS. Selecting one outline automatically
omits the region legend. Use a fresh output directory for this variant.

For an authorized submission from a clean pinned deployment:

```bash
export HWA_ARTIFACT_ROOT="$HOME/HW-analysis/artifacts"
export HWA_LOG_ROOT="$HOME/HW-analysis/logs"
export PREPARED_MAP_PATH="$HWA_ARTIFACT_ROOT/top_events_map/runs/c2387d4_20260910_pnw_bartusek_top10/top_events_map.nc"
export SOURCE_VALIDATION_PATH="$(dirname "$PREPARED_MAP_PATH")/source_validation.json"
export RUN_DIR="$HWA_ARTIFACT_ROOT/top_events_map/runs/<fresh-attempt>"
export LOG_DIR="$HWA_LOG_ROOT/top_events_map/<fresh-attempt>"
# Set PROJECT_ROOT and EXPECTED_COMMIT to the reviewed, clean deployment.
qsub -v PROJECT_ROOT,EXPECTED_COMMIT,HWA_ARTIFACT_ROOT,HWA_LOG_ROOT,PREPARED_MAP_PATH,SOURCE_VALIDATION_PATH,RUN_DIR,LOG_DIR \
  scripts/top_events_map/schedule_top_events_map.sh
```

For the smoke, additionally export `PLOT_RANKS=1` and include `PLOT_RANKS` in
`qsub -v`. The scale still comes from the full saved product. Use a separate
fresh run directory for the full ten-map reproduction. The new manifest
identifies the reused scientific audit; the copied source report retains its
original timestamp and validator commit.
