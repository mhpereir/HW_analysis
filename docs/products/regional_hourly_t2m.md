# Regional hourly ERA5 T2m

Prepare regional hourly surface temperature once with
`scripts/prepare_regional_hourly_t2m.py`. This reusable upstream ERA5 product
is independent of EHB pressure layers, thresholds and event selection. Spatial
reduction belongs here, not in each Stage-1 assembly or figure run.

## Sources and reduction

The default source directory is
`/home/mhpereir/downloads-mhpereir/REANALYSIS/ERA5/hourly/2mT`, containing
`2mT_hour_ERA5_<year>.nc`. The source root, region, inclusive year range and
time block size are configurable. Each source must contain every UTC hour of
its calendar year, including leap days, exactly once and in order.

Use the native 0.25-degree grid and the inclusive grid-center selection and
cosine-latitude weighting in `src.preprocess`, with region bounds from
`src.config.REGIONS`. Normalize longitude conventions, handle either latitude
direction, require the complete regional grid, and retain both land and ocean
cells. Reduce in float64. Reject nonfinite or missing source cells rather than
varying the effective averaging area between hours. No temporal averaging,
interpolation, regridding, or event filtering occurs.

Only a regional spatial slice and one time block are materialized at a time.
By default, block length follows the file's stored time chunk (676 hours in
the inspected 1940 and 2024 files); contiguous inputs use 168 hours. This
avoids repeatedly decompressing large source chunks for small hour requests.
Only the compact one-dimensional results are retained between annual files.

## Product and storage

An explicitly supplied, new output directory contains:

```text
tas_hourly_<region>_ERA5_025deg_<start_year>_<end_year>.nc
manifest.json
README.md
```

The standard directory is
`$HWA_ARTIFACT_ROOT/regional_hourly_t2m/<region>/<attempt>/`, with the shared
root defaulting to `~/HW-analysis/artifacts`. This prepared product follows
the same storage policy as Stage 1 and Stage 2. NetCDF contains `tas_region(time)`
in kelvin, `native_time_resolution="hourly"`, UTC timestamps, region bounds,
grid resolution, spatial weighting, selection and missing-data policy.
`tas_region` is the existing Stage-1 regional surface-temperature name. Keep
that name in prepared regional datasets; the native resolution is recorded
in the field metadata. The raw ERA5 input retains its source name `t2m`,
recorded as `source_variable="t2m"` on the prepared field. No duplicate `t2m`
field is written. This naming is part of the new, unreleased cache contract;
no production cache has been published under the earlier draft name.
Dataset metadata includes `pipeline_stage="era5_regional_hourly_t2m"`,
`regional_hourly_t2m_contract_version=1`, region, years and producer commit.
The manifest records execution identity/environment, resolved source paths,
sizes and modification times, coverage, selected grid and decoded regional
input hashes, output SHA-256 and validation. The regional input hash covers
the exact selected float64 Kelvin arrays in time/latitude/longitude order;
the grid/time hashes specify those coordinates. It is not a full-global-file
checksum. Raw-file stats must remain unchanged while each file is processed.

Publish NetCDF atomically only after round-trip validation. The output
directory is exclusive; existing attempts are never overwritten or silently
reused. Failed attempts are retained separately from successful products;
`manifest.json` is written last and identifies a completed preparation.

## Consumers and compatibility

Stage 1 optionally accepts `--hourly-t2m-path <prepared.nc>`, reads its hourly
`tas_region`, and stores it as
`tas_region_hourly(time)` on its existing EHB time axis using exact timestamp
lookup. Region, bounds, units, native resolution and all required hours must
match. Input path and checksum are recorded. No raw grids are opened by this
option. The field is optional, including when full diagnostics are requested;
Stage-1 version 2 and existing readers remain compatible. Pressure-layer
builds retain it from their accepted regional reference.

The `_hourly` suffix is needed inside Stage 1 because the existing daily
`tas_region` coexists there. In the standalone hourly product the canonical
field remains `tas_region`. The plotting CLI reads that field by default;
its variable override is available for explicitly supplied external products.
Diagnostic tables use `tas_region` and `tas_region_hourly` column names and
`tas_region` statistical keys, with time scale recorded separately.

`tas_region` remains the authoritative daily event-selection temperature and
is never replaced or regenerated. Its existing 2-degree grid and the new
native 0.25-degree reduction are different spatial operators. Their daily
means need not agree exactly. The temperature diagnostic reports that
disagreement and supports either `--hourly-t2m-path <prepared.nc>` or
`--stage1-hourly-t2m` after optional ingestion into Stage 1. No existing Stage-1
campaign needs rebuilding merely to run the diagnostic.

## Execution and validation

Run production preparation through
`schedulers/schedule_prepare_regional_hourly_t2m.sh`. Use the normal
`HWA_ARTIFACT_ROOT` and set `OUTPUT_DIR` to a fresh directory under
`regional_hourly_t2m/<region>/`. Scheduler logs use `HWA_LOG_ROOT` as usual.
Subsequent Stage-1 jobs consume the prepared cache by its absolute
`HOURLY_T2M_PATH`, which must also be beneath the shared artifact root.
No preparation-specific root override is needed. Existing explicit root
overrides remain supported; pass them through `qsub -v` when submission is
authorized, consistently for preparation and consumption.

The equivalent Python arguments inside that PBS job are:

```bash
python scripts/prepare_regional_hourly_t2m.py \
  --region pnw_hotz --start-year 1940 --end-year 2024 \
  --output-dir "${HWA_ARTIFACT_ROOT:-$HOME/HW-analysis/artifacts}/regional_hourly_t2m/pnw_hotz/new_attempt"
```

Local tests use small synthetic spatial grids over complete calendar years.
They check independently weighted means, source float64 reductions, reversed
latitude and longitude conventions, leap/year coverage, missing cells/hours,
grid completeness, block-size invariance, input immutability, the canonical
`tas_region` NetCDF name and its mapping into Stage 1, output reuse
refusal, manifest hashes and exact optional Stage-1 alignment. Scheduler
checks enforce the shared artifact root for both cache output and Stage-1
cache input, including rejection of paths and symlinks escaping that root. Run the full
`dev_env` suite, Ruff and shell checks before handoff. Production acceptance
still requires an authorized clean-commit PBS smoke run, independent checks
against selected raw values, and validation of the reopened cache and its
downstream temperature plot.

Before a regional campaign, run
`schedulers/schedule_temperature_variability_smoke.sh` with an existing
Stage-1 `INPUT_PATH`, a fresh `OUTPUT_DIR`, `REGION` and `SMOKE_YEAR`.
It runs the relevant synthetic regressions, prepares one complete raw year,
checks selected hourly values with an independent direct weighted reduction,
and renders the JJA temperature diagnostic against the supplied Stage-1 file.
The smoke directory retains the cache, figures and numerical-check report.
Inspect its PBS resources, report and figure before submitting full runs.
For sequential campaigns, submit one preparation job per region and chain
them with PBS `afterok` dependencies, beginning with `pnw_bartusek`.
