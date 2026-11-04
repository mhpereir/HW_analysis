#!/bin/bash
#PBS -N temperature_smoke
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?Existing Stage-1 INPUT_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?Fresh smoke OUTPUT_DIR is required}"
REGION="${REGION:-pnw_bartusek}"
SMOKE_YEAR="${SMOKE_YEAR:-1940}"
SOURCE_ROOT="${SOURCE_ROOT:?Raw hourly ERA5 SOURCE_ROOT is required}"

hwa_start_log "temperature_variability_smoke"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_DIR
test "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -f "${INPUT_PATH}"
test ! -e "${OUTPUT_DIR}" && test ! -L "${OUTPUT_DIR}"
[[ "${SOURCE_ROOT}" == /* && -d "${SOURCE_ROOT}" ]]
mkdir -p -- "$(dirname -- "${OUTPUT_DIR}")"
mkdir -- "${OUTPUT_DIR}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
cd "${PROJECT_ROOT}"
echo "[start] $(date -Is) job=${PBS_JOBID} host=$(hostname) commit=${EXPECTED_COMMIT}"
echo "[paths] input=${INPUT_PATH} raw=${SOURCE_ROOT} output=${OUTPUT_DIR}"
python -c 'import sys, numpy, xarray, h5netcdf; print(sys.executable, sys.version); print("numpy", numpy.__version__, "xarray", xarray.__version__, "h5netcdf", h5netcdf.__version__)'
/usr/bin/time -v python -m pytest -q -W error \
    tests/test_regional_hourly_t2m.py tests/test_plot_temperature_variability.py \
    tests/test_stage1_pressure_layer.py tests/test_artifact_paths.py
/usr/bin/time -v python scripts/prepare_regional_hourly_t2m.py \
    --region "${REGION}" --start-year "${SMOKE_YEAR}" --end-year "${SMOKE_YEAR}" \
    --source-root "${SOURCE_ROOT}" --output-dir "${OUTPUT_DIR}/cache"

python - "${OUTPUT_DIR}" "${SOURCE_ROOT}" "${REGION}" "${SMOKE_YEAR}" "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import REGIONS

output, source_root, region, year, commit = sys.argv[1:]
year = int(year)
output = Path(output)
manifest = json.loads((output / "cache/manifest.json").read_text())
path = Path(manifest["output"]["path"])
assert manifest["execution"]["commit"] == commit
assert manifest["execution"]["dirty"] is False
with path.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["output"]["sha256"]
times = pd.to_datetime([f"{year}-01-01T00", f"{year}-02-28T23", f"{year}-06-21T15", f"{year}-12-31T23"])
lat, lon = REGIONS[region]
with xr.open_dataset(Path(source_root) / f"2mT_hour_ERA5_{year}.nc", engine="h5netcdf", cache=False) as raw:
    latitude = raw.latitude.values
    longitude = (raw.longitude.values + 180) % 360 - 180
    lat_indices = np.flatnonzero((latitude >= lat.start) & (latitude <= lat.stop))
    lon_indices = np.flatnonzero((longitude >= lon.start) & (longitude <= lon.stop))
    values = raw.t2m.isel(latitude=lat_indices, longitude=lon_indices).sel(valid_time=times).transpose("valid_time", "latitude", "longitude").values.astype(np.float64)
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude[lat_indices]))[:, None], values.shape[1:])
    expected = (values * weights).sum(axis=(1, 2)) / weights.sum()
    assert np.isfinite(values).all()
with xr.open_dataset(path, engine="h5netcdf") as cached:
    assert set(cached.data_vars) == {"tas_region"}
    assert cached.tas_region.attrs["units"] == "K"
    assert cached.tas_region.attrs["native_time_resolution"] == "hourly"
    full_time = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    assert pd.DatetimeIndex(cached.time.values).equals(full_time)
    assert np.isfinite(cached.tas_region.values).all()
    actual = cached.tas_region.sel(time=times).values
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
report = {"commit": commit, "cache_sha256": manifest["output"]["sha256"], "hours": len(full_time), "sample_times": [str(t) for t in times], "expected_K": expected.tolist(), "actual_K": actual.tolist(), "max_abs_error_K": float(np.max(np.abs(actual - expected))), "absolute_tolerance_K": 1e-10}
with (output / "independent_validation.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(report, indent=2))
PY

/usr/bin/time -v python scripts/plot_temperature_variability.py \
    --input-path "${INPUT_PATH}" \
    --hourly-t2m-path "${OUTPUT_DIR}/cache/tas_hourly_${REGION}_ERA5_025deg_${SMOKE_YEAR}_${SMOKE_YEAR}.nc" \
    --start-year "${SMOKE_YEAR}" --end-year "${SMOKE_YEAR}" \
    --output-dir "${OUTPUT_DIR}/figure"
echo "[done] $(date -Is) job=${PBS_JOBID}"
