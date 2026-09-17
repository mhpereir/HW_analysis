#!/bin/bash
#PBS -N era5_daily_climatology
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

DAILY_DIR="${DAILY_DIR:-${HWA_ARTIFACT_ROOT}/spatial_composites/daily}"
OUTPUT_PATH="${OUTPUT_PATH:-${HWA_ARTIFACT_ROOT}/spatial_composites/climatology/era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc}"

# Runtime validation and logging.
hwa_start_log "era5_daily_climatology"
hwa_validate_artifact_paths DAILY_DIR OUTPUT_PATH

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -d "${DAILY_DIR}"
test ! -e "${OUTPUT_PATH}"

cd "${PROJECT_ROOT}"
echo "[info] $(date -Is) starting global ERA5 daily climatology on $(hostname)"
/usr/bin/time -v scripts/spatial_composites/build_era5_daily_doy_climatology.sh \
    --daily-dir "${DAILY_DIR}" \
    --output-path "${OUTPUT_PATH}"
echo "[info] $(date -Is) done"
