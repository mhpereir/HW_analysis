#!/bin/bash
#PBS -N era5_daily_spatial
#PBS -l select=1:ncpus=1:mem=2gb:host=venus05
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

YEAR=${PBS_ARRAY_INDEX:?Submit this scheduler as a PBS array job}
export OMP_NUM_THREADS=1
EXPECTED_HOST=venus05
ACTUAL_HOST=$(hostname -s)

OUTPUT_DIR="${OUTPUT_DIR:-${HWA_ARTIFACT_ROOT}/spatial_composites/daily}"

# Runtime validation and logging.
hwa_start_log "era5_daily_spatial"
hwa_validate_artifact_paths OUTPUT_DIR

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"

if [[ "$ACTUAL_HOST" != "$EXPECTED_HOST" ]]; then
    echo "[error] PBS placed this job on ${ACTUAL_HOST}; expected ${EXPECTED_HOST}." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
echo "[info] $(date -Is) starting global ERA5 daily preprocessing for ${YEAR} on $(hostname)"
/usr/bin/time -v scripts/spatial_composites/build_era5_daily_spatial_data.sh \
    --start-year "${YEAR}" \
    --end-year "${YEAR}" \
    --threads 1 \
    --output-dir "${OUTPUT_DIR}" \
    --skip-existing
echo "[info] $(date -Is) done"
