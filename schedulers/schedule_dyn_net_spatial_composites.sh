#!/bin/bash
#PBS -N dyn_net_spatial_composites
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

EVENT_FEATURES_PATH="${EVENT_FEATURES_PATH:-${HWA_ARTIFACT_ROOT}/stage2_event_features/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc}"
DAILY_DIR="${DAILY_DIR:-${HWA_ARTIFACT_ROOT}/spatial_composites/daily}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-${HWA_ARTIFACT_ROOT}/spatial_composites/climatology/era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc}"
COMPOSITE_OUTPUT_PATH="${COMPOSITE_OUTPUT_PATH:?COMPOSITE_OUTPUT_PATH is required}"
FIGURE_OUTPUT_PATH="${FIGURE_OUTPUT_PATH:?FIGURE_OUTPUT_PATH is required}"

# Runtime validation and logging.
hwa_start_log "dyn_net_spatial_composites"
hwa_validate_artifact_paths EVENT_FEATURES_PATH DAILY_DIR CLIMATOLOGY_PATH COMPOSITE_OUTPUT_PATH FIGURE_OUTPUT_PATH

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${EVENT_FEATURES_PATH}"
test -d "${DAILY_DIR}"
test -s "${CLIMATOLOGY_PATH}"
test ! -e "${COMPOSITE_OUTPUT_PATH}"
test ! -e "${FIGURE_OUTPUT_PATH}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

cd "${PROJECT_ROOT}"
echo "[info] $(date -Is) starting daily dynamical-sign composites on $(hostname)"
/usr/bin/time -v python scripts/spatial_composites/build_dyn_net_spatial_composites.py \
    --event-features-path "${EVENT_FEATURES_PATH}" \
    --daily-dir "${DAILY_DIR}" \
    --climatology-path "${CLIMATOLOGY_PATH}" \
    --output-path "${COMPOSITE_OUTPUT_PATH}"
/usr/bin/time -v python scripts/spatial_composites/plot_dyn_net_spatial_composites.py \
    --input-path "${COMPOSITE_OUTPUT_PATH}" \
    --output-path "${FIGURE_OUTPUT_PATH}"
test -s "${COMPOSITE_OUTPUT_PATH}"
test -s "${FIGURE_OUTPUT_PATH}"
echo "[info] $(date -Is) done"
