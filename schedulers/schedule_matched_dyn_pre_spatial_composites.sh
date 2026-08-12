#!/bin/bash
#PBS -N matched_idyn_spatial
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EVENT_FEATURES_PATH="${EVENT_FEATURES_PATH:-${PROJECT_ROOT}/results/stage2_event_features/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc}"
DAILY_DIR="${DAILY_DIR:-${PROJECT_ROOT}/results/spatial_composites/daily}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-${PROJECT_ROOT}/results/spatial_composites/climatology/era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc}"
MATCHING_SETTINGS_PATH="${MATCHING_SETTINGS_PATH:-${PROJECT_ROOT}/scripts/Idyn_matching_exploration/matching_settings.json}"
MATCHING_SPECIFICATION="${MATCHING_SPECIFICATION:-peak_anomaly_0p20}"
COMPOSITE_OUTPUT_PATH="${COMPOSITE_OUTPUT_PATH:-${PROJECT_ROOT}/results/spatial_composites/matched_dyn_pre_daily_spatial_composites_pnw_bartusek_tas_q90_1940_2024_peak_anomaly_0p20.nc}"
FIGURE_OUTPUT_PATH="${FIGURE_OUTPUT_PATH:-${PROJECT_ROOT}/results/spatial_composites/matched_dyn_pre_daily_t2m_z500_composites_pnw_bartusek_tas_q90_1940_2024_peak_anomaly_0p20.png}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(realpath "${PBS_O_WORKDIR}")" = "$(realpath "${PROJECT_ROOT}")"
test -s "${EVENT_FEATURES_PATH}"
test -d "${DAILY_DIR}"
test -s "${CLIMATOLOGY_PATH}"
test -s "${MATCHING_SETTINGS_PATH}"
test "${COMPOSITE_OUTPUT_PATH##*.}" = "nc"
test "${FIGURE_OUTPUT_PATH##*.}" = "png"
test ! -e "${COMPOSITE_OUTPUT_PATH}"
test ! -e "${FIGURE_OUTPUT_PATH}"

mkdir -p \
    "${LOG_DIR}" \
    "$(dirname "${COMPOSITE_OUTPUT_PATH}")" \
    "$(dirname "${FIGURE_OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_matched_dyn_pre_spatial_composites.log"
exec > >(tee -a "${LOGFILE}") 2>&1

COMPOSITE_STAGING_DIR="${COMPOSITE_OUTPUT_PATH}.staging.${PBS_JOBID}"
FIGURE_STAGING_DIR="${FIGURE_OUTPUT_PATH}.staging.${PBS_JOBID}"
test ! -e "${COMPOSITE_STAGING_DIR}"
test ! -e "${FIGURE_STAGING_DIR}"
mkdir "${COMPOSITE_STAGING_DIR}" "${FIGURE_STAGING_DIR}"
STAGED_COMPOSITE="${COMPOSITE_STAGING_DIR}/$(basename "${COMPOSITE_OUTPUT_PATH}")"
STAGED_FIGURE="${FIGURE_STAGING_DIR}/$(basename "${FIGURE_OUTPUT_PATH}")"
cleanup_staging() {
    rm -f -- "${STAGED_COMPOSITE}" "${STAGED_FIGURE}"
    rmdir "${COMPOSITE_STAGING_DIR}" 2>/dev/null || true
    rmdir "${FIGURE_STAGING_DIR}" 2>/dev/null || true
}
trap cleanup_staging EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=error
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(command -v python)"
echo "[info] event_features_path=${EVENT_FEATURES_PATH}"
echo "[info] event_features_sha256=$(sha256sum "${EVENT_FEATURES_PATH}" | cut -d ' ' -f 1)"
echo "[info] daily_dir=${DAILY_DIR}"
echo "[info] climatology_path=${CLIMATOLOGY_PATH}"
echo "[info] matching_settings_path=${MATCHING_SETTINGS_PATH}"
echo "[info] matching_settings_sha256=$(sha256sum "${MATCHING_SETTINGS_PATH}" | cut -d ' ' -f 1)"
echo "[info] matching_specification=${MATCHING_SPECIFICATION}"
echo "[info] composite_output_path=${COMPOSITE_OUTPUT_PATH}"
echo "[info] figure_output_path=${FIGURE_OUTPUT_PATH}"
echo "[info] started=$(date -Is)"

/usr/bin/time -v python \
    scripts/spatial_composites/build_matched_dyn_pre_spatial_composites.py \
    --event-features-path "${EVENT_FEATURES_PATH}" \
    --daily-dir "${DAILY_DIR}" \
    --climatology-path "${CLIMATOLOGY_PATH}" \
    --matching-settings-path "${MATCHING_SETTINGS_PATH}" \
    --matching-specification "${MATCHING_SPECIFICATION}" \
    --output-path "${STAGED_COMPOSITE}"

/usr/bin/time -v python \
    scripts/spatial_composites/plot_matched_dyn_pre_spatial_composites.py \
    --input-path "${STAGED_COMPOSITE}" \
    --output-path "${STAGED_FIGURE}" \
    --matching-specification "${MATCHING_SPECIFICATION}" \
    --plot-lags -2 0 2

test -s "${STAGED_COMPOSITE}"
test -s "${STAGED_FIGURE}"
test ! -e "${COMPOSITE_OUTPUT_PATH}"
test ! -e "${FIGURE_OUTPUT_PATH}"
mv "${STAGED_COMPOSITE}" "${COMPOSITE_OUTPUT_PATH}"
mv "${STAGED_FIGURE}" "${FIGURE_OUTPUT_PATH}"
rmdir "${COMPOSITE_STAGING_DIR}" "${FIGURE_STAGING_DIR}"
trap - EXIT

echo "[info] composite_sha256=$(sha256sum "${COMPOSITE_OUTPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] figure_sha256=$(sha256sum "${FIGURE_OUTPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] finished=$(date -Is)"
