#!/bin/bash
#PBS -N stage_2_plot_event_feature_split
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

REGION="pnw_bartusek"
THRESHOLD_VARIABLE="tas"
QUANTILE=90
TIME_START=1940
TIME_END=2024
SELECTION_VARIABLE="duration"
SELECTION_QUANTILE=0.9

# f_adiabatic_pre
# f_diabatic_pre
# f_advection_pre
# sqrt_I_lwa_a_pre_peak
# T_anom_mean_ant
# cos_days_from_solstice
# duration
# tas_anom_peak
# tas_peak

INPUT_PATH="${INPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_event_features/hw_event_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_q${QUANTILE}_${TIME_START}_${TIME_END}.nc}"
OUTPUT_PATH="${OUTPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_event_features/diagnostics/${REGION}/event_feature_tendency_scatter.png}"

# Runtime validation and logging.
hwa_start_log "plot_event_feature_split"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_PATH

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${OUTPUT_PATH}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

cd "${PROJECT_ROOT}/scripts"

echo "[info] $(date -Is) starting split event-feature plot generation on host $(hostname)"
/usr/bin/time -v python event_features/plot_event_feature_split.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --selection-variable "${SELECTION_VARIABLE}" \
    --selection-quantile "${SELECTION_QUANTILE}"
echo "[info] $(date -Is) done"
