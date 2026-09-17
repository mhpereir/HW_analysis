#!/bin/bash
#PBS -N stage_2_plot_event_vs_baseline
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
QUANTILE_THRESHOLD="q90"
TIME_START=1940
TIME_END=2024

INPUT_PATH="${INPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_baseline_features/non_event_day_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}.nc}"
EVENT_INPUT_PATH="${EVENT_INPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_event_features/hw_event_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}_${TIME_START}_${TIME_END}.nc}"
OUTPUT_PATH="${OUTPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_baseline_features/diagnostics/${REGION}/${THRESHOLD_VARIABLE}/event_vs_clean_baseline_diabatic_advection_scatter.png}"
POINT_SIZE=24.0
ALPHA=0.2
EVENT_POINT_SIZE=24.0
EVENT_ALPHA=0.7

# Runtime validation and logging.
hwa_start_log "plot_event_vs_baseline"
hwa_validate_artifact_paths INPUT_PATH EVENT_INPUT_PATH OUTPUT_PATH

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test -s "${EVENT_INPUT_PATH}"
test ! -e "${OUTPUT_PATH}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

cd "${PROJECT_ROOT}/scripts"

echo "[info] $(date -Is) starting event-versus-clean-baseline plot generation on host $(hostname)"
/usr/bin/time -v python event_features/plot_adiabatic_diabatic_advection_baseline.py \
    --input-path "${INPUT_PATH}" \
    --event-input-path "${EVENT_INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --point-size "${POINT_SIZE}" \
    --alpha "${ALPHA}" \
    --event-point-size "${EVENT_POINT_SIZE}" \
    --event-alpha "${EVENT_ALPHA}"
echo "[info] $(date -Is) done"
