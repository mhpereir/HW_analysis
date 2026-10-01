#!/bin/bash
#PBS -N plot_diurnal_cycle
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

REGION="${REGION:-pnw_bartusek}"
BOTTOM_BOUNDARY="${BOTTOM_BOUNDARY:-surface}"
TOP_BOUNDARY="${TOP_BOUNDARY:-700}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"
QUANTILE="${QUANTILE:-90}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"

INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
COMPOSITE_OUTPUT_PATH="${COMPOSITE_OUTPUT_PATH:?COMPOSITE_OUTPUT_PATH is required}"
VALIDATION_OUTPUT_PATH="${VALIDATION_OUTPUT_PATH:?VALIDATION_OUTPUT_PATH is required}"

# Runtime validation and logging.
hwa_start_log "plot_diurnal_cycle"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_PATH COMPOSITE_OUTPUT_PATH VALIDATION_OUTPUT_PATH
EXPECTED_INPUT_SHA256="${EXPECTED_INPUT_SHA256:?EXPECTED_INPUT_SHA256 is required}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${OUTPUT_PATH}"
test ! -L "${OUTPUT_PATH}"
test ! -e "${COMPOSITE_OUTPUT_PATH}"
test ! -L "${COMPOSITE_OUTPUT_PATH}"
test "${OUTPUT_PATH}" != "${COMPOSITE_OUTPUT_PATH}"
test ! -e "${VALIDATION_OUTPUT_PATH}"
test ! -L "${VALIDATION_OUTPUT_PATH}"
test "${VALIDATION_OUTPUT_PATH}" != "${OUTPUT_PATH}"
test "${VALIDATION_OUTPUT_PATH}" != "${COMPOSITE_OUTPUT_PATH}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
python -c 'import sys; print("[info] python=" + sys.executable)'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "${PROJECT_ROOT}/scripts"

echo "[info] $(date -Is) starting plot generation on host $(hostname)"
echo "[info] job=${PBS_JOBID} commit=${actual_commit}"
echo "[info] input=${INPUT_PATH} figure=${OUTPUT_PATH} composite=${COMPOSITE_OUTPUT_PATH}"
/usr/bin/time -v python plot_diurnal_cycle.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --composite-output-path "${COMPOSITE_OUTPUT_PATH}" \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --season-months 6 7 8 \
    --local-utc-offset-hours 0
python validate_diurnal_cycle.py \
    --input-path "${INPUT_PATH}" \
    --composite-path "${COMPOSITE_OUTPUT_PATH}" \
    --figure-path "${OUTPUT_PATH}" \
    --expected-input-sha256 "${EXPECTED_INPUT_SHA256}" \
    --output-path "${VALIDATION_OUTPUT_PATH}"
echo "[info] $(date -Is) done"
