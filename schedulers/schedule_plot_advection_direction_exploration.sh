#!/bin/bash
#PBS -N plot_advection_direction
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
REGION="${REGION:?REGION is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-24}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
SMOOTHED_OUTPUT_PATH="${OUTPUT_PATH%.*}_smoothed.${OUTPUT_PATH##*.}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test "${OUTPUT_PATH##*.}" = "png"
test ! -e "${OUTPUT_PATH}"
test ! -e "${SMOOTHED_OUTPUT_PATH}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_advection_direction_exploration.log"
exec > >(tee -a "${LOGFILE}") 2>&1

STAGING_DIR="${OUTPUT_PATH}.staging.${PBS_JOBID}"
test ! -e "${STAGING_DIR}"
mkdir "${STAGING_DIR}"
STAGED_OUTPUT="${STAGING_DIR}/$(basename "${OUTPUT_PATH}")"
STAGED_SMOOTHED_OUTPUT="${STAGED_OUTPUT%.*}_smoothed.${STAGED_OUTPUT##*.}"
published_outputs=()
cleanup_staging() {
    rm -f -- "${STAGED_OUTPUT}" "${STAGED_SMOOTHED_OUTPUT}"
    for published_output in "${published_outputs[@]}"; do
        rm -f -- "${published_output}"
    done
    rmdir "${STAGING_DIR}" 2>/dev/null || true
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
python_executable=$(command -v python)
test -x "${python_executable}"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] environment=${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] python=${python_executable}"
echo "[info] region=${REGION}"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] input_sha256=$(sha256sum "${INPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] smoothed_output_path=${SMOOTHED_OUTPUT_PATH}"
echo "[info] smoothing_window=${SMOOTHING_WINDOW}"
echo "[info] staged_output=${STAGED_OUTPUT}"
echo "[info] staged_smoothed_output=${STAGED_SMOOTHED_OUTPUT}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v "${python_executable}" \
  scripts/plot_advection_direction_exploration.py \
  --region "${REGION}" \
  --bottom-boundary surface \
  --top-boundary 700 \
  --threshold-variable tas \
  --quantile 90 \
  --start-year 1940 \
  --end-year 2024 \
  --input-path "${INPUT_PATH}" \
  --output-path "${STAGED_OUTPUT}" \
  --window-days 7 \
  --smoothing-window "${SMOOTHING_WINDOW}" \
  --season-months 6 7 8 \
  --require-full-event

test -s "${STAGED_OUTPUT}"
test -s "${STAGED_SMOOTHED_OUTPUT}"
test ! -e "${OUTPUT_PATH}"
test ! -e "${SMOOTHED_OUTPUT_PATH}"
mv --no-clobber "${STAGED_OUTPUT}" "${OUTPUT_PATH}"
published_outputs+=("${OUTPUT_PATH}")
mv --no-clobber "${STAGED_SMOOTHED_OUTPUT}" "${SMOOTHED_OUTPUT_PATH}"
published_outputs+=("${SMOOTHED_OUTPUT_PATH}")
rmdir "${STAGING_DIR}"
trap - EXIT
test -s "${OUTPUT_PATH}"
test -s "${SMOOTHED_OUTPUT_PATH}"
echo "[info] output_sha256=$(sha256sum "${OUTPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] smoothed_output_sha256=$(sha256sum "${SMOOTHED_OUTPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] finished=$(date -Is)"
