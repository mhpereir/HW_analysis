#!/bin/bash
#PBS -N advection_top_events
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
REGION="${REGION:?REGION is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
BOTTOM_BOUNDARY="${BOTTOM_BOUNDARY:-surface}"
TOP_BOUNDARY="${TOP_BOUNDARY:-700}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"
QUANTILE="${QUANTILE:-90}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
TOP_N="${TOP_N:-10}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-24}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${OUTPUT_DIR}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_DIR}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_advection_top_events.log"
exec > >(tee -a "${LOGFILE}") 2>&1

STAGING_DIR="${OUTPUT_DIR}.staging.${PBS_JOBID}"
test ! -e "${STAGING_DIR}"
cleanup_staging() {
    if test -d "${STAGING_DIR}"; then
        find "${STAGING_DIR}" -maxdepth 1 -type f -name '*.png' -delete
        rmdir "${STAGING_DIR}" 2>/dev/null || true
    fi
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
echo "[info] conda_prefix=${CONDA_PREFIX}"
echo "[info] python=${python_executable}"
echo "[info] region=${REGION}"
echo "[info] boundaries=${BOTTOM_BOUNDARY}-${TOP_BOUNDARY}hPa"
echo "[info] threshold=${THRESHOLD_VARIABLE}_q${QUANTILE}"
echo "[info] years=${TIME_START}-${TIME_END}"
echo "[info] top_n=${TOP_N}"
echo "[info] window_days=${WINDOW_DAYS}"
echo "[info] smoothing_window=${SMOOTHING_WINDOW}"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] input_sha256=$(sha256sum "${INPUT_PATH}" | cut -d ' ' -f 1)"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] staging_dir=${STAGING_DIR}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v "${python_executable}" \
  scripts/plot_advection_direction_exploration_top_events.py \
  --region "${REGION}" \
  --bottom-boundary "${BOTTOM_BOUNDARY}" \
  --top-boundary "${TOP_BOUNDARY}" \
  --threshold-variable "${THRESHOLD_VARIABLE}" \
  --quantile "${QUANTILE}" \
  --start-year "${TIME_START}" \
  --end-year "${TIME_END}" \
  --input-path "${INPUT_PATH}" \
  --output-dir "${STAGING_DIR}" \
  --top-n "${TOP_N}" \
  --window-days "${WINDOW_DAYS}" \
  --smoothing-window "${SMOOTHING_WINDOW}" \
  --season-months 6 7 8 \
  --require-full-event

test -d "${STAGING_DIR}"
expected_png_count=$((2 * TOP_N))
actual_png_count=$(find "${STAGING_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)
test "${actual_png_count}" -eq "${expected_png_count}"
empty_png_count=$(find "${STAGING_DIR}" -maxdepth 1 -type f -name '*.png' -size 0c | wc -l)
test "${empty_png_count}" -eq 0
test ! -e "${OUTPUT_DIR}"
mv --no-clobber -T "${STAGING_DIR}" "${OUTPUT_DIR}"
test ! -e "${STAGING_DIR}"
trap - EXIT
test -d "${OUTPUT_DIR}"
echo "[info] output_sha256:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.png' -print0 \
  | sort -z \
  | xargs -0 sha256sum
echo "[info] finished=$(date -Is)"
