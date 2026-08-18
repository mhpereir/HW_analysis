#!/bin/bash
#PBS -N composite_timeseries_all
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
REGION="${REGION:-pnw_bartusek}"
BOTTOM_BOUNDARY="${BOTTOM_BOUNDARY:-surface}"
TOP_BOUNDARY="${TOP_BOUNDARY:-700}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"
QUANTILE="${QUANTILE:-90}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-24}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
SMOOTHED_OUTPUT_PATH="${OUTPUT_PATH%.*}_smoothed.${OUTPUT_PATH##*.}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${OUTPUT_PATH}"
test ! -e "${SMOOTHED_OUTPUT_PATH}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_composite_timeseries_all.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(command -v python)"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/plot_composite_timeseries_all.py \
  --region "${REGION}" \
  --bottom-boundary "${BOTTOM_BOUNDARY}" \
  --top-boundary "${TOP_BOUNDARY}" \
  --threshold-variable "${THRESHOLD_VARIABLE}" \
  --quantile "${QUANTILE}" \
  --start-year "${TIME_START}" \
  --end-year "${TIME_END}" \
  --input-path "${INPUT_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --window-days "${WINDOW_DAYS}" \
  --smoothing-window "${SMOOTHING_WINDOW}" \
  --season-months 6 7 8 \
  --require-full-event \
  --plot-extended-variables

test -s "${OUTPUT_PATH}"
test -s "${SMOOTHED_OUTPUT_PATH}"
echo "[info] finished=$(date -Is)"
