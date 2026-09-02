#!/bin/bash
#PBS -N plot_adiabatic_advection_base
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
BASELINE_INPUT_PATH="${BASELINE_INPUT_PATH:?BASELINE_INPUT_PATH is required}"
EVENT_INPUT_PATH="${EVENT_INPUT_PATH:?EVENT_INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
COLOR_VARIABLE="${COLOR_VARIABLE:-tas_anom_peak}"
LAYOUT="${LAYOUT:-full}"
POINT_SIZE="${POINT_SIZE:-24.0}"
ALPHA="${ALPHA:-0.2}"
EVENT_POINT_SIZE="${EVENT_POINT_SIZE:-24.0}"
EVENT_ALPHA="${EVENT_ALPHA:-0.7}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${BASELINE_INPUT_PATH}"
test -s "${EVENT_INPUT_PATH}"
test ! -e "${OUTPUT_PATH}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_adiabatic_advection_comparison_baseline.log"
exec > >(tee -a "${LOGFILE}") 2>&1

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
echo "[info] baseline_input_path=${BASELINE_INPUT_PATH}"
echo "[info] event_input_path=${EVENT_INPUT_PATH}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] color_variable=${COLOR_VARIABLE}"
echo "[info] layout=${LAYOUT}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/event_features/plot_adiabatic_advection_comparison_baseline.py \
  --input-path "${BASELINE_INPUT_PATH}" \
  --event-input-path "${EVENT_INPUT_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --color-variable "${COLOR_VARIABLE}" \
  --layout "${LAYOUT}" \
  --point-size "${POINT_SIZE}" \
  --alpha "${ALPHA}" \
  --event-point-size "${EVENT_POINT_SIZE}" \
  --event-alpha "${EVENT_ALPHA}"

test -s "${OUTPUT_PATH}"
echo "[info] finished=$(date -Is)"
