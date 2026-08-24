#!/bin/bash
#PBS -N plot_inter_region_budget
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EVENT_INPUT_ROOT="${EVENT_INPUT_ROOT:?EVENT_INPUT_ROOT is required}"
BASELINE_INPUT_ROOT="${BASELINE_INPUT_ROOT:?BASELINE_INPUT_ROOT is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"
QUANTILE_THRESHOLD="${QUANTILE_THRESHOLD:-q90}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
FIGURE_TITLE="${FIGURE_TITLE:-Regional Pre-Peak Heat-Budget Composition}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -d "${EVENT_INPUT_ROOT}"
test -d "${BASELINE_INPUT_ROOT}"
test ! -e "${OUTPUT_PATH}"

regions=(
  alaska
  central_china
  eastern_canada
  gulf_usa
  pnw_bartusek
  pnw_hotz
  western_eu
)
plot_arguments=()
for region in "${regions[@]}"; do
  event_path="${EVENT_INPUT_ROOT}/hw_event_features_fixed_windows_${region}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}_${TIME_START}_${TIME_END}.nc"
  baseline_path="${BASELINE_INPUT_ROOT}/non_event_day_features_fixed_windows_${region}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}.nc"
  test -s "${event_path}"
  test -s "${baseline_path}"
  plot_arguments+=(--region-input "${region}" "${event_path}" "${baseline_path}")
done

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_inter_region_budget_fractions.log"
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
echo "[info] event_input_root=${EVENT_INPUT_ROOT}"
echo "[info] baseline_input_root=${BASELINE_INPUT_ROOT}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] regions=${regions[*]}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/event_features/plot_inter_region_budget_fractions.py \
  "${plot_arguments[@]}" \
  --output-path "${OUTPUT_PATH}" \
  --title "${FIGURE_TITLE}"

test -s "${OUTPUT_PATH}"
echo "[info] finished=$(date -Is)"
