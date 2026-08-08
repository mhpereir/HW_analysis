#!/bin/bash
#PBS -N composite_split_clim_anom
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
SPLIT_QUANTILE="${SPLIT_QUANTILE:-0.75}"
SPLIT_YEAR="${SPLIT_YEAR:-1982}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
OUTPUT_DIRECTORY="$(dirname "${OUTPUT_PATH}")"
OUTPUT_FILENAME="$(basename "${OUTPUT_PATH}")"
OUTPUT_STEM="${OUTPUT_FILENAME%.*}"
OUTPUT_SUFFIX="${OUTPUT_FILENAME##*.}"

split_variable_list=(
  "duration"
  "tas_anom_peak"
  "tas_excess_integral"
  "tas_excess_peak"
  "tas_peak"
)

split_output_path() {
  local split_variable="$1"
  printf '%s/%s_%s.%s\n' \
    "${OUTPUT_DIRECTORY}" \
    "${OUTPUT_STEM}" \
    "${split_variable}" \
    "${OUTPUT_SUFFIX}"
}

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test -s "${CLIMATOLOGY_PATH}"
for split_variable in "${split_variable_list[@]}" peak_time; do
  derived_output_path="$(split_output_path "${split_variable}")"
  test ! -e "${derived_output_path}"
  test ! -e "${derived_output_path%.*}_smoothed.${OUTPUT_SUFFIX}"
done

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_composite_split_clim_anom.log"
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
echo "[info] climatology_path=${CLIMATOLOGY_PATH}"
echo "[info] output_base_path=${OUTPUT_PATH}"
echo "[info] split_quantile=${SPLIT_QUANTILE}"
echo "[info] split_year=${SPLIT_YEAR}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
for split_variable in "${split_variable_list[@]}"; do
  echo "[info] split_variable=${split_variable}"
  /usr/bin/time -v python scripts/plot_composite_timeseries_split_clim_anom.py \
    --region pnw_bartusek \
    --bottom-boundary surface \
    --top-boundary 700 \
    --threshold-variable tas \
    --quantile 90 \
    --start-year 1940 \
    --end-year 2024 \
    --input-path "${INPUT_PATH}" \
    --climatology-path "${CLIMATOLOGY_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --window-days 7 \
    --split-variable "${split_variable}" \
    --split-quantiles "${SPLIT_QUANTILE}" \
    --season-months 6 7 8 \
    --require-full-event \
    --plot-extended-variables
  derived_output_path="$(split_output_path "${split_variable}")"
  test -s "${derived_output_path}"
  test -s "${derived_output_path%.*}_smoothed.${OUTPUT_SUFFIX}"
done

echo "[info] split_variable=peak_time"
/usr/bin/time -v python scripts/plot_composite_timeseries_split_clim_anom.py \
  --region pnw_bartusek \
  --bottom-boundary surface \
  --top-boundary 700 \
  --threshold-variable tas \
  --quantile 90 \
  --start-year 1940 \
  --end-year 2024 \
  --input-path "${INPUT_PATH}" \
  --climatology-path "${CLIMATOLOGY_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --window-days 7 \
  --split-variable peak_time \
  --split-years "${SPLIT_YEAR}" \
  --season-months 6 7 8 \
  --require-full-event \
  --plot-extended-variables
derived_output_path="$(split_output_path peak_time)"
test -s "${derived_output_path}"
test -s "${derived_output_path%.*}_smoothed.${OUTPUT_SUFFIX}"
echo "[info] finished=$(date -Is)"
