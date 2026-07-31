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

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_advection_direction_exploration.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

INPUT_PATH="${PROJECT_ROOT}/results/stage1/advection_direction_exploration/harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc"
OUTPUT_PATH="${PROJECT_ROOT}/results/plots_advection_direction_exploration/region_pnw_bartusek/boundary_surface_700hPa/time_range_1940_2024/advection_face_contributions_two_panel.png"

test -s "${INPUT_PATH}"
if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "[error] refusing to overwrite existing output: ${OUTPUT_PATH}" >&2
    exit 1
fi

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(python -c 'import sys; print(sys.executable)')"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}/scripts"
/usr/bin/time -v python plot_advection_direction_exploration.py \
  --region pnw_bartusek \
  --bottom-boundary surface \
  --top-boundary 700 \
  --threshold-variable tas \
  --quantile 90 \
  --start-year 1940 \
  --end-year 2024 \
  --input-path "${INPUT_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --window-days 7 \
  --season-months 6 7 8 \
  --require-full-event

test -s "${OUTPUT_PATH}"
echo "[info] finished=$(date -Is)"
