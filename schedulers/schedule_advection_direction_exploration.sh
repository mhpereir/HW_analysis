#!/bin/bash
#PBS -N advection_direction_exploration
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=01:00:00
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
LOGFILE="${LOG_DIR}/${PBS_JOBID}_advection_direction_exploration.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

REGION="pnw_bartusek"
BOTTOM_BOUNDARY="surface"
TOP_BOUNDARY=700
THRESHOLD_VARIABLE="tas"
QUANTILE=90
TIME_START=1940
TIME_END=2024
EHB_TIME_START=1940
EHB_TIME_END=2025

STAGE1_INPUT="${PROJECT_ROOT}/results/stage1/harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc"
EXPLORATION_DIR="${PROJECT_ROOT}/results/stage1/advection_direction_exploration"
STAGE1_OUTPUT="${EXPLORATION_DIR}/harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc"
PLOT_OUTPUT="${PROJECT_ROOT}/results/plots_advection_direction_exploration/region_pnw_bartusek/boundary_surface_700hPa/time_range_1940_2024/advection_face_contributions.png"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(python -c 'import sys; print(sys.executable)')"
echo "[info] stage1_input=${STAGE1_INPUT}"
echo "[info] stage1_output=${STAGE1_OUTPUT}"
echo "[info] plot_output=${PLOT_OUTPUT}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}/scripts"

/usr/bin/time -v python build_stage1_advection_exploration.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --start-year-ehb "${EHB_TIME_START}" \
    --end-year-ehb "${EHB_TIME_END}" \
    --input-path "${STAGE1_INPUT}" \
    --output-path "${STAGE1_OUTPUT}"

/usr/bin/time -v python plot_advection_direction_exploration.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --input-path "${STAGE1_OUTPUT}" \
    --output-path "${PLOT_OUTPUT}" \
    --window-days 7 \
    --ratio-epsilon 0.005 \
    --season-months 6 7 8 \
    --require-full-event

echo "[info] finished=$(date -Is)"
