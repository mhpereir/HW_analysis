#!/bin/bash
#PBS -N adv_direction_smoke
#PBS -l select=1:ncpus=4:mem=24gb
#PBS -l walltime=00:45:00
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
LOGFILE="${LOG_DIR}/${PBS_JOBID}_advection_direction_smoke.log"
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
TIME_START=2024
TIME_END=2024
EHB_TIME_START=1940
EHB_TIME_END=2025
LEGACY_CLOUD_ROOT="/home/mhpereir/data-mhpereir/arco_era5/CloudCover_download/outputs"

SMOKE_DIR="${PROJECT_ROOT}/results/stage1/advection_direction_exploration/smoke_2024"
STAGE1_BASE_PATH="${SMOKE_DIR}/base_stage1/harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_2024_2024.nc"
STAGE1_ENHANCED_PATH="${SMOKE_DIR}/harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_2024_2024.nc"
PLOT_OUTPUT_PATH="${PROJECT_ROOT}/results/plots_advection_direction_exploration/smoke_2024/advection_face_contributions_three_panel.png"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(python -c 'import sys; print(sys.executable)')"
echo "[info] started=$(date -Is)"

export PROJECT_ROOT
export REGION
export BOTTOM_BOUNDARY
export TOP_BOUNDARY
export THRESHOLD_VARIABLE
export QUANTILE
export TIME_START
export TIME_END
export EHB_TIME_START
export EHB_TIME_END
export LEGACY_CLOUD_ROOT
export STAGE1_BASE_PATH
export STAGE1_ENHANCED_PATH
export PLOT_OUTPUT_PATH

"${PROJECT_ROOT}/scripts/run_advection_direction_pipeline.sh"

echo "[info] finished=$(date -Is)"
