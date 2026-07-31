#!/bin/bash
#PBS -N stage1_global_cloud
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be supplied by the submission workflow}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be supplied by the submission workflow}"
REGION="${REGION:?REGION must be supplied by the submission workflow}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH must be supplied by the submission workflow}"

TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
QUANTILE="${QUANTILE:-90}"
CLOUD_COVER_ROOT="${CLOUD_COVER_ROOT:-/home/mhpereir/downloads-mhpereir/REANALYSIS/ERA5/hourly/cloud_cover}"
LOG_DIR="${LOG_DIR:-/home/mhpereir/HW_analysis/logs}"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-dev_env}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] checkout commit ${actual_commit} does not match ${EXPECTED_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "[error] runtime checkout is dirty: ${PROJECT_ROOT}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID:-manual}_stage1_${REGION}_global_cloud.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV}"
PYTHON_EXECUTABLE=$(command -v python)

cd "${PROJECT_ROOT}"

echo "[info] $(date -Is) starting Stage 1 build on host $(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] expected_commit=${EXPECTED_COMMIT}"
echo "[info] python=${PYTHON_EXECUTABLE}"
echo "[info] region=${REGION}"
echo "[info] cloud_cover_source_layout=global-hourly-grid"
echo "[info] cloud_cover_root=${CLOUD_COVER_ROOT}"
echo "[info] output_path=${OUTPUT_PATH}"

/usr/bin/time -v "${PYTHON_EXECUTABLE}" scripts/build_stage1_harmonized_timeseries.py \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --quantile "${QUANTILE}" \
    --region "${REGION}" \
    --bottom-boundary "surface" \
    --top-boundary 700 \
    --start-year-ehb 1940 \
    --end-year-ehb 2025 \
    --threshold-variable "tas" \
    --add-full-diagnostics \
    --cloud-cover-source-layout "global-hourly-grid" \
    --cloud-cover-root "${CLOUD_COVER_ROOT}" \
    --output-path "${OUTPUT_PATH}"

echo "[info] $(date -Is) done"
