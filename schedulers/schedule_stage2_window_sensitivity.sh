#!/bin/bash
#PBS -N stage2_window_sensitivity
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
INTEGRATION_HOURS="${INTEGRATION_HOURS:?INTEGRATION_HOURS is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"
hwa_start_log "stage2_window_sensitivity"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_DIR
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${OUTPUT_DIR}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error MPLBACKEND=Agg
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID} host=$(hostname) commit=${actual_commit}"
echo "[info] python=$(command -v python)"
echo "[info] input_path=${INPUT_PATH} output_dir=${OUTPUT_DIR} integration_hours=${INTEGRATION_HOURS}"
echo "[info] started=$(date -Is)"
cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/event_features/run_window_sensitivity.py \
    --input-path "${INPUT_PATH}" --output-dir "${OUTPUT_DIR}" \
    --integration-hours "${INTEGRATION_HOURS}"
test -s "${OUTPUT_DIR}/manifest.json"
echo "[info] finished=$(date -Is)"
