#!/bin/bash
#PBS -N stage2_campaign_validation
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
hwa_start_log "stage2_campaign_validation"
hwa_validate_artifact_paths MANIFEST_PATH OUTPUT_DIR
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${MANIFEST_PATH}"
test ! -e "${OUTPUT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID} host=$(hostname) commit=${actual_commit}"
echo "[info] manifest=${MANIFEST_PATH} output=${OUTPUT_DIR} python=$(command -v python)"
echo "[info] started=$(date -Is)"
cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/event_features/validate_stage2_campaign.py \
    --manifest "${MANIFEST_PATH}" --output-dir "${OUTPUT_DIR}"
test -s "${OUTPUT_DIR}/validation.json"
echo "[info] finished=$(date -Is)"
