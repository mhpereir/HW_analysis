#!/bin/bash
#PBS -N stage1_pressure_layer
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
REFERENCE_PATH="${REFERENCE_PATH:?REFERENCE_PATH is required}"
REFERENCE_SHA256="${REFERENCE_SHA256:?REFERENCE_SHA256 is required}"
HEAT_BUDGET_MANIFEST="${HEAT_BUDGET_MANIFEST:?HEAT_BUDGET_MANIFEST is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
LOG_DIR="${LOG_DIR:?LOG_DIR is required}"
REGION="${REGION:?REGION is required}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:?THRESHOLD_VARIABLE is required}"
BOTTOM_HPA="${BOTTOM_HPA:?BOTTOM_HPA is required}"
TOP_HPA="${TOP_HPA:?TOP_HPA is required}"
START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2024}"
QUANTILE="${QUANTILE:-90}"
VALIDATE_ALL_SOURCE_YEARS="${VALIDATE_ALL_SOURCE_YEARS:-1}"

test "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/${PBS_JOBID}_stage1_pressure_layer.log") 2>&1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
cd "${PROJECT_ROOT}"
echo "[start] $(date -Is) job=${PBS_JOBID} host=$(hostname) commit=${EXPECTED_COMMIT}"
echo "[python] $(command -v python)"
echo "[inputs] reference=${REFERENCE_PATH} ehb_manifest=${HEAT_BUDGET_MANIFEST}"
echo "[output] ${OUTPUT_DIR}"
SOURCE_VALIDATION_ARGS=()
case "${VALIDATE_ALL_SOURCE_YEARS}" in
    1) SOURCE_VALIDATION_ARGS=(--validate-all-source-years) ;;
    0) ;;
    *) echo "Invalid VALIDATE_ALL_SOURCE_YEARS" >&2; exit 2 ;;
esac
/usr/bin/time -v python -W error scripts/build_stage1_pressure_layer.py \
    --reference "${REFERENCE_PATH}" --reference-sha256 "${REFERENCE_SHA256}" \
    --heat-budget-manifest "${HEAT_BUDGET_MANIFEST}" \
    --region "${REGION}" --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" --bottom-hpa "${BOTTOM_HPA}" --top-hpa "${TOP_HPA}" \
    --start-year "${START_YEAR}" --end-year "${END_YEAR}" \
    "${SOURCE_VALIDATION_ARGS[@]}" --output-dir "${OUTPUT_DIR}"
echo "[done] $(date -Is) job=${PBS_JOBID}"
