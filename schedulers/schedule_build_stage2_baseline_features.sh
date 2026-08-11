#!/bin/bash
#PBS -N stage_2_baseline_features
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=00:10:00
#PBS -J 0-1
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
REGION="${REGION:-pnw_bartusek}"
THRESHOLD_VARIABLES=(tas lwa_a)
ARRAY_INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required}"
if [[ ! "${ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "[error] PBS_ARRAY_INDEX must be a non-negative integer: ${ARRAY_INDEX}" >&2
    exit 1
fi
if ((ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#THRESHOLD_VARIABLES[@]})); then
    echo "[error] unsupported PBS_ARRAY_INDEX: ${ARRAY_INDEX}" >&2
    exit 1
fi
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLES[ARRAY_INDEX]}"
QUANTILE_THRESHOLD="${QUANTILE_THRESHOLD:-q90}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/results/stage1/harmonized_regional_timeseries_${REGION}_surface_700hPa_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}_${TIME_START}_${TIME_END}.nc}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/stage2_baseline_features/non_event_day_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}.nc}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
STAGED_OUTPUT_PATH="${OUTPUT_PATH}.tmp.${PBS_JOBID}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test ! -e "${STAGED_OUTPUT_PATH}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_stage2_baseline_features.log"
exec > >(tee -a "${LOGFILE}") 2>&1
trap 'rm -f -- "${STAGED_OUTPUT_PATH}"' EXIT

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
echo "[info] threshold_variable=${THRESHOLD_VARIABLE}"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/event_features/build_stage2_baseline_features.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${STAGED_OUTPUT_PATH}" \
    --season-months 6 7 8

test -s "${STAGED_OUTPUT_PATH}"
mv -f -- "${STAGED_OUTPUT_PATH}" "${OUTPUT_PATH}"
test -s "${OUTPUT_PATH}"
echo "[info] finished=$(date -Is)"
