#!/bin/bash
#PBS -N temperature_variability
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
HOURLY_T2M_PATH="${HOURLY_T2M_PATH:?HOURLY_T2M_PATH is required}"
CACHE_MANIFEST_PATH="${CACHE_MANIFEST_PATH:?CACHE_MANIFEST_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2024}"
SEASON_MONTHS="${SEASON_MONTHS:-6 7 8}"

hwa_start_log "temperature_variability"
hwa_validate_artifact_paths INPUT_PATH HOURLY_T2M_PATH CACHE_MANIFEST_PATH OUTPUT_DIR
test "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}" && test -s "${HOURLY_T2M_PATH}" && test -s "${CACHE_MANIFEST_PATH}"
test ! -e "${OUTPUT_DIR}" && test ! -L "${OUTPUT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
cd "${PROJECT_ROOT}"
read -r -a months <<< "${SEASON_MONTHS}"
echo "[start] $(date -Is) job=${PBS_JOBID} host=$(hostname) commit=${EXPECTED_COMMIT}"
echo "[paths] stage1=${INPUT_PATH} hourly=${HOURLY_T2M_PATH} manifest=${CACHE_MANIFEST_PATH} output=${OUTPUT_DIR}"
python -c 'import sys; print(sys.executable, sys.version)'
/usr/bin/time -v python scripts/plot_temperature_variability.py \
    --input-path "${INPUT_PATH}" --hourly-t2m-path "${HOURLY_T2M_PATH}" \
    --start-year "${START_YEAR}" --end-year "${END_YEAR}" \
    --season-months "${months[@]}" --output-dir "${OUTPUT_DIR}"
/usr/bin/time -v python scripts/validate_temperature_variability.py \
    --output-dir "${OUTPUT_DIR}" --cache-manifest "${CACHE_MANIFEST_PATH}"
echo "[done] $(date -Is) job=${PBS_JOBID}"
