#!/bin/bash
#PBS -N regional_hourly_t2m
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
REGION="${REGION:?REGION is required}"
# Use the standard artifact root: regional_hourly_t2m/<region>/<attempt>/.
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2024}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
TIME_CHUNK="${TIME_CHUNK:-}"

hwa_start_log "regional_hourly_t2m"
hwa_validate_artifact_paths OUTPUT_DIR
test "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test ! -e "${OUTPUT_DIR}" && test ! -L "${OUTPUT_DIR}"
if [[ -n "${SOURCE_ROOT}" ]]; then
    [[ "${SOURCE_ROOT}" == /* && -d "${SOURCE_ROOT}" ]] || {
        echo "[error] SOURCE_ROOT must be an absolute raw-data directory" >&2
        exit 2
    }
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
cd "${PROJECT_ROOT}"
EXTRA_ARGS=()
if [[ -n "${SOURCE_ROOT}" ]]; then EXTRA_ARGS+=(--source-root "${SOURCE_ROOT}"); fi
if [[ -n "${TIME_CHUNK}" ]]; then EXTRA_ARGS+=(--time-chunk "${TIME_CHUNK}"); fi
echo "[start] $(date -Is) job=${PBS_JOBID} host=$(hostname) commit=${EXPECTED_COMMIT}"
echo "[python] $(command -v python)"
echo "[input] source_root=${SOURCE_ROOT:-configured-native-ERA5} region=${REGION} years=${START_YEAR}-${END_YEAR}"
echo "[output] ${OUTPUT_DIR}"
/usr/bin/time -v python -W error scripts/prepare_regional_hourly_t2m.py \
    --region "${REGION}" --start-year "${START_YEAR}" --end-year "${END_YEAR}" \
    "${EXTRA_ARGS[@]}" --output-dir "${OUTPUT_DIR}"
echo "[done] $(date -Is) job=${PBS_JOBID}"
