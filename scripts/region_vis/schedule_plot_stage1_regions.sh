#!/bin/bash
#PBS -N plot_stage1_regions
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -j oe

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PBS_O_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

RUN_ID="${RUN_ID:-bf232281_20260819}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/stage1/runs/${RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/region_vis/stage1_regional_domains_${RUN_ID}.png}"
EXPECTED_REGION_COUNT="${EXPECTED_REGION_COUNT:-7}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
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
if [[ "$(realpath "${PBS_O_WORKDIR}")" != "$(realpath "${PROJECT_ROOT}")" ]]; then
    echo "[error] PBS_O_WORKDIR must resolve to PROJECT_ROOT" >&2
    exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[error] Stage 1 run directory does not exist: ${RUN_DIR}" >&2
    exit 2
fi
if [[ ! "${EXPECTED_REGION_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] EXPECTED_REGION_COUNT must be a positive integer" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID:-manual}_plot_stage1_regions.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV}"
python_executable=$(command -v python)

cd "${PROJECT_ROOT}"

echo "[info] job_id=${PBS_JOBID:-manual}"
echo "[info] host=$(hostname -f)"
echo "[info] commit=${actual_commit}"
echo "[info] environment=${VENUS_MAMBA_ENV}"
echo "[info] python=${python_executable}"
echo "[info] run_dir=${RUN_DIR}"
echo "[info] expected_region_count=${EXPECTED_REGION_COUNT}"
echo "[info] output=${OUTPUT_PATH}"
echo "[info] start=$(date -Is)"
"${python_executable}" --version

/usr/bin/time -v "${python_executable}" \
    scripts/region_vis/plot_stage1_regions.py \
    --run-dir "${RUN_DIR}" \
    --output-path "${OUTPUT_PATH}" \
    --expected-region-count "${EXPECTED_REGION_COUNT}"

test -s "${OUTPUT_PATH}"
echo "[info] output_bytes=$(stat -c %s "${OUTPUT_PATH}")"
echo "[info] end=$(date -Is)"
