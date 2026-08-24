#!/bin/bash
#PBS -N pbl_700hpa_figure
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
MAP_MARGIN_DEGREES="${MAP_MARGIN_DEGREES:-2.5}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(realpath "${PBS_O_WORKDIR}")" = "$(realpath "${PROJECT_ROOT}")"
test -s "${INPUT_PATH}"
if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "[error] refusing to overwrite existing output: ${OUTPUT_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_pbl_700hpa_justification.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=error
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
python_executable=$(command -v python)

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname -f)"
echo "[info] commit=${actual_commit}"
echo "[info] python=${python_executable}"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] map_margin_degrees=${MAP_MARGIN_DEGREES}"
echo "[info] started=$(date -Is)"
"${python_executable}" --version

cd "${PROJECT_ROOT}"
/usr/bin/time -v "${python_executable}" \
    scripts/pbl_justification/plot_pbl_700hpa_justification.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --map-margin-degrees "${MAP_MARGIN_DEGREES}"

test -s "${OUTPUT_PATH}"
echo "[info] output_bytes=$(stat -c %s "${OUTPUT_PATH}")"
echo "[info] finished=$(date -Is)"
