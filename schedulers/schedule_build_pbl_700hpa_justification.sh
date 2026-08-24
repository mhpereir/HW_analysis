#!/bin/bash
#PBS -N pbl_700hpa_product
#PBS -l select=1:ncpus=4:mem=48gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
STAGE1_PATH="${STAGE1_PATH:?STAGE1_PATH is required}"
PBL_ROOT="${PBL_ROOT:?PBL_ROOT is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
REGION="${REGION:-pnw_bartusek}"
TIME_START="${TIME_START:-1940}"
TIME_END="${TIME_END:-2024}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"
QUANTILE="${QUANTILE:-90}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(realpath "${PBS_O_WORKDIR}")" = "$(realpath "${PROJECT_ROOT}")"
test -s "${STAGE1_PATH}"
test -d "${PBL_ROOT}/${REGION}"
if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "[error] refusing to overwrite existing output: ${OUTPUT_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_build_pbl_700hpa_justification.log"
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
echo "[info] stage1_path=${STAGE1_PATH}"
echo "[info] pbl_root=${PBL_ROOT}"
echo "[info] output_path=${OUTPUT_PATH}"
echo "[info] analysis_years=${TIME_START}-${TIME_END}"
echo "[info] started=$(date -Is)"
"${python_executable}" --version

cd "${PROJECT_ROOT}"
/usr/bin/time -v "${python_executable}" \
    scripts/pbl_justification/build_pbl_700hpa_justification.py \
    --stage1-path "${STAGE1_PATH}" \
    --pbl-root "${PBL_ROOT}" \
    --output-path "${OUTPUT_PATH}" \
    --region "${REGION}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --season-months 6 7 8 \
    --window-days "${WINDOW_DAYS}" \
    --source-commit "${EXPECTED_COMMIT}"

test -s "${OUTPUT_PATH}"
echo "[info] output_bytes=$(stat -c %s "${OUTPUT_PATH}")"
echo "[info] finished=$(date -Is)"
