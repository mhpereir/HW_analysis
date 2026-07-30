#!/bin/bash
#PBS -N plot_dyn_net_spatial
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PBS_O_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(realpath "${PBS_O_WORKDIR}")" = "$(realpath "${PROJECT_ROOT}")"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_dyn_net_spatial_composites.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-dev_env}"
mamba activate "${VENUS_MAMBA_ENV}"
python_executable=$(command -v python)

INPUT_PATH="${PROJECT_ROOT}/results/spatial_composites/dyn_net_daily_spatial_composites_pnw_bartusek_tas_q90_1940_2024.nc"
OUTPUT_PATH="${PROJECT_ROOT}/results/spatial_composites/dyn_net_daily_t2m_z500_composites_pnw_bartusek_tas_q90_1940_2024.png"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname -f)"
echo "[info] commit=${actual_commit}"
echo "[info] environment=${VENUS_MAMBA_ENV}"
echo "[info] python=${python_executable}"
echo "[info] input=${INPUT_PATH}"
echo "[info] output=${OUTPUT_PATH}"
echo "[info] start=$(date -Is)"
"${python_executable}" --version
test -s "${INPUT_PATH}"

cd "${PBS_O_WORKDIR}"
/usr/bin/time -v "${python_executable}" \
    scripts/spatial_composites/plot_dyn_net_spatial_composites.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --plot-lags -2 0 2

test -s "${OUTPUT_PATH}"
echo "[info] output_bytes=$(stat -c %s "${OUTPUT_PATH}")"
echo "[info] end=$(date -Is)"
