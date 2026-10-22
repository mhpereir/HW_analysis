#!/bin/bash
#PBS -N plot_dyn_net_spatial
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PBS_O_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

INPUT_PATH="${INPUT_PATH:-${HWA_ARTIFACT_ROOT}/spatial_composites/dyn_net_daily_spatial_composites_pnw_bartusek_tas_q90_1940_2024.nc}"
OUTPUT_PATH="${OUTPUT_PATH:-${HWA_ARTIFACT_ROOT}/spatial_composites/dyn_net_daily_t2m_z500_composites_pnw_bartusek_tas_q90_1940_2024.png}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

# Runtime validation and logging.
hwa_start_log "plot_dyn_net_spatial_composites"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_PATH

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(realpath "${PBS_O_WORKDIR}")" = "$(realpath "${PROJECT_ROOT}")"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-dev_env}"
mamba activate "${VENUS_MAMBA_ENV}"
python_executable=$(command -v python)

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
test ! -e "${OUTPUT_PATH}"

cd "${PBS_O_WORKDIR}"
/usr/bin/time -v "${python_executable}" \
    scripts/spatial_composites/plot_dyn_net_spatial_composites.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --plot-lags -2 0 2

test -s "${OUTPUT_PATH}"
echo "[info] output_bytes=$(stat -c %s "${OUTPUT_PATH}")"
echo "[info] end=$(date -Is)"
