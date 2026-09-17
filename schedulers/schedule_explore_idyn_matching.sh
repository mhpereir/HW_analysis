#!/bin/bash
#PBS -N idyn_matching_exploration
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:-${HWA_ARTIFACT_ROOT}/stage2_event_features/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc}"
SETTINGS_PATH="${SETTINGS_PATH:-${PROJECT_ROOT}/scripts/idyn_matching_exploration/matching_settings.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${HWA_ARTIFACT_ROOT}/Idyn_matching_exploration}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"

# Runtime validation and logging.
hwa_start_log "idyn_matching_exploration"
hwa_validate_artifact_paths INPUT_PATH OUTPUT_DIR
STAGED_OUTPUT_DIR="${OUTPUT_DIR}.tmp.${PBS_JOBID}"
ARTIFACT_FILENAMES=(
    idyn_population_overview.png
    tas_anom_matching_diagnostics.png
    covariate_balance_and_sensitivity.png
    matching_specification_tradeoff.png
    matching_summary.json
)

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test -s "${SETTINGS_PATH}"
hwa_validate_artifact_paths STAGED_OUTPUT_DIR
test ! -e "${STAGED_OUTPUT_DIR}"
for filename in "${ARTIFACT_FILENAMES[@]}"; do
    test ! -e "${OUTPUT_DIR}/${filename}"
done

cleanup() {
    if [[ -d "${STAGED_OUTPUT_DIR}" ]]; then
        rm -rf -- "${STAGED_OUTPUT_DIR}"
    fi
}
trap cleanup EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=error
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(command -v python)"
echo "[info] input_path=${INPUT_PATH}"
echo "[info] settings_path=${SETTINGS_PATH}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] started=$(date -Is)"

mkdir -p "${STAGED_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
/usr/bin/time -v python scripts/idyn_matching_exploration/explore_idyn_matching.py \
    --input-path "${INPUT_PATH}" \
    --settings-path "${SETTINGS_PATH}" \
    --output-dir "${STAGED_OUTPUT_DIR}"

for filename in "${ARTIFACT_FILENAMES[@]}"; do
    test -s "${STAGED_OUTPUT_DIR}/${filename}"
done

mkdir -p "${OUTPUT_DIR}"
for filename in "${ARTIFACT_FILENAMES[@]}"; do
    mv -n -- "${STAGED_OUTPUT_DIR}/${filename}" "${OUTPUT_DIR}/${filename}"
    test ! -e "${STAGED_OUTPUT_DIR}/${filename}"
done
test -z "$(find "${STAGED_OUTPUT_DIR}" -mindepth 1 -print -quit)"
rmdir -- "${STAGED_OUTPUT_DIR}"

echo "[info] finished=$(date -Is)"
