#!/bin/bash
#PBS -N stage2_component_anomalies
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"
REFERENCE_DIR="${REFERENCE_DIR:?REFERENCE_DIR is required}"
INTEGRATION_HOURS="${INTEGRATION_HOURS:?INTEGRATION_HOURS is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"
RUN_TESTS="${RUN_TESTS:-0}"
hwa_start_log "stage2_component_anomalies"
hwa_validate_artifact_paths INPUT_PATH CLIMATOLOGY_PATH REFERENCE_DIR OUTPUT_DIR
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
for input in "${INPUT_PATH}" "${CLIMATOLOGY_PATH}" "${REFERENCE_DIR}/event_features.nc" "${REFERENCE_DIR}/baseline_features.nc" "${REFERENCE_DIR}/manifest.json"; do
    test -s "${input}"
done
test ! -e "${OUTPUT_DIR}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error MPLBACKEND=Agg
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID} host=$(hostname) commit=${actual_commit}"
echo "[info] python=$(command -v python)"
echo "[info] input_path=${INPUT_PATH} climatology_path=${CLIMATOLOGY_PATH}"
echo "[info] reference_dir=${REFERENCE_DIR} output_dir=${OUTPUT_DIR} integration_hours=${INTEGRATION_HOURS}"
echo "[info] started=$(date -Is)"
cd "${PROJECT_ROOT}"
if [[ "${RUN_TESTS}" = "1" ]]; then
    python -m pytest -q -W error tests/test_stage2_component_anomalies.py tests/test_plot_adiabatic_advection_comparison_baseline.py
fi
/usr/bin/time -v python scripts/integration_window_analysis/build_component_anomalies.py \
    --input-path "${INPUT_PATH}" --climatology-path "${CLIMATOLOGY_PATH}" \
    --reference-dir "${REFERENCE_DIR}" --integration-hours "${INTEGRATION_HOURS}" \
    --output-dir "${OUTPUT_DIR}"
test -s "${OUTPUT_DIR}/manifest.json"
echo "[info] finished=$(date -Is)"
