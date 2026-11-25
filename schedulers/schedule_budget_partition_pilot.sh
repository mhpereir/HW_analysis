#!/bin/bash
#PBS -N budget_partition_pilot
#PBS -l select=1:ncpus=1:mem=40gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
REFERENCE_DIR="${REFERENCE_DIR:?REFERENCE_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
RUN_MODE="${RUN_MODE:-pilot}"
case "${RUN_MODE}" in smoke|pilot) ;; *) echo "Invalid RUN_MODE" >&2; exit 2 ;; esac
hwa_start_log "budget_partition_${RUN_MODE}"
hwa_validate_artifact_paths INPUT_PATH REFERENCE_DIR OUTPUT_DIR
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test ! -e "${OUTPUT_DIR}"
for input in "${INPUT_PATH}" "${REFERENCE_DIR}/event_features.nc" "${REFERENCE_DIR}/baseline_features.nc" "${REFERENCE_DIR}/manifest.json"; do
    test -s "${input}"
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error MPLBACKEND=Agg
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
cd "${PROJECT_ROOT}"
echo "[info] job=${PBS_JOBID} host=$(hostname) commit=${actual_commit} mode=${RUN_MODE}"
echo "[info] python=$(command -v python) started=$(date -Is)"
echo "[info] input=${INPUT_PATH} reference=${REFERENCE_DIR} output=${OUTPUT_DIR}"
python -c 'import sys, xarray, matplotlib; print(sys.version); print(xarray.__version__, matplotlib.__version__)'
if [[ "${RUN_MODE}" == "smoke" ]]; then
    mkdir -p "$(dirname "${OUTPUT_DIR}")"
    mkdir "${OUTPUT_DIR}"
    /usr/bin/time -v python -m pytest -q -W error \
        tests/test_budget_partition.py tests/test_harmonize.py \
        tests/test_stage2_component_anomalies.py tests/test_climatology_anomaly_scripts.py \
        tests/test_plot_adiabatic_advection_comparison_baseline.py \
        > "${OUTPUT_DIR}/pytest.txt" 2>&1
    cat "${OUTPUT_DIR}/pytest.txt"
else
    /usr/bin/time -v python scripts/run_budget_partition_pilot.py \
        --input-path "${INPUT_PATH}" --reference-dir "${REFERENCE_DIR}" \
        --output-dir "${OUTPUT_DIR}"
    test -s "${OUTPUT_DIR}/manifest.json"
fi
echo "[info] finished=$(date -Is)"
