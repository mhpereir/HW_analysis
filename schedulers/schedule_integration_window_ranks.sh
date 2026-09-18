#!/bin/bash
#PBS -N integration_window_ranks
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"
hwa_start_log "integration_window_ranks"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
COMPARISON_DIR="${COMPARISON_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
REGIONS="${REGIONS:-pnw_bartusek,pnw_hotz}"
RUN_TESTS="${RUN_TESTS:-0}"
hwa_validate_artifact_paths OUTPUT_DIR
if [[ -n "${COMPARISON_DIR}" ]]; then
    hwa_validate_artifact_paths COMPARISON_DIR
else
    STAGE1_DIR="${STAGE1_DIR:?STAGE1_DIR is required}"
    CLIMATOLOGY_DIR="${CLIMATOLOGY_DIR:?CLIMATOLOGY_DIR is required}"
    REFERENCE_DIR="${REFERENCE_DIR:?REFERENCE_DIR is required}"
    hwa_validate_artifact_paths STAGE1_DIR CLIMATOLOGY_DIR REFERENCE_DIR
fi
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test ! -e "${OUTPUT_DIR}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error MPLBACKEND=Agg
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID} host=$(hostname) commit=${actual_commit}"
echo "[info] python=$(command -v python)"
if [[ -n "${COMPARISON_DIR}" ]]; then
    echo "[info] plotting existing comparisons=${COMPARISON_DIR}"
else
    echo "[info] stage1=${STAGE1_DIR} climatology=${CLIMATOLOGY_DIR} reference=${REFERENCE_DIR}"
fi
echo "[info] output=${OUTPUT_DIR} regions=${REGIONS} integration_days=4..31"
echo "[info] started=$(date -Is)"
cd "${PROJECT_ROOT}"
if [[ "${RUN_TESTS}" = "1" ]]; then
    python -m pytest -q -W error tests/test_integration_window_analysis.py tests/test_stage2_window_sensitivity.py
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"
mkdir "${OUTPUT_DIR}"
IFS=',' read -r -a regions <<< "${REGIONS}"
tables=()
for region in "${regions[@]}"; do
    case "${region}" in
        pnw_bartusek) target_id=1127 ;;
        pnw_hotz) target_id=1217 ;;
        *) echo "Unsupported PNW region: ${region}" >&2; exit 2 ;;
    esac
    if [[ -n "${COMPARISON_DIR}" ]]; then
        table="${COMPARISON_DIR}/${region}/heating_comparison.nc"
        test -s "${table}"
    else
        python scripts/integration_window_analysis/build_heating_comparison.py \
            --input-path "${STAGE1_DIR}/harmonized_regional_timeseries_${region}_surface_700hPa_tas_q90_1940_2024.nc" \
            --climatology-path "${CLIMATOLOGY_DIR}/regional_hourly_climatology_${region}_surface_700hPa_1940_2024.nc" \
            --reference-dir "${REFERENCE_DIR}/${region}" \
            --output-dir "${OUTPUT_DIR}/${region}" \
            --min-days 4 --max-days 31 --target-peak "2021-06-29T00:00:00" --target-event-id "${target_id}"
        table="${OUTPUT_DIR}/${region}/heating_comparison.nc"
    fi
    tables+=("${table}")
done
python scripts/integration_window_analysis/plot_heating_comparison.py \
    --input-paths "${tables[@]}" --output-stem "${OUTPUT_DIR}/rank_sensitivity"
test -s "${OUTPUT_DIR}/rank_sensitivity.json"
echo "[info] finished=$(date -Is)"
