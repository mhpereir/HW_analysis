#!/bin/bash
#PBS -N composite_timeseries_split_q75
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_plot_composite_timeseries_split_q75.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"

REGION="pnw_bartusek"
BOTTOM_BOUNDARY="surface"
TOP_BOUNDARY=700
THRESHOLD_VARIABLE="tas"
QUANTILE=90
TIME_START=1940
TIME_END=2024
SPLIT_QUANTILE=0.75
OUTPUT_DIR="${PROJECT_ROOT}/results/plots_composite_timeseries_split/region_${REGION}/boundary_surface_700hPa/time_range_${TIME_START}_${TIME_END}/split_q75"

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=$(python -c 'import sys; print(sys.executable)')"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}/scripts"

split_variable_list=(
    "duration"
    "tas_anom_peak"
    "tas_excess_integral"
    "tas_excess_peak"
    "tas_peak"
)

for split_variable in "${split_variable_list[@]}"; do
    echo "[info] $(date -Is) starting ${split_variable}"
    /usr/bin/time -v python plot_composite_timeseries_split.py \
        --region "${REGION}" \
        --bottom-boundary "${BOTTOM_BOUNDARY}" \
        --top-boundary "${TOP_BOUNDARY}" \
        --threshold-variable "${THRESHOLD_VARIABLE}" \
        --quantile "${QUANTILE}" \
        --start-year "${TIME_START}" \
        --end-year "${TIME_END}" \
        --output-path "${OUTPUT_DIR}/hw_events_composite.png" \
        --window-days 7 \
        --split-variable "${split_variable}" \
        --split-quantiles "${SPLIT_QUANTILE}" \
        --season-months 6 7 8 \
        --require-full-event \
        --plot-extended-variables
    echo "[info] $(date -Is) finished ${split_variable}"
done

echo "[info] finished=$(date -Is)"
