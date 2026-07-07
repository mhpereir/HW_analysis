#!/bin/bash
#PBS -N stage_2_baseline_features
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_stage2_baseline_features.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

REGION="pnw_bartusek"
THRESHOLD_VARIABLE="tas"
QUANTILE_THRESHOLD="q90"

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage1/harmonized_regional_timeseries_${REGION}_surface_700hPa_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}_1940_2024.nc"
OUTPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_baseline_features/non_event_day_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}.nc"
SEASON_MONTHS=(6 7 8)

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting baseline-day feature extraction on host $(hostname)"
/usr/bin/time -v python event_features/build_stage2_baseline_features.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --season-months "${SEASON_MONTHS[@]}"
echo "[info] $(date -Is) done"

# CLI defaults intentionally omitted:
#   --csv-output-path: no CSV output
#   --use-extended-variables: disabled
#   --allow-missing-extended: disabled
#   --all-seasons: disabled; mutually exclusive with --season-months
#   --overwrite: disabled
