#!/bin/bash
#PBS -N stage_2_event_features
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_stage2_event_features.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage1/harmonized_regional_timeseries_pnw_hotz_surface_700hPa_tas_q90_1940_2024.nc"
OUTPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_event_features/hw_event_features_fixed_windows_pnw_hotz_tas_q90_1940_2024.nc"

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting event feature extraction on host $(hostname)"
/usr/bin/time -v python event_features/build_stage2_event_features.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --season-months 6 7 8 \
    --require-full-event \
    --overwrite
echo "[info] $(date -Is) done"



# Add this flag only when the Stage-1 input was built with --add-full-diagnostics:
#    --use-extended-variables \
