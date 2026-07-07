#!/bin/bash
#PBS -N stage_2_plot_event_vs_baseline
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_plot_event_vs_baseline.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

REGION="pnw_hotz"
THRESHOLD_VARIABLE="lwa_a"
QUANTILE_THRESHOLD="q90"
TIME_START=1940
TIME_END=2024

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_baseline_features/non_event_day_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}.nc"
EVENT_INPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_event_features/hw_event_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_${QUANTILE_THRESHOLD}_${TIME_START}_${TIME_END}.nc"
OUTPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_baseline_features/diagnostics/${REGION}/${THRESHOLD_VARIABLE}/event_vs_clean_baseline_diabatic_advection_scatter.png"
POINT_SIZE=24.0
ALPHA=0.2
EVENT_POINT_SIZE=24.0
EVENT_ALPHA=0.7

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting event-versus-clean-baseline plot generation on host $(hostname)"
/usr/bin/time -v python event_features/plot_adiabatic_diabatic_advection_baseline.py \
    --input-path "${INPUT_PATH}" \
    --event-input-path "${EVENT_INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --point-size "${POINT_SIZE}" \
    --alpha "${ALPHA}" \
    --event-point-size "${EVENT_POINT_SIZE}" \
    --event-alpha "${EVENT_ALPHA}"
echo "[info] $(date -Is) done"
