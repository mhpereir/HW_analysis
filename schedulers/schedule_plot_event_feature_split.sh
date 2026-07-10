#!/bin/bash
#PBS -N stage_2_plot_event_feature_split
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_plot_event_feature_split.log"
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
QUANTILE=90
TIME_START=1940
TIME_END=2024
SELECTION_VARIABLE="duration"
SELECTION_QUANTILE=0.9

# f_adiabatic_pre
# f_diabatic_pre
# f_advection_pre
# sqrt_I_lwa_a_pre_peak
# T_anom_mean_ant
# cos_days_from_solstice
# duration
# tas_anom_peak
# tas_peak

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_event_features/hw_event_features_fixed_windows_${REGION}_${THRESHOLD_VARIABLE}_q${QUANTILE}_${TIME_START}_${TIME_END}.nc"
OUTPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_event_features/diagnostics/${REGION}/event_feature_tendency_scatter.png"

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting split event-feature plot generation on host $(hostname)"
/usr/bin/time -v python event_features/plot_event_feature_split.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --selection-variable "${SELECTION_VARIABLE}" \
    --selection-quantile "${SELECTION_QUANTILE}"
echo "[info] $(date -Is) done"
