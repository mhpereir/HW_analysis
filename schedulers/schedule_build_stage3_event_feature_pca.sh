#!/bin/bash
#PBS -N stage_3_event_feature_pca
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_stage3_event_feature_pca.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage2_event_features/hw_event_features_fixed_windows_pnw_hotz_tas_q90_1940_2024.nc"
OUTPUT_PATH="/home/mhpereir/HW_analysis/results/stage3_event_feature_pca/hw_event_feature_pca_pnw_hotz_tas_q90_1940_2024.nc"

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting event feature PCA on host $(hostname)"
/usr/bin/time -v python event_features/build_stage3_event_feature_pca.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --scaler standard \
    --overwrite
echo "[info] $(date -Is) done"
