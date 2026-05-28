#!/bin/bash
#PBS -N stage_4_event_feature_clusters
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_stage4_event_feature_clusters.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

INPUT_PATH="/home/mhpereir/HW_analysis/results/stage3_event_feature_pca/hw_event_feature_pca_pnw_hotz_tas_q90_1940_2024.nc"
OUTPUT_DIR="/home/mhpereir/HW_analysis/results/stage4_event_feature_clusters/pnw_hotz/"
METHODS=(ward kmeans gmm)
PCS=(PC1 PC2 PC3)
N_CLUSTERS=3
RANDOM_STATE=0
TRACKED_VARIABLES=(
    PC1
    PC2
    PC3
    I_dTdt_pre
    I_adiabatic_pre
    I_diabatic_pre
    I_advection_pre
    f_adiabatic_pre
    f_diabatic_pre
    f_advection_pre
    sqrt_I_lwa_a_pre_peak
    T_anom_mean_ant
    duration
    tas_anom_peak
    tas_excess_integral
)

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting event-feature clustering on host $(hostname)"
/usr/bin/time -v python event_features/build_stage4_event_feature_clusters.py \
    --input-path "${INPUT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --methods "${METHODS[@]}" \
    --pcs "${PCS[@]}" \
    --n-clusters "${N_CLUSTERS}" \
    --random-state "${RANDOM_STATE}" \
    --tracked-variables "${TRACKED_VARIABLES[@]}" \
    --overwrite
echo "[info] $(date -Is) done"
