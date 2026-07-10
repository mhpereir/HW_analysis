#!/bin/bash
#PBS -N stage_2_plot_top_events
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_plot_top_events.log"
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
BOTTOM_BOUNDARY="surface"
TOP_BOUNDARY=700
THRESHOLD_VARIABLE="tas"
QUANTILE=90
TIME_START=1940
TIME_END=2024

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting plot generation on host $(hostname)"
/usr/bin/time -v python plot_top_events.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --top-n 10 \
    --window-days 7 \
    --plot-extended-variables
    
echo "[info] $(date -Is) done"
