#!/bin/bash
#PBS -N threshold_timeseries
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_threshold_timeseries.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

TIME_START=1940
TIME_END=2024
REGION="pnw_bartusek"
BOTTOM_BOUNDARY="surface"
TOP_BOUNDARY=700
THRESHOLD_VARIABLE="tas"
QUANTILE=90

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting plot generation on host $(hostname)"
/usr/bin/time -v python plot_threshold_timeseries.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --years ${TIME_START} ${TIME_END}
echo "[info] $(date -Is) done"
