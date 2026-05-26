#!/bin/bash
#PBS -N stage_2_plot_event_summary
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_plot_event_summary.log"
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
BOTTOM_BOUNDARY="surface"
TOP_BOUNDARY=700
THRESHOLD_VARIABLE="tas"
QUANTILE=90
TIME_START=1940
TIME_END=2024

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting event summary plotting on host $(hostname)"
/usr/bin/time -v python plot_event_summary.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --season-months 6 7 8 \
    --require-full-event
echo "[info] $(date -Is) done"
