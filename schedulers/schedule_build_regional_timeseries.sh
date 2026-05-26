#!/bin/bash
#PBS -N stage_1_build_regional_timeseries
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_build_regional_timeseries.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

TIME_START=1940
TIME_END=2024

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
/usr/bin/time -v python build_regional_timeseries.py \
    --start-year ${TIME_START} --end-year ${TIME_END} \
    --quantile 90 \
    --region "pnw_hotz" \
    --bottom-boundary "surface" \
    --top-boundary 700 \
    --start-year-ehb 1940 \
    --end-year-ehb 2025 \
    --threshold-variable "tas"

echo "[info] $(date -Is) done"



#    --add-full-diagnostics \
