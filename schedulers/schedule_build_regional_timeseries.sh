#!/bin/bash
#PBS -N stage_1_build_regional_timeseries
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_build_regional_timeseries.log"
exec > >(tee -a "${LOGFILE}") 2>&1

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

TIME_START=2000
TIME_END=2014

cd /home/mhpereir/HW_analysis/scripts

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
/usr/bin/time -v python build_regional_timeseries.py --start-year ${TIME_START} --end-year ${TIME_END}
echo "[info] $(date -Is) done"
