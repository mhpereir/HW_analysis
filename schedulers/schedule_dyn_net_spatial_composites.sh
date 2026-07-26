#!/bin/bash
#PBS -N dyn_net_spatial_composites
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_dyn_net_spatial_composites.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

cd /home/mhpereir/HW_analysis
echo "[info] $(date -Is) starting daily dynamical-sign composites on $(hostname)"
/usr/bin/time -v python scripts/spatial_composites/build_dyn_net_spatial_composites.py --overwrite
/usr/bin/time -v python scripts/spatial_composites/plot_dyn_net_spatial_composites.py
echo "[info] $(date -Is) done"
