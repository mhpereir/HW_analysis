#!/bin/bash
#PBS -N era5_daily_climatology
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_era5_daily_climatology.log"
exec > >(tee -a "${LOGFILE}") 2>&1

set -euo pipefail

cd /home/mhpereir/HW_analysis
echo "[info] $(date -Is) starting global ERA5 daily climatology on $(hostname)"
/usr/bin/time -v scripts/spatial_composites/build_era5_daily_doy_climatology.sh
echo "[info] $(date -Is) done"
