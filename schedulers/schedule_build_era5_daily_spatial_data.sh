#!/bin/bash
#PBS -N era5_daily_spatial
#PBS -l select=1:ncpus=1:mem=2gb:host=venus05
#PBS -j oe
#PBS -o /dev/null

LOGFILE="/home/mhpereir/HW_analysis/logs/${PBS_JOBID}_era5_daily_spatial.log"
exec > >(tee -a "${LOGFILE}") 2>&1

set -euo pipefail

YEAR=${PBS_ARRAY_INDEX:?Submit this scheduler as a PBS array job}
export OMP_NUM_THREADS=1
EXPECTED_HOST=venus05
ACTUAL_HOST=$(hostname -s)

if [[ "$ACTUAL_HOST" != "$EXPECTED_HOST" ]]; then
    echo "[error] PBS placed this job on ${ACTUAL_HOST}; expected ${EXPECTED_HOST}." >&2
    exit 1
fi

cd /home/mhpereir/HW_analysis
echo "[info] $(date -Is) starting global ERA5 daily preprocessing for ${YEAR} on $(hostname)"
/usr/bin/time -v scripts/spatial_composites/build_era5_daily_spatial_data.sh \
    --start-year "${YEAR}" \
    --end-year "${YEAR}" \
    --threads 1 \
    --skip-existing
echo "[info] $(date -Is) done"
