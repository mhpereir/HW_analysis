#!/bin/bash
#PBS -N season_selection_validation
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
hwa_start_log "season_selection_validation"
hwa_validate_artifact_paths MANIFEST_PATH OUTPUT_DIR
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${MANIFEST_PATH}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error
export MPLCONFIGDIR="${OUTPUT_DIR}/matplotlib"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID} host=$(hostname) commit=${actual_commit}"
echo "[info] manifest=${MANIFEST_PATH} output=${OUTPUT_DIR}"
echo "[info] started=$(date -Is)"
cd "${PROJECT_ROOT}"
python -c 'import sys, matplotlib, numpy, xarray; print(sys.executable, sys.version); print("matplotlib", matplotlib.__version__, "numpy", numpy.__version__, "xarray", xarray.__version__)'
# dev_env is shared across projects. Keep its whole-environment diagnostic,
# while required project imports and the complete suite remain hard gates.
if ! python -m pip check > "${OUTPUT_DIR}/pip-check.txt" 2>&1; then
    echo "[info] Shared-environment dependency diagnostic follows; project imports and tests must still pass."
fi
cat "${OUTPUT_DIR}/pip-check.txt"
/usr/bin/time -v python -m pytest -q -W error -o cache_dir="${OUTPUT_DIR}/pytest-cache" \
    --basetemp="${OUTPUT_DIR}/pytest" --junitxml="${OUTPUT_DIR}/junit.xml"
/usr/bin/time -v python scripts/validate_season_selection.py \
    --manifest "${MANIFEST_PATH}" --output-dir "${OUTPUT_DIR}/audit"
test -s "${OUTPUT_DIR}/audit/validation.json"
echo "[info] finished=$(date -Is)"
