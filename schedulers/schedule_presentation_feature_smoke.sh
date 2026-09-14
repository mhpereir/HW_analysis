#!/bin/bash
#PBS -N presentation_feature_smoke
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"

mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${PBS_JOBID}_presentation_feature_smoke.log"
test ! -e "${LOGFILE}"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=error
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
python_executable=$(command -v python)

echo "[info] job_id=${PBS_JOBID}"
echo "[info] host=$(hostname)"
echo "[info] commit=${actual_commit}"
echo "[info] python=${python_executable}"
echo "[info] environment=${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] inputs=synthetic unit-test fixtures only"
echo "[info] started=$(date -Is)"

cd "${PROJECT_ROOT}"
"${python_executable}" -c 'import matplotlib, numpy, xarray; print("matplotlib", matplotlib.__version__, "numpy", numpy.__version__, "xarray", xarray.__version__)'
/usr/bin/time -v "${python_executable}" -m pytest -q -W error \
  tests/test_plot_style.py \
  tests/test_presentation_budget_comparisons.py \
  tests/test_selectors.py \
  tests/test_plot_adiabatic_advection_comparison.py \
  tests/test_plot_adiabatic_advection_comparison_baseline.py

echo "[info] finished=$(date -Is)"
