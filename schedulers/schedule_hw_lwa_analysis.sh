#!/bin/bash
#PBS -N hw_lwa_analysis
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"
RUN_MODE="${RUN_MODE:-plot}"
hwa_start_log "hw_lwa_analysis"
trap 'status=$?; echo "[info] finished=$(date -Is) exit_status=${status}"' EXIT

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test "$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel)" = "${PROJECT_ROOT}"
echo "[info] started=$(date -Is) host=$(hostname) job=${PBS_JOBID} commit=${actual_commit} mode=${RUN_MODE}"

case "${RUN_MODE}" in
    smoke) ;;
    plot)
        INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
        OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
        EXPECTED_INPUT_SHA256="${EXPECTED_INPUT_SHA256:?EXPECTED_INPUT_SHA256 is required}"
        hwa_validate_artifact_paths INPUT_PATH OUTPUT_DIR
        test -s "${INPUT_PATH}"
        test ! -e "${OUTPUT_DIR}"
        test ! -L "${OUTPUT_DIR}"
        actual_input_sha256=$(sha256sum "${INPUT_PATH}")
        test "${actual_input_sha256%% *}" = "${EXPECTED_INPUT_SHA256}"
        echo "[info] input=${INPUT_PATH} input_sha256=${EXPECTED_INPUT_SHA256} output=${OUTPUT_DIR}"
        ;;
    *) echo "[error] RUN_MODE must be smoke or plot" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=error MPLBACKEND=Agg
export MPLCONFIGDIR="${LOG_DIR}/matplotlib-${PBS_JOBID}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
python_executable=$(command -v python)
echo "[info] environment=${VENUS_MAMBA_ENV:-dev_env} python=${python_executable} prefix=${CONDA_PREFIX}"
"${python_executable}" -c 'import sys, numpy, pandas, xarray, matplotlib, h5netcdf; print(sys.version); print({m.__name__: m.__version__ for m in (numpy, pandas, xarray, matplotlib, h5netcdf)})'
cd "${PROJECT_ROOT}"
if [[ "${RUN_MODE}" == smoke ]]; then
    /usr/bin/time -v "${python_executable}" -m pytest -q -W error \
        tests/test_plot_hw_lwa_analysis.py tests/test_artifact_paths.py
else
    args=(--input-path "${INPUT_PATH}" --output-dir "${OUTPUT_DIR}")
    if [[ -n "${LWA_UNITS:-}" ]]; then
        args+=(--lwa-units "${LWA_UNITS}" --lwa-units-evidence "${LWA_UNITS_EVIDENCE:?LWA_UNITS_EVIDENCE is required with LWA_UNITS}")
    fi
    season_args=()
    if [[ -n "${HWA_SEASON_OPTIONS:-}" ]]; then
        read -r -a season_args <<< "${HWA_SEASON_OPTIONS}"
    fi
    /usr/bin/time -v "${python_executable}" scripts/plot_hw_lwa_analysis.py "${args[@]}" "${season_args[@]}"
    sha256sum "${OUTPUT_DIR}/hw_lwa_analysis.png" "${OUTPUT_DIR}/daily_triplets.csv" "${OUTPUT_DIR}/summary.json"
fi
