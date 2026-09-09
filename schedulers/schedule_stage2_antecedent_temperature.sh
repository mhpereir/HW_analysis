#!/bin/bash
#PBS -N stage2_antecedent_temperature
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required; use a fresh region-specific run directory}"

actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)"
test -s "${INPUT_PATH}"
test -s "${CLIMATOLOGY_PATH}"
test ! -e "${RUN_DIR}"
mkdir "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/production.log") 2>&1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${RUN_DIR}/mpl-cache"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
echo "[info] job_id=${PBS_JOBID}; host=$(hostname); commit=${actual_commit}"
echo "[info] python=$(command -v python); started=$(date -Is)"
echo "[info] Stage-1=${INPUT_PATH}; climatology=${CLIMATOLOGY_PATH}; outputs=${RUN_DIR}"
cd "${PROJECT_ROOT}"

python scripts/event_features/build_stage2_event_features.py \
    --input-path "${INPUT_PATH}" --climatology-path "${CLIMATOLOGY_PATH}" \
    --output-path "${RUN_DIR}/event_features.nc" \
    --csv-output-path "${RUN_DIR}/event_features.csv" \
    --season-months 6 7 8 --require-full-event
python scripts/event_features/build_stage2_baseline_features.py \
    --input-path "${INPUT_PATH}" --climatology-path "${CLIMATOLOGY_PATH}" \
    --output-path "${RUN_DIR}/baseline_features.nc" \
    --csv-output-path "${RUN_DIR}/baseline_features.csv" --season-months 6 7 8
python scripts/event_features/validate_stage2_temperature.py \
    --input-path "${INPUT_PATH}" --climatology-path "${CLIMATOLOGY_PATH}" \
    --event-path "${RUN_DIR}/event_features.nc" --baseline-path "${RUN_DIR}/baseline_features.nc" \
    --output-path "${RUN_DIR}/validation.json" --expected-commit "${EXPECTED_COMMIT}"
python scripts/event_features/plot_antecedent_temperature.py \
    --input-path "${RUN_DIR}/event_features.nc" \
    --output-path "${RUN_DIR}/antecedent_temperature.png"
python scripts/event_features/plot_adiabatic_advection_comparison_baseline.py \
    --input-path "${RUN_DIR}/baseline_features.nc" --event-input-path "${RUN_DIR}/event_features.nc" \
    --layout presentation --output-path "${RUN_DIR}/event_vs_clean_baseline_presentation.png"

cd "${RUN_DIR}"
sha256sum event_features.nc event_features.csv baseline_features.nc baseline_features.csv \
    validation.json antecedent_temperature.png event_vs_clean_baseline_presentation.png > SHA256SUMS
echo "[info] finished=$(date -Is)"
