#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
REGION="${REGION:?REGION is required}"
BOTTOM_BOUNDARY="${BOTTOM_BOUNDARY:?BOTTOM_BOUNDARY is required}"
TOP_BOUNDARY="${TOP_BOUNDARY:?TOP_BOUNDARY is required}"
THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:?THRESHOLD_VARIABLE is required}"
QUANTILE="${QUANTILE:?QUANTILE is required}"
TIME_START="${TIME_START:?TIME_START is required}"
TIME_END="${TIME_END:?TIME_END is required}"
EHB_TIME_START="${EHB_TIME_START:?EHB_TIME_START is required}"
EHB_TIME_END="${EHB_TIME_END:?EHB_TIME_END is required}"
LEGACY_CLOUD_ROOT="${LEGACY_CLOUD_ROOT:?LEGACY_CLOUD_ROOT is required}"
STAGE1_BASE_PATH="${STAGE1_BASE_PATH:?STAGE1_BASE_PATH is required}"
STAGE1_ENHANCED_PATH="${STAGE1_ENHANCED_PATH:?STAGE1_ENHANCED_PATH is required}"
PLOT_OUTPUT_PATH="${PLOT_OUTPUT_PATH:?PLOT_OUTPUT_PATH is required}"

for output_path in \
    "${STAGE1_BASE_PATH}" \
    "${STAGE1_ENHANCED_PATH}" \
    "${PLOT_OUTPUT_PATH}"
do
    if [[ -e "${output_path}" ]]; then
        echo "[error] refusing to overwrite existing output: ${output_path}" >&2
        exit 1
    fi
done

echo "[info] stage1_base_path=${STAGE1_BASE_PATH}"
echo "[info] stage1_enhanced_path=${STAGE1_ENHANCED_PATH}"
echo "[info] plot_output_path=${PLOT_OUTPUT_PATH}"
echo "[info] cloud_cover_source_layout=legacy-regional"
echo "[info] cloud_cover_root=${LEGACY_CLOUD_ROOT}"

cd "${PROJECT_ROOT}/scripts"

/usr/bin/time -v python build_stage1_harmonized_timeseries.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --start-year-ehb "${EHB_TIME_START}" \
    --end-year-ehb "${EHB_TIME_END}" \
    --output-path "${STAGE1_BASE_PATH}" \
    --add-full-diagnostics \
    --cloud-cover-source-layout legacy-regional \
    --cloud-cover-root "${LEGACY_CLOUD_ROOT}"

/usr/bin/time -v python build_stage1_advection_exploration.py \
    --region "${REGION}" \
    --bottom-boundary "${BOTTOM_BOUNDARY}" \
    --top-boundary "${TOP_BOUNDARY}" \
    --threshold-variable "${THRESHOLD_VARIABLE}" \
    --quantile "${QUANTILE}" \
    --start-year "${TIME_START}" \
    --end-year "${TIME_END}" \
    --start-year-ehb "${EHB_TIME_START}" \
    --end-year-ehb "${EHB_TIME_END}" \
    --input-path "${STAGE1_BASE_PATH}" \
    --output-path "${STAGE1_ENHANCED_PATH}"

/usr/bin/time -v python plot_advection_direction_exploration.py \
  --region "${REGION}" \
  --bottom-boundary "${BOTTOM_BOUNDARY}" \
  --top-boundary "${TOP_BOUNDARY}" \
  --threshold-variable "${THRESHOLD_VARIABLE}" \
  --quantile "${QUANTILE}" \
  --start-year "${TIME_START}" \
  --end-year "${TIME_END}" \
  --input-path "${STAGE1_ENHANCED_PATH}" \
  --output-path "${PLOT_OUTPUT_PATH}" \
  --window-days 7 \
  --season-months 6 7 8 \
  --require-full-event

test -s "${STAGE1_BASE_PATH}"
test -s "${STAGE1_ENHANCED_PATH}"
test -s "${PLOT_OUTPUT_PATH}"
