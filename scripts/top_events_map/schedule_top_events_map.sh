#!/bin/bash
#PBS -N top_events_map
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
PREPARED_MAP_PATH="${PREPARED_MAP_PATH:-}"
SOURCE_VALIDATION_PATH="${SOURCE_VALIDATION_PATH:-}"
PLOT_RANKS="${PLOT_RANKS:-}"
export PREPARED_MAP_PATH SOURCE_VALIDATION_PATH PLOT_RANKS
RUN_DIR="${RUN_DIR:?RUN_DIR is required; choose a new run directory}"
CLIMATOLOGY_START_YEAR="${CLIMATOLOGY_START_YEAR:-1940}"
CLIMATOLOGY_END_YEAR="${CLIMATOLOGY_END_YEAR:-2024}"
TOP_N="${TOP_N:-1}"
RANK_METRIC="${RANK_METRIC:-tas_peak}"
MAP_EXTENT="${MAP_EXTENT:--170 -40 10 80}"

hwa_start_log "top_events_map"
hwa_validate_artifact_paths RUN_DIR
if [[ -n "${PREPARED_MAP_PATH}" ]]; then
    : "${SOURCE_VALIDATION_PATH:?SOURCE_VALIDATION_PATH is required for replotting}"
    hwa_validate_artifact_paths PREPARED_MAP_PATH SOURCE_VALIDATION_PATH
else
    : "${EVENT_FEATURES_PATH:?EVENT_FEATURES_PATH is required}"
    : "${REGION:?REGION is required}"
    : "${DAILY_DIR:?DAILY_DIR is required}"
    : "${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"
    hwa_validate_artifact_paths EVENT_FEATURES_PATH DAILY_DIR CLIMATOLOGY_PATH
fi
cd "${PROJECT_ROOT}"
actual_commit=$(git rev-parse HEAD)
test "${actual_commit}" = "${EXPECTED_COMMIT}"
test -z "$(git status --porcelain --untracked-files=normal)"
if [[ -n "${PREPARED_MAP_PATH}" ]]; then
    test -s "${PREPARED_MAP_PATH}"
    test -s "${SOURCE_VALIDATION_PATH}"
else
    test -s "${EVENT_FEATURES_PATH}"
    test -d "${DAILY_DIR}"
    test -s "${CLIMATOLOGY_PATH}"
fi
test ! -e "${RUN_DIR}"
test ! -L "${RUN_DIR}"
mkdir -p "$(dirname "${RUN_DIR}")"

STAGING_DIR=$(mktemp -d "${RUN_DIR}.staging.XXXXXX")
trap 'status=$?; echo "[info] exit_status=${status} finished=$(date -Is)"; if (( status != 0 )); then echo "[info] retained_staging=${STAGING_DIR}"; fi' EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=error
export MPLBACKEND=Agg
export MPLCONFIGDIR="${STAGING_DIR}/.matplotlib"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${VENUS_MAMBA_ENV:-dev_env}"
python -c 'import sys, numpy, xarray, h5netcdf, matplotlib, cartopy; print("[info] python=" + sys.executable)'

echo "[info] started=$(date -Is) job_id=${PBS_JOBID} host=$(hostname)"
echo "[info] commit=${actual_commit} project_root=${PROJECT_ROOT}"
echo "[info] event_features_path=${EVENT_FEATURES_PATH:-not_used} region=${REGION:-from_product}"
echo "[info] daily_dir=${DAILY_DIR:-not_used} climatology_path=${CLIMATOLOGY_PATH:-not_used}"
echo "[info] prepared_map_path=${PREPARED_MAP_PATH} source_validation_path=${SOURCE_VALIDATION_PATH}"
echo "[info] run_dir=${RUN_DIR} log=${LOGFILE}"
echo "[info] resources=select=1:ncpus=1:mem=8gb walltime=00:20:00"

if [[ -n "${PREPARED_MAP_PATH}" ]]; then
    python - "${PREPARED_MAP_PATH}" "${SOURCE_VALIDATION_PATH}" "${STAGING_DIR}" <<'PYTHON'
import shutil
import sys
from pathlib import Path
from src.top_events_map_validation import validate_saved_source_report

product, report, staging = map(Path, sys.argv[1:])
validate_saved_source_report(product, report)
for source, name in ((product, "top_events_map.nc"), (report, "source_validation.json")):
    with source.open("rb") as src, (staging / name).open("xb") as dest:
        shutil.copyfileobj(src, dest)
validate_saved_source_report(staging / "top_events_map.nc", staging / "source_validation.json")
PYTHON
else
    read -r -a extent_args <<< "${MAP_EXTENT}"
    build_args=(
        --event-features-path "${EVENT_FEATURES_PATH}" --region "${REGION}"
        --daily-dir "${DAILY_DIR}" --climatology-path "${CLIMATOLOGY_PATH}"
        --climatology-start-year "${CLIMATOLOGY_START_YEAR}"
        --climatology-end-year "${CLIMATOLOGY_END_YEAR}"
        --top-n "${TOP_N}" --rank-metric "${RANK_METRIC}"
        --extent "${extent_args[@]}" --output-path "${STAGING_DIR}/top_events_map.nc"
    )
    if [[ -n "${PEAK_YEAR:-}" ]]; then
        build_args+=(--peak-year "${PEAK_YEAR}")
    fi
    /usr/bin/time -v python scripts/top_events_map/build_top_events_map.py "${build_args[@]}"
fi

plot_args=(--input-path "${STAGING_DIR}/top_events_map.nc" --output-dir "${STAGING_DIR}/figures"
           --height-contour-interval "${HEIGHT_CONTOUR_INTERVAL:-50}")
if [[ -n "${TEMPERATURE_LIMIT:-}" ]]; then
    plot_args+=(--temperature-limit "${TEMPERATURE_LIMIT}")
fi
if [[ -n "${OUTLINE_REGIONS:-}" ]]; then
    read -r -a outline_args <<< "${OUTLINE_REGIONS}"
    plot_args+=(--outline-regions "${outline_args[@]}")
fi
if [[ -n "${PLOT_RANKS}" ]]; then
    read -r -a rank_args <<< "${PLOT_RANKS}"
    plot_args+=(--event-ranks "${rank_args[@]}")
fi
/usr/bin/time -v python scripts/top_events_map/plot_top_events_map.py "${plot_args[@]}"

if [[ -z "${PREPARED_MAP_PATH}" ]]; then
    /usr/bin/time -v python scripts/top_events_map/validate_top_events_map.py \
        --input-path "${STAGING_DIR}/top_events_map.nc" \
        --output-path "${STAGING_DIR}/source_validation.json"
fi

python - "${STAGING_DIR}" "${RUN_DIR}" "${LOGFILE}" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src import analysis_io, plot_style
from src.top_events_map import sha256_file
from src.top_events_map_plotting import MAP_FIGURE_ASPECT, event_indices_for_ranks, map_filename
from src.top_events_map_validation import validate_saved_source_report

staging, destination, logfile = map(Path, sys.argv[1:])
figure_size = plot_style.publication_figsize("single", aspect=MAP_FIGURE_ASPECT)
with analysis_io.open_top_event_maps(staging / "top_events_map.nc") as ds:
    prepared_path = os.environ["PREPARED_MAP_PATH"]
    if prepared_path:
        assert sha256_file(prepared_path) == sha256_file(staging / "top_events_map.nc")
        assert sha256_file(os.environ["SOURCE_VALIDATION_PATH"]) == sha256_file(staging / "source_validation.json")
    else:
        assert ds.attrs["source_commit"] == os.environ["EXPECTED_COMMIT"]
    validate_saved_source_report(staging / "top_events_map.nc", staging / "source_validation.json")
    assert ds.attrs["source_tree_dirty"] == 0
    ranks = [int(rank) for rank in os.environ["PLOT_RANKS"].split()] or None
    indices = event_indices_for_ranks(ds, ranks)
    figures = [staging / "figures" / map_filename(ds, i) for i in indices]
    assert set(figures) == set((staging / "figures").glob("*.png"))
    for path in figures:
        with Image.open(path) as image:
            image.load()
            assert image.size == tuple(round(size * plot_style.DPI) for size in figure_size)
            assert np.asarray(image.convert("RGB")).std() > 1, f"Blank figure: {path}"
    artifacts = [staging / "top_events_map.nc", staging / "source_validation.json", *figures]
    manifest = {
        "pipeline_stage": "top_events_map",
        "commit": os.environ["EXPECTED_COMMIT"],
        "product_commit": ds.attrs["source_commit"],
        "mode": "prepared_product_replot" if prepared_path else "build_and_plot",
        "prepared_input": prepared_path or None,
        "saved_source_validation": os.environ["SOURCE_VALIDATION_PATH"] or None,
        "source_validation_reused": bool(prepared_path),
        "rendered_ranks": ds.selection_rank.values[indices].tolist(),
        "rendered_event_ids": ds.event_id.values[indices].tolist(),
        "typography": {"width_inches": figure_size[0],
                       "height_inches": figure_size[1],
                       "label_font_pt": plot_style.SINGLE_COLUMN_FONT_SIZE_PT,
                       "legend_font_pt": plot_style.SINGLE_COLUMN_LEGEND_FONT_SIZE_PT},
        "resolved_temperature_limit_K": float(os.environ.get("TEMPERATURE_LIMIT") or max(1.0, np.ceil(abs(ds.t2m_anomaly).max().item()))),
        "resolved_height_contour_interval_m": float(os.environ.get("HEIGHT_CONTOUR_INTERVAL", "50")),
        "project_root": os.environ["PROJECT_ROOT"],
        "scheduler": "scripts/top_events_map/schedule_top_events_map.sh",
        "job_id": os.environ["PBS_JOBID"],
        "host": socket.gethostname(),
        "python": sys.executable,
        "environment": os.environ.get("VENUS_MAMBA_ENV", "dev_env"),
        "requested_resources": "select=1:ncpus=1:mem=8gb,walltime=00:20:00",
        "run_dir": str(destination), "log_path": str(logfile),
        "region": ds.attrs["region"], "rank_metric": ds.attrs["rank_metric"],
        "peak_year_filter": ds.attrs["peak_year_filter"],
        "event_ids": ds.event_id.values.tolist(),
        "sample_dates": ds.sample_date.values.astype("datetime64[D]").astype(str).tolist(),
        "climatology_years": [int(ds.attrs["climatology_start_year"]), int(ds.attrs["climatology_end_year"])],
        "map_extent": list(ds.attrs["map_extent"]),
        "plot_options": {key: os.environ.get(key) for key in
                         ("OUTLINE_REGIONS", "TEMPERATURE_LIMIT", "HEIGHT_CONTOUR_INTERVAL")},
        "input_files": json.loads(ds.attrs["source_files"]),
        "event_features_sha256": ds.attrs["event_features_sha256"],
        "artifacts": {str(path.relative_to(staging)): sha256_file(path) for path in artifacts},
        "validation": ("product contract, unchanged prepared input, retained independent-source audit with matching product hash, PNG decode and nonblank"
                       if prepared_path else "product contract, independent ranking and daily-source sums, finite arrays, anomaly identities, PNG decode and nonblank"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
with (staging / "manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(json.dumps(manifest, indent=2))
PY

mv -T --no-clobber -- "${STAGING_DIR}" "${RUN_DIR}"
test ! -d "${STAGING_DIR}"
echo "[info] published=${RUN_DIR}"
