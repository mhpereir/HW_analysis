#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
source "${PROJECT_ROOT}/config/artifact_paths.sh"
SCHEDULER="${PROJECT_ROOT}/schedulers/schedule_build_era5_daily_spatial_data.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${HWA_ARTIFACT_ROOT}/spatial_composites/daily}"
LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"
START_YEAR=1940
END_YEAR=2024
MAX_CONCURRENT=8
DRY_RUN=0

usage() {
    cat <<'EOF'
Submit annual ERA5 daily aggregation as a throttled PBS array.

Usage: submit_era5_daily_spatial_array.sh [options]

Options:
  --start-year YEAR       First array year (default: 1940).
  --end-year YEAR         Last array year (default: 2024).
  --max-concurrent N      Maximum simultaneously running years (default: 8).
  --dry-run               Print the qsub command without submitting.
  -h, --help              Show this help.
EOF
}

die() {
    echo "[error] $*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-year)
            require_value "$@"; START_YEAR="$2"; shift 2 ;;
        --end-year)
            require_value "$@"; END_YEAR="$2"; shift 2 ;;
        --max-concurrent)
            require_value "$@"; MAX_CONCURRENT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "Unknown option: $1" ;;
    esac
done

[[ "$START_YEAR" =~ ^[0-9]{4}$ ]] || die "--start-year must be a four-digit year."
[[ "$END_YEAR" =~ ^[0-9]{4}$ ]] || die "--end-year must be a four-digit year."
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || \
    die "--max-concurrent must be a positive integer."
(( START_YEAR <= END_YEAR )) || die "--start-year must be <= --end-year."
[[ -f "$SCHEDULER" ]] || die "Missing PBS scheduler: $SCHEDULER"

array_spec="${START_YEAR}-${END_YEAR}%${MAX_CONCURRENT}"
actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
EXPECTED_COMMIT="${EXPECTED_COMMIT:-${actual_commit}}"
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || die "EXPECTED_COMMIT does not match PROJECT_ROOT."
hwa_validate_artifact_paths OUTPUT_DIR
hwa_validate_log_dir
export PROJECT_ROOT EXPECTED_COMMIT HWA_ARTIFACT_ROOT HWA_LOG_ROOT OUTPUT_DIR LOG_DIR
export_names=PROJECT_ROOT,EXPECTED_COMMIT,HWA_ARTIFACT_ROOT,HWA_LOG_ROOT,OUTPUT_DIR,LOG_DIR
qsub_args=(-J "$array_spec" -v "$export_names" "$SCHEDULER")
if (( DRY_RUN )); then
    for name in PROJECT_ROOT EXPECTED_COMMIT HWA_ARTIFACT_ROOT HWA_LOG_ROOT OUTPUT_DIR LOG_DIR; do
        printf '[dry-run] export %s=%q\n' "$name" "${!name}"
    done
    printf '[dry-run] qsub'
    printf ' %q' "${qsub_args[@]}"
    printf '\n'
else
    [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]] || \
        die "Refusing submission from a dirty source checkout."
    command -v qsub >/dev/null 2>&1 || die "qsub was not found on PATH."
    cd "${PROJECT_ROOT}"
    job_id="$(qsub "${qsub_args[@]}")"
    echo "[info] Submitted ERA5 daily array ${array_spec}: ${job_id}"
fi
