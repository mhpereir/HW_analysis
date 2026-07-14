#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEDULER="${REPO_ROOT}/schedulers/schedule_build_era5_daily_spatial_data.sh"
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
if (( DRY_RUN )); then
    printf '[dry-run] qsub -J %q %q\n' "$array_spec" "$SCHEDULER"
else
    command -v qsub >/dev/null 2>&1 || die "qsub was not found on PATH."
    job_id="$(qsub -J "$array_spec" "$SCHEDULER")"
    echo "[info] Submitted ERA5 daily array ${array_spec}: ${job_id}"
fi
