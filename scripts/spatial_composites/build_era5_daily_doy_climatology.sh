#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DAILY_DIR="${REPO_ROOT}/results/spatial_composites/daily"
OUTPUT_PATH="${REPO_ROOT}/results/spatial_composites/climatology/era5_daily_doy_climatology_t2m_z500_global_1940_2024.nc"
START_YEAR=1940
END_YEAR=2024
THREADS=4
OVERWRITE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Build a global daily ERA5 day-of-year climatology with CDO.

Usage: build_era5_daily_doy_climatology.sh [options]

Options:
  --daily-dir PATH      Directory containing combined annual daily files.
  --output-path PATH    Climatology NetCDF output path.
  --start-year YEAR     First climatology year (default: 1940).
  --end-year YEAR       Last climatology year (default: 2024).
  --threads N           CDO OpenMP thread count (default: 4).
  --overwrite           Replace an existing climatology file.
  --dry-run             Validate inputs and print the CDO command only.
  -h, --help            Show this help.
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
        --daily-dir)
            require_value "$@"; DAILY_DIR="$2"; shift 2 ;;
        --output-path)
            require_value "$@"; OUTPUT_PATH="$2"; shift 2 ;;
        --start-year)
            require_value "$@"; START_YEAR="$2"; shift 2 ;;
        --end-year)
            require_value "$@"; END_YEAR="$2"; shift 2 ;;
        --threads)
            require_value "$@"; THREADS="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
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
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer."
(( START_YEAR <= END_YEAR )) || die "--start-year must be <= --end-year."
if (( ! DRY_RUN )); then
    command -v cdo >/dev/null 2>&1 || die "CDO is required but was not found on PATH."
fi

if [[ -e "$OUTPUT_PATH" && "$OVERWRITE" -ne 1 ]]; then
    die "Output exists: $OUTPUT_PATH (pass --overwrite to replace it)."
fi

daily_files=()
for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    path="${DAILY_DIR}/ERA5_daily_t2m_z500_${year}.nc"
    [[ -f "$path" ]] || die "Missing annual daily input: $path"
    daily_files+=("$path")
done

run() {
    if (( DRY_RUN )); then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

validate_climatology() {
    local path=$1
    local expected_days=365
    local ntime names levels xsize ysize previous date_value time_value
    for (( year=START_YEAR; year<=END_YEAR; year++ )); do
        if (( (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 )); then
            expected_days=366
            break
        fi
    done
    ntime="$(cdo -s ntime "$path" | tr -d '[:space:]')"
    [[ "$ntime" == "$expected_days" ]] || \
        die "$path has $ntime days; expected $expected_days."
    names=" $(cdo -s showname "$path") "
    [[ "$names" == *" t2m "* && "$names" == *" z "* ]] || \
        die "$path must contain t2m and z."
    levels=" $(cdo -s showlevel -selname,z "$path") "
    [[ "$levels" == *" 500 "* ]] || die "$path does not contain Z at 500 hPa."
    xsize="$(cdo -s griddes -selname,t2m "$path" | awk '$1 == "xsize" {print $3; exit}')"
    ysize="$(cdo -s griddes -selname,t2m "$path" | awk '$1 == "ysize" {print $3; exit}')"
    [[ "$xsize" == "1440" && "$ysize" == "721" ]] || \
        die "$path is not on the expected 1440x721 global grid."
    for time_value in $(cdo -s showtime "$path"); do
        [[ "$time_value" == "00:00:00" ]] || \
            die "$path contains non-midnight daily timestamp $time_value."
    done
    for date_value in $(cdo -s showdate "$path"); do
        if [[ -n "${previous:-}" && ! "$date_value" > "$previous" ]]; then
            die "$path dates are not strictly increasing."
        fi
        previous="$date_value"
    done
}

if (( DRY_RUN )); then
    temp_output="${OUTPUT_PATH}.DRY_RUN"
else
    output_dir="$(dirname "$OUTPUT_PATH")"
    mkdir -p "$output_dir"
    work_dir="$(mktemp -d "${output_dir}/.daily_climatology.XXXXXX")"
    trap 'rm -rf "${work_dir}"' EXIT
    temp_output="${work_dir}/$(basename "$OUTPUT_PATH")"
fi

echo "[info] Building global daily climatology for ${START_YEAR}-${END_YEAR}"
run cdo -O -L -P "$THREADS" -f nc4c -b F32 -z zip_4 \
    -setattribute,pipeline_stage=era5_daily_doy_climatology,climatology_start_year="$START_YEAR",climatology_end_year="$END_YEAR",climatology_method=CDO_ydaymean,daily_aggregation=UTC_calendar_day_arithmetic_mean \
    -ydaymean -mergetime "${daily_files[@]}" "$temp_output"

if (( ! DRY_RUN )); then
    validate_climatology "$temp_output"
    mv -f "$temp_output" "$OUTPUT_PATH"
    echo "[info] Wrote $OUTPUT_PATH"
fi

echo "[info] Daily ERA5 climatology complete."
