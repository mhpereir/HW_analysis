#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

T2M_ROOT="/home/mhpereir/downloads-mhpereir/REANALYSIS/ERA5/hourly/2mT"
Z500_ROOT="/home/mhpereir/downloads-mhpereir/REANALYSIS/ERA5/hourly/z500"
OUTPUT_DIR="${REPO_ROOT}/results/spatial_composites/daily"
START_YEAR=1940
END_YEAR=2024
THREADS=1
OVERWRITE=0
SKIP_EXISTING=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Build global native-grid daily-mean ERA5 T2m/Z500 files with CDO.

Usage: build_era5_daily_spatial_data.sh [options]

Options:
  --t2m-root PATH       Hourly T2m annual-file directory.
  --z500-root PATH      Hourly Z500 annual-file directory.
  --output-dir PATH     Directory for combined annual daily files.
  --start-year YEAR     First year to process (default: 1940).
  --end-year YEAR       Last year to process (default: 2024).
  --threads N           CDO OpenMP thread count (default: 1).
  --overwrite           Replace existing annual daily files.
  --skip-existing       Leave validated existing annual outputs unchanged.
  --dry-run             Validate inputs and print commands without writing.
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
        --t2m-root)
            require_value "$@"; T2M_ROOT="$2"; shift 2 ;;
        --z500-root)
            require_value "$@"; Z500_ROOT="$2"; shift 2 ;;
        --output-dir)
            require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
        --start-year)
            require_value "$@"; START_YEAR="$2"; shift 2 ;;
        --end-year)
            require_value "$@"; END_YEAR="$2"; shift 2 ;;
        --threads)
            require_value "$@"; THREADS="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --skip-existing)
            SKIP_EXISTING=1; shift ;;
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
(( OVERWRITE == 0 || SKIP_EXISTING == 0 )) || \
    die "--overwrite and --skip-existing are mutually exclusive."
command -v cdo >/dev/null 2>&1 || die "CDO is required but was not found on PATH."
command -v nccopy >/dev/null 2>&1 || die "nccopy is required but was not found on PATH."

run() {
    if (( DRY_RUN )); then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

is_leap_year() {
    local year=$1
    (( (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 ))
}

validate_names() {
    local path=$1
    local names
    names=" $(cdo -s showname "$path") "
    [[ "$names" == *" t2m "* ]] || die "$path is missing variable t2m."
    [[ "$names" == *" z "* ]] || die "$path is missing variable z."
}

validate_daily_file() {
    local path=$1
    local year=$2
    local expected_days=365
    local ntime xsize ysize levels previous date_value time_values time_value

    is_leap_year "$year" && expected_days=366
    ntime="$(cdo -s ntime "$path" | tr -d '[:space:]')"
    [[ "$ntime" == "$expected_days" ]] || die "$path has $ntime days; expected $expected_days."
    validate_names "$path"

    levels=" $(cdo -s showlevel -selname,z "$path") "
    [[ "$levels" == *" 500 "* ]] || die "$path does not contain Z at 500 hPa."

    xsize="$(cdo -s griddes -selname,t2m "$path" | awk '$1 == "xsize" {print $3; exit}')"
    ysize="$(cdo -s griddes -selname,t2m "$path" | awk '$1 == "ysize" {print $3; exit}')"
    [[ "$xsize" == "1440" && "$ysize" == "721" ]] || \
        die "$path is not on the expected 1440x721 global grid."

    time_values="$(cdo -s showtime "$path")"
    for time_value in $time_values; do
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

if (( ! DRY_RUN )); then
    mkdir -p "$OUTPUT_DIR"
    WORK_DIR="$(mktemp -d "${OUTPUT_DIR}/.daily_era5.XXXXXX")"
    trap 'rm -rf -- "${WORK_DIR}"' EXIT
else
    WORK_DIR="${OUTPUT_DIR}/.daily_era5.DRY_RUN"
fi

for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    t2m_input="${T2M_ROOT}/2mT_hour_ERA5_${year}.nc"
    z500_input="${Z500_ROOT}/z500_hour_ERA5_${year}.nc"
    output_path="${OUTPUT_DIR}/ERA5_daily_t2m_z500_${year}.nc"

    [[ -f "$t2m_input" ]] || die "Missing hourly T2m input: $t2m_input"
    [[ -f "$z500_input" ]] || die "Missing hourly Z500 input: $z500_input"
    if [[ -e "$output_path" ]]; then
        if (( SKIP_EXISTING )); then
            echo "[info] Skipping existing output: $output_path"
            continue
        elif (( ! OVERWRITE )); then
            die "Output exists: $output_path (pass --overwrite to replace it)."
        fi
    fi

    t2m_rechunked="${WORK_DIR}/t2m_hourly_rechunked_${year}.nc"
    z500_rechunked="${WORK_DIR}/z500_hourly_rechunked_${year}.nc"
    t2m_daily="${WORK_DIR}/t2m_daily_${year}.nc"
    z500_daily="${WORK_DIR}/z500_daily_${year}.nc"
    merged_daily="${WORK_DIR}/ERA5_daily_t2m_z500_${year}.nc"

    echo "[info] Building daily ERA5 spatial data for $year"
    echo "[info] Rechunking hourly T2m input for daily aggregation"
    run /usr/bin/time -v nccopy -d 1 -s \
        -c valid_time/24,latitude/180,longitude/180 \
        "$t2m_input" "$t2m_rechunked"
    run cdo -O -L -P "$THREADS" -f nc4c -b F32 -z zip_4 \
        settime,00:00:00 -daymean -selname,t2m \
        "$t2m_rechunked" "$t2m_daily"
    run rm -f -- "$t2m_rechunked"

    echo "[info] Rechunking hourly Z500 input for daily aggregation"
    run /usr/bin/time -v nccopy -d 1 -s \
        -c valid_time/24,pressure_level/1,latitude/180,longitude/180 \
        "$z500_input" "$z500_rechunked"
    run cdo -O -L -P "$THREADS" -f nc4c -b F32 -z zip_4 \
        settime,00:00:00 -daymean -sellevel,500 -selname,z \
        "$z500_rechunked" "$z500_daily"
    run rm -f -- "$z500_rechunked"

    run cdo -O -L -P "$THREADS" -f nc4c -b F32 -z zip_4 \
        -setattribute,pipeline_stage=era5_daily_spatial_data,daily_aggregation=UTC_calendar_day_arithmetic_mean,daily_source_samples=24 \
        -merge "$t2m_daily" "$z500_daily" "$merged_daily"

    if (( ! DRY_RUN )); then
        validate_daily_file "$merged_daily" "$year"
        mv -f "$merged_daily" "$output_path"
        rm -f "$t2m_daily" "$z500_daily"
        echo "[info] Wrote $output_path"
    fi
done

echo "[info] Daily ERA5 preprocessing complete."
