# Shared by active PBS entrypoints and the spatial preparation shell CLIs.
# Source this file after PROJECT_ROOT (PBS) or REPO_ROOT (direct CLI) is known.
# Keep defaults aligned with src/artifact_paths.py. Sourcing creates no files.

hwa_path_error() {
    echo "[error] $*" >&2
    return 1
}

hwa_storage_root() {
    local name="$1" resolved source_root
    [[ "${!name}" == /* ]] || {
        hwa_path_error "${name} must be an absolute path: ${!name}"
        return 1
    }
    resolved=$(realpath -m -- "${!name}") || return 1
    if [[ "${resolved}" == / || "${resolved}" == "$(realpath -m -- "${HOME}")" ]]; then
        hwa_path_error "${name} must be a dedicated storage directory: ${resolved}"
        return 1
    fi
    source_root="${PROJECT_ROOT:-${REPO_ROOT:-}}"
    if [[ -n "${source_root}" ]]; then
        [[ "${source_root}" == /* ]] || {
            hwa_path_error "Source checkout must be absolute: ${source_root}"
            return 1
        }
        source_root=$(realpath -m -- "${source_root}") || return 1
        case "${resolved}" in
            "${source_root}"|"${source_root}"/*)
                hwa_path_error "${name} must be outside the source checkout: ${resolved}"
                return 1 ;;
        esac
        case "${source_root}" in
            "${resolved}"/*)
                hwa_path_error "${name} must not contain the source checkout: ${resolved}"
                return 1 ;;
        esac
    fi
    printf -v "${name}" '%s' "${resolved}"
    export "${name}"
}

HWA_ARTIFACT_ROOT="${HWA_ARTIFACT_ROOT-${HOME:?HOME is required}/HW-analysis/artifacts}"
HWA_LOG_ROOT="${HWA_LOG_ROOT-${HOME:?HOME is required}/HW-analysis/logs}"
hwa_storage_root HWA_ARTIFACT_ROOT || return 1
hwa_storage_root HWA_LOG_ROOT || return 1

hwa_validate_artifact_paths() {
    # Pass variable names, not values, so errors identify the offending setting.
    local name resolved
    for name in "$@"; do
        [[ "${!name}" == /* ]] || {
            hwa_path_error "${name} must be an absolute prepared-product path: ${!name}"
            return 1
        }
        resolved=$(realpath -m -- "${!name}") || return 1
        case "${resolved}" in
            "${HWA_ARTIFACT_ROOT}"/*) ;;
            *)
                hwa_path_error "${name} must be beneath HWA_ARTIFACT_ROOT=${HWA_ARTIFACT_ROOT}: ${resolved}"
                return 1 ;;
        esac
        printf -v "${name}" '%s' "${resolved}"
    done
}

hwa_validate_log_dir() {
    local resolved
    LOG_DIR="${LOG_DIR-${HWA_LOG_ROOT}}"
    [[ "${LOG_DIR}" == /* ]] || {
        hwa_path_error "LOG_DIR must be absolute: ${LOG_DIR}"
        return 1
    }
    resolved=$(realpath -m -- "${LOG_DIR}") || return 1
    case "${resolved}" in
        "${HWA_LOG_ROOT}"|"${HWA_LOG_ROOT}"/*) ;;
        *)
            hwa_path_error "LOG_DIR must be within HWA_LOG_ROOT=${HWA_LOG_ROOT}: ${resolved}"
            return 1 ;;
    esac
    LOG_DIR="${resolved}"
}

hwa_start_log() {
    local suffix="$1" job_id="${PBS_JOBID:?PBS_JOBID is required}"
    hwa_validate_log_dir || return 1
    [[ "${job_id}" != */* && "${suffix}" != */* ]] || {
        hwa_path_error "Log job ID and suffix must not contain slashes."
        return 1
    }
    LOGFILE="${LOG_DIR}/${job_id}_${suffix}.log"
    mkdir -p -- "${LOG_DIR}"
    # Atomic no-clobber creation also rejects existing files and symlinks.
    (set -o noclobber; : > "${LOGFILE}") || {
        hwa_path_error "Cannot create a fresh log: ${LOGFILE}"
        return 1
    }
    exec > >(tee -a "${LOGFILE}") 2>&1
    echo "[info] project_root=${PROJECT_ROOT:-${REPO_ROOT:-unset}}"
    echo "[info] artifact_root=${HWA_ARTIFACT_ROOT}"
    echo "[info] log_root=${HWA_LOG_ROOT}"
    echo "[info] logfile=${LOGFILE}"
}
