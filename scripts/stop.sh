#!/usr/bin/env bash

set -Eeuo pipefail
set -o errtrace

readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly SCRIPT_NAME="${SCRIPT_PATH##*/}"
readonly SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" >/dev/null 2>&1 && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
readonly DEFAULT_RUNTIME_DIR="${REPO_ROOT}/.runtime/demo"

RUNTIME_DIR="${DEFAULT_RUNTIME_DIR}"
CONFIG_PATH=""

fct_log_info() {
	printf '%s\n' "[INFO] $*"
}

fct_log_error() {
	printf '%s\n' "[ERROR] $*" >&2
}

fct_die() {
	local message="${1}"
	local exit_code="${2:-1}"
	fct_log_error "${message}"
	exit "${exit_code}"
}

fct_usage() {
	cat <<EOF
${SCRIPT_NAME}
Stop and clean the local demo environment created by scripts/start.sh.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  --runtime-dir PATH   Runtime directory to remove.
                       Default: ${DEFAULT_RUNTIME_DIR}
  --config PATH        Config file inside a runtime directory.
                       When provided, runtime-dir defaults to the config's parent directory.
  -h, --help           Show this help and exit

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} --runtime-dir "${REPO_ROOT}/.runtime/demo-alt"
EOF
}

fct_parse_arguments() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--runtime-dir)
			[[ $# -ge 2 ]] || fct_die "Option --runtime-dir requires a path." 2
			RUNTIME_DIR="$2"
			shift 2
			;;
		--config)
			[[ $# -ge 2 ]] || fct_die "Option --config requires a path." 2
			CONFIG_PATH="$2"
			shift 2
			;;
		-h | --help)
			fct_usage
			exit 0
			;;
		*)
			fct_die "Unknown option: $1" 2
			;;
		esac
	done
}

fct_resolve_runtime_dir() {
	if [[ -n "${CONFIG_PATH}" ]]; then
		RUNTIME_DIR="$(realpath -m "$(dirname "${CONFIG_PATH}")")"
		return 0
	fi
	RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"
}

fct_validate_runtime_dir() {
	[[ -n "${RUNTIME_DIR}" ]] || fct_die "Runtime directory must not be empty."
	[[ "${RUNTIME_DIR}" != "/" ]] || fct_die "Refusing to remove '/'."
	[[ "${RUNTIME_DIR}" != "${REPO_ROOT}" ]] || fct_die "Refusing to remove the repository root."
	if [[ -e "${RUNTIME_DIR}" ]] && [[ ! -f "${RUNTIME_DIR}/session.env" ]]; then
		fct_die "Refusing to remove a directory that was not created by scripts/start.sh."
	fi
}

fct_cleanup_runtime_dir() {
	if [[ ! -e "${RUNTIME_DIR}" ]]; then
		fct_log_info "Nothing to stop. Runtime directory does not exist: ${RUNTIME_DIR}"
		return 0
	fi

	rm -rf "${RUNTIME_DIR}"
	fct_log_info "Removed runtime directory: ${RUNTIME_DIR}"
}

fct_main() {
	fct_parse_arguments "$@"
	fct_resolve_runtime_dir
	fct_validate_runtime_dir
	fct_cleanup_runtime_dir
}

fct_main "$@"
