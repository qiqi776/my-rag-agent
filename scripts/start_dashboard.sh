#!/usr/bin/env bash

set -Eeuo pipefail
set -o errtrace

readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly SCRIPT_NAME="${SCRIPT_PATH##*/}"
readonly SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" >/dev/null 2>&1 && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
readonly DEFAULT_CONFIG_PATH="${REPO_ROOT}/config/settings.yaml.example"
readonly DEFAULT_ADDRESS="127.0.0.1"
readonly DEFAULT_PORT="8501"

CONFIG_PATH="${DEFAULT_CONFIG_PATH}"
ADDRESS="${DEFAULT_ADDRESS}"
PORT="${DEFAULT_PORT}"
SKIP_SYNC=0

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
Start the local Streamlit dashboard for the minimal modular RAG project.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  --config PATH     Settings file passed to the dashboard.
                    Default: ${DEFAULT_CONFIG_PATH}
  --address HOST    Bind address for Streamlit.
                    Default: ${DEFAULT_ADDRESS}
  --port PORT       Bind port for Streamlit.
                    Default: ${DEFAULT_PORT}
  --skip-sync       Skip 'uv sync --extra dev --extra dashboard'
  -h, --help        Show this help and exit
EOF
}

fct_parse_arguments() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--config)
			[[ $# -ge 2 ]] || fct_die "Option --config requires a path." 2
			CONFIG_PATH="$2"
			shift 2
			;;
		--address)
			[[ $# -ge 2 ]] || fct_die "Option --address requires a value." 2
			ADDRESS="$2"
			shift 2
			;;
		--port)
			[[ $# -ge 2 ]] || fct_die "Option --port requires a value." 2
			PORT="$2"
			shift 2
			;;
		--skip-sync)
			SKIP_SYNC=1
			shift
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

fct_prepare_environment() {
	if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
		fct_log_info "Creating local virtual environment with uv."
		(
			cd "${REPO_ROOT}"
			uv venv
		)
	fi

	if [[ "${SKIP_SYNC}" -eq 0 ]]; then
		fct_log_info "Syncing project dependencies for the dashboard."
		(
			cd "${REPO_ROOT}"
			uv sync --extra dev --extra dashboard
		)
	fi
}

fct_start_dashboard() {
	local resolved_config
	resolved_config="$(realpath -m "${CONFIG_PATH}")"
	[[ -f "${resolved_config}" ]] || fct_die "Config file not found: ${resolved_config}"

	fct_log_info "Starting dashboard on http://${ADDRESS}:${PORT}"
	export MRAG_DASHBOARD_CONFIG="${resolved_config}"
	(
		cd "${REPO_ROOT}"
		uv run streamlit run src/observability/dashboard/app.py \
			--server.address "${ADDRESS}" \
			--server.port "${PORT}" \
			--browser.gatherUsageStats false
	)
}

fct_main() {
	fct_parse_arguments "$@"
	fct_prepare_environment
	fct_start_dashboard
}

fct_main "$@"
