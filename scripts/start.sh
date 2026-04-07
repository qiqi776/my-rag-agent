#!/usr/bin/env bash

set -Eeuo pipefail
set -o errtrace

readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly SCRIPT_NAME="${SCRIPT_PATH##*/}"
readonly SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" >/dev/null 2>&1 && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
readonly DEFAULT_RUNTIME_DIR="${REPO_ROOT}/.runtime/demo"
readonly DEFAULT_DOCS_DIR="${REPO_ROOT}/docs"
readonly DEFAULT_COLLECTION="knowledge"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

RUNTIME_DIR="${DEFAULT_RUNTIME_DIR}"
CONFIG_PATH=""
DOCS_DIR=""
COLLECTION="${DEFAULT_COLLECTION}"
SKIP_SYNC=0
SKIP_INGEST=0
UV_CACHE_DIR_PATH=""
CONFIG_EXPLICIT=0
LOADER_PROVIDER="text"
SUPPORTED_EXTENSIONS_BLOCK='    - ".txt"
    - ".md"'

fct_log_info() {
	printf '%s\n' "[INFO] $*"
}

fct_log_warn() {
	printf '%s\n' "[WARN] $*" >&2
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
Start a local demo environment for the minimal modular RAG project.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  --runtime-dir PATH   Runtime directory to create and manage.
                       Default: ${DEFAULT_RUNTIME_DIR}
  --config PATH        Config file path to create/use.
                       Default: <runtime-dir>/settings.yaml
  --docs-dir PATH      Docs directory to create/use.
                       Default: ${DEFAULT_DOCS_DIR}
  --collection NAME    Collection name used for initial ingest.
                       Default: ${DEFAULT_COLLECTION}
  --skip-sync          Skip 'uv sync --extra dev'
  --skip-ingest        Skip the initial ingest step
  -h, --help           Show this help and exit

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} --skip-sync
  ${SCRIPT_NAME} --runtime-dir "${REPO_ROOT}/.runtime/demo-alt" --collection knowledge
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
			CONFIG_EXPLICIT=1
			shift 2
			;;
		--docs-dir)
			[[ $# -ge 2 ]] || fct_die "Option --docs-dir requires a path." 2
			DOCS_DIR="$2"
			shift 2
			;;
		--collection)
			[[ $# -ge 2 ]] || fct_die "Option --collection requires a value." 2
			COLLECTION="$2"
			shift 2
			;;
		--skip-sync)
			SKIP_SYNC=1
			shift
			;;
		--skip-ingest)
			SKIP_INGEST=1
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

fct_resolve_runtime_paths() {
	RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"
	if [[ -z "${CONFIG_PATH}" ]]; then
		CONFIG_PATH="${RUNTIME_DIR}/settings.yaml"
	else
		CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
	fi
	if [[ -z "${DOCS_DIR}" ]]; then
		DOCS_DIR="${DEFAULT_DOCS_DIR}"
	else
		DOCS_DIR="$(realpath -m "${DOCS_DIR}")"
	fi
	UV_CACHE_DIR_PATH="${RUNTIME_DIR}/uv-cache"
}

fct_require_command() {
	local command_name="${1}"
	command -v "${command_name}" >/dev/null 2>&1 || fct_die "Required command not found: ${command_name}"
}

fct_prepare_python_environment() {
	if [[ ! -x "${PYTHON_BIN}" ]]; then
		fct_log_info "Creating local virtual environment with uv."
		(
			cd "${REPO_ROOT}"
			uv venv
		)
	fi

	if [[ "${SKIP_SYNC}" -eq 0 ]]; then
		fct_log_info "Syncing project dependencies."
		(
			cd "${REPO_ROOT}"
			uv sync --extra dev
		)
	else
		fct_log_info "Skipping dependency sync."
	fi
}

fct_prepare_runtime_dirs() {
	mkdir -p "${RUNTIME_DIR}" "${UV_CACHE_DIR_PATH}" "$(dirname "${CONFIG_PATH}")"
}

fct_prepare_docs_dir() {
	if [[ ! -d "${DOCS_DIR}" ]]; then
		mkdir -p "${DOCS_DIR}"
		fct_log_info "Created docs directory at ${DOCS_DIR}."
	fi
}

fct_configure_uv_environment() {
	export UV_CACHE_DIR="${UV_CACHE_DIR_PATH}"
}

fct_write_demo_doc_if_missing() {
	local demo_doc_path="${DOCS_DIR}/getting-started.txt"
	if find "${DOCS_DIR}" -maxdepth 1 -type f | grep -q .; then
		fct_log_info "Using existing documents in ${DOCS_DIR}."
		return 0
	fi

	cat >"${demo_doc_path}" <<EOF
Minimal Modular RAG Demo

This local demo environment is ready.
Use semantic embeddings, hybrid retrieval, and agent workflows to explore the project.
EOF
	fct_log_info "Created demo document at ${demo_doc_path}."
}

fct_detect_loader_profile() {
	local has_pdf=0
	local has_text=0

	if find "${DOCS_DIR}" -type f \( -iname '*.pdf' \) | grep -q .; then
		has_pdf=1
	fi
	if find "${DOCS_DIR}" -type f \( -iname '*.txt' -o -iname '*.md' \) | grep -q .; then
		has_text=1
	fi

	if [[ "${has_pdf}" -eq 1 ]]; then
		LOADER_PROVIDER="pdf"
		SUPPORTED_EXTENSIONS_BLOCK='    - ".pdf"'
		if [[ "${has_text}" -eq 1 ]]; then
			fct_log_warn "Detected PDF and text documents. Demo config will use the PDF loader and ignore text files."
		fi
		return 0
	fi

	LOADER_PROVIDER="text"
	SUPPORTED_EXTENSIONS_BLOCK='    - ".txt"
    - ".md"'
}

fct_write_config() {
	local store_path="${RUNTIME_DIR}/store.json"
	local trace_path="${RUNTIME_DIR}/trace.jsonl"

	if [[ "${CONFIG_EXPLICIT}" -eq 1 ]] && [[ -f "${CONFIG_PATH}" ]]; then
		fct_log_info "Using existing config at ${CONFIG_PATH}."
		return 0
	fi

	cat >"${CONFIG_PATH}" <<EOF
project:
  name: "minimal-modular-rag"
  environment: "local-demo"
ingestion:
  default_collection: "${COLLECTION}"
  chunk_size: 200
  chunk_overlap: 20
  supported_extensions:
${SUPPORTED_EXTENSIONS_BLOCK}
retrieval:
  mode: "hybrid"
  dense_top_k: 3
  sparse_top_k: 3
  rrf_k: 60
generation:
  max_context_results: 2
  max_answer_chars: 240
adapters:
  loader:
    provider: "${LOADER_PROVIDER}"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "${store_path}"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "${trace_path}"
  log_level: "INFO"
EOF
	fct_log_info "Wrote demo config to ${CONFIG_PATH} using loader=${LOADER_PROVIDER}."
}

fct_write_session_file() {
	local session_file="${RUNTIME_DIR}/session.env"
	cat >"${session_file}" <<EOF
RUNTIME_DIR="${RUNTIME_DIR}"
CONFIG_PATH="${CONFIG_PATH}"
DOCS_DIR="${DOCS_DIR}"
COLLECTION="${COLLECTION}"
EOF
}

fct_run_initial_ingest() {
	if [[ "${SKIP_INGEST}" -eq 1 ]]; then
		fct_log_info "Skipping initial ingest."
		return 0
	fi

	fct_log_info "Running initial ingest from ${DOCS_DIR}."
	(
		cd "${REPO_ROOT}"
		"${PYTHON_BIN}" -m src.interfaces.cli.ingest "${DOCS_DIR}" --collection "${COLLECTION}" --config "${CONFIG_PATH}"
	)
}

fct_print_next_steps() {
	cat <<EOF
[OK] Demo environment ready.

Runtime:
  runtime_dir: ${RUNTIME_DIR}
  config:      ${CONFIG_PATH}
  docs:        ${DOCS_DIR}
  collection:  ${COLLECTION}

Next commands:
  ${PYTHON_BIN} -m src.interfaces.cli.query "semantic embeddings" --collection "${COLLECTION}" --config "${CONFIG_PATH}"
  ${PYTHON_BIN} -m src.interfaces.cli.answer "semantic embeddings" --collection "${COLLECTION}" --config "${CONFIG_PATH}"
  ${PYTHON_BIN} -m src.interfaces.cli.agent run-workflow research_and_answer "semantic embeddings" --collection "${COLLECTION}" --mode hybrid --config "${CONFIG_PATH}"
  ${PYTHON_BIN} -m src.interfaces.cli.traces stats --config "${CONFIG_PATH}"

To clean up this demo environment:
  bash scripts/stop.sh --runtime-dir "${RUNTIME_DIR}"
EOF
}

fct_main() {
	fct_parse_arguments "$@"
	fct_resolve_runtime_paths
	fct_require_command "uv"
	fct_prepare_runtime_dirs
	fct_prepare_docs_dir
	fct_configure_uv_environment
	fct_prepare_python_environment
	fct_write_demo_doc_if_missing
	fct_detect_loader_profile
	fct_write_config
	fct_write_session_file
	fct_run_initial_ingest
	fct_print_next_steps
}

fct_main "$@"
