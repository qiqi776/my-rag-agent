"""CLI entry point for reading and summarizing traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.errors import ConfigError
from src.core.settings import load_settings
from src.observability.trace_reader import TraceReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect ingestion/query/answer traces.")
    parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help="Optional trace file override.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List traces.")
    list_parser.add_argument(
        "--trace-type",
        choices=("ingestion", "query", "answer"),
        default=None,
        help="Filter by trace type.",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of traces to return.",
    )

    show_parser = subparsers.add_parser("show", help="Show a single trace.")
    show_parser.add_argument("trace_id", help="Trace identifier.")

    stats_parser = subparsers.add_parser("stats", help="Summarize traces.")
    stats_parser.add_argument(
        "--trace-type",
        choices=("ingestion", "query", "answer"),
        default=None,
        help="Filter by trace type.",
    )

    return parser


def _resolve_trace_file(config_path: str, explicit_trace_file: str | None) -> str:
    if explicit_trace_file:
        return explicit_trace_file
    return load_settings(config_path).observability.trace_file


def main() -> int:
    args = build_parser().parse_args()
    try:
        trace_file = _resolve_trace_file(args.config, args.trace_file)
        reader = TraceReader(trace_file)
    except ConfigError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    if args.command == "list":
        payload = {
            "traces": [
                trace.summary_dict()
                for trace in reader.list_traces(trace_type=args.trace_type, limit=args.limit)
            ]
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "show":
        trace = reader.get_trace(args.trace_id)
        if trace is None:
            print(
                json.dumps(
                    {"error": f"Trace not found: {args.trace_id}"},
                    ensure_ascii=False,
                )
            )
            return 1
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
        return 0

    payload = reader.summarize(trace_type=args.trace_type).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
