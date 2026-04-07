"""CLI entry point for agent-ready tools and workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.dependencies import build_agent_dependencies
from src.agent.tools import create_tool_registry
from src.agent.workflows import create_workflow_runner
from src.core.errors import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-ready tools and workflows.")
    parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tools", help="List agent-ready tools.")
    subparsers.add_parser("list-workflows", help="List available workflows.")

    run_parser = subparsers.add_parser("run-workflow", help="Run a workflow.")
    run_parser.add_argument("name", help="Workflow name.")
    run_parser.add_argument("query", help="Workflow query input.")
    run_parser.add_argument("--collection", default=None, help="Optional collection name.")
    run_parser.add_argument(
        "--mode",
        choices=("dense", "hybrid"),
        default=None,
        help="Optional retrieval mode override.",
    )
    run_parser.add_argument(
        "--search-top-k",
        type=int,
        default=None,
        help="Optional top-k override for the search tool.",
    )
    run_parser.add_argument(
        "--answer-top-k",
        type=int,
        default=None,
        help="Optional top-k override for the answer tool.",
    )

    return parser


def _format_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> int:
    args = build_parser().parse_args()
    try:
        dependencies = build_agent_dependencies(args.config)
        registry = create_tool_registry(dependencies)
        runner = create_workflow_runner(registry)
    except ConfigError as exc:
        print(_format_json({"error": str(exc)}))
        return 1

    if args.command == "list-tools":
        print(
            _format_json(
                {
                    "kind": "agent_tool_list",
                    "tools": [tool.to_dict() for tool in registry.list_tools()],
                }
            )
        )
        return 0

    if args.command == "list-workflows":
        print(
            _format_json(
                {
                    "kind": "workflow_list",
                    "workflows": [workflow.to_dict() for workflow in runner.list_workflows()],
                }
            )
        )
        return 0

    try:
        state = runner.run(
            args.name,
            {
                key: value
                for key, value in {
                    "query": args.query,
                    "collection": args.collection,
                    "mode": args.mode,
                    "search_top_k": args.search_top_k,
                    "answer_top_k": args.answer_top_k,
                }.items()
                if value is not None
            },
        )
    except ValueError as exc:
        print(_format_json({"error": str(exc)}))
        return 1

    print(_format_json(state.to_dict()))
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
