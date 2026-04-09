"""CLI entry point for retrieval and answer regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.errors import ConfigError, EmptyQueryError, UnsupportedRetrievalModeError
from src.evaluation.fixtures import DEFAULT_ANSWER_FIXTURES, DEFAULT_RETRIEVAL_FIXTURES
from src.evaluation.runtime import (
    run_all_evaluations,
    run_answer_evaluation,
    run_retrieval_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval and answer regression suites.")
    parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval regression.")
    retrieval_parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_RETRIEVAL_FIXTURES),
        help="Path to retrieval fixtures JSON.",
    )

    answer_parser = subparsers.add_parser("answer", help="Run answer regression.")
    answer_parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_ANSWER_FIXTURES),
        help="Path to answer fixtures JSON.",
    )

    all_parser = subparsers.add_parser("all", help="Run all regression suites.")
    all_parser.add_argument(
        "--retrieval-fixtures",
        default=str(DEFAULT_RETRIEVAL_FIXTURES),
        help="Path to retrieval fixtures JSON.",
    )
    all_parser.add_argument(
        "--answer-fixtures",
        default=str(DEFAULT_ANSWER_FIXTURES),
        help="Path to answer fixtures JSON.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "retrieval":
            report = run_retrieval_evaluation(args.config, args.fixtures)
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.passed_cases == report.total_cases else 1

        if args.command == "answer":
            report = run_answer_evaluation(args.config, args.fixtures)
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.passed_cases == report.total_cases else 1

        payload = run_all_evaluations(
            args.config,
            args.retrieval_fixtures,
            args.answer_fixtures,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        retrieval_report = payload["retrieval"]
        answer_report = payload["answer"]
        if retrieval_report["passed_cases"] != retrieval_report["total_cases"]:
            return 1
        if answer_report["passed_cases"] != answer_report["total_cases"]:
            return 1
        return 0
    except (
        ConfigError,
        EmptyQueryError,
        UnsupportedRetrievalModeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
