"""CLI entry point for retrieval and answer regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.llm.factory import create_llm
from src.adapters.reranker.factory import create_reranker
from src.adapters.vector_store.factory import create_vector_store
from src.application.answer_service import AnswerService
from src.application.search_service import SearchService
from src.core.errors import ConfigError, EmptyQueryError, UnsupportedRetrievalModeError
from src.core.settings import load_settings
from src.evaluation.answer_eval import AnswerEvalRunner
from src.evaluation.fixtures import (
    DEFAULT_ANSWER_FIXTURES,
    DEFAULT_RETRIEVAL_FIXTURES,
    load_answer_cases,
    load_retrieval_cases,
)
from src.evaluation.retrieval_eval import RetrievalEvalRunner
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


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


def _build_search_service(config_path: str) -> SearchService:
    settings = load_settings(config_path)
    trace_store = (
        TraceStore(settings.observability.trace_file)
        if settings.observability.trace_enabled
        else None
    )
    vector_store = create_vector_store(settings)
    return SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )


def _build_answer_service(config_path: str) -> AnswerService:
    settings = load_settings(config_path)
    trace_store = (
        TraceStore(settings.observability.trace_file)
        if settings.observability.trace_enabled
        else None
    )
    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    return AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=create_reranker(settings),
        llm=create_llm(settings),
        trace_store=trace_store,
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "retrieval":
            report = RetrievalEvalRunner(_build_search_service(args.config)).run(
                load_retrieval_cases(args.fixtures)
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.passed_cases == report.total_cases else 1

        if args.command == "answer":
            report = AnswerEvalRunner(_build_answer_service(args.config)).run(
                load_answer_cases(args.fixtures)
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.passed_cases == report.total_cases else 1

        retrieval_report = RetrievalEvalRunner(_build_search_service(args.config)).run(
            load_retrieval_cases(args.retrieval_fixtures)
        )
        answer_report = AnswerEvalRunner(_build_answer_service(args.config)).run(
            load_answer_cases(args.answer_fixtures)
        )
        payload = {
            "kind": "evaluation_summary",
            "retrieval": retrieval_report.to_dict(),
            "answer": answer_report.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if retrieval_report.passed_cases != retrieval_report.total_cases:
            return 1
        if answer_report.passed_cases != answer_report.total_cases:
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
