"""Reusable evaluation runtime helpers for CLI and dashboard consumers."""

from __future__ import annotations

from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.llm.factory import create_llm
from src.adapters.reranker.factory import create_reranker
from src.adapters.vector_store.factory import create_vector_store
from src.application.answer_service import AnswerService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.evaluation.answer_eval import AnswerEvalRunner
from src.evaluation.fixtures import (
    DEFAULT_ANSWER_FIXTURES,
    DEFAULT_RETRIEVAL_FIXTURES,
    load_answer_cases,
    load_retrieval_cases,
)
from src.evaluation.models import AnswerEvalReport, RetrievalEvalReport
from src.evaluation.retrieval_eval import RetrievalEvalRunner
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def default_fixture_paths() -> dict[str, str]:
    """Return the default retrieval and answer fixture paths."""

    return {
        "retrieval": str(DEFAULT_RETRIEVAL_FIXTURES),
        "answer": str(DEFAULT_ANSWER_FIXTURES),
    }


def build_search_service(config_path: str | Path | None = None) -> SearchService:
    """Construct the SearchService used by evaluation flows."""

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


def build_answer_service(config_path: str | Path | None = None) -> AnswerService:
    """Construct the AnswerService used by evaluation flows."""

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


def run_retrieval_evaluation(
    config_path: str | Path | None,
    fixtures_path: str | Path,
) -> RetrievalEvalReport:
    """Run retrieval regression cases and return the report."""

    return RetrievalEvalRunner(build_search_service(config_path)).run(
        load_retrieval_cases(fixtures_path)
    )


def run_answer_evaluation(
    config_path: str | Path | None,
    fixtures_path: str | Path,
) -> AnswerEvalReport:
    """Run answer regression cases and return the report."""

    return AnswerEvalRunner(build_answer_service(config_path)).run(load_answer_cases(fixtures_path))


def run_all_evaluations(
    config_path: str | Path | None,
    retrieval_fixtures: str | Path,
    answer_fixtures: str | Path,
) -> dict[str, object]:
    """Run retrieval and answer regressions and return a combined payload."""

    retrieval_report = run_retrieval_evaluation(config_path, retrieval_fixtures)
    answer_report = run_answer_evaluation(config_path, answer_fixtures)
    return {
        "kind": "evaluation_summary",
        "retrieval": retrieval_report.to_dict(),
        "answer": answer_report.to_dict(),
    }
