from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.llm.factory import create_llm
from src.adapters.loader.factory import create_loader
from src.adapters.reranker.factory import create_reranker
from src.adapters.vector_store.factory import create_vector_store
from src.application.answer_service import AnswerService
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.evaluation.answer_eval import AnswerEvalRunner
from src.evaluation.fixtures import load_answer_cases, load_retrieval_cases
from src.evaluation.retrieval_eval import RetrievalEvalRunner
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever

EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"
INGEST_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


def _write_text_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 90
  chunk_overlap: 15
  supported_extensions:
    - ".txt"
retrieval:
  mode: "hybrid"
  dense_top_k: 2
  sparse_top_k: 4
  dense_candidate_multiplier: 3
  sparse_candidate_multiplier: 4
  max_candidate_top_k: 10
  rrf_k: 60
generation:
  max_context_results: 2
  candidate_results: 4
  max_context_chars: 180
  max_answer_chars: 260
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


def _write_pdf_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 200
  chunk_overlap: 20
  supported_extensions:
    - ".pdf"
  transforms:
    enabled: true
    order:
      - "metadata_enrichment"
      - "chunk_refinement"
    metadata_enrichment:
      enabled: true
      section_title_max_length: 80
    chunk_refinement:
      enabled: true
      collapse_whitespace: true
retrieval:
  mode: "hybrid"
  dense_top_k: 2
  sparse_top_k: 4
  dense_candidate_multiplier: 3
  sparse_candidate_multiplier: 4
  max_candidate_top_k: 10
  rrf_k: 60
generation:
  max_context_results: 1
  candidate_results: 3
  max_context_chars: 180
  max_answer_chars: 260
adapters:
  loader:
    provider: "pdf"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_answer_quality_regression_passes_for_golden_text_cases(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_text_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    vector_store = create_vector_store(settings)
    embedding = create_embedding(settings)
    trace_store = TraceStore(trace_path)
    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
        trace_store=trace_store,
    )
    ingest_service.ingest_path(EVAL_FIXTURE_DIR / "corpus", collection="knowledge")

    search_service = SearchService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    answer_service = AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=create_reranker(settings),
        llm=create_llm(settings),
        trace_store=trace_store,
    )

    retrieval_report = RetrievalEvalRunner(search_service).run(
        load_retrieval_cases(EVAL_FIXTURE_DIR / "retrieval_cases.json")
    )
    answer_report = AnswerEvalRunner(answer_service).run(
        load_answer_cases(EVAL_FIXTURE_DIR / "answer_cases.json")
    )

    assert retrieval_report.passed_cases == retrieval_report.total_cases
    assert retrieval_report.average_recall_at_k == 1.0
    assert answer_report.passed_cases == answer_report.total_cases
    assert answer_report.average_keyword_coverage == 1.0
    assert answer_report.average_source_coverage == 1.0


@pytest.mark.integration
def test_answer_quality_regression_keeps_pdf_page_citations(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_pdf_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    vector_store = create_vector_store(settings)
    embedding = create_embedding(settings)
    trace_store = TraceStore(trace_path)
    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
        trace_store=trace_store,
    )
    ingest_service.ingest_path(INGEST_FIXTURE_DIR / "multi_page.pdf", collection="knowledge")

    search_service = SearchService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    answer_service = AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=create_reranker(settings),
        llm=create_llm(settings),
        trace_store=trace_store,
    )

    output = answer_service.answer("page citations", collection="knowledge", top_k=1, mode="hybrid")

    assert "page citations" in output.answer.lower()
    assert output.citations
    assert output.citations[0].page == 2
    assert output.citations[0].source_path.endswith("multi_page.pdf")
