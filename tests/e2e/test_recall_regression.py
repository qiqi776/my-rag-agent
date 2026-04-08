from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.evaluation.fixtures import load_retrieval_cases
from src.evaluation.retrieval_eval import RetrievalEvalRunner
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever

EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"


def _write_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
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


@pytest.mark.e2e
def test_recall_regression_meets_minimum_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    vector_store = create_vector_store(settings)
    embedding = create_embedding(settings)
    trace_store = TraceStore(trace_path)
    IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
        trace_store=trace_store,
    ).ingest_path(EVAL_FIXTURE_DIR / "corpus", collection="knowledge")

    search_service = SearchService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    report = RetrievalEvalRunner(search_service).run(
        load_retrieval_cases(EVAL_FIXTURE_DIR / "retrieval_cases.json")
    )

    assert report.total_cases >= 3
    assert report.average_hit_at_k >= 1.0
    assert report.average_recall_at_k >= 1.0
    assert report.passed_cases == report.total_cases
