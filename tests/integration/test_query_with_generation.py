from __future__ import annotations

import json
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
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def _write_settings(
    path: Path,
    storage_path: Path,
    trace_path: Path,
    retrieval_mode: str = "hybrid",
) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  mode: "{retrieval_mode}"
  dense_top_k: 3
  sparse_top_k: 3
  rrf_k: 60
generation:
  max_context_results: 2
  max_answer_chars: 240
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


@pytest.mark.integration
def test_answer_service_generates_answer_from_hybrid_results(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path, retrieval_mode="hybrid")
    settings = load_settings(config_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Semantic embeddings help Python retrieval systems answer questions.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    ingest_service.ingest_path(text_file, collection="knowledge")

    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=TraceStore(trace_path),
    )
    answer_service = AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=create_reranker(settings),
        llm=create_llm(settings),
        trace_store=TraceStore(trace_path),
    )

    output = answer_service.answer(
        "semantic embeddings",
        collection="knowledge",
        top_k=1,
        mode="hybrid",
    )

    assert output.answer
    assert output.retrieval_mode == "hybrid"
    assert output.supporting_results
    assert output.citations[0].source_path.endswith("python.txt")

    parsed = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").strip().splitlines()]
    assert {item["trace_type"] for item in parsed} >= {"ingestion", "query", "answer"}
