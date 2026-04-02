from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore


def _write_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
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
  dense_top_k: 3
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_ingest_then_query_returns_relevant_chunks_and_traces(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "python.txt").write_text(
        "Python retrieval systems use embeddings for semantic search.",
        encoding="utf-8",
    )
    (docs_dir / "cooking.md").write_text(
        "Cooking recipes focus on ingredients, heat, and timing.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )

    ingest_results = ingest_service.ingest_path(docs_dir, collection="knowledge")
    response = search_service.search("semantic embeddings", collection="knowledge", top_k=2)

    assert len(ingest_results) == 2
    assert response.results
    assert response.results[0].source_path.endswith("python.txt")
    assert response.citations[0].source_path.endswith("python.txt")
    assert response.result_count == len(response.results)

    trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 3
    parsed = [json.loads(line) for line in trace_lines]
    assert {item["trace_type"] for item in parsed} == {"ingestion", "query"}


@pytest.mark.integration
def test_factory_wiring_supports_cross_instance_query(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Python retrieval systems use embeddings for semantic search.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )

    ingest_service.ingest_path(text_file, collection="knowledge")
    response = search_service.search("semantic embeddings", collection="knowledge", top_k=1)

    assert response.results
    assert response.results[0].source_path.endswith("python.txt")
