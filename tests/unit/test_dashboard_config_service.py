from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.document_service import DocumentService
from src.core.errors import ConfigError
from src.core.settings import load_settings
from src.core.types import ChunkRecord
from src.observability.dashboard.services.config_service import ConfigService


def _write_settings(path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "knowledge"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  mode: "hybrid"
  dense_top_k: 3
  sparse_top_k: 4
  rrf_k: 60
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
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


def _record(record_id: str, doc_id: str, collection: str, source_path: str, chunk_index: int) -> ChunkRecord:
    return ChunkRecord(
        id=record_id,
        doc_id=doc_id,
        text=f"text-{record_id}",
        embedding=[1.0, 0.0],
        metadata={
            "source_path": source_path,
            "collection": collection,
            "chunk_index": chunk_index,
            "doc_id": doc_id,
        },
    )


@pytest.mark.unit
def test_config_service_returns_provider_cards_and_overview_stats(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, trace_path)
    trace_path.write_text('{"trace_id":"t-1"}\n{"trace_id":"t-2"}\n', encoding="utf-8")

    settings = load_settings(config_path)
    store = InMemoryVectorStore()
    store.upsert(
        "knowledge",
        [
            _record("k-1", "doc-1", "knowledge", "/tmp/doc-1.txt", 0),
            _record("k-2", "doc-1", "knowledge", "/tmp/doc-1.txt", 1),
        ],
    )
    store.upsert(
        "notes",
        [
            _record("n-1", "doc-2", "notes", "/tmp/doc-2.txt", 0),
        ],
    )
    service = ConfigService(
        settings=settings,
        vector_store=store,
        document_service=DocumentService(settings, store),
    )

    cards = service.get_provider_cards()
    snapshot = service.get_overview_snapshot()

    assert [card.name for card in cards[:5]] == [
        "Loader",
        "Embedding",
        "LLM",
        "Reranker",
        "Vector Store",
    ]
    assert cards[0].provider == "text"
    assert cards[1].details["dimensions"] == 16
    assert snapshot.collection_count == 2
    assert snapshot.document_count == 2
    assert snapshot.chunk_count == 3
    assert snapshot.trace_exists
    assert snapshot.trace_line_count == 2


@pytest.mark.unit
def test_config_service_missing_config_is_readable(tmp_path: Path) -> None:
    service = ConfigService(tmp_path / "missing.yaml")

    with pytest.raises(ConfigError, match="Configuration file not found"):
        _ = service.settings
