from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.document_service import DocumentService
from src.core.settings import load_settings
from src.core.types import ChunkRecord
from src.observability.dashboard.services.data_service import DataService


def _write_settings(path: Path) -> None:
    path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "knowledge"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


def _record(record_id: str, chunk_index: int) -> ChunkRecord:
    return ChunkRecord(
        id=record_id,
        doc_id="doc-1",
        text=f"chunk {chunk_index} text",
        embedding=[1.0, 0.0],
        metadata={
            "source_path": "/tmp/doc-1.txt",
            "collection": "knowledge",
            "doc_id": "doc-1",
            "chunk_index": chunk_index,
            "doc_type": "text",
        },
    )


@pytest.mark.unit
def test_data_service_lists_documents_chunks_and_delete(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)

    store = InMemoryVectorStore()
    store.upsert("knowledge", [_record("c-1", 0), _record("c-2", 1)])
    data_service = DataService(
        settings=settings,
        vector_store=store,
        document_service=DocumentService(settings, store),
    )

    collections = data_service.list_collections()
    documents = data_service.list_documents("knowledge")
    detail = data_service.get_document_detail("doc-1", "knowledge")
    chunks = data_service.get_document_chunks("doc-1", "knowledge")
    delete_result = data_service.delete_document("doc-1", "knowledge")

    assert collections == ["knowledge"]
    assert len(documents) == 1
    assert detail is not None
    assert detail["metadata"]["doc_type"] == "text"
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert delete_result["deleted"]
    assert data_service.list_documents("knowledge") == []
