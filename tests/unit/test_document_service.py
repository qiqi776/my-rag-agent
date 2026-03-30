from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.document_service import DocumentService
from src.core.settings import load_settings
from src.core.types import ChunkRecord


def _write_settings(path: Path, storage_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 40
  chunk_overlap: 5
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
    provider: "memory"
    storage_path: "{storage_path}"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


def _record(
    record_id: str,
    doc_id: str,
    collection: str,
    chunk_index: int,
    source_path: str,
) -> ChunkRecord:
    return ChunkRecord(
        id=record_id,
        doc_id=doc_id,
        text=f"text-{record_id}",
        embedding=[1.0, 0.0],
        metadata={
            "source_path": source_path,
            "collection": collection,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
        },
    )


@pytest.mark.unit
def test_document_service_lists_documents_across_collections(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    _write_settings(config_path, storage_path)
    settings = load_settings(config_path)

    store = InMemoryVectorStore(storage_path)
    store.upsert(
        "alpha",
        [
            _record("a-1", "doc-a", "alpha", 0, "/tmp/alpha.txt"),
            _record("a-2", "doc-a", "alpha", 1, "/tmp/alpha.txt"),
            _record("b-1", "doc-b", "alpha", 0, "/tmp/beta.txt"),
        ],
    )
    store.upsert(
        "beta",
        [
            _record("c-1", "doc-c", "beta", 0, "/tmp/gamma.txt"),
        ],
    )

    service = DocumentService(settings=settings, vector_store=store)

    documents = service.list_documents()

    assert [(item.collection, item.doc_id, item.chunk_count) for item in documents] == [
        ("alpha", "doc-a", 2),
        ("alpha", "doc-b", 1),
        ("beta", "doc-c", 1),
    ]


@pytest.mark.unit
def test_document_service_delete_only_removes_target_collection(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    _write_settings(config_path, storage_path)
    settings = load_settings(config_path)

    store = InMemoryVectorStore(storage_path)
    store.upsert("alpha", [_record("a-1", "doc-a", "alpha", 0, "/tmp/alpha.txt")])
    store.upsert("beta", [_record("b-1", "doc-a", "beta", 0, "/tmp/alpha.txt")])
    service = DocumentService(settings=settings, vector_store=store)

    result = service.delete_document("doc-a", collection="alpha")

    assert result.deleted
    assert result.deleted_chunks == 1
    assert service.list_documents("alpha") == []
    assert [(item.collection, item.doc_id) for item in service.list_documents("beta")] == [
        ("beta", "doc-a")
    ]
