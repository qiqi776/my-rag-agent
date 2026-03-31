from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.vector_store.local_json_store import LocalJsonVectorStore
from src.core.types import ChunkRecord


def _record(record_id: str, doc_id: str, text: str, embedding: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=record_id,
        doc_id=doc_id,
        text=text,
        embedding=embedding,
        metadata={
            "source_path": f"/tmp/{doc_id}.txt",
            "collection": "default",
            "doc_id": doc_id,
            "chunk_index": 0,
        },
    )


@pytest.mark.unit
def test_local_json_store_persists_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "store.json"
    original = LocalJsonVectorStore(snapshot)
    original.upsert("default", [_record("chunk-1", "doc-1", "python", [1.0, 0.0])])

    restored = LocalJsonVectorStore(snapshot)

    assert len(restored.list_records("default")) == 1
    assert restored.list_records("default")[0].doc_id == "doc-1"


@pytest.mark.unit
def test_local_json_store_removes_empty_collection_after_delete(tmp_path: Path) -> None:
    snapshot = tmp_path / "store.json"
    store = LocalJsonVectorStore(snapshot)
    store.upsert("default", [_record("chunk-1", "doc-1", "python", [1.0, 0.0])])

    deleted = store.delete_doc("default", "doc-1")
    restored = LocalJsonVectorStore(snapshot)

    assert deleted == 1
    assert restored.list_collections() == []


@pytest.mark.unit
def test_local_json_store_refreshes_state_across_instances(tmp_path: Path) -> None:
    snapshot = tmp_path / "store.json"
    writer = LocalJsonVectorStore(snapshot)
    reader = LocalJsonVectorStore(snapshot)

    writer.upsert("default", [_record("chunk-1", "doc-1", "python", [1.0, 0.0])])

    assert len(reader.list_records("default")) == 1
