from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
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
def test_in_memory_store_query_returns_highest_score(tmp_path: Path) -> None:
    store = InMemoryVectorStore(tmp_path / "store.json")
    store.upsert(
        "default",
        [
            _record("chunk-1", "doc-1", "python", [1.0, 0.0]),
            _record("chunk-2", "doc-2", "java", [0.0, 1.0]),
        ],
    )

    results = store.query("default", [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


@pytest.mark.unit
def test_in_memory_store_persists_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "store.json"
    original = InMemoryVectorStore(snapshot)
    original.upsert("default", [_record("chunk-1", "doc-1", "python", [1.0, 0.0])])

    restored = InMemoryVectorStore(snapshot)

    assert len(restored.list_records("default")) == 1
    assert restored.list_records("default")[0].doc_id == "doc-1"

