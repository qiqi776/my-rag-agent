"""In-memory vector store with optional local snapshot persistence."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.settings import resolve_path
from src.core.types import ChunkRecord, RetrievalResult


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore:
    """Simple in-memory store with JSON snapshot support."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = resolve_path(storage_path or "./data/db/vector_store.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, dict[str, ChunkRecord]] = {}
        self._load()

    def upsert(self, collection: str, records: list[ChunkRecord]) -> int:
        bucket = self._collections.setdefault(collection, {})
        for record in records:
            bucket[record.id] = record
        self._flush()
        return len(records)

    def query(self, collection: str, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        bucket = self._collections.get(collection, {})
        scored = [
            RetrievalResult(
                chunk_id=record.id,
                doc_id=record.doc_id,
                score=_cosine_similarity(query_vector, record.embedding),
                text=record.text,
                metadata=record.metadata.copy(),
            )
            for record in bucket.values()
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:top_k]

    def list_collections(self) -> list[str]:
        return sorted(self._collections)

    def list_records(self, collection: str) -> list[ChunkRecord]:
        return list(self._collections.get(collection, {}).values())

    def delete_doc(self, collection: str, doc_id: str) -> int:
        bucket = self._collections.get(collection, {})
        to_delete = [chunk_id for chunk_id, record in bucket.items() if record.doc_id == doc_id]
        for chunk_id in to_delete:
            del bucket[chunk_id]
        if not bucket and collection in self._collections:
            del self._collections[collection]
        self._flush()
        return len(to_delete)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self._collections = {
            collection: {
                record_data["id"]: ChunkRecord.from_dict(record_data)
                for record_data in records
            }
            for collection, records in raw.items()
        }

    def _flush(self) -> None:
        payload = {
            collection: [record.to_dict() for record in records.values()]
            for collection, records in self._collections.items()
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
