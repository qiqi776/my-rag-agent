"""Pure in-memory vector store."""

from __future__ import annotations

from typing import Any

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.types import ChunkRecord, Metadata, RetrievalResult


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _metadata_matches(metadata: Metadata, filters: Metadata | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory store without any persistence side effects."""

    def __init__(self, initial_collections: dict[str, dict[str, ChunkRecord]] | None = None) -> None:
        self._collections = initial_collections or {}

    def upsert(self, collection: str, records: list[ChunkRecord]) -> int:
        bucket = self._collections.setdefault(collection, {})
        for record in records:
            bucket[record.id] = record
        return len(records)

    def query(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: Metadata | None = None,
    ) -> list[RetrievalResult]:
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
            if _metadata_matches(record.metadata, filters)
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
        return len(to_delete)

    def to_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            collection: [record.to_dict() for record in record_map.values()]
            for collection, record_map in self._collections.items()
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, list[dict[str, Any]]] | dict[str, dict[str, Any]]) -> InMemoryVectorStore:
        collections: dict[str, dict[str, ChunkRecord]] = {}
        for collection, records in payload.items():
            if isinstance(records, list):
                collections[collection] = {
                    record_data["id"]: ChunkRecord.from_dict(record_data)
                    for record_data in records
                }
                continue

            collections[collection] = {
                record_id: ChunkRecord.from_dict(record_data)
                for record_id, record_data in records.items()
            }
        return cls(initial_collections=collections)
