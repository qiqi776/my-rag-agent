"""Local JSON-backed vector store."""

from __future__ import annotations

import json
from pathlib import Path

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.core.settings import resolve_path
from src.core.types import ChunkRecord, RetrievalResult


class LocalJsonVectorStore(BaseVectorStore):
    """Persist vector-store state to a JSON snapshot on local disk."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = resolve_path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._store = self._load()

    def upsert(self, collection: str, records: list[ChunkRecord]) -> int:
        self._refresh()
        upserted = self._store.upsert(collection, records)
        self._flush()
        return upserted

    def query(self, collection: str, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        self._refresh()
        return self._store.query(collection, query_vector, top_k)

    def list_collections(self) -> list[str]:
        self._refresh()
        return self._store.list_collections()

    def list_records(self, collection: str) -> list[ChunkRecord]:
        self._refresh()
        return self._store.list_records(collection)

    def delete_doc(self, collection: str, doc_id: str) -> int:
        self._refresh()
        deleted = self._store.delete_doc(collection, doc_id)
        self._flush()
        return deleted

    def _load(self) -> InMemoryVectorStore:
        if not self.storage_path.exists():
            return InMemoryVectorStore()
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return InMemoryVectorStore.from_snapshot(raw)

    def _refresh(self) -> None:
        self._store = self._load()

    def _flush(self) -> None:
        self.storage_path.write_text(
            json.dumps(self._store.to_snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
