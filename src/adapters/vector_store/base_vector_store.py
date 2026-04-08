"""Base contract for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import ChunkRecord, Metadata, RetrievalResult


class BaseVectorStore(ABC):
    """Minimal vector-store surface required by current application services."""

    @abstractmethod
    def upsert(self, collection: str, records: list[ChunkRecord]) -> int:
        """Insert or replace chunk records in a collection."""

    @abstractmethod
    def query(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: Metadata | None = None,
    ) -> list[RetrievalResult]:
        """Return the highest scoring matches for a query vector."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List known collection names."""

    @abstractmethod
    def list_records(self, collection: str) -> list[ChunkRecord]:
        """List all chunk records for a collection."""

    @abstractmethod
    def delete_doc(self, collection: str, doc_id: str) -> int:
        """Delete all chunks belonging to a document in a collection."""
