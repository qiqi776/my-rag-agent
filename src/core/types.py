"""Core domain types for the minimal modular RAG project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Metadata = dict[str, Any]
EmbeddingVector = list[float]


def _require_source_path(metadata: Metadata, type_name: str) -> None:
    if not metadata.get("source_path"):
        raise ValueError(f"{type_name} metadata must contain 'source_path'")


@dataclass(slots=True)
class Document:
    """A source document loaded from local storage."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_source_path(self.metadata, "Document")

    def to_dict(self) -> Metadata:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Metadata) -> Document:
        return cls(**data)


@dataclass(slots=True)
class Chunk:
    """A deterministic slice of a document."""

    id: str
    doc_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_source_path(self.metadata, "Chunk")
        if "chunk_index" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'chunk_index'")

    def to_dict(self) -> Metadata:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Metadata) -> Chunk:
        return cls(**data)


@dataclass(slots=True)
class ChunkRecord:
    """A persisted chunk with its dense embedding."""

    id: str
    doc_id: str
    text: str
    embedding: EmbeddingVector
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_source_path(self.metadata, "ChunkRecord")
        if "collection" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'collection'")

    def to_dict(self) -> Metadata:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Metadata) -> ChunkRecord:
        return cls(**data)


@dataclass(slots=True)
class ProcessedQuery:
    """A normalized query ready for retrieval."""

    original_query: str
    normalized_query: str
    collection: str
    top_k: int

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    """A single search result."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return asdict(self)

