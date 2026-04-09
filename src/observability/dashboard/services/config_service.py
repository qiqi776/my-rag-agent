"""Configuration and overview stats service for the dashboard."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.factory import create_vector_store
from src.application.document_service import DocumentService
from src.core.settings import Settings, load_settings, resolve_path

_DASHBOARD_CONFIG_ENV = "MRAG_DASHBOARD_CONFIG"


def _resolve_settings_path(path: str | Path | None) -> str | Path | None:
    return path or os.environ.get(_DASHBOARD_CONFIG_ENV)


@dataclass(frozen=True, slots=True)
class ProviderCard:
    """Summary of a configured provider or subsystem."""

    name: str
    provider: str
    model: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CollectionStat:
    """Aggregate statistics for a collection."""

    name: str
    document_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OverviewSnapshot:
    """Dashboard overview payload."""

    provider_cards: list[ProviderCard]
    collections: list[CollectionStat]
    collection_count: int
    document_count: int
    chunk_count: int
    trace_file: str
    trace_exists: bool
    trace_line_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_cards": [card.to_dict() for card in self.provider_cards],
            "collections": [collection.to_dict() for collection in self.collections],
            "collection_count": self.collection_count,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "trace_file": self.trace_file,
            "trace_exists": self.trace_exists,
            "trace_line_count": self.trace_line_count,
        }


class ConfigService:
    """Read settings and aggregate dashboard overview data."""

    def __init__(
        self,
        settings_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        vector_store: BaseVectorStore | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._settings_path = _resolve_settings_path(settings_path)
        self._settings = settings
        self._vector_store = vector_store
        self._document_service = document_service

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self._settings_path)
        return self._settings

    def reload(self) -> None:
        """Force settings and cached dependencies to reload."""

        self._settings = None
        self._vector_store = None
        self._document_service = None

    def get_provider_cards(self) -> list[ProviderCard]:
        """Return configured provider cards for the overview page."""

        settings = self.settings
        return [
            ProviderCard(
                name="Loader",
                provider=settings.adapters.loader.provider,
                model="-",
                details={
                    "supported_extensions": settings.ingestion.supported_extensions,
                },
            ),
            ProviderCard(
                name="Embedding",
                provider=settings.adapters.embedding.provider,
                model=settings.adapters.embedding.model or "-",
                details={
                    "dimensions": settings.adapters.embedding.dimensions,
                },
            ),
            ProviderCard(
                name="LLM",
                provider=settings.adapters.llm.provider,
                model=settings.adapters.llm.model or "-",
                details={
                    "temperature": settings.adapters.llm.temperature,
                },
            ),
            ProviderCard(
                name="Reranker",
                provider=settings.adapters.reranker.provider,
                model=settings.adapters.reranker.model or "-",
            ),
            ProviderCard(
                name="Vector Store",
                provider=settings.adapters.vector_store.provider,
                model=settings.adapters.vector_store.storage_path,
            ),
            ProviderCard(
                name="Retrieval",
                provider=settings.retrieval.mode,
                model="dense + hybrid",
                details={
                    "dense_top_k": settings.retrieval.dense_top_k,
                    "sparse_top_k": settings.retrieval.sparse_top_k,
                    "rrf_k": settings.retrieval.rrf_k,
                },
            ),
        ]

    def get_overview_snapshot(self) -> OverviewSnapshot:
        """Return aggregate overview data for the dashboard."""

        vector_store = self._get_vector_store()
        document_service = self._get_document_service()
        documents = document_service.list_documents()

        stats_by_collection: dict[str, dict[str, int]] = {
            name: {"documents": 0, "chunks": 0}
            for name in vector_store.list_collections()
        }
        for document in documents:
            bucket = stats_by_collection.setdefault(
                document.collection,
                {"documents": 0, "chunks": 0},
            )
            bucket["documents"] += 1
            bucket["chunks"] += document.chunk_count

        collections = [
            CollectionStat(
                name=name,
                document_count=stats["documents"],
                chunk_count=stats["chunks"],
            )
            for name, stats in sorted(stats_by_collection.items())
        ]

        trace_path = resolve_path(self.settings.observability.trace_file)
        trace_line_count = 0
        if trace_path.exists():
            with trace_path.open("r", encoding="utf-8") as handle:
                trace_line_count = sum(1 for line in handle if line.strip())

        return OverviewSnapshot(
            provider_cards=self.get_provider_cards(),
            collections=collections,
            collection_count=len(collections),
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            trace_file=str(trace_path),
            trace_exists=trace_path.exists(),
            trace_line_count=trace_line_count,
        )

    def _get_vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            self._vector_store = create_vector_store(self.settings)
        return self._vector_store

    def _get_document_service(self) -> DocumentService:
        if self._document_service is None:
            self._document_service = DocumentService(self.settings, self._get_vector_store())
        return self._document_service

