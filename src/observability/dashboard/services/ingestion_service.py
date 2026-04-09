"""Dashboard ingestion and preview service."""

from __future__ import annotations

import os
from pathlib import Path

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.base_loader import BaseLoader
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import (
    IngestService,
    discover_ingestion_files,
    first_non_empty_excerpt,
)
from src.core.settings import Settings, load_settings
from src.ingestion.pipeline import IngestionPipeline, create_ingestion_pipeline
from src.observability.trace_store import TraceStore

_DASHBOARD_CONFIG_ENV = "MRAG_DASHBOARD_CONFIG"


def _resolve_settings_path(path: str | Path | None) -> str | Path | None:
    return path or os.environ.get(_DASHBOARD_CONFIG_ENV)


class DashboardIngestionService:
    """Run preview and ingestion flows for dashboard consumers."""

    def __init__(
        self,
        settings_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        loader: BaseLoader | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        pipeline: IngestionPipeline | None = None,
        ingest_service: IngestService | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._settings_path = _resolve_settings_path(settings_path)
        self._settings = settings
        self._loader = loader
        self._embedding = embedding
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._ingest_service = ingest_service
        self._trace_store = trace_store

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self._settings_path)
        return self._settings

    def preview_path(
        self,
        path: str | Path,
        collection: str | None = None,
        *,
        max_chars: int = 240,
    ) -> list[dict[str, object]]:
        """Preview document extraction and transform results."""

        target_collection = collection or self.settings.ingestion.default_collection
        files = discover_ingestion_files(path, self.settings.ingestion.supported_extensions)
        results: list[dict[str, object]] = []

        for file_path in files:
            document = self._get_loader().load(file_path)
            prepared = self._get_pipeline().prepare(document, target_collection)
            preview_text = first_non_empty_excerpt(
                [unit.text for unit in prepared.units] or [document.text],
                max_chars=max_chars,
            )
            quality_status = str(document.metadata.get("quality_status") or "n/a")
            warnings = document.metadata.get("quality_warnings", [])
            results.append(
                {
                    "source_path": str(document.metadata.get("source_path", file_path)),
                    "doc_id": document.id,
                    "collection": target_collection,
                    "page_count": int(document.metadata.get("page_count", 1)),
                    "quality_status": quality_status,
                    "warnings": list(warnings) if isinstance(warnings, list) else [],
                    "will_ingest": preview_text != "<empty>" and quality_status != "bad",
                    "transforms": prepared.transforms_applied,
                    "preview": preview_text,
                }
            )

        return results

    def ingest_path(
        self,
        path: str | Path,
        collection: str | None = None,
        *,
        max_chars: int = 240,
    ) -> list[dict[str, object]]:
        """Run ingestion and merge the result summary with preview metadata."""

        target_collection = collection or self.settings.ingestion.default_collection
        preview_map = {
            Path(item["source_path"]).resolve(): item
            for item in self.preview_path(path, target_collection, max_chars=max_chars)
        }
        ingested = self._get_ingest_service().ingest_path(path, collection=target_collection)

        merged: list[dict[str, object]] = []
        for item in ingested:
            preview = preview_map.get(Path(item.source_path).resolve(), {})
            merged.append(
                {
                    "source_path": item.source_path,
                    "doc_id": item.doc_id,
                    "collection": item.collection,
                    "chunk_count": item.chunk_count,
                    "quality_status": preview.get("quality_status", "n/a"),
                    "warnings": preview.get("warnings", []),
                    "preview": preview.get("preview", "<empty>"),
                }
            )
        return merged

    def _get_loader(self) -> BaseLoader:
        if self._loader is None:
            self._loader = create_loader(self.settings)
        return self._loader

    def _get_embedding(self) -> BaseEmbedding:
        if self._embedding is None:
            self._embedding = create_embedding(self.settings)
        return self._embedding

    def _get_vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            self._vector_store = create_vector_store(self.settings)
        return self._vector_store

    def _get_pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = create_ingestion_pipeline(self.settings)
        return self._pipeline

    def _get_trace_store(self) -> TraceStore | None:
        if not self.settings.observability.trace_enabled:
            return None
        if self._trace_store is None:
            self._trace_store = TraceStore(self.settings.observability.trace_file)
        return self._trace_store

    def _get_ingest_service(self) -> IngestService:
        if self._ingest_service is None:
            self._ingest_service = IngestService(
                settings=self.settings,
                loader=self._get_loader(),
                embedding=self._get_embedding(),
                vector_store=self._get_vector_store(),
                trace_store=self._get_trace_store(),
                pipeline=self._get_pipeline(),
            )
        return self._ingest_service

