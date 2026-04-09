"""Data service for browsing documents and chunks in the dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.factory import create_vector_store
from src.application.document_service import DocumentService
from src.core.settings import Settings, load_settings

_DASHBOARD_CONFIG_ENV = "MRAG_DASHBOARD_CONFIG"


def _resolve_settings_path(path: str | Path | None) -> str | Path | None:
    return path or os.environ.get(_DASHBOARD_CONFIG_ENV)


class DataService:
    """Read-only facade for document, chunk, and metadata browsing."""

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

    def list_collections(self) -> list[str]:
        """Return available collections, always including the default collection."""

        collections = self._get_vector_store().list_collections()
        default_collection = self.settings.ingestion.default_collection
        if default_collection not in collections:
            collections = [default_collection, *collections]
        return sorted(set(collections))

    def list_documents(self, collection: str | None = None) -> list[dict[str, object]]:
        """Return document list payloads for the selected collection."""

        return [
            document.to_dict()
            for document in self._get_document_service().list_documents(collection)
        ]

    def get_document_detail(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> dict[str, object] | None:
        """Return a document detail payload or None."""

        detail = self._get_document_service().get_document_summary(doc_id, collection)
        return detail.to_dict() if detail is not None else None

    def get_document_chunks(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> list[dict[str, object]]:
        """Return ordered chunk payloads for a document."""

        return [
            chunk.to_dict()
            for chunk in self._get_document_service().get_document_chunks(doc_id, collection)
        ]

    def delete_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> dict[str, object]:
        """Delete a document via DocumentService and return the result payload."""

        return self._get_document_service().delete_document(doc_id, collection).to_dict()

    def _get_vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            self._vector_store = create_vector_store(self.settings)
        return self._vector_store

    def _get_document_service(self) -> DocumentService:
        if self._document_service is None:
            self._document_service = DocumentService(self.settings, self._get_vector_store())
        return self._document_service

