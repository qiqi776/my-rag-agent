"""Document lifecycle application service for the MVP."""

from __future__ import annotations

from dataclasses import dataclass

from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.core.settings import Settings


@dataclass(slots=True)
class DocumentSummary:
    """Aggregated document information derived from stored chunks."""

    doc_id: str
    source_path: str
    collection: str
    chunk_count: int


@dataclass(slots=True)
class DeleteDocumentResult:
    """Outcome of deleting a document from a collection."""

    doc_id: str
    collection: str
    deleted_chunks: int

    @property
    def deleted(self) -> bool:
        return self.deleted_chunks > 0


class DocumentService:
    """Expose list and delete operations over stored documents."""

    def __init__(self, settings: Settings, vector_store: InMemoryVectorStore) -> None:
        self.settings = settings
        self.vector_store = vector_store

    def list_documents(self, collection: str | None = None) -> list[DocumentSummary]:
        target_collections = (
            [collection]
            if collection is not None
            else self.vector_store.list_collections()
        )
        documents: list[DocumentSummary] = []

        for collection_name in target_collections:
            counts: dict[str, int] = {}
            source_paths: dict[str, str] = {}

            for record in self.vector_store.list_records(collection_name):
                counts[record.doc_id] = counts.get(record.doc_id, 0) + 1
                source_paths.setdefault(
                    record.doc_id,
                    str(record.metadata.get("source_path", "")),
                )

            for doc_id, chunk_count in counts.items():
                documents.append(
                    DocumentSummary(
                        doc_id=doc_id,
                        source_path=source_paths[doc_id],
                        collection=collection_name,
                        chunk_count=chunk_count,
                    )
                )

        documents.sort(key=lambda item: (item.collection, item.source_path, item.doc_id))
        return documents

    def delete_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DeleteDocumentResult:
        target_collection = collection or self.settings.ingestion.default_collection
        deleted_chunks = self.vector_store.delete_doc(target_collection, doc_id)
        return DeleteDocumentResult(
            doc_id=doc_id,
            collection=target_collection,
            deleted_chunks=deleted_chunks,
        )
