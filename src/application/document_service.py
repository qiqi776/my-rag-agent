"""Document lifecycle application service for the MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.settings import Settings
from src.core.types import ChunkRecord, Metadata


@dataclass(slots=True)
class DocumentSummary:
    """Aggregated document information derived from stored chunks."""

    doc_id: str
    source_path: str
    collection: str
    chunk_count: int

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class DeleteDocumentResult:
    """Outcome of deleting a document from a collection."""

    doc_id: str
    collection: str
    deleted_chunks: int

    @property
    def deleted(self) -> bool:
        return self.deleted_chunks > 0

    def to_dict(self) -> Metadata:
        payload = asdict(self)
        payload["deleted"] = self.deleted
        return payload


@dataclass(slots=True)
class DocumentDetail:
    """Detailed document view derived from stored chunks."""

    doc_id: str
    source_path: str
    collection: str
    chunk_count: int
    preview: str
    metadata: Metadata

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class DocumentChunkDetail:
    """Detailed chunk view for dashboard and document browser consumers."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Metadata

    def to_dict(self) -> Metadata:
        return asdict(self)


class DocumentService:
    """Expose list and delete operations over stored documents."""

    def __init__(self, settings: Settings, vector_store: BaseVectorStore) -> None:
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

    def list_collections(self) -> list[str]:
        return self.vector_store.list_collections()

    def get_document_summary(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DocumentDetail | None:
        match = self._resolve_document_records(doc_id, collection)
        if match is None:
            return None
        collection_name, records = match
        return self._build_document_detail(collection_name, records)

    def get_document_chunks(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> list[DocumentChunkDetail]:
        """Return ordered chunk details for a document."""

        match = self._resolve_document_records(doc_id, collection)
        if match is None:
            return []
        _, records = match
        return self._build_chunk_details(records)

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

    def _resolve_document_records(
        self,
        doc_id: str,
        collection: str | None,
    ) -> tuple[str, list[ChunkRecord]] | None:
        target_collections = (
            [collection]
            if collection is not None
            else self.vector_store.list_collections()
        )

        matches: list[tuple[str, list[ChunkRecord]]] = []
        for collection_name in target_collections:
            matching_records = [
                record
                for record in self.vector_store.list_records(collection_name)
                if record.doc_id == doc_id
            ]
            if not matching_records:
                continue
            matches.append((collection_name, matching_records))

        if not matches:
            return None

        if collection is None and len(matches) > 1:
            collections = ", ".join(sorted(collection_name for collection_name, _ in matches))
            raise ValueError(
                f"Document '{doc_id}' exists in multiple collections: {collections}. "
                "Specify a collection explicitly."
            )

        return matches[0]

    def _build_document_detail(
        self,
        collection: str,
        records: list[ChunkRecord],
    ) -> DocumentDetail:
        ordered_records = sorted(
            records,
            key=lambda record: (
                int(record.metadata.get("chunk_index", 0)),
                record.id,
            ),
        )
        first = ordered_records[0]
        preview = next(
            (
                " ".join(record.text.split()).strip()[:240]
                for record in ordered_records
                if record.text.strip()
            ),
            "",
        )
        metadata: Metadata = {
            "source_path": first.metadata.get("source_path", ""),
            "collection": collection,
            "doc_type": first.metadata.get("doc_type"),
            "page_count": first.metadata.get("page_count"),
        }
        pages = sorted(
            {
                value
                for value in (record.metadata.get("page") for record in ordered_records)
                if isinstance(value, int)
            }
        )
        if pages:
            metadata["pages"] = pages
        return DocumentDetail(
            doc_id=first.doc_id,
            source_path=str(first.metadata.get("source_path", "")),
            collection=collection,
            chunk_count=len(ordered_records),
            preview=preview,
            metadata=metadata,
        )

    def _build_chunk_details(self, records: list[ChunkRecord]) -> list[DocumentChunkDetail]:
        ordered_records = sorted(
            records,
            key=lambda record: (
                int(record.metadata.get("chunk_index", 0)),
                record.id,
            ),
        )
        details: list[DocumentChunkDetail] = []
        for record in ordered_records:
            details.append(
                DocumentChunkDetail(
                    chunk_id=record.id,
                    doc_id=record.doc_id,
                    chunk_index=int(record.metadata.get("chunk_index", 0)),
                    text=record.text,
                    metadata=record.metadata.copy(),
                )
            )
        return details
