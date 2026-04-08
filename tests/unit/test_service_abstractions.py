from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.loader.base_loader import BaseLoader
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.application.document_service import DocumentService
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.core.types import ChunkRecord, Document, RetrievalResult


def _write_settings(path: Path) -> None:
    path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 2
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class StubLoader(BaseLoader):
    def load(self, path: str | Path) -> Document:
        return Document(
            id="stub-doc",
            text="semantic embeddings support retrieval",
            metadata={"source_path": str(Path(path).resolve())},
        )


class StubEmbedding(BaseEmbedding):
    @property
    def dimensions(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class StubVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, ChunkRecord]] = {}

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
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        del filters
        records = list(self._collections.get(collection, {}).values())[:top_k]
        return [
            RetrievalResult(
                chunk_id=record.id,
                doc_id=record.doc_id,
                score=1.0,
                text=record.text,
                metadata=record.metadata.copy(),
            )
            for record in records
        ]

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


@pytest.mark.unit
def test_services_work_with_base_contract_implementations(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)

    text_file = tmp_path / "doc.txt"
    text_file.write_text("semantic embeddings support retrieval", encoding="utf-8")

    vector_store = StubVectorStore()
    ingest_service = IngestService(
        settings=settings,
        loader=StubLoader(),
        embedding=StubEmbedding(),
        vector_store=vector_store,
    )
    search_service = SearchService(
        settings=settings,
        embedding=StubEmbedding(),
        vector_store=vector_store,
    )
    document_service = DocumentService(settings=settings, vector_store=vector_store)

    ingested = ingest_service.ingest_path(text_file, collection="knowledge")
    response = search_service.search("semantic embeddings", collection="knowledge", top_k=1)
    documents = document_service.list_documents("knowledge")
    deleted = document_service.delete_document(ingested[0].doc_id, collection="knowledge")

    assert ingested[0].doc_id == "stub-doc"
    assert response.results
    assert documents[0].doc_id == "stub-doc"
    assert deleted.deleted
