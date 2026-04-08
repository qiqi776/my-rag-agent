from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.adapters.loader.text_loader import TextLoader
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.errors import EmptyQueryError
from src.core.settings import load_settings
from src.core.types import ChunkRecord, RetrievalResult
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def _write_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 40
  chunk_overlap: 5
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  dense_top_k: 3
  sparse_top_k: 3
  dense_candidate_multiplier: 4
  sparse_candidate_multiplier: 5
  max_candidate_top_k: 9
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "memory"
    storage_path: "{storage_path}"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class RecordingSparseRetriever(SparseRetriever):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, collection: str, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append({"collection": collection, "query": query, "top_k": top_k})
        return []


class RecordingVectorStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.query_top_ks: list[int] = []

    def query(self, collection: str, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        self.query_top_ks.append(top_k)
        return super().query(collection, query_vector, top_k)


@pytest.mark.unit
def test_search_service_rejects_empty_query(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    service = SearchService(
        settings=settings,
        embedding=FakeEmbedding(settings.adapters.embedding.dimensions),
        vector_store=InMemoryVectorStore(),
        trace_store=TraceStore(trace_path),
    )

    with pytest.raises(EmptyQueryError):
        service.search("   ")


@pytest.mark.unit
def test_search_service_returns_stable_response_output(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    store = InMemoryVectorStore()
    store.upsert(
        "knowledge",
        [
            ChunkRecord(
                id="chunk-1",
                doc_id="doc-1",
                text="semantic embeddings support retrieval",
                embedding=[1.0] * settings.adapters.embedding.dimensions,
                metadata={
                    "source_path": str((tmp_path / "doc.txt").resolve()),
                    "collection": "knowledge",
                    "chunk_index": 0,
                },
            )
        ],
    )

    service = SearchService(
        settings=settings,
        embedding=FakeEmbedding(settings.adapters.embedding.dimensions),
        vector_store=store,
        trace_store=TraceStore(trace_path),
    )

    response = service.search("semantic embeddings", collection="knowledge", top_k=1)

    assert response.query == "semantic embeddings"
    assert response.normalized_query == "semantic embeddings"
    assert response.collection == "knowledge"
    assert response.result_count == 1
    assert response.results[0].rank == 1
    assert response.citations[0].chunk_id == "chunk-1"


@pytest.mark.unit
def test_search_service_uses_candidate_top_k_boundaries_for_hybrid_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    store = RecordingVectorStore()
    store.upsert(
        "knowledge",
        [
            ChunkRecord(
                id="chunk-1",
                doc_id="doc-1",
                text="semantic embeddings support retrieval",
                embedding=[1.0] * settings.adapters.embedding.dimensions,
                metadata={
                    "source_path": str((tmp_path / "doc.txt").resolve()),
                    "collection": "knowledge",
                    "chunk_index": 0,
                },
            )
        ],
    )
    sparse_retriever = RecordingSparseRetriever()
    service = SearchService(
        settings=settings,
        embedding=FakeEmbedding(settings.adapters.embedding.dimensions),
        vector_store=store,
        sparse_retriever=sparse_retriever,
        trace_store=TraceStore(trace_path),
    )

    response = service.search("semantic embeddings", collection="knowledge", top_k=2, mode="hybrid")

    assert response.results
    assert store.query_top_ks == [8]
    assert sparse_retriever.calls == [
        {
            "collection": "knowledge",
            "query": "semantic embeddings",
            "top_k": 9,
        }
    ]


@pytest.mark.unit
def test_ingest_service_creates_deterministic_chunk_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    text_file = tmp_path / "doc.txt"
    text_file.write_text("python retrieval system for testing", encoding="utf-8")

    service = IngestService(
        settings=settings,
        loader=TextLoader(settings.ingestion.supported_extensions),
        embedding=FakeEmbedding(settings.adapters.embedding.dimensions),
        vector_store=InMemoryVectorStore(),
        trace_store=TraceStore(trace_path),
    )

    first = service.ingest_path(text_file)[0]
    second = service.ingest_path(text_file)[0]

    assert first.doc_id == second.doc_id
    assert first.chunk_count == second.chunk_count
