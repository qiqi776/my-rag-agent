from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.document_service import DocumentService
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore


def _write_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
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
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_duplicate_ingestion_is_idempotent_in_document_listing(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Python retrieval systems use embeddings for semantic search.",
        encoding="utf-8",
    )

    first_ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    second_ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    document_service = DocumentService(settings=settings, vector_store=create_vector_store(settings))

    first = first_ingest_service.ingest_path(text_file, collection="knowledge")
    second = second_ingest_service.ingest_path(text_file, collection="knowledge")
    documents = document_service.list_documents("knowledge")

    assert first[0].doc_id == second[0].doc_id
    assert len(documents) == 1
    assert documents[0].chunk_count == first[0].chunk_count


@pytest.mark.integration
def test_delete_document_keeps_other_collection_searchable(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)
    settings = load_settings(config_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Semantic embeddings help Python retrieval systems answer questions.",
        encoding="utf-8",
    )

    trace_store = TraceStore(trace_path)
    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=trace_store,
    )

    ingest_result = ingest_service.ingest_path(text_file, collection="alpha")[0]
    IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=trace_store,
    ).ingest_path(text_file, collection="beta")

    document_service = DocumentService(
        settings=settings,
        vector_store=create_vector_store(settings),
    )
    delete_result = document_service.delete_document(ingest_result.doc_id, collection="alpha")
    alpha_docs = document_service.list_documents("alpha")
    beta_docs = document_service.list_documents("beta")
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=trace_store,
    )
    beta_response = search_service.search("semantic embeddings", collection="beta", top_k=1)

    assert delete_result.deleted
    assert alpha_docs == []
    assert len(beta_docs) == 1
    assert beta_response.results
    assert beta_response.results[0].source_path.endswith("python.txt")
    assert beta_response.citations[0].collection == "beta"
