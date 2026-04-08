from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def _write_settings(
    path: Path,
    storage_path: Path,
    trace_path: Path,
    retrieval_mode: str = "hybrid",
) -> None:
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
  mode: "{retrieval_mode}"
  dense_top_k: 3
  sparse_top_k: 3
  dense_candidate_multiplier: 3
  sparse_candidate_multiplier: 4
  max_candidate_top_k: 10
  rrf_k: 60
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
def test_hybrid_search_runs_sparse_retrieval_and_rrf_fusion(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path, retrieval_mode="hybrid")
    settings = load_settings(config_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "hybrid.txt").write_text(
        "Hybrid retrieval uses raretokenalpha embeddings for ranking and matching.",
        encoding="utf-8",
    )
    (docs_dir / "other.md").write_text(
        "Cooking recipes focus on ingredients, heat, and timing.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    ingest_service.ingest_path(docs_dir, collection="knowledge")

    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=TraceStore(trace_path),
    )

    response = search_service.search(
        "raretokenalpha embeddings",
        collection="knowledge",
        top_k=2,
    )

    assert response.retrieval_mode == "hybrid"
    assert response.results
    assert response.results[0].source_path.endswith("hybrid.txt")
    assert set(response.results[0].metadata["rrf_sources"]) == {"dense", "sparse"}
    assert response.citations[0].source_path.endswith("hybrid.txt")

    query_trace = json.loads(trace_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert [stage["stage"] for stage in query_trace["stages"]] == [
        "embed_query",
        "dense_retrieve",
        "sparse_retrieve",
        "rrf_fuse",
    ]
    assert query_trace["stages"][1]["data"]["candidate_top_k"] == 6
    assert query_trace["stages"][2]["data"]["candidate_top_k"] == 8


@pytest.mark.integration
def test_dense_mode_override_still_works_with_hybrid_capable_setup(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path, retrieval_mode="hybrid")
    settings = load_settings(config_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Python retrieval systems use semantic embeddings for search.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    ingest_service.ingest_path(text_file, collection="knowledge")

    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=TraceStore(trace_path),
    )

    response = search_service.search(
        "semantic embeddings",
        collection="knowledge",
        top_k=1,
        mode="dense",
    )

    assert response.retrieval_mode == "dense"
    assert response.results
    assert "rrf_sources" not in response.results[0].metadata
    assert response.citations[0].collection == "knowledge"

    query_trace = json.loads(trace_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert [stage["stage"] for stage in query_trace["stages"]] == [
        "embed_query",
        "dense_retrieve",
    ]
