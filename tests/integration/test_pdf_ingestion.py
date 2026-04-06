from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


def _write_settings(path: Path) -> None:
    path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 200
  chunk_overlap: 20
  supported_extensions:
    - ".pdf"
  transforms:
    enabled: true
    order:
      - "metadata_enrichment"
      - "chunk_refinement"
    metadata_enrichment:
      enabled: true
      section_title_max_length: 80
    chunk_refinement:
      enabled: true
      collapse_whitespace: true
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "pdf"
  embedding:
    provider: "fake"
    dimensions: 16
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


@pytest.mark.integration
def test_pdf_ingestion_keeps_page_metadata_searchable(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)
    vector_store = InMemoryVectorStore()
    embedding = FakeEmbedding(settings.adapters.embedding.dimensions)

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
    )
    search_service = SearchService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
    )

    ingest_service.ingest_path(FIXTURE_DIR / "multi_page.pdf", collection="knowledge")
    response = search_service.search("transform ordering citations", collection="knowledge", top_k=1)

    assert response.results
    assert response.results[0].source_path.endswith("multi_page.pdf")
    assert response.results[0].page == 2
    assert response.citations[0].page == 2
    assert response.results[0].metadata["page"] == 2
