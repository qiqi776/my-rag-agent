from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.ingest_service import IngestService
from src.core.settings import load_settings

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


def _write_settings(path: Path, *, transforms_enabled: bool) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 200
  chunk_overlap: 20
  supported_extensions:
    - ".txt"
  transforms:
    enabled: {str(transforms_enabled).lower()}
    order:
      - "chunk_refinement"
      - "metadata_enrichment"
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
    provider: "text"
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
def test_transform_pipeline_enriches_text_ingestion_without_losing_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, transforms_enabled=True)
    settings = load_settings(config_path)
    vector_store = InMemoryVectorStore()
    embedding = FakeEmbedding(settings.adapters.embedding.dimensions)

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
    )

    source_file = tmp_path / "sample.txt"
    source_file.write_text(
        (FIXTURE_DIR / "refinement_sample.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    ingest_service.ingest_path(source_file, collection="knowledge")
    records = vector_store.list_records("knowledge")

    assert len(records) == 1
    assert records[0].metadata["source_path"].endswith("sample.txt")
    assert records[0].metadata["section_title"] == "Minimal Modular RAG"
    assert records[0].metadata["refinement_applied"] is True
    assert records[0].text == "Minimal Modular RAG\n\nTransform cleanup should normalize spacing."


@pytest.mark.integration
def test_disabled_transform_pipeline_degrades_to_original_text_ingestion(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, transforms_enabled=False)
    settings = load_settings(config_path)
    vector_store = InMemoryVectorStore()
    embedding = FakeEmbedding(settings.adapters.embedding.dimensions)

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=embedding,
        vector_store=vector_store,
    )

    source_file = tmp_path / "sample.txt"
    raw_text = (FIXTURE_DIR / "refinement_sample.txt").read_text(encoding="utf-8")
    source_file.write_text(raw_text, encoding="utf-8")

    ingest_service.ingest_path(source_file, collection="knowledge")
    records = vector_store.list_records("knowledge")

    assert len(records) == 1
    assert records[0].text == raw_text
    assert "section_title" not in records[0].metadata
