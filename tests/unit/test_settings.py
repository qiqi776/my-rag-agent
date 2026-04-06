from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_settings(
    path: Path,
    chunk_overlap: int = 50,
    retrieval_mode: str = "dense",
) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 500
  chunk_overlap: {chunk_overlap}
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  mode: "{retrieval_mode}"
  dense_top_k: 5
  sparse_top_k: 5
  rrf_k: 60
generation:
  max_context_results: 3
  max_answer_chars: 400
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "./data/traces/app.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_load_settings_success(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)

    settings = load_settings(config_path)

    assert settings.project.name == "minimal-modular-rag"
    assert settings.ingestion.chunk_size == 500
    assert settings.adapters.embedding.dimensions == 16
    assert settings.retrieval.mode == "dense"
    assert settings.retrieval.sparse_top_k == 5
    assert settings.retrieval.rrf_k == 60
    assert settings.adapters.llm.provider == "fake"
    assert settings.adapters.reranker.provider == "fake"
    assert settings.generation.max_context_results == 3
    assert settings.generation.max_answer_chars == 400
    assert settings.ingestion.transforms.enabled is False
    assert settings.ingestion.transforms.order == [
        "metadata_enrichment",
        "chunk_refinement",
        "image_captioning",
    ]


@pytest.mark.unit
def test_load_settings_rejects_invalid_overlap(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, chunk_overlap=500)

    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_settings(config_path)


@pytest.mark.unit
def test_load_settings_rejects_invalid_retrieval_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, retrieval_mode="unknown")

    with pytest.raises(ConfigError, match="retrieval.mode"):
        load_settings(config_path)


@pytest.mark.unit
def test_load_settings_parses_transform_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 500
  chunk_overlap: 50
  supported_extensions:
    - ".pdf"
  transforms:
    enabled: true
    order:
      - "chunk_refinement"
      - "metadata_enrichment"
    metadata_enrichment:
      enabled: true
      section_title_max_length: 64
    chunk_refinement:
      enabled: true
      collapse_whitespace: true
    image_captioning:
      enabled: false
      stub_caption: "stub caption"
retrieval:
  mode: "dense"
  dense_top_k: 5
generation:
  max_context_results: 3
  max_answer_chars: 400
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
  trace_enabled: true
  trace_file: "./data/traces/app.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.adapters.loader.provider == "pdf"
    assert settings.ingestion.transforms.enabled is True
    assert settings.ingestion.transforms.order == [
        "chunk_refinement",
        "metadata_enrichment",
    ]
    assert settings.ingestion.transforms.metadata_enrichment.section_title_max_length == 64
    assert settings.ingestion.transforms.image_captioning.stub_caption == "stub caption"


@pytest.mark.unit
def test_load_settings_rejects_unknown_transform_order(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 500
  chunk_overlap: 50
  supported_extensions:
    - ".txt"
  transforms:
    enabled: true
    order:
      - "not_real"
retrieval:
  mode: "dense"
  dense_top_k: 5
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
  trace_enabled: true
  trace_file: "./data/traces/app.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported transforms"):
        load_settings(config_path)
