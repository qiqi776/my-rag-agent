from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline, create_ingestion_pipeline
from src.ingestion.transforms.image_captioning import ImageCaptioningTransform


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
    - ".pdf"
  transforms:
    enabled: true
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


@pytest.mark.unit
def test_ingestion_pipeline_applies_configured_order_and_preserves_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)
    pipeline = create_ingestion_pipeline(settings)
    document = Document(
        id="doc-1",
        text="unused once pages are present",
        metadata={
            "source_path": "/tmp/demo.pdf",
            "doc_type": "pdf",
            "pages": [
                {"page": 1, "text": "  Section   One  \n\nPage 1\n\nDetails  here."},
                {"page": 2, "text": "Second Section\n\nMore context."},
            ],
        },
    )

    run = pipeline.prepare(document, collection="knowledge")

    assert run.transforms_applied == ["chunk_refinement", "metadata_enrichment"]
    assert run.source_unit_count == 2
    assert run.output_unit_count == 2
    assert run.units[0].metadata["collection"] == "knowledge"
    assert run.units[0].metadata["doc_id"] == "doc-1"
    assert run.units[0].metadata["page"] == 1
    assert run.units[0].metadata["section_title"] == "Section One"
    assert run.units[0].metadata["refinement_applied"] is True
    assert "Page 1" not in run.units[0].text


@pytest.mark.unit
def test_ingestion_pipeline_allows_noop_transforms_without_losing_units() -> None:
    pipeline = IngestionPipeline(transforms=[ImageCaptioningTransform("stub caption")])
    document = Document(
        id="doc-1",
        text="plain text",
        metadata={"source_path": "/tmp/demo.txt"},
    )

    run = pipeline.prepare(document, collection="knowledge")

    assert run.transforms_applied == ["image_captioning"]
    assert len(run.units) == 1
    assert run.units[0].text == "plain text"
    assert "caption" not in run.units[0].metadata
