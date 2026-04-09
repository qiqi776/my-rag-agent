from __future__ import annotations

from pathlib import Path

import pytest

from src.application.ingest_service import IngestedDocument
from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.models import IngestionUnit
from src.ingestion.pipeline import PipelineRun
from src.observability.dashboard.services.ingestion_service import DashboardIngestionService


def _write_settings(path: Path) -> None:
    path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "knowledge"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
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


class FakeLoader:
    def load(self, path: Path) -> Document:
        return Document(
            id="doc-1",
            text="fake preview body",
            metadata={
                "source_path": str(path),
                "page_count": 1,
                "quality_status": "warning",
                "quality_warnings": ["low printable ratio"],
            },
        )


class FakePipeline:
    def prepare(self, document: Document, collection: str) -> PipelineRun:
        return PipelineRun(
            units=[
                IngestionUnit(
                    unit_id=f"{document.id}:unit:1",
                    doc_id=document.id,
                    text=document.text,
                    metadata={"source_path": document.metadata["source_path"], "collection": collection},
                )
            ],
            transforms_applied=["metadata_enrichment"],
            source_unit_count=1,
            output_unit_count=1,
        )


class FakeIngestService:
    def ingest_path(self, path: str | Path, collection: str | None = None) -> list[IngestedDocument]:
        return [
            IngestedDocument(
                source_path=str(Path(path).resolve()),
                doc_id="doc-1",
                collection=collection or "knowledge",
                chunk_count=2,
            )
        ]


@pytest.mark.unit
def test_dashboard_ingestion_service_maps_preview_and_ingest_results(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)
    text_file = tmp_path / "doc.txt"
    text_file.write_text("hello dashboard", encoding="utf-8")

    service = DashboardIngestionService(
        settings=settings,
        loader=FakeLoader(),
        pipeline=FakePipeline(),
        ingest_service=FakeIngestService(),
    )

    preview = service.preview_path(text_file, "knowledge")
    ingested = service.ingest_path(text_file, "knowledge")

    assert preview[0]["quality_status"] == "warning"
    assert preview[0]["warnings"] == ["low printable ratio"]
    assert preview[0]["transforms"] == ["metadata_enrichment"]
    assert ingested[0]["chunk_count"] == 2
    assert ingested[0]["quality_status"] == "warning"
