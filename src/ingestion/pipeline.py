"""Composable document-to-unit pipeline used by the ingestion service."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.settings import Settings
from src.core.types import Document, Metadata
from src.ingestion.models import IngestionUnit
from src.ingestion.transforms import (
    BaseTransform,
    ChunkRefinementTransform,
    ImageCaptioningTransform,
    MetadataEnrichmentTransform,
)


@dataclass(slots=True)
class PipelineRun:
    """Prepared ingestion units plus a small execution summary."""

    units: list[IngestionUnit]
    transforms_applied: list[str] = field(default_factory=list)
    source_unit_count: int = 0
    output_unit_count: int = 0

    def to_trace_payload(self) -> Metadata:
        return {
            "source_unit_count": self.source_unit_count,
            "output_unit_count": self.output_unit_count,
            "transforms_applied": self.transforms_applied,
        }


class IngestionPipeline:
    """Expand loaded documents into transformable pre-chunk units."""

    def __init__(self, transforms: list[BaseTransform] | None = None) -> None:
        self.transforms = transforms or []

    def prepare(self, document: Document, collection: str) -> PipelineRun:
        units = self._document_units(document, collection)
        original_count = len(units)
        applied: list[str] = []
        for transform in self.transforms:
            units = transform.transform(units)
            applied.append(transform.name)
        return PipelineRun(
            units=units,
            transforms_applied=applied,
            source_unit_count=original_count,
            output_unit_count=len(units),
        )

    def _document_units(self, document: Document, collection: str) -> list[IngestionUnit]:
        base_metadata = {
            key: value
            for key, value in document.metadata.items()
            if key not in {"pages"}
        }
        base_metadata["collection"] = collection
        base_metadata["doc_id"] = document.id

        pages = document.metadata.get("pages")
        if isinstance(pages, list) and pages:
            units: list[IngestionUnit] = []
            for index, page_payload in enumerate(pages, start=1):
                if not isinstance(page_payload, dict):
                    continue
                page_text = page_payload.get("text")
                if not isinstance(page_text, str):
                    continue
                page_number = page_payload.get("page")
                metadata = base_metadata.copy()
                if isinstance(page_number, int):
                    metadata["page"] = page_number
                units.append(
                    IngestionUnit(
                        unit_id=f"{document.id}:page:{index:04d}",
                        doc_id=document.id,
                        text=page_text,
                        metadata=metadata,
                    )
                )
            if units:
                return units

        return [
            IngestionUnit(
                unit_id=document.id,
                doc_id=document.id,
                text=document.text,
                metadata=base_metadata,
            )
        ]


def create_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    """Build the configured ingestion pipeline."""

    config = settings.ingestion.transforms
    if not config.enabled:
        return IngestionPipeline()

    transforms: list[BaseTransform] = []
    for transform_name in config.order:
        if transform_name == "metadata_enrichment" and config.metadata_enrichment.enabled:
            transforms.append(
                MetadataEnrichmentTransform(
                    section_title_max_length=config.metadata_enrichment.section_title_max_length
                )
            )
        elif transform_name == "chunk_refinement" and config.chunk_refinement.enabled:
            transforms.append(
                ChunkRefinementTransform(
                    collapse_whitespace=config.chunk_refinement.collapse_whitespace
                )
            )
        elif transform_name == "image_captioning" and config.image_captioning.enabled:
            transforms.append(
                ImageCaptioningTransform(
                    stub_caption=config.image_captioning.stub_caption,
                    append_to_text=config.image_captioning.append_to_text,
                )
            )
    return IngestionPipeline(transforms=transforms)
