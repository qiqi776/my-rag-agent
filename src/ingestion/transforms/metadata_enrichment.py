"""Metadata enrichment transform for ingestion units."""

from __future__ import annotations

from src.ingestion.models import IngestionUnit
from src.ingestion.transforms.base import BaseTransform


class MetadataEnrichmentTransform(BaseTransform):
    """Attach stable section-level metadata without changing content."""

    name = "metadata_enrichment"

    def __init__(self, section_title_max_length: int = 120) -> None:
        self.section_title_max_length = section_title_max_length

    def transform(self, units: list[IngestionUnit]) -> list[IngestionUnit]:
        enriched: list[IngestionUnit] = []
        for unit in units:
            metadata = unit.metadata.copy()
            section_title = self._section_title(unit.text)
            if section_title:
                metadata.setdefault("section_title", section_title)
            metadata["metadata_enriched"] = True
            enriched.append(
                IngestionUnit(
                    unit_id=unit.unit_id,
                    doc_id=unit.doc_id,
                    text=unit.text,
                    metadata=metadata,
                )
            )
        return enriched

    def _section_title(self, text: str) -> str | None:
        for line in text.splitlines():
            candidate = " ".join(line.split()).strip()
            if candidate:
                return candidate[: self.section_title_max_length]
        return None
