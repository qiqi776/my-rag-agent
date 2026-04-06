"""Stub image captioning transform for future multimodal enrichment."""

from __future__ import annotations

from src.ingestion.models import IngestionUnit
from src.ingestion.transforms.base import BaseTransform


class ImageCaptioningTransform(BaseTransform):
    """Attach a placeholder caption when upstream metadata references images."""

    name = "image_captioning"

    def __init__(self, stub_caption: str, append_to_text: bool = False) -> None:
        self.stub_caption = stub_caption
        self.append_to_text = append_to_text

    def transform(self, units: list[IngestionUnit]) -> list[IngestionUnit]:
        captioned: list[IngestionUnit] = []
        for unit in units:
            metadata = unit.metadata.copy()
            text = unit.text
            images = metadata.get("images")
            if images:
                metadata["caption"] = self.stub_caption
                metadata["caption_generated_by"] = "stub"
                if self.append_to_text and self.stub_caption not in text:
                    text = f"{text.rstrip()}\n\n[Image Caption]\n{self.stub_caption}".strip()
            captioned.append(
                IngestionUnit(
                    unit_id=unit.unit_id,
                    doc_id=unit.doc_id,
                    text=text,
                    metadata=metadata,
                )
            )
        return captioned
