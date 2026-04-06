from __future__ import annotations

import pytest

from src.ingestion.models import IngestionUnit
from src.ingestion.transforms.chunk_refinement import ChunkRefinementTransform
from src.ingestion.transforms.image_captioning import ImageCaptioningTransform
from src.ingestion.transforms.metadata_enrichment import MetadataEnrichmentTransform


@pytest.mark.unit
def test_metadata_enrichment_transform_adds_section_title() -> None:
    transform = MetadataEnrichmentTransform(section_title_max_length=64)
    units = [
        IngestionUnit(
            unit_id="unit-1",
            doc_id="doc-1",
            text="Section Title\n\nBody text",
            metadata={"source_path": "/tmp/doc.txt"},
        )
    ]

    enriched = transform.transform(units)

    assert enriched[0].metadata["section_title"] == "Section Title"
    assert enriched[0].metadata["metadata_enriched"] is True


@pytest.mark.unit
def test_chunk_refinement_transform_normalizes_spacing_and_page_noise() -> None:
    transform = ChunkRefinementTransform(collapse_whitespace=True)
    units = [
        IngestionUnit(
            unit_id="unit-1",
            doc_id="doc-1",
            text="  Section   Title  \n\nPage 1\n\nBody    text   here. ",
            metadata={"source_path": "/tmp/doc.txt"},
        )
    ]

    refined = transform.transform(units)

    assert refined[0].text == "Section Title\n\nBody text here."
    assert refined[0].metadata["refinement_applied"] is True


@pytest.mark.unit
def test_image_captioning_transform_is_noop_without_images() -> None:
    transform = ImageCaptioningTransform("stub caption", append_to_text=True)
    units = [
        IngestionUnit(
            unit_id="unit-1",
            doc_id="doc-1",
            text="Body text",
            metadata={"source_path": "/tmp/doc.txt"},
        )
    ]

    transformed = transform.transform(units)

    assert transformed[0].text == "Body text"
    assert "caption" not in transformed[0].metadata


@pytest.mark.unit
def test_image_captioning_transform_adds_stub_caption_when_images_exist() -> None:
    transform = ImageCaptioningTransform("stub caption", append_to_text=True)
    units = [
        IngestionUnit(
            unit_id="unit-1",
            doc_id="doc-1",
            text="Body text",
            metadata={
                "source_path": "/tmp/doc.txt",
                "images": [{"id": "img-1"}],
            },
        )
    ]

    transformed = transform.transform(units)

    assert transformed[0].metadata["caption"] == "stub caption"
    assert transformed[0].metadata["caption_generated_by"] == "stub"
    assert "stub caption" in transformed[0].text
