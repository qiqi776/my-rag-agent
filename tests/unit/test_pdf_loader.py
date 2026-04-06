from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.loader.pdf_loader import PdfLoader

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


@pytest.mark.unit
def test_pdf_loader_reads_single_page_pdf_and_extracts_title() -> None:
    loader = PdfLoader([".pdf"])

    document = loader.load(FIXTURE_DIR / "simple.pdf")

    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["page_count"] == 1
    assert document.metadata["title"] == "Minimal Modular RAG PDF"
    assert "PDF ingestion keeps source metadata." in document.text
    assert document.metadata["pages"][0]["page"] == 1


@pytest.mark.unit
def test_pdf_loader_preserves_page_level_text() -> None:
    loader = PdfLoader([".pdf"])

    document = loader.load(FIXTURE_DIR / "multi_page.pdf")

    assert document.metadata["page_count"] == 2
    assert len(document.metadata["pages"]) == 2
    assert document.metadata["pages"][1]["page"] == 2
    assert "Second Page Details" in document.metadata["pages"][1]["text"]
    assert "page citations" in document.metadata["pages"][1]["text"].lower()


@pytest.mark.unit
def test_pdf_loader_rejects_unsupported_suffix(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-pdf.txt"
    file_path.write_text("hello", encoding="utf-8")

    loader = PdfLoader([".pdf"])

    with pytest.raises(Exception, match="Unsupported file type"):
        loader.load(file_path)
