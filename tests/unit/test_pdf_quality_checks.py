from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.loader.pdf_loader import PdfLoader, PDFPage


@pytest.mark.unit
def test_pdf_loader_flags_clean_text_as_good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "clean.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    loader = PdfLoader([".pdf"])
    monkeypatch.setattr(
        loader,
        "_extract_pages",
        lambda _path, _raw: [
            PDFPage(page=1, text="Virtual memory splits an address space into pages."),
            PDFPage(page=2, text="Paging avoids external fragmentation in many systems."),
        ],
    )

    document = loader.load(pdf_path)

    assert document.metadata["quality_status"] == "good"
    assert document.metadata["non_empty_page_ratio"] == 1.0
    assert document.metadata["printable_char_ratio"] == 1.0
    assert document.metadata["quality_warnings"] == []


@pytest.mark.unit
def test_pdf_loader_flags_garbled_text_as_bad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "garbled.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    loader = PdfLoader([".pdf"])
    monkeypatch.setattr(
        loader,
        "_extract_pages",
        lambda _path, _raw: [
            PDFPage(page=1, text="û $ÀÈ³1°#1 CB+ÄT¦Gý B+?± ´ _È² Ì ´a _L5 AêÃÈL5B+"),
            PDFPage(page=2, text="¼ ´ È û $ÀÈ³1°#1 CB+ÄT¦Gý B+?±,´ _È²"),
        ],
    )

    document = loader.load(pdf_path)

    assert document.metadata["quality_status"] == "bad"
    assert document.metadata["latin_extended_ratio"] > 0.2
    assert "high latin-extended ratio" in document.metadata["quality_warnings"]


@pytest.mark.unit
def test_pdf_loader_flags_sparse_pages_as_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "sparse.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    loader = PdfLoader([".pdf"])
    monkeypatch.setattr(
        loader,
        "_extract_pages",
        lambda _path, _raw: [
            PDFPage(page=1, text="Address translation maps virtual memory to physical memory."),
            PDFPage(page=2, text=""),
        ],
    )

    document = loader.load(pdf_path)

    assert document.metadata["quality_status"] == "warning"
    assert document.metadata["non_empty_page_ratio"] == 0.5
    assert "high empty-page ratio" in document.metadata["quality_warnings"]
