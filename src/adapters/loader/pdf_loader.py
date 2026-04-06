"""Minimal PDF loader with page-aware metadata support."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.adapters.loader.base_loader import BaseLoader
from src.core.errors import UnsupportedFileTypeError
from src.core.types import Document, Metadata


def _load_pdf_reader() -> type[Any] | None:
    if importlib.util.find_spec("pypdf") is None:  # pragma: no cover - environment-dependent
        return None

    module = importlib.import_module("pypdf")  # pragma: no cover - environment-dependent
    reader = getattr(module, "PdfReader", None)
    return reader if isinstance(reader, type) else None


PdfReader = _load_pdf_reader()

_OBJECT_PATTERN = re.compile(rb"(?P<object_id>\d+)\s+\d+\s+obj(?P<body>.*?)endobj", re.S)
_STREAM_PATTERN = re.compile(rb"stream\r?\n(?P<stream>.*?)\r?\nendstream", re.S)
_PAGE_PATTERN = re.compile(rb"/Type\s*/Page\b")
_CONTENTS_PATTERN = re.compile(
    rb"/Contents\s*(?P<value>\[(?:.*?)\]|\d+\s+\d+\s+R)",
    re.S,
)
_REF_PATTERN = re.compile(rb"(\d+)\s+\d+\s+R")
_TEXT_TOKEN_PATTERN = re.compile(
    r"\[(?P<array>.*?)\]\s*TJ|"
    r"\((?P<literal>(?:\\.|[^\\)])*)\)\s*(?:Tj|')|"
    r"<(?P<hex>[0-9A-Fa-f]+)>\s*Tj",
    re.S,
)
_ARRAY_LITERAL_PATTERN = re.compile(r"\((?P<literal>(?:\\.|[^\\)])*)\)")
_ARRAY_HEX_PATTERN = re.compile(r"<(?P<hex>[0-9A-Fa-f]+)>")


@dataclass(frozen=True, slots=True)
class PDFPage:
    page: int
    text: str


class PdfLoader(BaseLoader):
    """Load a PDF into a normalized document with page-level metadata."""

    def __init__(self, supported_extensions: list[str] | None = None) -> None:
        self.supported_extensions = {
            ext.lower() for ext in (supported_extensions or [".pdf"])
        }

    def load(self, path: str | Path) -> Document:
        file_path = Path(path).resolve()
        if file_path.suffix.lower() not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix.lower()}. "
                f"Supported: {sorted(self.supported_extensions)}"
            )

        raw = file_path.read_bytes()
        doc_id = hashlib.sha256(raw).hexdigest()
        pages = self._extract_pages(file_path, raw)
        combined_text = "\n\n".join(page.text for page in pages if page.text.strip()).strip()

        metadata: Metadata = {
            "source_path": str(file_path),
            "doc_type": "pdf",
            "page_count": len(pages),
            "pages": [
                {
                    "page": page.page,
                    "text": page.text,
                }
                for page in pages
            ],
        }
        title = self._title_from_pages(pages)
        if title:
            metadata["title"] = title

        return Document(
            id=doc_id,
            text=combined_text,
            metadata=metadata,
        )

    def _extract_pages(self, file_path: Path, raw: bytes) -> list[PDFPage]:
        if PdfReader is not None:
            parsed = self._extract_pages_with_pypdf(file_path)
            if parsed:
                return parsed

        parsed = self._extract_pages_from_raw(raw)
        if parsed:
            return parsed

        return [PDFPage(page=1, text="")]

    def _extract_pages_with_pypdf(self, file_path: Path) -> list[PDFPage]:
        if PdfReader is None:  # pragma: no cover - optional dependency guard
            return []

        try:
            reader = PdfReader(str(file_path))
        except Exception:  # pragma: no cover - parser fallback path
            return []

        pages: list[PDFPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # pragma: no cover - parser fallback path
                text = ""
            pages.append(PDFPage(page=index, text=text.strip()))
        return pages

    def _extract_pages_from_raw(self, raw: bytes) -> list[PDFPage]:
        objects: dict[int, bytes] = {}
        page_object_ids: list[int] = []

        for match in _OBJECT_PATTERN.finditer(raw):
            object_id = int(match.group("object_id"))
            body = match.group("body")
            objects[object_id] = body
            if _PAGE_PATTERN.search(body):
                page_object_ids.append(object_id)

        pages: list[PDFPage] = []
        for page_number, object_id in enumerate(page_object_ids, start=1):
            page_body = objects[object_id]
            contents = self._content_object_ids(page_body)
            text_fragments: list[str] = []
            for content_id in contents:
                body = objects.get(content_id)
                if body is None:
                    continue
                decoded_stream = self._decode_stream(body)
                if decoded_stream is None:
                    continue
                fragment = self._extract_text_from_stream(decoded_stream)
                if fragment:
                    text_fragments.append(fragment)
            pages.append(PDFPage(page=page_number, text="\n".join(text_fragments).strip()))

        if pages:
            return pages

        fallback_text = []
        for body in objects.values():
            decoded_stream = self._decode_stream(body)
            if decoded_stream is None:
                continue
            fragment = self._extract_text_from_stream(decoded_stream)
            if fragment:
                fallback_text.append(fragment)
        if fallback_text:
            return [PDFPage(page=1, text="\n".join(fallback_text).strip())]
        return []

    def _content_object_ids(self, page_body: bytes) -> list[int]:
        match = _CONTENTS_PATTERN.search(page_body)
        if match is None:
            return []
        refs = [int(item) for item in _REF_PATTERN.findall(match.group("value"))]
        return refs

    def _decode_stream(self, object_body: bytes) -> bytes | None:
        match = _STREAM_PATTERN.search(object_body)
        if match is None:
            return None

        stream = match.group("stream")
        if b"/Filter" not in object_body:
            return stream
        if b"/FlateDecode" in object_body:
            try:
                return zlib.decompress(stream)
            except zlib.error:
                return None
        return None

    def _extract_text_from_stream(self, stream: bytes) -> str:
        content = stream.decode("latin-1", errors="ignore")
        fragments: list[str] = []
        for match in _TEXT_TOKEN_PATTERN.finditer(content):
            array_value = match.group("array")
            literal_value = match.group("literal")
            hex_value = match.group("hex")

            if array_value is not None:
                parts = [
                    self._decode_pdf_literal(literal.group("literal"))
                    for literal in _ARRAY_LITERAL_PATTERN.finditer(array_value)
                ]
                parts.extend(
                    self._decode_pdf_hex_string(hex_match.group("hex"))
                    for hex_match in _ARRAY_HEX_PATTERN.finditer(array_value)
                )
                text = "".join(part for part in parts if part)
            elif literal_value is not None:
                text = self._decode_pdf_literal(literal_value)
            else:
                text = self._decode_pdf_hex_string(hex_value or "")

            cleaned = " ".join(text.split()).strip()
            if cleaned:
                fragments.append(cleaned)
        return "\n".join(fragments)

    def _decode_pdf_literal(self, value: str) -> str:
        decoded: list[str] = []
        index = 0
        while index < len(value):
            char = value[index]
            if char != "\\":
                decoded.append(char)
                index += 1
                continue

            index += 1
            if index >= len(value):
                break
            escaped = value[index]
            escapes = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "b": "\b",
                "f": "\f",
                "(": "(",
                ")": ")",
                "\\": "\\",
            }
            if escaped in escapes:
                decoded.append(escapes[escaped])
                index += 1
                continue
            if escaped in "\n\r":
                while index < len(value) and value[index] in "\n\r":
                    index += 1
                continue
            if escaped in "01234567":
                octal = [escaped]
                index += 1
                for _ in range(2):
                    if index < len(value) and value[index] in "01234567":
                        octal.append(value[index])
                        index += 1
                    else:
                        break
                decoded.append(chr(int("".join(octal), 8)))
                continue

            decoded.append(escaped)
            index += 1

        return "".join(decoded)

    def _decode_pdf_hex_string(self, value: str) -> str:
        if not value:
            return ""
        normalized = value if len(value) % 2 == 0 else f"{value}0"
        try:
            return bytes.fromhex(normalized).decode("latin-1", errors="ignore")
        except ValueError:
            return ""

    def _title_from_pages(self, pages: list[PDFPage]) -> str | None:
        for page in pages:
            for line in page.text.splitlines():
                candidate = line.strip()
                if candidate:
                    return candidate
        return None
