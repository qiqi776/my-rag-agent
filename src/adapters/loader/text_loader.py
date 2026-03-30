"""Local text and markdown loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.errors import UnsupportedFileTypeError
from src.core.types import Document


class TextLoader:
    """Load `.txt` and `.md` documents from local storage."""

    def __init__(self, supported_extensions: list[str]) -> None:
        self.supported_extensions = {ext.lower() for ext in supported_extensions}

    def load(self, path: str | Path) -> Document:
        file_path = Path(path).resolve()
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {suffix}. Supported: {sorted(self.supported_extensions)}"
            )

        raw = file_path.read_bytes()
        text = raw.decode("utf-8")
        doc_id = hashlib.sha256(raw).hexdigest()
        return Document(
            id=doc_id,
            text=text,
            metadata={
                "source_path": str(file_path),
                "doc_type": suffix.lstrip("."),
            },
        )

