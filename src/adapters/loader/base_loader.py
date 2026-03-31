"""Base contract for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.types import Document


class BaseLoader(ABC):
    """Minimal loader interface required by the ingestion flow."""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load a source file into a normalized document object."""
