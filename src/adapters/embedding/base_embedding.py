"""Base contract for embedding adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Minimal embedding interface required by ingestion and search."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensionality."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
