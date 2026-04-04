"""Base contract for reranker adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.response.response_builder import SearchResultItem


class BaseReranker(ABC):
    """Minimal reranker surface required for M6."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the configured provider name."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        """Return reranked search result items."""
