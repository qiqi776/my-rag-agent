"""Base contract for answer-generation LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Minimal LLM surface required for answer synthesis."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the configured provider name."""

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        contexts: list[str],
        max_chars: int,
    ) -> str:
        """Generate an answer from query text and supporting contexts."""
