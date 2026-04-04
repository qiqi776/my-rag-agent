"""Deterministic fake LLM for local testing."""

from __future__ import annotations

import re

from src.adapters.llm.base_llm import BaseLLM

_WHITESPACE = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class FakeLLM(BaseLLM):
    """Simple deterministic LLM stub."""

    @property
    def provider(self) -> str:
        return "fake"

    def generate_answer(
        self,
        query: str,
        contexts: list[str],
        max_chars: int,
    ) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")

        normalized_query = _collapse_whitespace(query)
        collapsed_contexts = [_collapse_whitespace(text) for text in contexts if text.strip()]

        if not collapsed_contexts:
            return f"No supporting context available for '{normalized_query}'."[:max_chars]

        pieces = [
            f"[{index}] {context[:120]}"
            for index, context in enumerate(collapsed_contexts[:2], start=1)
        ]
        answer = (
            f"For '{normalized_query}', the retrieved context indicates: "
            + " ".join(pieces)
        )
        return answer[:max_chars].rstrip()
