"""Deterministic fake reranker for local testing."""

from __future__ import annotations

import re
from dataclasses import replace

from src.adapters.reranker.base_reranker import BaseReranker
from src.response.response_builder import SearchResultItem

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


class FakeReranker(BaseReranker):
    """Rerank results using lexical overlap as a deterministic heuristic."""

    @property
    def provider(self) -> str:
        return "fake"

    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_terms = _tokens(query)
        rescored: list[tuple[int, SearchResultItem]] = []
        for item in results:
            overlap = len(query_terms & _tokens(item.text))
            metadata = item.metadata.copy()
            metadata["original_rank"] = item.rank
            metadata["rerank_overlap"] = overlap
            rescored.append(
                (
                    overlap,
                    replace(item, metadata=metadata),
                )
            )

        rescored.sort(key=lambda pair: (-pair[0], -pair[1].score, pair[1].chunk_id))
        reranked = [
            replace(item, rank=index)
            for index, (_, item) in enumerate(
                rescored[: top_k or len(rescored)],
                start=1,
            )
        ]
        return reranked
