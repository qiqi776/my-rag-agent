"""Fusion utilities for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from src.core.types import RetrievalResult


def rrf_fuse(
    result_sets: list[tuple[str, list[RetrievalResult]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result sets using Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("k must be > 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0")

    aggregated: dict[str, dict[str, Any]] = {}

    for source_name, results in result_sets:
        for rank, result in enumerate(results, start=1):
            entry = aggregated.setdefault(
                result.chunk_id,
                {
                    "chunk_id": result.chunk_id,
                    "doc_id": result.doc_id,
                    "text": result.text,
                    "metadata": result.metadata.copy(),
                    "score": 0.0,
                    "sources": set(),
                    "source_ranks": {},
                },
            )
            entry["score"] += 1.0 / (k + rank)
            entry["sources"].add(source_name)
            entry["source_ranks"][source_name] = rank

    fused_results: list[RetrievalResult] = []
    for entry in aggregated.values():
        metadata = entry["metadata"].copy()
        metadata["rrf_sources"] = sorted(entry["sources"])
        metadata["rrf_source_ranks"] = dict(sorted(entry["source_ranks"].items()))

        fused_results.append(
            RetrievalResult(
                chunk_id=entry["chunk_id"],
                doc_id=entry["doc_id"],
                score=round(entry["score"], 8),
                text=entry["text"],
                metadata=metadata,
            )
        )

    fused_results.sort(key=lambda item: (-item.score, item.chunk_id))
    if top_k is not None:
        return fused_results[:top_k]
    return fused_results
