"""Cross-encoder reranker with optional runtime dependency loading."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from src.adapters.reranker.base_reranker import BaseReranker
from src.core.errors import ConfigError
from src.response.response_builder import SearchResultItem


def _load_cross_encoder_class() -> type[Any]:
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ConfigError(
            "The 'sentence_transformers' package is required for the cross_encoder reranker."
        )

    module = importlib.import_module("sentence_transformers")
    cross_encoder = getattr(module, "CrossEncoder", None)
    if not isinstance(cross_encoder, type):
        raise ConfigError(
            "sentence_transformers.CrossEncoder is not available in the installed package"
        )
    return cross_encoder


class CrossEncoderReranker(BaseReranker):
    """Rerank results with a local cross-encoder model."""

    def __init__(self, *, model: str) -> None:
        cross_encoder_class = _load_cross_encoder_class()
        self._model = cross_encoder_class(model)
        self._model_name = model

    @property
    def provider(self) -> str:
        return "cross_encoder"

    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not results:
            return []

        pairs = [(query, item.text) for item in results]
        raw_scores = self._model.predict(pairs)
        scores = self._normalize_scores(raw_scores)
        if len(scores) != len(results):
            raise ValueError("Cross-encoder reranker returned an unexpected score payload")

        rescored: list[tuple[float, SearchResultItem]] = []
        for score, item in zip(scores, results, strict=True):
            if not isinstance(score, int | float):
                raise ValueError("Cross-encoder reranker scores must be numeric")
            metadata = item.metadata.copy()
            metadata["original_rank"] = item.rank
            metadata["rerank_score"] = float(score)
            rescored.append((float(score), replace(item, metadata=metadata)))

        rescored.sort(key=lambda pair: (-pair[0], -pair[1].score, pair[1].chunk_id))
        return [
            replace(item, rank=index)
            for index, (_, item) in enumerate(
                rescored[: top_k or len(rescored)],
                start=1,
            )
        ]

    def _normalize_scores(self, scores: Any) -> list[float]:
        if hasattr(scores, "tolist"):
            converted = scores.tolist()
            if isinstance(converted, list):
                return [float(score) for score in converted]

        if isinstance(scores, list):
            return [float(score) for score in scores]

        if isinstance(scores, Iterable) and not isinstance(scores, str | bytes):
            return [float(score) for score in scores]

        raise ValueError("Cross-encoder reranker returned an unexpected score payload")
