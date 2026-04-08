"""Structured query normalization and filter preparation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.core.types import Metadata, ProcessedQuery

_TOKEN_PATTERN = re.compile(r"\w+")


class QueryProcessor:
    """Normalize query text and shape retrieval filters for downstream services."""

    def __init__(self, default_collection: str) -> None:
        self.default_collection = default_collection

    def process(
        self,
        query: str,
        *,
        collection: str | None,
        top_k: int,
        filters: Metadata | None = None,
    ) -> ProcessedQuery:
        normalized_query = " ".join(query.split()).strip()
        normalized_filters = self._normalize_filters(filters)
        resolved_collection = self._resolve_collection(collection, normalized_filters)
        keywords = self._extract_keywords(normalized_query)
        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            collection=resolved_collection,
            top_k=top_k,
            keywords=keywords,
            filters=normalized_filters,
        )

    def _resolve_collection(self, collection: str | None, filters: Metadata) -> str:
        if collection is not None and collection.strip():
            return collection.strip()
        filtered_collection = filters.get("collection")
        if isinstance(filtered_collection, str) and filtered_collection.strip():
            return filtered_collection.strip()
        return self.default_collection

    def _normalize_filters(self, filters: Metadata | None) -> Metadata:
        if not filters:
            return {}

        normalized: Metadata = {}
        for key, value in filters.items():
            if key == "collection":
                if isinstance(value, str) and value.strip():
                    normalized["collection"] = value.strip()
                continue
            if key == "doc_type":
                if isinstance(value, str) and value.strip():
                    normalized["doc_type"] = value.strip().lower()
                continue
            if isinstance(value, bool | int | float):
                normalized[key] = value
                continue
            if isinstance(value, str) and value.strip():
                normalized[key] = value.strip()
                continue
            if isinstance(value, Iterable) and not isinstance(value, str):
                values = [
                    item.strip()
                    for item in value
                    if isinstance(item, str) and item.strip()
                ]
                if values:
                    normalized[key] = values
        return normalized

    def _extract_keywords(self, query: str) -> list[str]:
        seen: set[str] = set()
        keywords: list[str] = []
        for token in _TOKEN_PATTERN.findall(query.lower()):
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords
