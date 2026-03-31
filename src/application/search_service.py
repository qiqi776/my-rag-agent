"""Search application service for the M1 MVP."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.errors import EmptyQueryError
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.core.types import ProcessedQuery, RetrievalResult
from src.observability.logger import get_logger
from src.observability.trace_store import TraceStore


@dataclass(slots=True)
class SearchResponse:
    """Structured search response for the MVP."""

    query: ProcessedQuery
    results: list[RetrievalResult]


class SearchService:
    """Coordinate query normalization, embedding, retrieval, and trace output."""

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding = embedding
        self.vector_store = vector_store
        self.trace_store = trace_store
        self.logger = get_logger("minimal-rag.search", settings.observability.log_level)

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> SearchResponse:
        """Execute a dense-only query."""

        normalized = query.strip()
        if not normalized:
            raise EmptyQueryError("Query cannot be empty")

        processed = ProcessedQuery(
            original_query=query,
            normalized_query=normalized,
            collection=collection or self.settings.ingestion.default_collection,
            top_k=top_k or self.settings.retrieval.dense_top_k,
        )
        trace = TraceContext(
            trace_type="query",
            metadata={
                "query": processed.original_query,
                "collection": processed.collection,
                "top_k": processed.top_k,
            },
        )

        embed_started = time.monotonic()
        query_vector = self.embedding.embed_text(processed.normalized_query)
        trace.record_stage(
            "embed_query",
            {"dimensions": self.embedding.dimensions},
            elapsed_ms=(time.monotonic() - embed_started) * 1000.0,
        )

        retrieve_started = time.monotonic()
        results = self.vector_store.query(
            processed.collection,
            query_vector=query_vector,
            top_k=processed.top_k,
        )
        trace.record_stage(
            "dense_retrieve",
            {"result_count": len(results)},
            elapsed_ms=(time.monotonic() - retrieve_started) * 1000.0,
        )

        trace.finish()
        if self.trace_store is not None:
            self.trace_store.append(trace)

        return SearchResponse(query=processed, results=results)
