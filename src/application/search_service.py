"""Search application service for dense-only and hybrid retrieval."""

from __future__ import annotations

import time

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.errors import EmptyQueryError, UnsupportedRetrievalModeError
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.core.types import ProcessedQuery
from src.observability.logger import get_logger
from src.observability.trace_store import TraceStore
from src.response.response_builder import ResponseBuilder, SearchOutput
from src.retrieval.fusion import rrf_fuse
from src.retrieval.sparse_retriever import SparseRetriever


class SearchService:
    """Coordinate query normalization, retrieval, fusion, and trace output."""

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        sparse_retriever: SparseRetriever | None = None,
        response_builder: ResponseBuilder | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding = embedding
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.response_builder = response_builder or ResponseBuilder()
        self.trace_store = trace_store
        self.logger = get_logger("minimal-rag.search", settings.observability.log_level)

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput:
        """Execute dense-only or hybrid retrieval."""

        normalized = query.strip()
        if not normalized:
            raise EmptyQueryError("Query cannot be empty")

        retrieval_mode = (mode or self.settings.retrieval.mode).strip().lower()
        if retrieval_mode not in {"dense", "hybrid"}:
            raise UnsupportedRetrievalModeError(
                f"Unsupported retrieval mode: {retrieval_mode}"
            )

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
                "mode": retrieval_mode,
            },
        )

        embed_started = time.monotonic()
        query_vector = self.embedding.embed_text(processed.normalized_query)
        trace.record_stage(
            "embed_query",
            {"dimensions": self.embedding.dimensions},
            elapsed_ms=(time.monotonic() - embed_started) * 1000.0,
        )

        dense_started = time.monotonic()
        dense_candidate_k = max(processed.top_k, self.settings.retrieval.dense_top_k)
        dense_results = self.vector_store.query(
            processed.collection,
            query_vector=query_vector,
            top_k=dense_candidate_k,
        )
        final_results = dense_results[: processed.top_k]
        trace.record_stage(
            "dense_retrieve",
            {
                "candidate_result_count": len(dense_results),
                "final_result_count": len(final_results) if retrieval_mode == "dense" else None,
                "candidate_top_k": dense_candidate_k,
            },
            elapsed_ms=(time.monotonic() - dense_started) * 1000.0,
        )

        if retrieval_mode == "hybrid":
            if self.sparse_retriever is None:
                raise UnsupportedRetrievalModeError(
                    "Hybrid retrieval requested but sparse retriever is not configured"
                )

            sparse_started = time.monotonic()
            sparse_candidate_k = max(processed.top_k, self.settings.retrieval.sparse_top_k)
            sparse_results = self.sparse_retriever.retrieve(
                processed.collection,
                processed.normalized_query,
                top_k=sparse_candidate_k,
            )
            trace.record_stage(
                "sparse_retrieve",
                {
                    "candidate_result_count": len(sparse_results),
                    "candidate_top_k": sparse_candidate_k,
                },
                elapsed_ms=(time.monotonic() - sparse_started) * 1000.0,
            )

            if sparse_results:
                fuse_started = time.monotonic()
                final_results = rrf_fuse(
                    [
                        ("dense", dense_results),
                        ("sparse", sparse_results),
                    ],
                    k=self.settings.retrieval.rrf_k,
                    top_k=processed.top_k,
                )
                trace.record_stage(
                    "rrf_fuse",
                    {
                        "final_result_count": len(final_results),
                        "rrf_k": self.settings.retrieval.rrf_k,
                    },
                    elapsed_ms=(time.monotonic() - fuse_started) * 1000.0,
                )
            else:
                trace.record_stage(
                    "rrf_fuse",
                    {
                        "final_result_count": len(final_results),
                        "rrf_k": self.settings.retrieval.rrf_k,
                        "skipped": True,
                    },
                    elapsed_ms=0.0,
                )

        trace.finish()
        if self.trace_store is not None:
            self.trace_store.append(trace)

        return self.response_builder.build(
            query=processed,
            results=final_results,
            retrieval_mode=retrieval_mode,
        )
