"""Answer generation application service for M6."""

from __future__ import annotations

import time

from src.adapters.llm.base_llm import BaseLLM
from src.adapters.reranker.base_reranker import BaseReranker
from src.application.search_service import SearchService
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.observability.logger import get_logger
from src.observability.trace_store import TraceStore
from src.response.answer_builder import AnswerBuilder, AnswerOutput


class AnswerService:
    """Coordinate retrieval, reranking, answer synthesis, and answer traces."""

    def __init__(
        self,
        settings: Settings,
        search_service: SearchService,
        reranker: BaseReranker,
        llm: BaseLLM,
        answer_builder: AnswerBuilder | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.settings = settings
        self.search_service = search_service
        self.reranker = reranker
        self.llm = llm
        self.answer_builder = answer_builder or AnswerBuilder()
        self.trace_store = trace_store
        self.logger = get_logger("minimal-rag.answer", settings.observability.log_level)

    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput:
        """Execute retrieve -> rerank -> build answer."""

        context_limit = top_k or self.settings.generation.max_context_results
        trace = TraceContext(
            trace_type="answer",
            metadata={
                "query": query,
                "collection": collection or self.settings.ingestion.default_collection,
                "top_k": context_limit,
                "mode": mode or self.settings.retrieval.mode,
            },
        )

        retrieve_started = time.monotonic()
        search_output = self.search_service.search(
            query=query,
            collection=collection,
            top_k=max(context_limit, self.settings.generation.max_context_results),
            mode=mode,
        )
        trace.record_stage(
            "retrieve",
            {
                "result_count": search_output.result_count,
                "retrieval_mode": search_output.retrieval_mode,
            },
            elapsed_ms=(time.monotonic() - retrieve_started) * 1000.0,
        )

        rerank_started = time.monotonic()
        supporting_results = self.reranker.rerank(
            search_output.normalized_query,
            search_output.results,
            top_k=context_limit,
        )
        trace.record_stage(
            "rerank",
            {
                "input_count": search_output.result_count,
                "result_count": len(supporting_results),
                "provider": self.reranker.provider,
            },
            elapsed_ms=(time.monotonic() - rerank_started) * 1000.0,
        )

        context_started = time.monotonic()
        contexts = [result.text for result in supporting_results]
        trace.record_stage(
            "assemble_context",
            {
                "context_count": len(contexts),
                "char_count": sum(len(text) for text in contexts),
            },
            elapsed_ms=(time.monotonic() - context_started) * 1000.0,
        )

        generate_started = time.monotonic()
        answer = self.llm.generate_answer(
            search_output.normalized_query,
            contexts,
            max_chars=self.settings.generation.max_answer_chars,
        )
        trace.record_stage(
            "generate_answer",
            {
                "provider": self.llm.provider,
                "answer_chars": len(answer),
            },
            elapsed_ms=(time.monotonic() - generate_started) * 1000.0,
        )

        trace.finish()
        if self.trace_store is not None:
            self.trace_store.append(trace)

        return self.answer_builder.build(search_output, supporting_results, answer)
