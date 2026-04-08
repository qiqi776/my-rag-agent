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
        candidate_limit = max(context_limit, self.settings.generation.candidate_results)
        trace = TraceContext(
            trace_type="answer",
            metadata={
                "query": query,
                "collection": collection or self.settings.ingestion.default_collection,
                "top_k": context_limit,
                "candidate_results": candidate_limit,
                "mode": mode or self.settings.retrieval.mode,
            },
        )

        retrieve_started = time.monotonic()
        search_output = self.search_service.search(
            query=query,
            collection=collection,
            top_k=candidate_limit,
            mode=mode,
        )
        trace.record_stage(
            "retrieve",
            {
                "result_count": search_output.result_count,
                "retrieval_mode": search_output.retrieval_mode,
                "candidate_results": candidate_limit,
            },
            elapsed_ms=(time.monotonic() - retrieve_started) * 1000.0,
        )

        rerank_started = time.monotonic()
        rerank_fallback = False
        rerank_error: str | None = None
        try:
            reranked_results = self.reranker.rerank(
                search_output.normalized_query,
                search_output.results,
                top_k=candidate_limit,
            )
        except Exception as exc:
            rerank_fallback = True
            rerank_error = str(exc)
            self.logger.warning(
                "Reranker '%s' failed, falling back to search order: %s",
                self.reranker.provider,
                exc,
            )
            reranked_results = search_output.results[:candidate_limit]
        supporting_results = reranked_results[:context_limit]
        trace.record_stage(
            "rerank",
            {
                "input_count": search_output.result_count,
                "candidate_result_count": len(reranked_results),
                "selected_result_count": len(supporting_results),
                "provider": self.reranker.provider,
                "fallback": rerank_fallback,
                "error": rerank_error,
            },
            elapsed_ms=(time.monotonic() - rerank_started) * 1000.0,
        )

        context_started = time.monotonic()
        contexts, context_char_count, truncated_contexts = self._assemble_contexts(supporting_results)
        trace.record_stage(
            "assemble_context",
            {
                "context_count": len(contexts),
                "char_count": context_char_count,
                "max_context_chars": self.settings.generation.max_context_chars,
                "truncated_contexts": truncated_contexts,
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

    def _assemble_contexts(self, supporting_results: list) -> tuple[list[str], int, int]:
        max_context_chars = self.settings.generation.max_context_chars
        contexts: list[str] = []
        used_chars = 0
        truncated_contexts = 0

        for result in supporting_results:
            normalized_text = " ".join(result.text.split()).strip()
            if not normalized_text:
                continue

            remaining_chars = max_context_chars - used_chars
            if remaining_chars <= 0:
                truncated_contexts += 1
                break

            if len(normalized_text) > remaining_chars:
                if not contexts:
                    snippet = normalized_text[:remaining_chars].rstrip()
                    if snippet:
                        contexts.append(snippet)
                        used_chars += len(snippet)
                truncated_contexts += 1
                break

            contexts.append(normalized_text)
            used_chars += len(normalized_text)

        return contexts, used_chars, truncated_contexts
