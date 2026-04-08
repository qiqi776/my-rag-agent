from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.llm.base_llm import BaseLLM
from src.adapters.reranker.base_reranker import BaseReranker
from src.application.answer_service import AnswerService
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


def _write_settings(path: Path, trace_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
retrieval:
  mode: "dense"
  dense_top_k: 3
generation:
  max_context_results: 2
  candidate_results: 3
  max_context_chars: 200
  max_answer_chars: 200
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class _StubSearchService:
    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict[str, object] | None = None,
    ) -> SearchOutput:
        del filters
        results = [
            SearchResultItem(
                rank=1,
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.9,
                text="Virtual memory gives each process an isolated address space.",
                source_path="/tmp/ostep.txt",
                collection=collection or "default",
                chunk_index=0,
                metadata={},
            )
        ]
        return SearchOutput(
            query=query,
            normalized_query=query.strip(),
            collection=collection or "default",
            retrieval_mode=mode or "dense",
            result_count=len(results),
            results=results,
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/ostep.txt",
                    collection=collection or "default",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


class _ExplodingReranker(BaseReranker):
    @property
    def provider(self) -> str:
        return "exploding"

    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        del query, results, top_k
        raise RuntimeError("reranker failed")


class _StubLLM(BaseLLM):
    @property
    def provider(self) -> str:
        return "stub"

    def generate_answer(self, query: str, contexts: list[str], max_chars: int) -> str:
        return f"Answer from fallback for {query}"[:max_chars]


@pytest.mark.integration
def test_answer_service_falls_back_when_reranker_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, trace_path)
    settings = load_settings(config_path)

    service = AnswerService(
        settings=settings,
        search_service=_StubSearchService(),
        reranker=_ExplodingReranker(),
        llm=_StubLLM(),
        trace_store=TraceStore(trace_path),
    )

    output = service.answer("What is virtual memory?", collection="knowledge", top_k=1)

    assert "fallback" in output.answer
    trace = json.loads(trace_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    rerank_stage = next(stage for stage in trace["stages"] if stage["stage"] == "rerank")
    assert rerank_stage["data"]["fallback"] is True
    assert rerank_stage["data"]["provider"] == "exploding"
