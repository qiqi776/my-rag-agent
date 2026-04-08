from __future__ import annotations

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
    - ".md"
retrieval:
  mode: "dense"
  dense_top_k: 3
  sparse_top_k: 3
  rrf_k: 60
generation:
  max_context_results: 2
  candidate_results: 4
  max_context_chars: 60
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
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class StubSearchService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput:
        self.calls.append(
            {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "mode": mode,
            }
        )
        return SearchOutput(
            query=query,
            normalized_query=query.strip(),
            collection=collection or "default",
            retrieval_mode=mode or "dense",
            result_count=3,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="semantic embeddings support retrieval with long supporting context",
                    source_path="/tmp/python.txt",
                    collection=collection or "default",
                    chunk_index=0,
                    metadata={},
                ),
                SearchResultItem(
                    rank=2,
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    score=0.8,
                    text="semantic search combines lexical matching and vector similarity",
                    source_path="/tmp/retrieval.txt",
                    collection=collection or "default",
                    chunk_index=1,
                    metadata={},
                ),
                SearchResultItem(
                    rank=3,
                    chunk_id="chunk-3",
                    doc_id="doc-3",
                    score=0.7,
                    text="extra context that should not be selected once top_k is constrained",
                    source_path="/tmp/extra.txt",
                    collection=collection or "default",
                    chunk_index=2,
                    metadata={},
                ),
            ],
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/python.txt",
                    collection=collection or "default",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


class StubReranker(BaseReranker):
    @property
    def provider(self) -> str:
        return "stub"

    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        return results[: top_k or len(results)]


class StubLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def provider(self) -> str:
        return "stub"

    def generate_answer(self, query: str, contexts: list[str], max_chars: int) -> str:
        self.calls.append(
            {
                "query": query,
                "contexts": contexts,
                "max_chars": max_chars,
            }
        )
        return f"answer for {query}"[:max_chars]


@pytest.mark.unit
def test_answer_service_builds_answer_output_and_trace(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, trace_path)
    settings = load_settings(config_path)

    search_service = StubSearchService()
    llm = StubLLM()
    service = AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=StubReranker(),
        llm=llm,
        trace_store=TraceStore(trace_path),
    )

    output = service.answer("semantic embeddings", collection="knowledge", top_k=1)

    assert output.answer == "answer for semantic embeddings"
    assert output.citations[0].collection == "knowledge"
    assert search_service.calls == [
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "top_k": 4,
            "mode": None,
        }
    ]
    assert len(output.supporting_results) == 1
    assert llm.calls
    assert len(llm.calls[0]["contexts"]) == 1
    assert len(llm.calls[0]["contexts"][0]) <= settings.generation.max_context_chars
    trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert trace_lines
    assert '"trace_type": "answer"' in trace_lines[-1]
