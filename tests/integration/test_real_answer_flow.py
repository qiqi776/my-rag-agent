from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.adapters.llm.factory import create_llm
from src.adapters.reranker.base_reranker import BaseReranker
from src.application.answer_service import AnswerService
from src.core.settings import load_settings
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


def _write_settings(path: Path) -> None:
    path.write_text(
        """
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
  sparse_top_k: 3
  rrf_k: 60
generation:
  max_context_results: 2
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
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "llm-key"
    base_url: "https://api.openai.com/v1"
    temperature: 0.0
  reranker:
    provider: "fake"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
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
    ) -> SearchOutput:
        return SearchOutput(
            query=query,
            normalized_query=query.strip(),
            collection=collection or "default",
            retrieval_mode=mode or "dense",
            result_count=1,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="Virtual memory gives each process the illusion of a large private address space.",
                    source_path="/tmp/ostep.pdf",
                    collection=collection or "default",
                    chunk_index=0,
                    metadata={},
                )
            ],
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/ostep.pdf",
                    collection=collection or "default",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


class _StubReranker(BaseReranker):
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


@pytest.mark.integration
def test_real_answer_flow_uses_configured_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)
    settings = load_settings(config_path)

    class _FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Virtual memory gives each process an isolated address space backed by physical memory and disk."
                        )
                    )
                ]
            )

    class _FakeOpenAIClient:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "src.adapters.llm.openai_llm.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeOpenAIClient),
    )

    llm = create_llm(settings)
    service = AnswerService(
        settings=settings,
        search_service=_StubSearchService(),
        reranker=_StubReranker(),
        llm=llm,
    )

    output = service.answer("What is virtual memory?", collection="knowledge", top_k=1)

    assert "isolated address space" in output.answer
    assert output.citations[0].source_path == "/tmp/ostep.pdf"
