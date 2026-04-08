from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.llm.base_llm import BaseLLM
from src.adapters.llm.factory import create_llm
from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_settings(path: Path, llm_block: str) -> None:
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
  dense_top_k: 3
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
{llm_block}
  reranker:
    provider: "fake"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class _StubLLM(BaseLLM):
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    @property
    def provider(self) -> str:
        return "stub"

    def generate_answer(self, query: str, contexts: list[str], max_chars: int) -> str:
        return "stub answer"


@pytest.mark.unit
def test_llm_factory_creates_openai_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(
        config_path,
        """
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "llm-key"
    base_url: "https://api.openai.com/v1"
    temperature: 0.3
""".rstrip(),
    )
    settings = load_settings(config_path)
    captured: dict[str, object] = {}

    def _build_stub(**kwargs: object) -> BaseLLM:
        captured.update(kwargs)
        return _StubLLM(**kwargs)

    monkeypatch.setattr("src.adapters.llm.factory.OpenAILLM", _build_stub)

    llm = create_llm(settings)

    assert isinstance(llm, _StubLLM)
    assert captured == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "llm-key",
        "temperature": 0.3,
        "base_url": "https://api.openai.com/v1",
        "azure_endpoint": None,
        "deployment_name": None,
        "api_version": None,
    }


@pytest.mark.unit
def test_llm_factory_rejects_real_provider_without_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(
        config_path,
        """
    provider: "openai"
    model: "gpt-4o-mini"
""".rstrip(),
    )

    with pytest.raises(ConfigError, match="api_key"):
        load_settings(config_path)
