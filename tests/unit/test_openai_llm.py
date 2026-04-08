from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.llm.openai_llm import OpenAILLM
from src.core.errors import ConfigError


class _FakeChatCompletionsAPI:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                )
            ]
        )


class _FakeOpenAIClient:
    init_kwargs: list[dict[str, object]] = []
    next_content = "answer"

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs.append(kwargs)
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsAPI(type(self).next_content))


class _FakeAzureOpenAIClient(_FakeOpenAIClient):
    pass


@pytest.mark.unit
def test_openai_llm_builds_chat_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "src.adapters.llm.openai_llm.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.init_kwargs.clear()
    _FakeOpenAIClient.next_content = "Grounded answer from context."

    llm = OpenAILLM(
        provider="openai",
        model="gpt-4o-mini",
        api_key="llm-key",
        base_url="https://api.openai.com/v1",
        temperature=0.2,
    )

    answer = llm.generate_answer(
        "What is virtual memory?",
        ["Virtual memory gives each process an address space."],
        max_chars=120,
    )

    assert answer == "Grounded answer from context."
    assert _FakeOpenAIClient.init_kwargs == [
        {"api_key": "llm-key", "base_url": "https://api.openai.com/v1"}
    ]
    assert llm._client.chat.completions.calls[0]["model"] == "gpt-4o-mini"  # pyright: ignore[reportPrivateUsage]
    assert llm._client.chat.completions.calls[0]["temperature"] == pytest.approx(0.2)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
def test_openai_llm_uses_azure_client_and_deployment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "src.adapters.llm.openai_llm.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeAzureOpenAIClient.init_kwargs.clear()
    _FakeAzureOpenAIClient.next_content = "Azure answer."

    llm = OpenAILLM(
        provider="azure",
        model="gpt-4o",
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        deployment_name="azure-deployment",
        api_version="2024-02-15-preview",
    )

    answer = llm.generate_answer("Explain paging.", ["Paging splits memory into pages."], 80)

    assert answer == "Azure answer."
    assert _FakeAzureOpenAIClient.init_kwargs == [
        {
            "api_key": "azure-key",
            "azure_endpoint": "https://example.openai.azure.com/",
            "api_version": "2024-02-15-preview",
        }
    ]
    assert llm._client.chat.completions.calls[0]["model"] == "azure-deployment"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
def test_openai_llm_handles_missing_context_without_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "src.adapters.llm.openai_llm.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.init_kwargs.clear()

    llm = OpenAILLM(
        provider="openai",
        model="gpt-4o-mini",
        api_key="llm-key",
    )

    answer = llm.generate_answer("Explain paging.", [], 80)

    assert "No supporting context available" in answer
    assert llm._client.chat.completions.calls == []  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
def test_openai_llm_requires_openai_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: None)

    with pytest.raises(ConfigError, match="openai"):
        OpenAILLM(provider="openai", model="gpt-4o-mini", api_key="llm-key")


@pytest.mark.unit
def test_openai_llm_rejects_missing_text_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.adapters.llm.openai_llm.importlib.util.find_spec", lambda name: object())

    class _NoContentCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    class _NoContentClient:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=_NoContentCompletions())

    monkeypatch.setattr(
        "src.adapters.llm.openai_llm.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_NoContentClient, AzureOpenAI=_NoContentClient),
    )

    llm = OpenAILLM(provider="openai", model="gpt-4o-mini", api_key="llm-key")

    with pytest.raises(ValueError, match="text content"):
        llm.generate_answer("Explain paging.", ["Paging splits memory into pages."], 80)
