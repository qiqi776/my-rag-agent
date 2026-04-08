"""OpenAI-compatible LLM adapter."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from src.adapters.llm.base_llm import BaseLLM
from src.core.errors import ConfigError


def _load_openai_client_class(class_name: str) -> type[Any]:
    if importlib.util.find_spec("openai") is None:
        raise ConfigError(
            "The 'openai' package is required for real LLM providers. "
            "Install project dependencies with the runtime extras you intend to use."
        )

    module = importlib.import_module("openai")
    client_class = getattr(module, class_name, None)
    if not isinstance(client_class, type):
        raise ConfigError(f"openai.{class_name} is not available in the installed package")
    return client_class


class OpenAILLM(BaseLLM):
    """LLM adapter for OpenAI-compatible and Azure OpenAI chat completions."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._deployment_name = deployment_name
        self._temperature = temperature

        if provider == "openai":
            client_class = _load_openai_client_class("OpenAI")
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            self._client = client_class(**client_kwargs)
            return

        if provider == "azure":
            client_class = _load_openai_client_class("AzureOpenAI")
            self._client = client_class(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
            return

        raise ConfigError(f"Unsupported llm provider: {provider}")

    @property
    def provider(self) -> str:
        return self._provider

    def generate_answer(
        self,
        query: str,
        contexts: list[str],
        max_chars: int,
    ) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")

        cleaned_contexts = [text.strip() for text in contexts if text.strip()]
        if not cleaned_contexts:
            return f"No supporting context available for '{query.strip()}'."[:max_chars]

        prompt = self._build_prompt(query, cleaned_contexts)
        response = self._client.chat.completions.create(
            model=self._deployment_name or self._model,
            temperature=self._temperature,
            messages=prompt,
        )

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response did not include any choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not include text content")

        return content.strip()[:max_chars].rstrip()

    def _build_prompt(self, query: str, contexts: list[str]) -> list[dict[str, str]]:
        context_block = "\n\n".join(
            f"[{index}] {context[:1500]}" for index, context in enumerate(contexts, start=1)
        )
        return [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided context. "
                    "If the context is insufficient, say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query.strip()}\n\n"
                    f"Context:\n{context_block}\n\n"
                    "Write a concise answer grounded in the context."
                ),
            },
        ]
