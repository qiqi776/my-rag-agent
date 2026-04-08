"""OpenAI-compatible embedding adapter."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.core.errors import ConfigError


def _load_openai_client_class(class_name: str) -> type[Any]:
    if importlib.util.find_spec("openai") is None:
        raise ConfigError(
            "The 'openai' package is required for real embedding providers. "
            "Install project dependencies with the runtime extras you intend to use."
        )

    module = importlib.import_module("openai")
    client_class = getattr(module, class_name, None)
    if not isinstance(client_class, type):
        raise ConfigError(f"openai.{class_name} is not available in the installed package")
    return client_class


class OpenAIEmbedding(BaseEmbedding):
    """Embedding adapter for OpenAI-compatible and Azure OpenAI endpoints."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dimensions: int,
        api_key: str,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")

        self._provider = provider
        self._model = model
        self._deployment_name = deployment_name
        self._dimensions = dimensions

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

        raise ConfigError(f"Unsupported embedding provider: {provider}")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = [text if text.strip() else " " for text in texts]
        response = self._client.embeddings.create(
            model=self._deployment_name or self._model,
            input=payload,
        )

        vectors: list[list[float]] = []
        for item in response.data:
            embedding = getattr(item, "embedding", None)
            if not isinstance(embedding, list) or not all(
                isinstance(value, int | float) for value in embedding
            ):
                raise ValueError("Embedding response item is missing a numeric embedding vector")
            vectors.append([float(value) for value in embedding])

        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding response size mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        return vectors
