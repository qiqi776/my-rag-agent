"""Factory for embedding adapters."""

from __future__ import annotations

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.adapters.embedding.openai_embedding import OpenAIEmbedding
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_embedding(settings: Settings) -> BaseEmbedding:
    """Instantiate the configured embedding adapter."""

    provider = settings.adapters.embedding.provider
    if provider == "fake":
        return FakeEmbedding(settings.adapters.embedding.dimensions)
    if provider in {"openai", "azure"}:
        embedding_settings = settings.adapters.embedding
        if embedding_settings.model is None or embedding_settings.api_key is None:
            raise ConfigError(
                f"Embedding provider '{provider}' requires model and api_key configuration"
            )
        return OpenAIEmbedding(
            provider=provider,
            model=embedding_settings.model,
            dimensions=embedding_settings.dimensions,
            api_key=embedding_settings.api_key,
            base_url=embedding_settings.base_url,
            azure_endpoint=embedding_settings.azure_endpoint,
            deployment_name=embedding_settings.deployment_name,
            api_version=embedding_settings.api_version,
        )
    raise ConfigError(f"Unsupported embedding provider: {provider}")
