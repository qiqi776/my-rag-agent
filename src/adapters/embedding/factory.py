"""Factory for embedding adapters."""

from __future__ import annotations

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_embedding(settings: Settings) -> BaseEmbedding:
    """Instantiate the configured embedding adapter."""

    provider = settings.adapters.embedding.provider
    if provider == "fake":
        return FakeEmbedding(settings.adapters.embedding.dimensions)
    raise ConfigError(f"Unsupported embedding provider: {provider}")
