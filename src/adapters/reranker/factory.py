"""Factory for reranker adapters."""

from __future__ import annotations

from src.adapters.reranker.base_reranker import BaseReranker
from src.adapters.reranker.fake_reranker import FakeReranker
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_reranker(settings: Settings) -> BaseReranker:
    """Instantiate the configured reranker adapter."""

    provider = settings.adapters.reranker.provider
    if provider == "fake":
        return FakeReranker()
    raise ConfigError(f"Unsupported reranker provider: {provider}")
