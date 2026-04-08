"""Factory for reranker adapters."""

from __future__ import annotations

from src.adapters.reranker.base_reranker import BaseReranker
from src.adapters.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.adapters.reranker.fake_reranker import FakeReranker
from src.adapters.reranker.llm_reranker import LLMReranker
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_reranker(settings: Settings) -> BaseReranker:
    """Instantiate the configured reranker adapter."""

    provider = settings.adapters.reranker.provider
    if provider == "fake":
        return FakeReranker()
    if provider == "llm":
        reranker_settings = settings.adapters.reranker
        if reranker_settings.model is None or reranker_settings.api_key is None:
            raise ConfigError("LLM reranker requires model and api_key configuration")
        return LLMReranker(
            model=reranker_settings.model,
            api_key=reranker_settings.api_key,
            base_url=reranker_settings.base_url,
            azure_endpoint=reranker_settings.azure_endpoint,
            deployment_name=reranker_settings.deployment_name,
            api_version=reranker_settings.api_version,
        )
    if provider == "cross_encoder":
        reranker_settings = settings.adapters.reranker
        if reranker_settings.model is None:
            raise ConfigError("Cross-encoder reranker requires model configuration")
        return CrossEncoderReranker(model=reranker_settings.model)
    raise ConfigError(f"Unsupported reranker provider: {provider}")
