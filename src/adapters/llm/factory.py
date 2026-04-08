"""Factory for LLM adapters."""

from __future__ import annotations

from src.adapters.llm.base_llm import BaseLLM
from src.adapters.llm.fake_llm import FakeLLM
from src.adapters.llm.openai_llm import OpenAILLM
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_llm(settings: Settings) -> BaseLLM:
    """Instantiate the configured LLM adapter."""

    provider = settings.adapters.llm.provider
    if provider == "fake":
        return FakeLLM()
    if provider in {"openai", "azure"}:
        llm_settings = settings.adapters.llm
        if llm_settings.model is None or llm_settings.api_key is None:
            raise ConfigError(f"LLM provider '{provider}' requires model and api_key configuration")
        return OpenAILLM(
            provider=provider,
            model=llm_settings.model,
            api_key=llm_settings.api_key,
            temperature=llm_settings.temperature,
            base_url=llm_settings.base_url,
            azure_endpoint=llm_settings.azure_endpoint,
            deployment_name=llm_settings.deployment_name,
            api_version=llm_settings.api_version,
        )
    raise ConfigError(f"Unsupported llm provider: {provider}")
