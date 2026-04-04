"""Factory for LLM adapters."""

from __future__ import annotations

from src.adapters.llm.base_llm import BaseLLM
from src.adapters.llm.fake_llm import FakeLLM
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_llm(settings: Settings) -> BaseLLM:
    """Instantiate the configured LLM adapter."""

    provider = settings.adapters.llm.provider
    if provider == "fake":
        return FakeLLM()
    raise ConfigError(f"Unsupported llm provider: {provider}")
