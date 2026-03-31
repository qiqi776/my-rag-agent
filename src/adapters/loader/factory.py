"""Factory for loader adapters."""

from __future__ import annotations

from src.adapters.loader.base_loader import BaseLoader
from src.adapters.loader.text_loader import TextLoader
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_loader(settings: Settings) -> BaseLoader:
    """Instantiate the configured loader adapter."""

    provider = settings.adapters.loader.provider
    if provider == "text":
        return TextLoader(settings.ingestion.supported_extensions)
    raise ConfigError(f"Unsupported loader provider: {provider}")
