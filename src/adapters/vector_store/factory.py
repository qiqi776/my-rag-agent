"""Factory for vector-store adapters."""

from __future__ import annotations

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.chroma_store import ChromaVectorStore
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.adapters.vector_store.local_json_store import LocalJsonVectorStore
from src.core.errors import ConfigError
from src.core.settings import Settings


def create_vector_store(settings: Settings) -> BaseVectorStore:
    """Instantiate the configured vector-store adapter."""

    provider = settings.adapters.vector_store.provider
    if provider == "memory":
        return InMemoryVectorStore()
    if provider in {"local_json", "json"}:
        return LocalJsonVectorStore(settings.adapters.vector_store.storage_path)
    if provider == "chroma":
        return ChromaVectorStore(settings.adapters.vector_store.storage_path)
    raise ConfigError(f"Unsupported vector_store provider: {provider}")
