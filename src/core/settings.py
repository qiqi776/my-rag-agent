"""Configuration loading for the minimal modular RAG project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml.example"


def resolve_path(path: str | Path) -> Path:
    """Resolve paths relative to the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _require_mapping(data: dict[str, Any], key: str, parent: str = "settings") -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid mapping: {parent}.{key}")
    return value


def _require_str(data: dict[str, Any], key: str, parent: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid string: {parent}.{key}")
    return value


def _require_int(data: dict[str, Any], key: str, parent: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"Missing or invalid integer: {parent}.{key}")
    return value


def _require_bool(data: dict[str, Any], key: str, parent: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Missing or invalid boolean: {parent}.{key}")
    return value


def _require_list(data: dict[str, Any], key: str, parent: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"Missing or invalid list: {parent}.{key}")
    return value


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    name: str
    environment: str


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    default_collection: str
    chunk_size: int
    chunk_overlap: int
    supported_extensions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    dense_top_k: int


@dataclass(frozen=True, slots=True)
class LoaderAdapterSettings:
    provider: str


@dataclass(frozen=True, slots=True)
class EmbeddingAdapterSettings:
    provider: str
    dimensions: int = 16


@dataclass(frozen=True, slots=True)
class VectorStoreAdapterSettings:
    provider: str
    storage_path: str = "./data/db/vector_store.json"


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    loader: LoaderAdapterSettings
    embedding: EmbeddingAdapterSettings
    vector_store: VectorStoreAdapterSettings


@dataclass(frozen=True, slots=True)
class ObservabilitySettings:
    trace_enabled: bool
    trace_file: str
    log_level: str


@dataclass(frozen=True, slots=True)
class Settings:
    project: ProjectSettings
    ingestion: IngestionSettings
    retrieval: RetrievalSettings
    adapters: AdapterSettings
    observability: ObservabilitySettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        if not isinstance(data, dict):
            raise ConfigError("Settings root must be a mapping")

        project = _require_mapping(data, "project")
        ingestion = _require_mapping(data, "ingestion")
        retrieval = _require_mapping(data, "retrieval")
        adapters = _require_mapping(data, "adapters")
        observability = _require_mapping(data, "observability")

        chunk_size = _require_int(ingestion, "chunk_size", "ingestion")
        chunk_overlap = _require_int(ingestion, "chunk_overlap", "ingestion")
        if chunk_size <= 0:
            raise ConfigError("ingestion.chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigError("ingestion.chunk_overlap must be >= 0 and < chunk_size")

        loader = _require_mapping(adapters, "loader", "adapters")
        embedding = _require_mapping(adapters, "embedding", "adapters")
        vector_store = _require_mapping(adapters, "vector_store", "adapters")

        return cls(
            project=ProjectSettings(
                name=_require_str(project, "name", "project"),
                environment=_require_str(project, "environment", "project"),
            ),
            ingestion=IngestionSettings(
                default_collection=_require_str(
                    ingestion, "default_collection", "ingestion"
                ),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                supported_extensions=_require_list(
                    ingestion, "supported_extensions", "ingestion"
                ),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval")
            ),
            adapters=AdapterSettings(
                loader=LoaderAdapterSettings(
                    provider=_require_str(loader, "provider", "adapters.loader")
                ),
                embedding=EmbeddingAdapterSettings(
                    provider=_require_str(embedding, "provider", "adapters.embedding"),
                    dimensions=int(embedding.get("dimensions", 16)),
                ),
                vector_store=VectorStoreAdapterSettings(
                    provider=_require_str(
                        vector_store, "provider", "adapters.vector_store"
                    ),
                    storage_path=str(
                        vector_store.get("storage_path", "./data/db/vector_store.json")
                    ),
                ),
            ),
            observability=ObservabilitySettings(
                trace_enabled=_require_bool(
                    observability, "trace_enabled", "observability"
                ),
                trace_file=_require_str(observability, "trace_file", "observability"),
                log_level=_require_str(observability, "log_level", "observability"),
            ),
        )


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from YAML."""

    settings_path = resolve_path(path or DEFAULT_SETTINGS_PATH)
    if not settings_path.exists():
        raise ConfigError(
            f"Configuration file not found: {settings_path}. "
            "Provide an explicit config path or create a local settings file."
        )

    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return Settings.from_dict(raw)
