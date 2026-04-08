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


def _require_mapping(
    data: dict[str, Any],
    key: str,
    parent: str = "settings",
) -> dict[str, Any]:
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


def _optional_mapping(data: dict[str, Any], key: str, parent: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid mapping: {parent}.{key}")
    return value


def _optional_bool(data: dict[str, Any], key: str, parent: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Missing or invalid boolean: {parent}.{key}")
    return value


def _optional_str(data: dict[str, Any], key: str, parent: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid string: {parent}.{key}")
    return value.strip()


def _optional_number(data: dict[str, Any], key: str, parent: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ConfigError(f"Missing or invalid number: {parent}.{key}")
    return float(value)


def _optional_positive_int(
    data: dict[str, Any],
    key: str,
    parent: str,
    default: int,
) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Missing or invalid integer: {parent}.{key}")
    return value


def _validate_real_provider_settings(
    adapter_path: str,
    provider: str,
    *,
    model: str | None,
    api_key: str | None,
    azure_endpoint: str | None,
    deployment_name: str | None,
    api_version: str | None,
) -> None:
    if provider == "fake":
        return

    if model is None:
        raise ConfigError(f"{adapter_path}.model is required for provider '{provider}'")

    if provider == "openai" and api_key is None:
        raise ConfigError(f"{adapter_path}.api_key is required for provider 'openai'")

    if provider == "azure":
        if api_key is None:
            raise ConfigError(f"{adapter_path}.api_key is required for provider 'azure'")
        if azure_endpoint is None:
            raise ConfigError(
                f"{adapter_path}.azure_endpoint is required for provider 'azure'"
            )
        if deployment_name is None:
            raise ConfigError(
                f"{adapter_path}.deployment_name is required for provider 'azure'"
            )
        if api_version is None:
            raise ConfigError(f"{adapter_path}.api_version is required for provider 'azure'")


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
    transforms: IngestionTransformsSettings = field(default_factory=lambda: IngestionTransformsSettings())


DEFAULT_TRANSFORM_ORDER = [
    "metadata_enrichment",
    "chunk_refinement",
    "image_captioning",
]


@dataclass(frozen=True, slots=True)
class MetadataEnrichmentSettings:
    enabled: bool = True
    section_title_max_length: int = 120


@dataclass(frozen=True, slots=True)
class ChunkRefinementSettings:
    enabled: bool = True
    collapse_whitespace: bool = True


@dataclass(frozen=True, slots=True)
class ImageCaptioningSettings:
    enabled: bool = False
    stub_caption: str = "image captioning not configured"
    append_to_text: bool = False


@dataclass(frozen=True, slots=True)
class IngestionTransformsSettings:
    enabled: bool = False
    order: list[str] = field(default_factory=lambda: DEFAULT_TRANSFORM_ORDER.copy())
    metadata_enrichment: MetadataEnrichmentSettings = field(
        default_factory=lambda: MetadataEnrichmentSettings()
    )
    chunk_refinement: ChunkRefinementSettings = field(
        default_factory=lambda: ChunkRefinementSettings()
    )
    image_captioning: ImageCaptioningSettings = field(
        default_factory=lambda: ImageCaptioningSettings()
    )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    dense_top_k: int
    mode: str = "dense"
    sparse_top_k: int = 5
    dense_candidate_multiplier: int = 3
    sparse_candidate_multiplier: int = 3
    max_candidate_top_k: int = 12
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class LoaderAdapterSettings:
    provider: str


@dataclass(frozen=True, slots=True)
class EmbeddingAdapterSettings:
    provider: str
    dimensions: int = 16
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None


@dataclass(frozen=True, slots=True)
class VectorStoreAdapterSettings:
    provider: str
    storage_path: str = "./data/db/vector_store.json"


@dataclass(frozen=True, slots=True)
class LLMAdapterSettings:
    provider: str = "fake"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class RerankerAdapterSettings:
    provider: str = "fake"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    loader: LoaderAdapterSettings
    embedding: EmbeddingAdapterSettings
    vector_store: VectorStoreAdapterSettings
    llm: LLMAdapterSettings
    reranker: RerankerAdapterSettings


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_context_results: int = 3
    candidate_results: int = 6
    max_context_chars: int = 1200
    max_answer_chars: int = 400


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
    generation: GenerationSettings
    observability: ObservabilitySettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        if not isinstance(data, dict):
            raise ConfigError("Settings root must be a mapping")

        project = _require_mapping(data, "project")
        ingestion = _require_mapping(data, "ingestion")
        retrieval = _require_mapping(data, "retrieval")
        adapters = _require_mapping(data, "adapters")
        generation = data.get("generation", {})
        if not isinstance(generation, dict):
            raise ConfigError("generation must be a mapping")
        observability = _require_mapping(data, "observability")

        chunk_size = _require_int(ingestion, "chunk_size", "ingestion")
        chunk_overlap = _require_int(ingestion, "chunk_overlap", "ingestion")
        if chunk_size <= 0:
            raise ConfigError("ingestion.chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigError("ingestion.chunk_overlap must be >= 0 and < chunk_size")

        transforms = _optional_mapping(ingestion, "transforms", "ingestion")
        transform_order_raw = transforms.get("order", DEFAULT_TRANSFORM_ORDER)
        if not isinstance(transform_order_raw, list) or not all(
            isinstance(item, str) and item.strip() for item in transform_order_raw
        ):
            raise ConfigError("ingestion.transforms.order must be a list of transform names")
        transform_order = [item.strip().lower() for item in transform_order_raw]
        supported_transforms = set(DEFAULT_TRANSFORM_ORDER)
        unknown_transforms = sorted(set(transform_order) - supported_transforms)
        if unknown_transforms:
            raise ConfigError(
                "ingestion.transforms.order contains unsupported transforms: "
                + ", ".join(unknown_transforms)
            )

        metadata_enrichment = _optional_mapping(
            transforms,
            "metadata_enrichment",
            "ingestion.transforms",
        )
        metadata_title_max_length = metadata_enrichment.get("section_title_max_length", 120)
        if not isinstance(metadata_title_max_length, int) or metadata_title_max_length <= 0:
            raise ConfigError(
                "ingestion.transforms.metadata_enrichment.section_title_max_length must be > 0"
            )

        chunk_refinement = _optional_mapping(
            transforms,
            "chunk_refinement",
            "ingestion.transforms",
        )
        image_captioning = _optional_mapping(
            transforms,
            "image_captioning",
            "ingestion.transforms",
        )
        image_caption_stub = image_captioning.get(
            "stub_caption",
            "image captioning not configured",
        )
        if not isinstance(image_caption_stub, str) or not image_caption_stub.strip():
            raise ConfigError("ingestion.transforms.image_captioning.stub_caption must be a string")

        dense_top_k = _require_int(retrieval, "dense_top_k", "retrieval")
        if dense_top_k <= 0:
            raise ConfigError("retrieval.dense_top_k must be > 0")

        retrieval_mode_raw = retrieval.get("mode", "dense")
        if not isinstance(retrieval_mode_raw, str):
            raise ConfigError("retrieval.mode must be one of: dense, hybrid")
        retrieval_mode = retrieval_mode_raw.strip().lower()
        if retrieval_mode not in {"dense", "hybrid"}:
            raise ConfigError("retrieval.mode must be one of: dense, hybrid")

        sparse_top_k = retrieval.get("sparse_top_k", dense_top_k)
        if not isinstance(sparse_top_k, int) or sparse_top_k <= 0:
            raise ConfigError("retrieval.sparse_top_k must be > 0")

        dense_candidate_multiplier = retrieval.get("dense_candidate_multiplier", 3)
        if not isinstance(dense_candidate_multiplier, int) or dense_candidate_multiplier <= 0:
            raise ConfigError("retrieval.dense_candidate_multiplier must be > 0")

        sparse_candidate_multiplier = retrieval.get("sparse_candidate_multiplier", 3)
        if not isinstance(sparse_candidate_multiplier, int) or sparse_candidate_multiplier <= 0:
            raise ConfigError("retrieval.sparse_candidate_multiplier must be > 0")

        max_candidate_top_k = retrieval.get("max_candidate_top_k", 12)
        if not isinstance(max_candidate_top_k, int) or max_candidate_top_k <= 0:
            raise ConfigError("retrieval.max_candidate_top_k must be > 0")

        rrf_k = retrieval.get("rrf_k", 60)
        if not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ConfigError("retrieval.rrf_k must be > 0")

        loader = _require_mapping(adapters, "loader", "adapters")
        embedding = _require_mapping(adapters, "embedding", "adapters")
        vector_store = _require_mapping(adapters, "vector_store", "adapters")
        llm = adapters.get("llm", {"provider": "fake"})
        if not isinstance(llm, dict):
            raise ConfigError("Missing or invalid mapping: adapters.llm")
        reranker = adapters.get("reranker", {"provider": "fake"})
        if not isinstance(reranker, dict):
            raise ConfigError("Missing or invalid mapping: adapters.reranker")

        embedding_provider = _require_str(embedding, "provider", "adapters.embedding")
        embedding_model = _optional_str(embedding, "model", "adapters.embedding")
        embedding_api_key = _optional_str(embedding, "api_key", "adapters.embedding")
        embedding_base_url = _optional_str(embedding, "base_url", "adapters.embedding")
        embedding_azure_endpoint = _optional_str(
            embedding,
            "azure_endpoint",
            "adapters.embedding",
        )
        embedding_deployment_name = _optional_str(
            embedding,
            "deployment_name",
            "adapters.embedding",
        )
        embedding_api_version = _optional_str(
            embedding,
            "api_version",
            "adapters.embedding",
        )
        _validate_real_provider_settings(
            "adapters.embedding",
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            azure_endpoint=embedding_azure_endpoint,
            deployment_name=embedding_deployment_name,
            api_version=embedding_api_version,
        )

        llm_provider = _require_str(llm, "provider", "adapters.llm")
        llm_model = _optional_str(llm, "model", "adapters.llm")
        llm_api_key = _optional_str(llm, "api_key", "adapters.llm")
        llm_base_url = _optional_str(llm, "base_url", "adapters.llm")
        llm_azure_endpoint = _optional_str(llm, "azure_endpoint", "adapters.llm")
        llm_deployment_name = _optional_str(llm, "deployment_name", "adapters.llm")
        llm_api_version = _optional_str(llm, "api_version", "adapters.llm")
        _validate_real_provider_settings(
            "adapters.llm",
            llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            azure_endpoint=llm_azure_endpoint,
            deployment_name=llm_deployment_name,
            api_version=llm_api_version,
        )

        reranker_provider = _require_str(reranker, "provider", "adapters.reranker")
        reranker_model = _optional_str(reranker, "model", "adapters.reranker")
        reranker_api_key = _optional_str(reranker, "api_key", "adapters.reranker")
        reranker_base_url = _optional_str(reranker, "base_url", "adapters.reranker")
        reranker_azure_endpoint = _optional_str(
            reranker,
            "azure_endpoint",
            "adapters.reranker",
        )
        reranker_deployment_name = _optional_str(
            reranker,
            "deployment_name",
            "adapters.reranker",
        )
        reranker_api_version = _optional_str(
            reranker,
            "api_version",
            "adapters.reranker",
        )
        _validate_real_provider_settings(
            "adapters.reranker",
            reranker_provider,
            model=reranker_model,
            api_key=reranker_api_key,
            azure_endpoint=reranker_azure_endpoint,
            deployment_name=reranker_deployment_name,
            api_version=reranker_api_version,
        )

        max_context_results = generation.get("max_context_results", dense_top_k)
        if not isinstance(max_context_results, int) or max_context_results <= 0:
            raise ConfigError("generation.max_context_results must be > 0")

        candidate_results = generation.get(
            "candidate_results",
            max(max_context_results * 2, max_context_results),
        )
        if not isinstance(candidate_results, int) or candidate_results <= 0:
            raise ConfigError("generation.candidate_results must be > 0")

        max_context_chars = generation.get("max_context_chars", 1200)
        if not isinstance(max_context_chars, int) or max_context_chars <= 0:
            raise ConfigError("generation.max_context_chars must be > 0")

        max_answer_chars = generation.get("max_answer_chars", 400)
        if not isinstance(max_answer_chars, int) or max_answer_chars <= 0:
            raise ConfigError("generation.max_answer_chars must be > 0")

        return cls(
            project=ProjectSettings(
                name=_require_str(project, "name", "project"),
                environment=_require_str(project, "environment", "project"),
            ),
            ingestion=IngestionSettings(
                default_collection=_require_str(
                    ingestion,
                    "default_collection",
                    "ingestion",
                ),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                supported_extensions=_require_list(
                    ingestion,
                    "supported_extensions",
                    "ingestion",
                ),
                transforms=IngestionTransformsSettings(
                    enabled=_optional_bool(transforms, "enabled", "ingestion.transforms", False),
                    order=transform_order,
                    metadata_enrichment=MetadataEnrichmentSettings(
                        enabled=_optional_bool(
                            metadata_enrichment,
                            "enabled",
                            "ingestion.transforms.metadata_enrichment",
                            True,
                        ),
                        section_title_max_length=metadata_title_max_length,
                    ),
                    chunk_refinement=ChunkRefinementSettings(
                        enabled=_optional_bool(
                            chunk_refinement,
                            "enabled",
                            "ingestion.transforms.chunk_refinement",
                            True,
                        ),
                        collapse_whitespace=_optional_bool(
                            chunk_refinement,
                            "collapse_whitespace",
                            "ingestion.transforms.chunk_refinement",
                            True,
                        ),
                    ),
                    image_captioning=ImageCaptioningSettings(
                        enabled=_optional_bool(
                            image_captioning,
                            "enabled",
                            "ingestion.transforms.image_captioning",
                            False,
                        ),
                        stub_caption=image_caption_stub.strip(),
                        append_to_text=_optional_bool(
                            image_captioning,
                            "append_to_text",
                            "ingestion.transforms.image_captioning",
                            False,
                        ),
                    ),
                ),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=dense_top_k,
                mode=retrieval_mode,
                sparse_top_k=sparse_top_k,
                dense_candidate_multiplier=dense_candidate_multiplier,
                sparse_candidate_multiplier=sparse_candidate_multiplier,
                max_candidate_top_k=max_candidate_top_k,
                rrf_k=rrf_k,
            ),
            adapters=AdapterSettings(
                loader=LoaderAdapterSettings(
                    provider=_require_str(loader, "provider", "adapters.loader")
                ),
                embedding=EmbeddingAdapterSettings(
                    provider=embedding_provider,
                    dimensions=_optional_positive_int(
                        embedding,
                        "dimensions",
                        "adapters.embedding",
                        16,
                    ),
                    model=embedding_model,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    azure_endpoint=embedding_azure_endpoint,
                    deployment_name=embedding_deployment_name,
                    api_version=embedding_api_version,
                ),
                vector_store=VectorStoreAdapterSettings(
                    provider=_require_str(
                        vector_store,
                        "provider",
                        "adapters.vector_store",
                    ),
                    storage_path=str(
                        vector_store.get("storage_path", "./data/db/vector_store.json")
                    ),
                ),
                llm=LLMAdapterSettings(
                    provider=llm_provider,
                    model=llm_model,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                    azure_endpoint=llm_azure_endpoint,
                    deployment_name=llm_deployment_name,
                    api_version=llm_api_version,
                    temperature=_optional_number(llm, "temperature", "adapters.llm", 0.0),
                ),
                reranker=RerankerAdapterSettings(
                    provider=reranker_provider,
                    model=reranker_model,
                    api_key=reranker_api_key,
                    base_url=reranker_base_url,
                    azure_endpoint=reranker_azure_endpoint,
                    deployment_name=reranker_deployment_name,
                    api_version=reranker_api_version,
                ),
            ),
            generation=GenerationSettings(
                max_context_results=max_context_results,
                candidate_results=candidate_results,
                max_context_chars=max_context_chars,
                max_answer_chars=max_answer_chars,
            ),
            observability=ObservabilitySettings(
                trace_enabled=_require_bool(
                    observability,
                    "trace_enabled",
                    "observability",
                ),
                trace_file=_require_str(
                    observability,
                    "trace_file",
                    "observability",
                ),
                log_level=_require_str(
                    observability,
                    "log_level",
                    "observability",
                ),
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
