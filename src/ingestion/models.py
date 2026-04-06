"""Shared ingestion-side models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.core.types import Metadata


@dataclass(slots=True)
class IngestionUnit:
    """A pre-chunk text unit emitted by the ingestion pipeline."""

    unit_id: str
    doc_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metadata.get("source_path"):
            raise ValueError("IngestionUnit metadata must contain 'source_path'")

    def to_dict(self) -> Metadata:
        return asdict(self)
