"""Base transform contract for ingestion pipeline plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.ingestion.models import IngestionUnit


class BaseTransform(ABC):
    """Transform a list of ingestion units into a new list of ingestion units."""

    name: str

    @abstractmethod
    def transform(self, units: list[IngestionUnit]) -> list[IngestionUnit]:
        """Return transformed ingestion units."""
