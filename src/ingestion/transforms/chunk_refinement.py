"""Rule-based content cleanup before chunk splitting."""

from __future__ import annotations

import re

from src.ingestion.models import IngestionUnit
from src.ingestion.transforms.base import BaseTransform

_INLINE_WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")
_PAGE_NOISE_PATTERN = re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE)


class ChunkRefinementTransform(BaseTransform):
    """Normalize whitespace and drop trivial page-noise lines."""

    name = "chunk_refinement"

    def __init__(self, collapse_whitespace: bool = True) -> None:
        self.collapse_whitespace = collapse_whitespace

    def transform(self, units: list[IngestionUnit]) -> list[IngestionUnit]:
        refined_units: list[IngestionUnit] = []
        for unit in units:
            refined_text = self._refine(unit.text)
            metadata = unit.metadata.copy()
            metadata["refinement_applied"] = refined_text != unit.text
            refined_units.append(
                IngestionUnit(
                    unit_id=unit.unit_id,
                    doc_id=unit.doc_id,
                    text=refined_text,
                    metadata=metadata,
                )
            )
        return refined_units

    def _refine(self, text: str) -> str:
        if not self.collapse_whitespace:
            return text

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for line in normalized.splitlines():
            collapsed = _INLINE_WHITESPACE_PATTERN.sub(" ", line).strip()
            if not collapsed:
                lines.append("")
                continue
            if _PAGE_NOISE_PATTERN.match(collapsed):
                continue
            lines.append(collapsed)
        refined = "\n".join(lines).strip()
        return _MULTI_BLANK_PATTERN.sub("\n\n", refined)
