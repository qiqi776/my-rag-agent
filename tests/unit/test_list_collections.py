from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.mcp.tools.list_collections import ListCollectionsTool


class _StubDocumentService:
    def list_collections(self) -> list[str]:
        return ["alpha", "knowledge"]


@pytest.mark.unit
def test_list_collections_tool_returns_collection_payload() -> None:
    tool = ListCollectionsTool(SimpleNamespace(document_service=_StubDocumentService()))

    result = tool.handle({})

    assert result.structured_content["kind"] == "collection_list"
    assert result.structured_content["collections"] == ["alpha", "knowledge"]
