from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.application.document_service import DocumentSummary
from src.interfaces.mcp.tools.list_documents import ListDocumentsTool


class StubDocumentService:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def list_documents(self, collection: str | None = None) -> list[DocumentSummary]:
        self.calls.append(collection)
        return [
            DocumentSummary(
                doc_id="doc-1",
                source_path="/tmp/doc.txt",
                collection=collection or "default",
                chunk_count=2,
            )
        ]


@pytest.mark.unit
def test_list_documents_tool_uses_document_service() -> None:
    document_service = StubDocumentService()
    tool = ListDocumentsTool(SimpleNamespace(document_service=document_service))

    result = tool.handle({"collection": "knowledge"})

    assert document_service.calls == ["knowledge"]
    assert result.structured_content["documents"][0]["collection"] == "knowledge"
    assert "Documents" in result.content[0].text


@pytest.mark.unit
def test_list_documents_tool_rejects_blank_collection() -> None:
    tool = ListDocumentsTool(SimpleNamespace(document_service=StubDocumentService()))

    with pytest.raises(ValueError, match="collection"):
        tool.handle({"collection": "  "})
