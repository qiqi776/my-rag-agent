from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.application.document_service import DeleteDocumentResult
from src.interfaces.mcp.tools.delete_document import DeleteDocumentTool


class StubDocumentService:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def delete_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DeleteDocumentResult:
        self.calls.append({"doc_id": doc_id, "collection": collection})
        return DeleteDocumentResult(
            doc_id=doc_id,
            collection=collection or "default",
            deleted_chunks=1,
        )


@pytest.mark.unit
def test_delete_document_tool_uses_document_service() -> None:
    document_service = StubDocumentService()
    tool = DeleteDocumentTool(SimpleNamespace(document_service=document_service))

    result = tool.handle({"doc_id": "doc-1", "collection": "knowledge"})

    assert document_service.calls == [{"doc_id": "doc-1", "collection": "knowledge"}]
    assert result.structured_content["deleted"] is True
    assert "Document Deleted" in result.content[0].text


@pytest.mark.unit
def test_delete_document_tool_requires_doc_id() -> None:
    tool = DeleteDocumentTool(SimpleNamespace(document_service=StubDocumentService()))

    with pytest.raises(ValueError, match="doc_id"):
        tool.handle({"doc_id": ""})
