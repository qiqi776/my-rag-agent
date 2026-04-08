from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.application.document_service import DocumentDetail
from src.interfaces.mcp.tools.get_document_summary import GetDocumentSummaryTool


class _StubDocumentService:
    def __init__(self, detail: DocumentDetail | None) -> None:
        self.detail = detail

    def get_document_summary(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DocumentDetail | None:
        del doc_id, collection
        return self.detail


@pytest.mark.unit
def test_get_document_summary_tool_returns_detail_payload() -> None:
    detail = DocumentDetail(
        doc_id="doc-1",
        source_path="/tmp/ostep.txt",
        collection="knowledge",
        chunk_count=3,
        preview="Virtual memory gives each process an address space.",
        metadata={"source_path": "/tmp/ostep.txt", "page_count": 2},
    )
    tool = GetDocumentSummaryTool(SimpleNamespace(document_service=_StubDocumentService(detail)))

    result = tool.handle({"doc_id": "doc-1"})

    assert result.structured_content["kind"] == "document_detail"
    assert result.structured_content["found"] is True
    assert result.structured_content["doc_id"] == "doc-1"


@pytest.mark.unit
def test_get_document_summary_tool_returns_not_found_payload() -> None:
    tool = GetDocumentSummaryTool(SimpleNamespace(document_service=_StubDocumentService(None)))

    result = tool.handle({"doc_id": "missing"})

    assert result.structured_content["found"] is False
