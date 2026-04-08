"""Mapping helpers from application outputs to MCP tool results."""

from __future__ import annotations

from typing import Any

from src.application.document_service import DeleteDocumentResult, DocumentDetail, DocumentSummary
from src.interfaces.mcp.models import MCPTextContent, MCPToolResult
from src.response.response_builder import SearchOutput


def _snippet(text: str, max_length: int = 140) -> str:
    collapsed = text.replace("\n", " ").strip()
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3] + "..."


def map_search_output(output: SearchOutput) -> MCPToolResult:
    """Convert SearchOutput into a human-readable plus structured MCP payload."""

    if not output.results:
        content = (
            "## No Results\n\n"
            f"Query: **{output.normalized_query}**\n"
            f"Collection: `{output.collection}`\n"
            f"Mode: `{output.retrieval_mode}`\n"
        )
        payload = output.to_dict()
        payload["kind"] = "search_output"
        return MCPToolResult(
            content=[MCPTextContent(content)],
            structured_content=payload,
        )

    lines = [
        "## Search Results",
        "",
        f"Query: **{output.normalized_query}**",
        f"Collection: `{output.collection}`",
        f"Mode: `{output.retrieval_mode}`",
        f"Returned: {output.result_count}",
        "",
    ]
    for item in output.results:
        page_suffix = f", page={item.page}" if item.page is not None else ""
        lines.append(
            f"{item.rank}. `{item.source_path}` "
            f"(score={item.score:.4f}, chunk_id={item.chunk_id}{page_suffix})"
        )
        lines.append(f"   {_snippet(item.text)}")
        lines.append("")

    if output.citations:
        lines.append("### Citations")
        lines.append("")
        for index, citation in enumerate(output.citations, start=1):
            page_suffix = f", page={citation.page}" if citation.page is not None else ""
            lines.append(
                f"[{index}] `{citation.source_path}` "
                f"(doc_id={citation.doc_id}, chunk_index={citation.chunk_index}{page_suffix})"
            )

    payload = output.to_dict()
    payload["kind"] = "search_output"
    return MCPToolResult(
        content=[MCPTextContent("\n".join(lines))],
        structured_content=payload,
    )


def map_document_list(
    documents: list[DocumentSummary],
    collection: str | None = None,
) -> MCPToolResult:
    """Convert document summaries into an MCP payload."""

    if not documents:
        label = collection or "all collections"
        return MCPToolResult(
            content=[MCPTextContent(f"## No Documents Found\n\nScope: `{label}`")],
            structured_content={
                "kind": "document_list",
                "collection": collection,
                "count": 0,
                "documents": [],
            },
        )

    lines = [
        "## Documents",
        "",
        f"Count: {len(documents)}",
    ]
    if collection is not None:
        lines.append(f"Collection: `{collection}`")
    lines.append("")

    for document in documents:
        lines.append(
            f"- `{document.doc_id}` collection=`{document.collection}` "
            f"chunks={document.chunk_count} source=`{document.source_path}`"
        )

    return MCPToolResult(
        content=[MCPTextContent("\n".join(lines))],
        structured_content={
            "kind": "document_list",
            "collection": collection,
            "count": len(documents),
            "documents": [document.to_dict() for document in documents],
        },
    )


def map_collection_list(collections: list[str]) -> MCPToolResult:
    """Convert collection names into a stable MCP payload."""

    if not collections:
        return MCPToolResult(
            content=[MCPTextContent("## No Collections Found")],
            structured_content={
                "kind": "collection_list",
                "count": 0,
                "collections": [],
            },
        )

    lines = ["## Collections", "", f"Count: {len(collections)}", ""]
    lines.extend(f"- `{collection}`" for collection in collections)
    return MCPToolResult(
        content=[MCPTextContent("\n".join(lines))],
        structured_content={
            "kind": "collection_list",
            "count": len(collections),
            "collections": collections,
        },
    )


def map_document_detail(detail: DocumentDetail | None) -> MCPToolResult:
    """Convert a document summary/detail payload into MCP output."""

    if detail is None:
        return MCPToolResult(
            content=[MCPTextContent("## Document Not Found")],
            structured_content={
                "kind": "document_detail",
                "found": False,
            },
        )

    lines = [
        "## Document Summary",
        "",
        f"Document: `{detail.doc_id}`",
        f"Collection: `{detail.collection}`",
        f"Source: `{detail.source_path}`",
        f"Chunks: {detail.chunk_count}",
    ]
    if detail.metadata.get("page_count") is not None:
        lines.append(f"Pages: {detail.metadata['page_count']}")
    if detail.preview:
        lines.extend(["", "### Preview", "", _snippet(detail.preview, max_length=300)])

    payload = detail.to_dict()
    payload["kind"] = "document_detail"
    payload["found"] = True
    return MCPToolResult(
        content=[MCPTextContent("\n".join(lines))],
        structured_content=payload,
    )


def map_delete_result(result: DeleteDocumentResult) -> MCPToolResult:
    """Convert a delete result into an MCP payload."""

    payload = result.to_dict()
    payload["kind"] = "delete_document_result"

    if result.deleted:
        content = (
            "## Document Deleted\n\n"
            f"Collection: `{result.collection}`\n"
            f"Document: `{result.doc_id}`\n"
            f"Deleted chunks: {result.deleted_chunks}"
        )
    else:
        content = (
            "## Document Not Found\n\n"
            f"Collection: `{result.collection}`\n"
            f"Document: `{result.doc_id}`\n"
            "Deleted chunks: 0"
        )

    return MCPToolResult(
        content=[MCPTextContent(content)],
        structured_content=payload,
    )


def map_error(message: str, *, code: str) -> MCPToolResult:
    """Convert an application or parameter error into an MCP payload."""

    return MCPToolResult(
        content=[MCPTextContent(f"Error: {message}")],
        structured_content={
            "kind": "error",
            "error": {
                "code": code,
                "message": message,
            },
        },
        is_error=True,
    )


def format_json_payload(payload: dict[str, Any]) -> str:
    """Return a stable JSON string for CLI-style MCP smoke output."""

    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
