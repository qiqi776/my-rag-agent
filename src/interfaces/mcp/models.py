"""Lightweight MCP-facing protocol models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MCPTextContent:
    """Human-readable text block returned by an MCP tool."""

    text: str
    type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MCPTool:
    """Definition of a callable MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inputSchema"] = payload.pop("input_schema")
        return payload


@dataclass(slots=True)
class MCPToolResult:
    """Standardized in-process MCP tool result."""

    content: list[MCPTextContent] = field(default_factory=list)
    structured_content: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": [item.to_dict() for item in self.content],
            "structuredContent": self.structured_content,
            "isError": self.is_error,
        }
