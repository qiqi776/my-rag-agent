"""In-process MCP server and CLI smoke entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.core.errors import ConfigError, EmptyQueryError, UnsupportedRetrievalModeError
from src.interfaces.mcp.dependencies import MCPDependencies, build_dependencies
from src.interfaces.mcp.mappers import format_json_payload, map_error
from src.interfaces.mcp.models import MCPTool, MCPToolResult
from src.interfaces.mcp.tools import (
    register_delete_document_tool,
    register_list_documents_tool,
    register_query_knowledge_tool,
)
from src.observability.logger import get_logger

ToolHandler = Callable[[dict[str, Any]], MCPToolResult]


class MCPServer:
    """Minimal local MCP server abstraction with tool registry."""

    def __init__(
        self,
        dependencies: MCPDependencies,
        *,
        server_name: str = "minimal-modular-rag-mcp",
        server_version: str = "0.1.0",
    ) -> None:
        self.dependencies = dependencies
        self.server_name = server_name
        self.server_version = server_version
        self._tools: dict[str, MCPTool] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self.logger = get_logger(
            "minimal-rag.mcp",
            dependencies.settings.observability.log_level,
        )

    def register_tool(self, tool: MCPTool, handler: ToolHandler) -> None:
        """Register a tool with a handler."""

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

    def list_tools(self) -> list[MCPTool]:
        """Return registered tools."""

        return [self._tools[name] for name in sorted(self._tools)]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a registered tool and map errors into MCP-safe responses."""

        if name not in self._tools:
            return map_error(f"Unknown tool: {name}", code="tool_not_found")

        try:
            return self._handlers[name](arguments or {})
        except (
            ConfigError,
            EmptyQueryError,
            UnsupportedRetrievalModeError,
            ValueError,
            TypeError,
        ) as exc:
            return map_error(str(exc), code="invalid_params")
        except Exception:
            self.logger.exception("Unhandled MCP tool error: %s", name)
            return map_error(
                f"Internal server error while executing '{name}'",
                code="internal_error",
            )


def create_mcp_server(config_path: str | Path | None = None) -> MCPServer:
    """Create an MCP server with default tools registered."""

    dependencies = build_dependencies(config_path)
    server = MCPServer(dependencies)
    register_query_knowledge_tool(server, dependencies)
    register_list_documents_tool(server, dependencies)
    register_delete_document_tool(server, dependencies)
    return server


def build_parser() -> argparse.ArgumentParser:
    """Build a small CLI for local MCP smoke testing."""

    parser = argparse.ArgumentParser(description="Run local MCP tool operations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tools", help="List registered MCP tools.")
    list_parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    call_parser = subparsers.add_parser("call-tool", help="Call an MCP tool locally.")
    call_parser.add_argument("name", help="Tool name.")
    call_parser.add_argument(
        "--arguments-json",
        default="{}",
        help="JSON object with tool arguments.",
    )
    call_parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    return parser


def main() -> int:
    """CLI entry point for local MCP smoke operations."""

    args = build_parser().parse_args()
    server = create_mcp_server(args.config)

    if args.command == "list-tools":
        payload = {
            "server": {
                "name": server.server_name,
                "version": server.server_version,
            },
            "tools": [tool.to_dict() for tool in server.list_tools()],
        }
        print(format_json_payload(payload))
        return 0

    try:
        arguments = json.loads(args.arguments_json)
    except json.JSONDecodeError as exc:
        print(format_json_payload(map_error(str(exc), code="invalid_json").to_dict()))
        return 1
    if not isinstance(arguments, dict):
        print(
            format_json_payload(
                map_error("arguments-json must decode to an object", code="invalid_json").to_dict()
            )
        )
        return 1

    result = server.call_tool(args.name, arguments)
    print(format_json_payload(result.to_dict()))
    return 1 if result.is_error else 0
