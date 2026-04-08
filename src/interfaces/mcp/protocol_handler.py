"""Formal MCP/JSON-RPC request handling for stdio transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.interfaces.mcp.models import JSONRPCError, JSONRPCResponse

if TYPE_CHECKING:
    from src.interfaces.mcp.server import MCPServer

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPProtocolHandler:
    """Handle MCP requests over a line-delimited JSON-RPC transport."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("id")
        try:
            self._validate_payload(payload)
            method = payload["method"]
            params = payload.get("params", {})
            if method == "initialize":
                return JSONRPCResponse(
                    id=request_id,
                    result={
                        "serverInfo": {
                            "name": self.server.server_name,
                            "version": self.server.server_version,
                        },
                        "capabilities": {
                            "tools": {
                                "listChanged": False,
                            }
                        },
                    },
                ).to_dict()
            if method == "tools/list":
                return JSONRPCResponse(
                    id=request_id,
                    result={
                        "tools": [tool.to_dict() for tool in self.server.list_tools()],
                    },
                ).to_dict()
            if method == "tools/call":
                name, arguments = self._extract_tool_call(params)
                result = self.server.call_tool(name, arguments)
                return JSONRPCResponse(id=request_id, result=result.to_dict()).to_dict()
            return self._error_response(
                request_id,
                METHOD_NOT_FOUND,
                f"Unknown method: {method}",
            )
        except ValueError as exc:
            return self._error_response(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:
            return self._error_response(request_id, INTERNAL_ERROR, str(exc))

    def parse_error(self, message: str) -> dict[str, Any]:
        return self._error_response(None, PARSE_ERROR, message)

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("jsonrpc") != "2.0":
            raise ValueError("jsonrpc must be '2.0'")
        method = payload.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        params = payload.get("params", {})
        if params is None:
            return
        if not isinstance(params, dict):
            raise ValueError("params must be an object when provided")

    def _extract_tool_call(self, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = params.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tools/call requires a non-empty 'name'")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tools/call 'arguments' must be an object")
        return name, arguments

    def _error_response(
        self,
        request_id: str | int | None,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        return JSONRPCResponse(
            id=request_id,
            error=JSONRPCError(code=code, message=message),
        ).to_dict()
