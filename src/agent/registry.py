"""Registry for agent-ready tools."""

from __future__ import annotations

from src.agent.contracts import AgentTool, ToolRequest, ToolResult, ToolSpec
from src.core.errors import ConfigError, EmptyQueryError, UnsupportedRetrievalModeError
from src.observability.logger import get_logger


class ToolRegistry:
    """Register, list, look up, and execute tools."""

    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        self.logger = get_logger("minimal-rag.agent.registry")
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        """Register a tool by name."""

        if tool.spec.name in self._tools:
            raise ValueError(f"Tool '{tool.spec.name}' is already registered")
        self._tools[tool.spec.name] = tool

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tool definitions."""

        return [self._tools[name].spec for name in sorted(self._tools)]

    def get(self, name: str) -> AgentTool | None:
        """Return a tool by name if present."""

        return self._tools.get(name)

    def call(self, request: ToolRequest) -> ToolResult:
        """Execute a registered tool and normalize common error cases."""

        tool = self.get(request.name)
        if tool is None:
            return ToolResult.error_result(
                request.name,
                f"Unknown tool: {request.name}",
                code="tool_not_found",
                metadata=request.context.to_dict(),
            )

        try:
            return tool.execute(request)
        except (
            ConfigError,
            EmptyQueryError,
            UnsupportedRetrievalModeError,
            ValueError,
            TypeError,
            FileNotFoundError,
            UnicodeDecodeError,
        ) as exc:
            return ToolResult.error_result(
                request.name,
                str(exc),
                code="invalid_params",
                metadata=request.context.to_dict(),
            )
        except Exception:
            self.logger.exception("Unhandled agent tool error: %s", request.name)
            return ToolResult.error_result(
                request.name,
                f"Internal error while executing '{request.name}'",
                code="internal_error",
                metadata=request.context.to_dict(),
            )
