from __future__ import annotations

import pytest

from src.agent.contracts import AgentTool, ToolRequest, ToolResult, ToolSpec
from src.agent.registry import ToolRegistry


class StubTool(AgentTool):
    def __init__(self, name: str = "stub_tool") -> None:
        self._spec = ToolSpec(
            name=name,
            description="stub tool",
            input_schema={"type": "object"},
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=f"handled {request.arguments.get('value', 'missing')}",
            structured_content={"kind": "stub", "arguments": request.arguments.copy()},
        )


@pytest.mark.unit
def test_tool_registry_registers_lists_and_calls_tools() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    result = registry.call(ToolRequest(name="stub_tool", arguments={"value": "ok"}))

    assert [tool.name for tool in registry.list_tools()] == ["stub_tool"]
    assert result.ok is True
    assert result.content == "handled ok"
    assert result.structured_content["kind"] == "stub"


@pytest.mark.unit
def test_tool_registry_rejects_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.register(StubTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(StubTool())


@pytest.mark.unit
def test_tool_registry_returns_error_for_unknown_tool() -> None:
    registry = ToolRegistry()

    result = registry.call(ToolRequest(name="missing_tool"))

    assert result.ok is False
    assert result.error_code == "tool_not_found"
    assert result.structured_content["error"]["message"] == "Unknown tool: missing_tool"
