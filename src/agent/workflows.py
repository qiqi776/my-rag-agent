"""Deterministic workflow runner and built-in workflow stubs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from src.agent.contracts import ToolRequest
from src.agent.registry import ToolRegistry
from src.agent.state import WorkflowState
from src.core.types import Metadata

WorkflowHandler = Callable[[dict[str, Any], WorkflowState, ToolRegistry], WorkflowState]


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Static description of a workflow entry point."""

    name: str
    description: str
    input_schema: Metadata

    def to_dict(self) -> Metadata:
        return asdict(self)


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string when provided")
    return value


def _optional_positive_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer when provided")
    return value


def research_and_answer_workflow(
    arguments: dict[str, Any],
    state: WorkflowState,
    registry: ToolRegistry,
) -> WorkflowState:
    """Run a deterministic search -> answer workflow."""

    query = _require_string(arguments, "query")
    collection = _optional_string(arguments, "collection")
    mode = _optional_string(arguments, "mode")
    search_top_k = _optional_positive_int(arguments, "search_top_k")
    answer_top_k = _optional_positive_int(arguments, "answer_top_k")

    search_request = ToolRequest(
        name="search_knowledge",
        arguments={
            key: value
            for key, value in {
                "query": query,
                "collection": collection,
                "top_k": search_top_k,
                "mode": mode,
            }.items()
            if value is not None
        },
        context=state.tool_context(),
    )
    search_result = registry.call(search_request)
    state.add_step(search_request, search_result)
    state.set_intermediate("search_output", search_result.structured_content)
    if not search_result.ok:
        state.fail(
            search_result.error or "search tool failed",
            final_output={
                "kind": "workflow_output",
                "workflow_name": state.workflow_name,
                "tools_used": state.tools_used,
            },
        )
        return state

    answer_request = ToolRequest(
        name="answer_question",
        arguments={
            key: value
            for key, value in {
                "query": query,
                "collection": collection,
                "top_k": answer_top_k,
                "mode": mode,
            }.items()
            if value is not None
        },
        context=state.tool_context(),
    )
    answer_result = registry.call(answer_request)
    state.add_step(answer_request, answer_result)
    state.set_intermediate("answer_output", answer_result.structured_content)
    if not answer_result.ok:
        state.fail(
            answer_result.error or "answer tool failed",
            final_output={
                "kind": "workflow_output",
                "workflow_name": state.workflow_name,
                "tools_used": state.tools_used,
            },
        )
        return state

    answer_payload = answer_result.structured_content.copy()
    state.complete(
        {
            "kind": "workflow_output",
            "workflow_name": state.workflow_name,
            "answer": answer_payload.get("answer", ""),
            "collection": answer_payload.get("collection", collection),
            "retrieval_mode": answer_payload.get("retrieval_mode", mode),
            "tools_used": state.tools_used,
            "citations": answer_payload.get("citations", []),
            "supporting_results": answer_payload.get("supporting_results", []),
            "metadata": {
                "workflow_id": state.workflow_id,
                "search_result_count": search_result.structured_content.get("result_count"),
            },
        }
    )
    return state


class WorkflowRunner:
    """Register and execute deterministic workflows against a tool registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._workflows: dict[str, tuple[WorkflowSpec, WorkflowHandler]] = {}

    def register(self, spec: WorkflowSpec, handler: WorkflowHandler) -> None:
        """Register a workflow by name."""

        if spec.name in self._workflows:
            raise ValueError(f"Workflow '{spec.name}' is already registered")
        self._workflows[spec.name] = (spec, handler)

    def list_workflows(self) -> list[WorkflowSpec]:
        """Return all registered workflow definitions."""

        return [self._workflows[name][0] for name in sorted(self._workflows)]

    def run(self, name: str, arguments: dict[str, Any]) -> WorkflowState:
        """Execute a workflow and return its final state."""

        entry = self._workflows.get(name)
        if entry is None:
            raise ValueError(f"Unknown workflow: {name}")

        _, handler = entry
        state = WorkflowState(
            workflow_name=name,
            user_input=str(arguments.get("query", "")),
            collection=arguments.get("collection")
            if isinstance(arguments.get("collection"), str)
            else None,
        )
        state.start()
        return handler(arguments, state, self.registry)


def create_workflow_runner(registry: ToolRegistry) -> WorkflowRunner:
    """Create the default workflow runner with built-in workflow stubs."""

    runner = WorkflowRunner(registry)
    runner.register(
        WorkflowSpec(
            name="research_and_answer",
            description="Search the knowledge base and synthesize a final answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question text."},
                    "collection": {"type": "string", "description": "Optional collection name."},
                    "mode": {
                        "type": "string",
                        "description": "Retrieval mode override.",
                        "enum": ["dense", "hybrid"],
                    },
                    "search_top_k": {
                        "type": "integer",
                        "description": "Maximum results used during search.",
                        "minimum": 1,
                    },
                    "answer_top_k": {
                        "type": "integer",
                        "description": "Maximum supporting results used for answer synthesis.",
                        "minimum": 1,
                    },
                },
                "required": ["query"],
            },
        ),
        research_and_answer_workflow,
    )
    return runner
