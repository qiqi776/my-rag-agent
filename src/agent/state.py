"""Workflow state models for deterministic agent-ready orchestration."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal

from src.agent.contracts import ToolContext, ToolRequest, ToolResult
from src.core.types import Metadata

WorkflowStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(slots=True)
class WorkflowStep:
    """A single recorded tool invocation inside a workflow."""

    index: int
    tool_name: str
    arguments: Metadata = field(default_factory=dict)
    ok: bool = True
    output_summary: str = ""
    structured_content: Metadata = field(default_factory=dict)
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class WorkflowState:
    """Mutable workflow state captured across deterministic orchestration steps."""

    workflow_name: str
    user_input: str
    collection: str | None = None
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = "pending"
    steps: list[WorkflowStep] = field(default_factory=list)
    intermediate_results: Metadata = field(default_factory=dict)
    final_output: Metadata = field(default_factory=dict)
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [step.tool_name for step in self.steps]

    def start(self) -> None:
        self.status = "running"
        self.started_at = datetime.now(UTC).isoformat()

    def tool_context(self) -> ToolContext:
        return ToolContext(
            workflow_id=self.workflow_id,
            metadata={
                "workflow_name": self.workflow_name,
                "collection": self.collection,
            },
        )

    def add_step(self, request: ToolRequest, result: ToolResult) -> WorkflowStep:
        step = WorkflowStep(
            index=len(self.steps) + 1,
            tool_name=request.name,
            arguments=request.arguments.copy(),
            ok=result.ok,
            output_summary=result.content,
            structured_content=result.structured_content.copy(),
            error=result.error,
            metadata=result.metadata.copy(),
        )
        self.steps.append(step)
        return step

    def set_intermediate(self, key: str, value: Metadata) -> None:
        self.intermediate_results[key] = value

    def complete(self, final_output: Metadata) -> None:
        self.status = "completed"
        self.final_output = final_output
        self.finished_at = datetime.now(UTC).isoformat()

    def fail(self, message: str, *, final_output: Metadata | None = None) -> None:
        self.status = "failed"
        self.error = message
        if final_output is not None:
            self.final_output = final_output
        self.finished_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> Metadata:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "user_input": self.user_input,
            "collection": self.collection,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "intermediate_results": self.intermediate_results,
            "final_output": self.final_output,
            "error": self.error,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
