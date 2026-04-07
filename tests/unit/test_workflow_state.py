from __future__ import annotations

import pytest

from src.agent.contracts import ToolRequest, ToolResult
from src.agent.state import WorkflowState


@pytest.mark.unit
def test_workflow_state_records_steps_and_completion() -> None:
    state = WorkflowState(
        workflow_name="research_and_answer",
        user_input="semantic embeddings",
        collection="knowledge",
    )
    state.start()
    request = ToolRequest(name="search_knowledge", arguments={"query": "semantic embeddings"})
    result = ToolResult(
        name="search_knowledge",
        ok=True,
        content="Found 1 search result(s).",
        structured_content={"kind": "search_output", "result_count": 1},
    )

    state.add_step(request, result)
    state.set_intermediate("search_output", result.structured_content)
    state.complete({"kind": "workflow_output", "answer": "done"})

    assert state.status == "completed"
    assert state.tools_used == ["search_knowledge"]
    assert state.steps[0].structured_content["result_count"] == 1
    assert state.final_output["answer"] == "done"


@pytest.mark.unit
def test_workflow_state_records_failure() -> None:
    state = WorkflowState(
        workflow_name="research_and_answer",
        user_input="semantic embeddings",
    )
    state.start()
    state.fail("tool failed")

    assert state.status == "failed"
    assert state.error == "tool failed"
