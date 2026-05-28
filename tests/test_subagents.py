from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.errors import GraphRecursionError

from superassist_plus.config import Settings
from superassist_plus.subagents import GENERAL_PURPOSE_PROMPT, RESEARCH_PROMPT, TASK_STORE, SubagentResult, SubagentStatus
from superassist_plus.subagents.config import build_builtin_subagents
import superassist_plus.subagents.executor as executor_module
from superassist_plus.subagents.executor import SubagentExecutor, _filter_tools
from superassist_plus.tools import default_tools
from superassist_plus.ui.server import create_app


def test_builtin_subagents_expose_expected_prompts() -> None:
    configs = build_builtin_subagents(timeout_seconds=900, max_turns=20)

    assert configs["general-purpose"].system_prompt == GENERAL_PURPOSE_PROMPT
    assert configs["research"].system_prompt == RESEARCH_PROMPT
    assert "Do not call the task tool" in configs["general-purpose"].system_prompt
    assert "Prioritize reliable primary or official sources" in configs["research"].system_prompt


def test_subagent_tool_filter_excludes_task() -> None:
    tools = _filter_tools(default_tools(include_task=True), allowed=None)

    assert "task" not in {tool.name for tool in tools}


def test_subagent_executor_runs_fallback_agent(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    config = build_builtin_subagents(timeout_seconds=30, max_turns=10)["general-purpose"]
    executor = SubagentExecutor(config=config, tools=default_tools(include_task=False), settings=settings)

    result = executor.run("Return a short confirmation.", description="smoke")

    assert result.status == SubagentStatus.COMPLETED
    assert "fallback mode" in result.result
    assert result.ai_messages


def test_subagent_executor_reports_streamed_ai_text(tmp_path, monkeypatch) -> None:
    class StreamingAgent:
        def stream(self, state, config=None, stream_mode=None):
            yield ("messages", (AIMessageChunk(content="I am checking", id="sub_msg_1"), {}))
            yield ("values", {"messages": [*state["messages"], AIMessage(content="done")]})

    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    events = []
    config = build_builtin_subagents(timeout_seconds=30, max_turns=10)["general-purpose"]
    monkeypatch.setattr(executor_module, "create_agent", lambda **kwargs: StreamingAgent())
    executor = SubagentExecutor(
        config=config,
        tools=default_tools(include_task=False),
        settings=settings,
        run_event_reporter=events.append,
    )

    result = executor.run("Do a streamed check.", description="stream check")

    assert result.status == SubagentStatus.COMPLETED
    assert result.result == "done"
    assert [event.type for event in events] == ["subagent_text", "subagent_text"]
    assert events[0].message == "I am checking"
    assert events[0].metadata["description"] == "stream check"


def test_subagent_executor_summarizes_when_recursion_limit_is_reached(tmp_path, monkeypatch) -> None:
    class RecursingAgent:
        def invoke(self, state, config):
            raise GraphRecursionError("limit")

    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    config = build_builtin_subagents(timeout_seconds=30, max_turns=1)["general-purpose"]
    monkeypatch.setattr(executor_module, "create_agent", lambda **kwargs: RecursingAgent())
    executor = SubagentExecutor(config=config, tools=default_tools(include_task=False), settings=settings)

    result = executor.run("Keep working until recursion limit.", description="recursion")

    assert result.status == SubagentStatus.COMPLETED
    assert "maximum recursion limit" in result.result
    assert result.ai_messages


def test_subagent_task_store_fastapi_endpoints(tmp_path) -> None:
    TASK_STORE.put(
        SubagentResult(
            task_id="task_1",
            description="demo",
            subagent_type="research",
            status=SubagentStatus.COMPLETED,
            result="done",
        )
    )
    settings = Settings(SUPERASSIST_PLUS_DATA_DIR=tmp_path, SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash")
    client = TestClient(create_app(settings=settings))

    listed = client.get("/api/subagents/tasks").json()
    assert listed["tasks"][0]["task_id"] == "task_1"

    detail = client.get("/api/subagents/tasks/task_1").json()
    assert detail["result"] == "done"

    assert client.get("/api/subagents/tasks/missing").status_code == 404
    assert client.delete("/api/subagents/tasks/task_1").json() == {"deleted": True}
    assert client.get("/api/subagents/tasks/task_1").status_code == 404


def test_last_ai_text_shape_for_subagent_results() -> None:
    message = AIMessage(content="final subagent result")

    assert str(message.content) == "final subagent result"
