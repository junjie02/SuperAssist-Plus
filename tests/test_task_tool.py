from __future__ import annotations

import importlib
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from superassist_plus.agent import AgentRuntime
from superassist_plus.agent.middleware import SubagentLimitMiddleware
from superassist_plus.config import Settings
from superassist_plus.llm import FallbackChatModel
from superassist_plus.subagents.store import SubagentResult, SubagentStatus
from superassist_plus.tools.task import task


def test_task_rejects_unknown_subagent(tmp_path, monkeypatch) -> None:
    task_module = importlib.import_module("superassist_plus.tools.task")
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_PLUS_SUBAGENTS_ENABLED=True,
    )
    monkeypatch.setattr(task_module, "get_settings", lambda: settings)

    result = task.invoke({"description": "bad", "prompt": "do it", "subagent_type": "missing"})

    assert "Unknown subagent type 'missing'" in result
    assert "general-purpose" in result
    assert "research" in result


def test_task_formats_success_failure_and_timeout(tmp_path, monkeypatch) -> None:
    task_module = importlib.import_module("superassist_plus.tools.task")
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_PLUS_SUBAGENTS_ENABLED=True,
    )
    monkeypatch.setattr(task_module, "get_settings", lambda: settings)

    class FakeExecutor:
        results = [
            SubagentResult("t1", "d", "general-purpose", status=SubagentStatus.COMPLETED, result="ok"),
            SubagentResult("t2", "d", "general-purpose", status=SubagentStatus.FAILED, error="boom"),
            SubagentResult("t3", "d", "general-purpose", status=SubagentStatus.TIMED_OUT, error="slow"),
        ]

        def __init__(self, **kwargs):
            pass

        def run(self, prompt, *, description):
            return self.results.pop(0)

    monkeypatch.setattr(task_module, "SubagentExecutor", FakeExecutor)

    args = {"description": "demo", "prompt": "do it", "subagent_type": "general-purpose"}
    assert task.invoke(args) == "Task Succeeded. Result: ok"
    assert task.invoke(args) == "Task failed. Error: boom"
    assert task.invoke(args) == "Task timed out. Error: slow"


def test_subagent_limit_middleware_keeps_first_three_task_calls() -> None:
    middleware = SubagentLimitMiddleware(max_concurrent=3)
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "task", "args": {"description": "a"}, "id": "call_1"},
            {"name": "web_search", "args": {"query": "x"}, "id": "call_2"},
            {"name": "task", "args": {"description": "b"}, "id": "call_3"},
            {"name": "task", "args": {"description": "c"}, "id": "call_4"},
            {"name": "task", "args": {"description": "d"}, "id": "call_5"},
        ],
    )

    update = middleware.after_model({"messages": [message], "tool_events": []}, SimpleNamespace())

    assert update is not None
    kept = update["messages"][0].tool_calls
    assert [call["id"] for call in kept] == ["call_1", "call_2", "call_3", "call_4"]
    assert update["tool_events"][0]["dropped"] == 1


def test_runtime_dispatches_parallel_task_calls_and_receives_results(tmp_path, monkeypatch) -> None:
    task_module = importlib.import_module("superassist_plus.tools.task")
    runtime_module = importlib.import_module("superassist_plus.agent.runtime")

    class LeadTaskModel(FallbackChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            if not tool_messages:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="I will delegate three checks.",
                                tool_calls=[
                                    {
                                        "name": "task",
                                        "args": {
                                            "description": "check one",
                                            "prompt": "Return a short confirmation for task one.",
                                            "subagent_type": "general-purpose",
                                        },
                                        "id": "call_1",
                                    },
                                    {
                                        "name": "task",
                                        "args": {
                                            "description": "check two",
                                            "prompt": "Return a short confirmation for task two.",
                                            "subagent_type": "general-purpose",
                                        },
                                        "id": "call_2",
                                    },
                                    {
                                        "name": "task",
                                        "args": {
                                            "description": "check three",
                                            "prompt": "Return a short confirmation for task three.",
                                            "subagent_type": "research",
                                        },
                                        "id": "call_3",
                                    },
                                ],
                            )
                        )
                    ]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"FINAL RECEIVED {len(tool_messages)} TASK RESULTS")
                    )
                ]
            )

    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_ENABLE_TOOLS=True,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_SUBAGENT_TIMEOUT_SECONDS=30,
        SUPERASSIST_PLUS_SUBAGENT_MAX_TURNS=10,
    )
    monkeypatch.setattr(task_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime_module, "create_chat_model", lambda _settings: LeadTaskModel())
    runtime = AgentRuntime(settings)

    result = runtime.run("delegate three tasks", user_id="u", thread_id="parallel-task-smoke")
    runtime.memory_queue.flush()

    assert result.answer == "FINAL RECEIVED 3 TASK RESULTS"
