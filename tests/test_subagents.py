from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.errors import GraphRecursionError

from superassist.config import Settings
from superassist.agent.prompts import compose_system_prompt
from superassist.llm import FallbackChatModel
from superassist.subagents import (
    TASK_STORE,
    SubagentConfigError,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
)
import superassist.subagents.executor as executor_module
from superassist.channels.daily_quiz import make_daily_quiz_tool
from superassist.subagents.executor import SubagentExecutor, _filter_tools
from superassist.tools import default_tools
from superassist.ui.server import create_app


def test_directory_subagents_expose_expected_prompts_and_default() -> None:
    registry = SubagentRegistry(Settings(SUPERASSIST_EMBEDDING_PROVIDER="hash"))
    configs = {config.name: config for config in registry.list()}

    assert registry.default().name == "general-purpose"
    assert registry.names() == ["general-purpose", "research", "shenlun-quiz"]
    assert "Do not call the task tool" in configs["general-purpose"].system_prompt
    assert "Prioritize reliable primary or official sources" in configs["research"].system_prompt
    assert configs["shenlun-quiz"].model_profile == "memory"
    assert configs["shenlun-quiz"].allowed_tools == ["daily_quiz_update", "read_file"]
    assert "task.parameters.question_count" in configs["shenlun-quiz"].description
    assert "/mnt/skills/public/huasheng13/SKILL.md" in configs["shenlun-quiz"].system_prompt
    assert "中高难度" in configs["shenlun-quiz"].system_prompt
    assert "唯一答案审计" in configs["shenlun-quiz"].system_prompt
    assert "命题人审计" in configs["shenlun-quiz"].system_prompt
    assert "绝对化措辞审计结果" in configs["shenlun-quiz"].system_prompt
    assert "只改变一个核心维度" in configs["shenlun-quiz"].system_prompt
    assert "近几日日报" in registry.available_agents_text()


def test_custom_agent_descriptions_are_injected_without_full_prompts(tmp_path) -> None:
    agents_dir = tmp_path / "agents"
    _write_agent(
        agents_dir,
        "default-worker",
        description="Default worker from disk.",
        system_prompt="PRIVATE DEFAULT SYSTEM PROMPT",
        is_default=True,
    )
    _write_agent(
        agents_dir,
        "domain-worker",
        description="Domain worker discovered dynamically.",
        system_prompt="PRIVATE DOMAIN SYSTEM PROMPT",
    )
    settings = Settings(
        SUPERASSIST_AGENTS_DIR=agents_dir,
        SUPERASSIST_SUBAGENTS_ENABLED=True,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )

    prompt = compose_system_prompt(settings)

    assert "- default-worker: Default worker from disk." in prompt
    assert "- domain-worker: Domain worker discovered dynamically." in prompt
    assert "PRIVATE DEFAULT SYSTEM PROMPT" not in prompt
    assert "PRIVATE DOMAIN SYSTEM PROMPT" not in prompt


def test_registry_requires_exactly_one_directory_default(tmp_path) -> None:
    agents_dir = tmp_path / "agents"
    _write_agent(agents_dir, "worker", description="Worker", system_prompt="Prompt")

    with pytest.raises(SubagentConfigError, match="Exactly one enabled agent"):
        SubagentRegistry(
            Settings(SUPERASSIST_AGENTS_DIR=agents_dir, SUPERASSIST_EMBEDDING_PROVIDER="hash")
        )


def test_subagent_tool_filter_excludes_task() -> None:
    tools = _filter_tools(default_tools(include_task=True), allowed=None)

    assert "task" not in {tool.name for tool in tools}


def test_memory_model_profile_reuses_existing_deepseek_configuration(tmp_path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_MEMORY_MODEL="deepseek-v4-flash",
        SUPERASSIST_MEMORY_API_KEY="memory-secret",
        SUPERASSIST_MEMORY_BASE_URL="https://api.deepseek.com/v1",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    config = SubagentRegistry(settings).get("shenlun-quiz")
    captured = {}

    def create_model(resolved):
        captured["settings"] = resolved
        return FallbackChatModel()

    monkeypatch.setattr(executor_module, "create_chat_model", create_model)
    executor = SubagentExecutor(
        config=config,
        tools=[*default_tools(include_task=False), make_daily_quiz_tool(settings)],
        settings=settings,
    )

    resolved = captured["settings"]
    assert resolved.model == "deepseek-v4-flash"
    assert resolved.api_key == "memory-secret"
    assert resolved.base_url == "https://api.deepseek.com/v1"
    assert resolved.use_responses_api is True
    assert {tool.name for tool in executor.tools} == {"daily_quiz_update", "read_file"}


def test_subagent_executor_runs_fallback_agent(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_CLAUDE_FALLBACK_API_KEY="",
        SUPERASSIST_DEEPSEEK_FALLBACK_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    config = replace(
        SubagentRegistry(settings).get("general-purpose"),
        timeout_seconds=30,
        max_turns=10,
    )
    executor = SubagentExecutor(config=config, tools=default_tools(include_task=False), settings=settings)

    result = executor.run("Return a short confirmation.", description="smoke")

    assert result.status == SubagentStatus.COMPLETED
    assert "fallback mode" in result.result
    assert result.ai_messages


def test_subagent_executor_injects_structured_task_parameters(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    config = SubagentRegistry(settings).get("shenlun-quiz")
    executor = SubagentExecutor(
        config=config,
        tools=default_tools(include_task=False),
        settings=settings,
        task_parameters={"question_count": 5},
    )

    prepared = executor._prepare(
        {
            "prompt": "生成一套练习题。",
            "task_parameters": executor.task_parameters,
            "result": SubagentResult("task-params", "demo", "shenlun-quiz"),
        }
    )

    content = str(prepared["messages"][1].content)
    assert '<DelegatedTaskParameters format="json">' in content
    assert '"question_count":5' in content
    assert "生成一套练习题。" in content


def test_subagent_executor_reports_streamed_ai_text(tmp_path, monkeypatch) -> None:
    class StreamingAgent:
        def stream(self, state, config=None, stream_mode=None):
            yield ("messages", (AIMessageChunk(content="I am checking", id="sub_msg_1"), {}))
            yield ("values", {"messages": [*state["messages"], AIMessage(content="done")]})

    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    events = []
    config = replace(
        SubagentRegistry(settings).get("general-purpose"),
        timeout_seconds=30,
        max_turns=10,
    )
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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    config = replace(
        SubagentRegistry(settings).get("general-purpose"),
        timeout_seconds=30,
        max_turns=1,
    )
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
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
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


def _write_agent(
    root,
    name: str,
    *,
    description: str,
    system_prompt: str,
    is_default: bool = False,
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "agent.toml").write_text(
        "\n".join(
            [
                f'name = "{name}"',
                f'description = "{description}"',
                'prompt_file = "system.md"',
                f"default = {'true' if is_default else 'false'}",
                'allowed_tools = "*"',
            ]
        ),
        encoding="utf-8",
    )
    (directory / "system.md").write_text(system_prompt, encoding="utf-8")
