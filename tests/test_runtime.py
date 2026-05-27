from superassist_plus.agent import AgentRuntime
from superassist_plus.agent.runtime import SYSTEM_PROMPT
from superassist_plus.agent.short_memory import (
    maybe_compress_short_memory,
    read_jsonl,
    split_records_for_compression,
    turn_records,
    write_jsonl,
)
from superassist_plus.agent.middleware import (
    DynamicContextMiddleware,
    MemoryAfterAgentMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolEventMiddleware,
    SubagentLimitMiddleware,
    build_middlewares,
    _merge_runtime_context,
)
from superassist_plus.config import Settings
from superassist_plus.config import PROJECT_ROOT
from superassist_plus.llm import FallbackChatModel, MiniMaxCompatibleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest


def test_middleware_chain_order_is_explicit() -> None:
    chain = build_middlewares()

    assert [type(middleware) for middleware in chain] == [
        DynamicContextMiddleware,
        ToolErrorMiddleware,
        ToolCallLimitMiddleware,
        SubagentLimitMiddleware,
        ToolEventMiddleware,
        MemoryAfterAgentMiddleware,
    ]


def test_project_root_env_file_is_configured() -> None:
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_system_prompt_uses_human_progress_notes_not_raw_tool_names() -> None:
    assert "<tool_use>" in SYSTEM_PROMPT
    assert "Progress notes should summarize" in SYSTEM_PROMPT
    assert "what the previous tool result showed" in SYSTEM_PROMPT
    assert "Before each tool or `task` call" in SYSTEM_PROMPT
    assert "After tools or subagents return" in SYSTEM_PROMPT
    assert "<citations>" in SYSTEM_PROMPT


def test_short_memory_defaults_are_configured() -> None:
    settings = Settings(
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )

    assert settings.short_memory_token_limit == 80000
    assert settings.short_memory_keep_recent_turns == 10
    assert settings.short_memory_summary_target_tokens == 6000
    assert settings.short_memory_enable_tool_events is True
    assert settings.feishu_domain == "https://open.feishu.cn"
    assert settings.feishu_mention_only is True


def test_dynamic_context_merges_with_existing_system_message() -> None:
    messages = [SystemMessage(content="Base system"), HumanMessage(content="Hi")]

    merged = _merge_runtime_context(messages, "Runtime context")

    assert len([message for message in merged if isinstance(message, SystemMessage)]) == 1
    assert "Base system" in str(merged[0].content)
    assert "Runtime context" in str(merged[0].content)


def test_dynamic_context_injects_current_time() -> None:
    middleware = DynamicContextMiddleware("Base")
    messages = [HumanMessage(content="Hi")]
    request_state = {
        "user_id": "u",
        "thread_id": "t",
        "memory_recall": {},
    }

    class Request:
        def override(self, **kwargs):
            return kwargs["messages"]

    request = Request()
    request.state = request_state
    request.messages = messages
    merged = middleware.wrap_model_call(request, lambda value: value)

    assert "current_time_utc:" in str(merged[0].content)


def test_middleware_accepts_system_prompt() -> None:
    chain = build_middlewares("Lead prompt")

    assert isinstance(chain[0], DynamicContextMiddleware)
    assert chain[0].system_prompt == "Lead prompt"


def test_tool_event_middleware_reports_start_and_result() -> None:
    reported = []
    middleware = ToolEventMiddleware(reported.append)

    class DummyTool:
        name = "echo"

    request = ToolCallRequest(
        tool_call={"name": "echo", "id": "call_1", "args": {"text": "hi"}},
        tool=DummyTool(),
        state={"tool_events": []},
        runtime=None,
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _request: ToolMessage(content="hi", tool_call_id="call_1", name="echo"),
    )

    assert result.content == "hi"
    assert [event["type"] for event in reported] == ["tool_start", "tool_result"]
    assert reported[0]["args"] == {"text": "hi"}
    assert reported[1]["args"] == {"text": "hi"}


def test_tool_event_middleware_reports_agent_tool_call_content() -> None:
    reported = []
    middleware = ToolEventMiddleware(reported.append)

    class Request:
        pass

    request = Request()
    response_message = AIMessage(
        content="I will read the file first.",
        tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1"}],
    )

    response = middleware.wrap_model_call(request, lambda _request: type("Response", (), {"result": [response_message]})())

    assert response.result == [response_message]
    assert reported == [
        {
            "type": "agent_tool_call",
            "content": "I will read the file first.",
            "tool_calls": [{"name": "read_file", "args": {"path": "README.md"}}],
        }
    ]


def test_tool_event_middleware_does_not_invent_agent_tool_call_content_when_missing() -> None:
    reported = []
    middleware = ToolEventMiddleware(reported.append)

    class Request:
        pass

    request = Request()
    response_message = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
    )

    middleware.wrap_model_call(request, lambda _request: type("Response", (), {"result": [response_message]})())

    assert response_message.content == ""
    assert reported[0]["content"] == ""


def test_tools_are_disabled_by_default(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_ENABLE_TOOLS=False,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    assert runtime.settings.enable_tools is False


def test_runtime_preloads_embedder(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    assert runtime.memory.embed("warm")


def test_runtime_runs_in_fallback_mode(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    result = runtime.run("Remember that I like concise answers.", user_id="u", thread_id="t")
    runtime.memory_queue.flush()

    assert result.thread_id == "t"
    assert "fallback mode" in result.answer
    assert result.metadata["dynamic_context_injected"] is True
    assert result.metadata["memory_ready"] is True
    assert (tmp_path / "threads" / "t" / "messages.jsonl").exists()


def test_runtime_loads_thread_history_on_followup(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    first = runtime.run("First message", user_id="u", thread_id="same-thread")
    second = runtime.run("Second message", user_id="u", thread_id="same-thread")
    runtime.memory_queue.flush()

    assert first.metadata["history_loaded"] is False
    assert second.metadata["history_loaded"] is True
    assert second.metadata["history_message_count"] == 2


def test_runtime_reports_coarse_run_events(tmp_path) -> None:
    events = []
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, run_event_reporter=events.append)

    runtime.run("hello", user_id="u", thread_id="t")

    assert [event.type for event in events] == ["preparing_context"]
    assert all(event.metadata["thread_id"] == "t" for event in events)


def test_runtime_forwards_agent_text_tool_call_events(tmp_path) -> None:
    events = []
    tool_events = []
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, tool_event_reporter=tool_events.append, run_event_reporter=events.append)

    runtime._report_tool_event({"type": "agent_tool_call", "content": "I will inspect the file.", "tool_calls": []})
    runtime._report_tool_event({"type": "tool_start", "tool": "read_file", "args": {"path": "README.md"}})
    runtime._report_tool_event({"type": "agent_tool_call", "content": "", "tool_calls": []})

    assert [event.type for event in events] == ["agent_text"]
    assert events[0].message == "I will inspect the file."
    assert [event["type"] for event in tool_events] == ["agent_tool_call", "tool_start", "agent_tool_call"]


def test_runtime_streaming_reports_thinking_after_context(tmp_path) -> None:
    events = []
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_ENABLE_TOOLS=False,
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, run_event_reporter=events.append)

    runtime.run_streaming("hello", user_id="u", thread_id="t")

    assert [event.type for event in events] == ["preparing_context", "thinking"]
    assert events[1].message == "Thinking..."


def test_runtime_accumulates_stream_text_and_ignores_tool_calls(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    buffers: dict[str, str] = {}

    first, message_id = runtime._accumulate_stream_text(
        buffers,
        None,
        (AIMessage(content="我查到", id="msg_1"), {}),
    )
    second, message_id = runtime._accumulate_stream_text(
        buffers,
        message_id,
        (AIMessage(content="一些线索。", id="msg_1"), {}),
    )
    tool_text, message_id = runtime._accumulate_stream_text(
        buffers,
        message_id,
        (
            AIMessage(
                content="",
                id="msg_2",
                tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
            ),
            {},
        ),
    )
    tool_result_text, message_id = runtime._accumulate_stream_text(
        buffers,
        message_id,
        (ToolMessage(content="工具返回的大段结果", tool_call_id="call_1", name="web_search"), {}),
    )

    assert first == "我查到"
    assert second == "我查到一些线索。"
    assert tool_text is None
    assert tool_result_text is None
    assert message_id == "msg_1"


def test_runtime_accumulates_text_even_when_ai_chunk_has_tool_calls(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    text, message_id = runtime._accumulate_stream_text(
        {},
        None,
        (
            AIMessageChunk(
                content="我先查两个方向。",
                id="msg_tool",
                tool_call_chunks=[{"name": "task", "args": "", "id": "call_1", "index": 0}],
            ),
            {},
        ),
    )

    assert text == "我先查两个方向。"
    assert message_id == "msg_tool"


def test_runtime_reports_agent_text_once_across_stream_and_tool_event(tmp_path) -> None:
    events = []
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, run_event_reporter=events.append)
    runtime._active_agent_text_seen = set()

    runtime._report_agent_text("我会并发派三个子任务。", thread_id="t")
    runtime._report_tool_event({"type": "agent_tool_call", "content": "我会并发派三个子任务。", "tool_calls": []})
    runtime._report_agent_text("我会并发派三个子任务。稍后整合结果。", thread_id="t")

    assert [event.message for event in events] == [
        "我会并发派三个子任务。",
        "我会并发派三个子任务。稍后整合结果。",
    ]


def test_runtime_accumulates_ai_message_chunks(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    text, message_id = runtime._accumulate_stream_text(
        {},
        None,
        (AIMessageChunk(content="chunk", id="chunk_1"), {}),
    )

    assert text == "chunk"
    assert message_id == "chunk_1"


def test_runtime_last_ai_text_skips_empty_tool_call_messages(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    messages = [
        AIMessage(content="visible answer"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}]),
    ]

    assert runtime._last_ai_text(messages) == "visible answer"


def test_runtime_persists_compact_tool_events_in_short_memory(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    state = {
        "thread_id": "t",
        "input": "search this",
        "answer": "done",
        "tool_events": [
            {"type": "tool_start", "tool": "web_search", "args": {"query": "x"}},
            {
                "type": "tool_result",
                "tool": "web_search",
                "args": {"query": "x"},
                "status": "success",
                "content": "very long result that should not be persisted",
            },
        ],
        "loaded_skills": [],
        "metadata": {},
    }

    runtime._persist_turn(state)
    records = read_jsonl(tmp_path / "threads" / "t" / "messages.jsonl")

    assert [record["role"] for record in records] == ["user", "tool_event", "assistant"]
    assert records[1]["tool"] == "web_search"
    assert records[1]["args"] == {"query": "x"}
    assert "very long result" not in str(records[1])


def test_short_memory_compression_keeps_recent_ten_turns(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(12):
        records.extend(
            turn_records(
                user_message=f"user {index}",
                assistant_answer=f"assistant {index}",
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)
    old_records, recent_records = split_records_for_compression(records, keep_recent_turns=10)

    assert old_records[0]["content"] == "user 0"
    assert old_records[-1]["content"] == "assistant 1"
    assert recent_records[0]["content"] == "user 2"
    assert recent_records[-1]["content"] == "assistant 11"


def test_short_memory_compression_writes_summary_and_prunes_old_records(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(12):
        records.extend(
            turn_records(
                user_message=f"old user {index} " + ("x" * 200),
                assistant_answer=f"old assistant {index} " + ("y" * 200),
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)

    update = maybe_compress_short_memory(
        messages_path=path,
        metadata={},
        model=FallbackChatModel(),
        token_limit=50,
        keep_recent_turns=10,
        summary_target_tokens=50,
        loaded_skills=["deep-research"],
    )

    assert update["short_memory_compressed"] is True
    assert "old user 0" in update["summary"]
    remaining = read_jsonl(path)
    assert len([record for record in remaining if record["role"] == "user"]) == 10
    assert remaining[0]["content"].startswith("old user 2")


def test_short_memory_compression_failure_does_not_prune(tmp_path) -> None:
    class BrokenModel(FallbackChatModel):
        def invoke(self, messages, config=None, **kwargs):
            raise RuntimeError("no summary")

    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(12):
        records.extend(
            turn_records(
                user_message=f"user {index} " + ("x" * 200),
                assistant_answer=f"assistant {index} " + ("y" * 200),
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)

    update = maybe_compress_short_memory(
        messages_path=path,
        metadata={},
        model=BrokenModel(),
        token_limit=50,
        keep_recent_turns=10,
        summary_target_tokens=50,
        loaded_skills=[],
    )

    assert "short_memory_compression_error" in update
    assert len(read_jsonl(path)) == len(records)


def test_runtime_sends_write_context_to_memory_writer(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    captured = []

    def capture(payload):
        captured.append(payload)

    runtime.memory_queue.add = capture

    runtime.run("Remember that I like concise answers.", user_id="u", thread_id="t")

    assert captured
    assert captured[0].memory_context is not None
    assert set(captured[0].memory_context) == {"immediate", "working", "background", "buffer"}


def test_runtime_returns_model_error_without_crashing(tmp_path, monkeypatch) -> None:
    class RefusingModel(FallbackChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("output new_sensitive (1027)")

    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist_plus.agent.runtime.create_chat_model", lambda settings: RefusingModel())
    runtime = AgentRuntime(settings)

    result = runtime.run("test sensitive provider refusal", user_id="u", thread_id="t")

    assert "模型服务拒绝" in result.answer
    assert result.metadata["model_error"] == "RuntimeError"


def test_minimax_tools_enabled_uses_compatibility_binding(tmp_path, monkeypatch) -> None:
    class LocalMiniMax(MiniMaxCompatibleChatModel):
        def __init__(self):
            super().__init__(
                model="MiniMax-M2.7",
                api_key="secret",
                base_url="https://api.minimaxi.com/v1",
                temperature=1.0,
            )
            object.__setattr__(self, "_fallback", FallbackChatModel())

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            assert "tools" in kwargs
            return self._fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    monkeypatch.setattr("superassist_plus.agent.runtime.create_chat_model", lambda settings: LocalMiniMax())
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="secret",
        SUPERASSIST_PLUS_MODEL="MiniMax-M2.7",
        SUPERASSIST_PLUS_BASE_URL="https://api.minimaxi.com/v1",
        SUPERASSIST_PLUS_ENABLE_TOOLS=True,
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    result = runtime.run("你好", user_id="u", thread_id="t")

    assert "fallback mode" in result.answer
    assert result.metadata["tool_calling_enabled"] is True
    assert result.metadata["tool_schema_binding"] == "openai_compatible_minimax"
