import json
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.agent import AgentRuntime
from superassist.agent.runtime import SYSTEM_PROMPT
from superassist.agent.short_memory import (
    build_summary_prompt,
    load_short_memory,
    maybe_compress_short_memory,
    read_jsonl,
    turn_records,
    write_jsonl,
)
from superassist.config import PROJECT_ROOT, Settings
from superassist.llm import FallbackChatModel, MiniMaxCompatibleChatModel
from superassist.memory.service import project_memory_recall, project_memory_write_context
from superassist.middlewares import (
    DynamicContextMiddleware,
    MemoryWriterMiddleware,
    ToolEventMiddleware,
)
from superassist.models import MemoryNode, MemoryRecall, NodeType


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
        _env_file=None,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )

    assert settings.short_memory_keep_recent_turns == 30
    assert settings.short_memory_token_limit == 80000
    assert settings.memory_llm_writer_enabled is True
    assert settings.memory_model == "deepseek-v4-flash"
    assert settings.feishu_domain == "https://open.feishu.cn"
    assert settings.feishu_mention_only is True


def test_dynamic_context_injects_runtime_section() -> None:
    middleware = DynamicContextMiddleware()
    base_messages = [SystemMessage(content="Base system"), HumanMessage(content="Hi")]
    state = {"user_id": "u", "thread_id": "t", "memory_recall": {}, "loaded_skills": []}

    captured: dict[str, object] = {}

    class Request:
        def __init__(self) -> None:
            self.state = state
            self.messages = base_messages

        def override(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return kwargs["messages"]

    middleware.wrap_model_call(Request(), lambda value: value)

    merged = captured["messages"]
    assert isinstance(merged[0], SystemMessage)
    assert merged[0].content == "Base system"
    assert isinstance(merged[-2], SystemMessage)
    assert isinstance(merged[-1], HumanMessage)
    assert merged[-1].content == "Hi"
    assert "current_time_utc:" not in str(merged[-2].content)
    assert "<TurnContext>" in str(merged[-2].content)
    assert "<RuntimeContext>" in str(merged[-2].content)
    assert '<LongTermMemory format="json">' in str(merged[-2].content)


def test_dynamic_context_preserves_legacy_system_order_when_cache_mode_is_disabled() -> None:
    middleware = DynamicContextMiddleware(preserve_static_prefix=False)
    base_messages = [SystemMessage(content="Base system"), HumanMessage(content="Hi")]

    class Request:
        state = {"user_id": "u", "thread_id": "t", "memory_recall": {}}
        messages = base_messages

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)

    assert len(merged) == 2
    assert isinstance(merged[0], SystemMessage)
    assert str(merged[0].content).startswith("Base system\n\n<TurnContext>")


def test_dynamic_context_keeps_tool_continuation_after_current_user() -> None:
    middleware = DynamicContextMiddleware()
    messages = [
        SystemMessage(content="Base system"),
        SystemMessage(content="<ShortMemory>summary</ShortMemory>"),
        HumanMessage(content="current question"),
        AIMessage(content="", tool_calls=[{"name": "echo", "args": {}, "id": "call_1"}]),
        ToolMessage(content="tool result", tool_call_id="call_1", name="echo"),
    ]

    class Request:
        state = {"user_id": "u", "thread_id": "t", "memory_recall": {}}

        def __init__(self) -> None:
            self.messages = messages

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)

    assert "<TurnContext>" in str(merged[2].content)
    assert merged[3].content == "current question"
    assert isinstance(merged[-1], ToolMessage)
    assert merged[-1].content == "tool result"


def test_dynamic_context_keeps_multimodal_image_during_skill_tool_continuation() -> None:
    middleware = DynamicContextMiddleware()
    image_url = "data:image/png;base64,aW1hZ2U="
    multimodal = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "请分析图片"},
        ]
    )
    messages = [
        SystemMessage(content="Base system"),
        multimodal,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "/mnt/skills/public/example/SKILL.md"},
                    "id": "call_skill",
                }
            ],
        ),
        ToolMessage(content="# Skill", tool_call_id="call_skill", name="read_file"),
    ]

    class Request:
        state = {"user_id": "u", "thread_id": "t", "memory_recall": {}}

        def __init__(self) -> None:
            self.messages = messages

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)

    retained = next(message for message in merged if isinstance(message, HumanMessage))
    assert retained.content == multimodal.content
    assert retained.content[0]["image_url"]["url"] == image_url
    assert isinstance(merged[-1], ToolMessage)


def test_initial_prompt_order_ends_with_current_user() -> None:
    middleware = DynamicContextMiddleware(clock=lambda: 0.0)
    messages = [
        SystemMessage(content="stable system"),
        SystemMessage(content="<ShortMemory>summary</ShortMemory>"),
        HumanMessage(content="older question"),
        AIMessage(content="older answer"),
        HumanMessage(content="current question"),
    ]

    class Request:
        state = {"user_id": "u", "thread_id": "t", "memory_recall": {}}

        def __init__(self) -> None:
            self.messages = messages

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)

    assert [type(message) for message in merged] == [
        SystemMessage,
        SystemMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        HumanMessage,
    ]
    assert "<TurnContext>" in str(merged[-2].content)
    assert merged[-1].content == "current question"


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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_ENABLE_TOOLS=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    assert runtime.settings.enable_tools is False


def test_runtime_preloads_embedder(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    assert runtime.memory.embed("warm")


def test_runtime_runs_in_fallback_mode(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, run_event_reporter=events.append)

    runtime.run("hello", user_id="u", thread_id="t")

    assert [event.type for event in events] == ["preparing_context"]
    assert all(event.metadata["thread_id"] == "t" for event in events)


def test_runtime_forwards_agent_text_tool_call_events(tmp_path) -> None:
    events = []
    tool_events = []
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_ENABLE_TOOLS=False,
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings, run_event_reporter=events.append)

    runtime.run_streaming("hello", user_id="u", thread_id="t")

    types = [event.type for event in events]
    assert types[0] == "preparing_context"
    assert types[1] == "thinking"
    assert events[1].message == "Thinking..."


def test_runtime_multimodal_content_keeps_text_input_for_memory(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_ENABLE_TOOLS=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    content = [
        {"type": "text", "text": "What is shown?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
    ]

    state = runtime._initial_state(
        "memory-safe text",
        user_id="u",
        thread_id="t",
        message_content=content,
    )

    assert state["input"] == "memory-safe text"
    rendered = state["messages"][-1].content
    assert rendered[1] == content[1]
    assert rendered[0]["text"].startswith("What is shown?")
    assert f"[系统时间: {state['message_created_at']}]" in rendered[0]["text"]
    assert content[0]["text"] == "What is shown?"


def test_runtime_current_user_message_has_stable_timestamp_and_raw_input(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_ENABLE_TOOLS=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    state = runtime._initial_state("raw question", user_id="u", thread_id="t")

    assert state["input"] == "raw question"
    assert state["messages"][-1].content == (
        f"raw question\n\n[系统时间: {state['message_created_at']}]"
    )
    datetime.fromisoformat(state["message_created_at"])


def test_runtime_accumulates_stream_text_and_ignores_tool_calls(tmp_path) -> None:
    from superassist.agent.streaming import accumulate_stream_text

    buffers: dict[str, str] = {}

    first, message_id = accumulate_stream_text(
        buffers,
        None,
        (AIMessage(content="我查到", id="msg_1"), {}),
    )
    second, message_id = accumulate_stream_text(
        buffers,
        message_id,
        (AIMessage(content="一些线索。", id="msg_1"), {}),
    )
    tool_text, message_id = accumulate_stream_text(
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
    tool_result_text, message_id = accumulate_stream_text(
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
    from superassist.agent.streaming import accumulate_stream_text

    text, message_id = accumulate_stream_text(
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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
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
    from superassist.agent.streaming import accumulate_stream_text

    text, message_id = accumulate_stream_text(
        {},
        None,
        (AIMessageChunk(content="chunk", id="chunk_1"), {}),
    )

    assert text == "chunk"
    assert message_id == "chunk_1"


def test_runtime_stream_parts_separates_structured_reasoning() -> None:
    from superassist.agent.streaming import StreamParts, accumulate_stream_parts

    buffers: dict[str, StreamParts] = {}
    reasoning, answer, message_id = accumulate_stream_parts(
        buffers,
        None,
        AIMessageChunk(
            content="最终答案",
            id="reasoning_1",
            additional_kwargs={"reasoning_content": "先分析问题"},
        ),
    )

    assert reasoning == "先分析问题"
    assert answer == "最终答案"
    assert message_id == "reasoning_1"


def test_runtime_stream_parts_reads_responses_reasoning_summary_blocks() -> None:
    from superassist.agent.streaming import accumulate_stream_parts

    reasoning, answer, message_id = accumulate_stream_parts(
        {},
        None,
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "先检查约束"}],
                }
            ],
            id="reasoning_response_1",
        ),
    )

    assert reasoning == "先检查约束"
    assert answer is None
    assert message_id == "reasoning_response_1"


def test_runtime_stream_parts_handles_split_inline_think_tags() -> None:
    from superassist.agent.streaming import StreamParts, accumulate_stream_parts

    buffers: dict[str, StreamParts] = {}
    reasoning, answer, message_id = accumulate_stream_parts(
        buffers, None, AIMessageChunk(content="<thi", id="reasoning_2")
    )
    assert reasoning is None
    assert answer is None

    reasoning, answer, message_id = accumulate_stream_parts(
        buffers, message_id, AIMessageChunk(content="nk>分析中</thi", id="reasoning_2")
    )
    assert reasoning == "分析中"
    assert answer is None

    reasoning, answer, _message_id = accumulate_stream_parts(
        buffers, message_id, AIMessageChunk(content="nk>这是正文", id="reasoning_2")
    )
    assert reasoning == "分析中"
    assert answer == "这是正文"


def test_runtime_last_ai_text_removes_inline_reasoning() -> None:
    from superassist.agent.runtime import _last_ai_text

    assert _last_ai_text([AIMessage(content="<think>内部推理</think>公开答案")]) == "公开答案"


def test_runtime_last_ai_text_extracts_responses_text_blocks() -> None:
    from superassist.agent.runtime import _last_ai_text

    message = AIMessage(
        content=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "内部摘要"}]},
            {"type": "text", "text": "公开答案", "phase": "final_answer"},
        ]
    )

    assert _last_ai_text([message]) == "公开答案"


def test_runtime_last_ai_text_skips_empty_tool_call_messages(tmp_path) -> None:
    from superassist.agent.runtime import _last_ai_text

    messages = [
        AIMessage(content="visible answer"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}]),
    ]

    assert _last_ai_text(messages) == "visible answer"


def test_runtime_persists_only_user_and_final_assistant_in_short_memory(tmp_path) -> None:

    from langchain_core.messages import AIMessage as _AIMessage

    from superassist.middlewares.short_memory_middleware import ShortMemoryMiddleware

    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    middleware = ShortMemoryMiddleware(settings, runtime.model)
    state = {
        "user_id": "user-owner",
        "thread_id": "t",
        "input": "search this",
        "message_created_at": "2026-08-01T08:00:00+00:00",
        "messages": [
            _AIMessage(
                content=(
                    "done\n\n"
                    "<ImageDescription>A geometry diagram with a labeled right triangle.</ImageDescription>"
                )
            )
        ],
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

    middleware.after_agent(state, runtime=None)
    records = read_jsonl(tmp_path / "threads" / "t" / "messages.jsonl")

    assert [record["role"] for record in records] == ["user", "assistant"]
    assert records[0]["created_at"] == "2026-08-01T08:00:00+00:00"
    assert "<ImageDescription>" in records[1]["content"]
    assert "very long result" not in str(records)
    assert "data:image" not in str(records)
    assert "web_search" not in str(records)
    metadata = json.loads((tmp_path / "threads" / "t" / "thread_meta.json").read_text(encoding="utf-8"))
    assert metadata["user_id"] == "user-owner"


def test_memory_writer_queue_receives_only_compact_tool_completion_fields() -> None:
    class Queue:
        def __init__(self) -> None:
            self.payload = None

        def add(self, payload) -> None:
            self.payload = payload

    queue = Queue()
    middleware = MemoryWriterMiddleware(queue)
    middleware.after_agent(
        {
            "user_id": "u",
            "thread_id": "t",
            "memory_event_id": "event_1",
            "input": "fetch it",
            "message_created_at": "2026-08-01T08:00:00+00:00",
            "messages": [AIMessage(content="finished")],
            "tool_events": [
                {"type": "tool_start", "tool": "web_fetch", "args": {"url": "secret"}},
                {
                    "type": "tool_result",
                    "tool": "web_fetch",
                    "args": {"url": "secret"},
                    "status": "error",
                    "content": "provider failed",
                },
            ],
            "memory_write_context": {},
        },
        runtime=None,
    )

    assert queue.payload is not None
    assert queue.payload.user_message_created_at == "2026-08-01T08:00:00+00:00"
    assert datetime.fromisoformat(queue.payload.assistant_message_created_at)
    assert queue.payload.tool_events == [
        {"name": "web_fetch", "status": "error", "error_summary": "provider failed"}
    ]


def test_runtime_persists_only_unexpired_skill_activations(tmp_path) -> None:
    import time

    from langchain_core.messages import AIMessage as _AIMessage
    from superassist.middlewares.short_memory_middleware import ShortMemoryMiddleware

    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS=300,
    )
    runtime = AgentRuntime(settings)
    middleware = ShortMemoryMiddleware(settings, runtime.model)
    activated_at = time.time()
    state = {
        "user_id": "u",
        "thread_id": "t",
        "input": "research this",
        "messages": [_AIMessage(content="done")],
        "tool_events": [],
        "loaded_skills": ["deep-research"],
        "skill_activations": {
            "deep-research": activated_at,
            "gongkao-huasheng13": activated_at - 301,
        },
        "metadata": {},
    }

    middleware.after_agent(state, runtime=None)
    metadata = json.loads((tmp_path / "threads" / "t" / "thread_meta.json").read_text(encoding="utf-8"))
    restored = runtime._initial_state("next", user_id="u", thread_id="t")

    assert metadata["loaded_skills"] == ["deep-research"]
    assert set(metadata["skill_activations"]) == {"deep-research"}
    assert restored["loaded_skills"] == ["deep-research"]


def test_short_memory_compression_waits_until_turn_limit(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(29):
        records.extend(
            turn_records(
                user_message=f"user {index}",
                assistant_answer=f"assistant {index}",
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)
    before_limit = maybe_compress_short_memory(
        messages_path=path,
        metadata={},
        model=FallbackChatModel(),
        token_limit=80000,
        keep_recent_turns=30,
        summary_target_tokens=6000,
        loaded_skills=[],
    )
    append_records = turn_records(
        user_message="user 29",
        assistant_answer="assistant 29",
        tool_events=[],
        include_tool_events=False,
    )
    write_jsonl(path, [*records, *append_records])
    at_limit = maybe_compress_short_memory(
        messages_path=path,
        metadata={},
        model=FallbackChatModel(),
        token_limit=80000,
        keep_recent_turns=30,
        summary_target_tokens=6000,
        loaded_skills=[],
    )

    assert before_limit == {}
    assert at_limit["short_memory_compressed"] is True
    assert at_limit["short_memory_compaction_trigger"] == "turns"
    assert at_limit["short_memory_compacted_records"] == 60


def test_short_memory_compression_triggers_at_token_limit(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = turn_records(
        user_message="large input " + ("x" * 1000),
        assistant_answer="large answer " + ("y" * 1000),
        tool_events=[],
        include_tool_events=False,
    )
    write_jsonl(path, records)

    update = maybe_compress_short_memory(
        messages_path=path,
        metadata={},
        model=FallbackChatModel(),
        token_limit=10,
        keep_recent_turns=30,
        summary_target_tokens=100,
        loaded_skills=[],
    )

    assert update["short_memory_compressed"] is True
    assert update["short_memory_compaction_trigger"] == "tokens"
    assert update["short_memory_compacted_records"] == len(records)
    assert read_jsonl(path) == records


def test_short_memory_loads_complete_active_segment_after_checkpoint(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(35):
        records.extend(
            turn_records(
                user_message=f"user {index}",
                assistant_answer=f"assistant {index}",
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)

    loaded = load_short_memory(path, {}, keep_recent_turns=30)
    checkpointed = load_short_memory(
        path,
        {"summary": "compressed first 30 turns", "short_memory_compacted_records": 60},
        keep_recent_turns=30,
    )

    assert loaded.summary == ""
    assert str(loaded.messages[0].content).startswith("user 0\n\n[系统时间: ")
    assert loaded.messages[-1].content == "assistant 34"
    assert checkpointed.summary == "compressed first 30 turns"
    assert str(checkpointed.messages[0].content).startswith("user 30\n\n[系统时间: ")
    assert checkpointed.messages[-1].content == "assistant 34"
    assert len([record for record in read_jsonl(path) if record["role"] == "user"]) == 35


def test_short_memory_replays_user_timestamp_and_compressor_sees_it(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = turn_records(
        user_message="time-sensitive question",
        assistant_answer="answer",
        tool_events=[],
        include_tool_events=False,
        user_created_at="2026-08-01T08:00:00+00:00",
        assistant_created_at="2026-08-01T08:01:00+00:00",
    )
    write_jsonl(path, records)

    loaded = load_short_memory(path, {}, keep_recent_turns=30)
    summary_prompt = build_summary_prompt(
        previous_summary="",
        records=records,
        summary_target_tokens=100,
        loaded_skills=[],
    )

    expected = "time-sensitive question\n\n[系统时间: 2026-08-01T08:00:00+00:00]"
    assert loaded.messages[0].content == expected
    assert expected in summary_prompt


def test_short_memory_load_does_not_silently_trim_at_token_limit(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    records = []
    for index in range(35):
        records.extend(
            turn_records(
                user_message=f"user {index}",
                assistant_answer=f"assistant {index} " + ("x" * 160),
                tool_events=[],
                include_tool_events=True,
            )
        )
    write_jsonl(path, records)
    loaded = load_short_memory(
        path,
        {},
        keep_recent_turns=30,
        token_limit=1,
    )

    assert loaded.records == records
    assert len(read_jsonl(path)) == len(records)


def test_model_memory_projection_uses_explicit_field_allowlist() -> None:
    created_at = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
    updated_at = datetime(2026, 7, 30, 8, 0, tzinfo=UTC)
    node = MemoryNode(
        id="concept_1",
        user_id="feishu:user_1",
        type=NodeType.CONCEPT,
        title="Preferred answer style",
        description="The user prefers concise answers.",
        importance=0.9,
        access_count=12,
        embedding=[0.1, 0.2, 0.3],
        reasoning="Created because the user stated a preference.",
        grounded_in=["event_1"],
        metadata={"source": "test"},
        created_at=created_at,
        updated_at=updated_at,
    )

    projected = project_memory_recall(MemoryRecall(immediate=[node]))

    assert projected["immediate"] == [
        {
            "tier": "immediate",
            "id": "concept_1",
            "type": "concept",
            "title": "Preferred answer style",
            "description": "The user prefers concise answers.",
            "user_id": "feishu:user_1",
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }
    ]
    serialized = json.dumps(projected)
    assert "embedding" not in serialized
    assert "reasoning" not in serialized
    assert "importance" not in serialized
    assert "grounded_in" not in serialized

    writer_projection = project_memory_write_context(MemoryRecall(immediate=[node]))
    assert writer_projection["immediate"] == [
        {
            "tier": "immediate",
            "id": "concept_1",
            "type": "concept",
            "title": "Preferred answer style",
            "description": "The user prefers concise answers.",
            "user_id": "feishu:user_1",
            "importance": 0.9,
            "grounded_in": ["event_1"],
            "source": "test",
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }
    ]
    writer_serialized = json.dumps(writer_projection)
    assert "embedding" not in writer_serialized
    assert "access_count" not in writer_serialized
    assert "reasoning" not in writer_serialized
    assert "thread_id" not in writer_serialized


def test_short_memory_compression_writes_checkpoint_without_pruning_jsonl(tmp_path) -> None:
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
    assert remaining == records
    assert update["short_memory_compacted_records"] == len(records)
    reloaded = load_short_memory(path, update, keep_recent_turns=10)
    assert reloaded.summary == update["summary"]
    assert reloaded.records == []


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
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
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
    records = read_jsonl(tmp_path / "threads" / "t" / "messages.jsonl")
    assert records[0]["created_at"] == captured[0].user_message_created_at
    assert records[1]["created_at"] == captured[0].assistant_message_created_at


def test_runtime_returns_model_error_without_crashing(tmp_path, monkeypatch) -> None:
    class RefusingModel(FallbackChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("output new_sensitive (1027)")

    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.agent.factory.create_chat_model", lambda settings: RefusingModel())
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

    monkeypatch.setattr("superassist.agent.factory.create_chat_model", lambda settings: LocalMiniMax())
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_MODEL="MiniMax-M2.7",
        SUPERASSIST_BASE_URL="https://api.minimaxi.com/v1",
        SUPERASSIST_ENABLE_TOOLS=True,
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)

    result = runtime.run("你好", user_id="u", thread_id="t")

    assert "fallback mode" in result.answer
    assert result.metadata["tool_calling_enabled"] is True
    assert result.metadata["tool_schema_binding"] == "openai_compatible_minimax"
