from superassist_plus.agent import AgentRuntime
from superassist_plus.agent.middleware import (
    DynamicContextMiddleware,
    MemoryAfterAgentMiddleware,
    ToolErrorMiddleware,
    ToolEventMiddleware,
    build_middlewares,
    _merge_runtime_context,
)
from superassist_plus.config import Settings
from superassist_plus.config import PROJECT_ROOT
from superassist_plus.llm import FallbackChatModel, MiniMaxCompatibleChatModel
from langchain_core.messages import HumanMessage, SystemMessage


def test_middleware_chain_order_is_explicit() -> None:
    chain = build_middlewares()

    assert [type(middleware) for middleware in chain] == [
        DynamicContextMiddleware,
        ToolErrorMiddleware,
        ToolEventMiddleware,
        MemoryAfterAgentMiddleware,
    ]


def test_project_root_env_file_is_configured() -> None:
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_dynamic_context_merges_with_existing_system_message() -> None:
    messages = [SystemMessage(content="Base system"), HumanMessage(content="Hi")]

    merged = _merge_runtime_context(messages, "Runtime context")

    assert len([message for message in merged if isinstance(message, SystemMessage)]) == 1
    assert "Base system" in str(merged[0].content)
    assert "Runtime context" in str(merged[0].content)


def test_middleware_accepts_system_prompt() -> None:
    chain = build_middlewares("Lead prompt")

    assert isinstance(chain[0], DynamicContextMiddleware)
    assert chain[0].system_prompt == "Lead prompt"


def test_tools_are_disabled_by_default(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
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
