"""Public entry point that drives one agent turn.

The actual agent graph (model + tools + middleware chain) is built by
``factory.build_agent`` and consists of a single LangChain ``create_agent``
call — there is no outer LangGraph state graph. This runtime only:

* loads thread history from disk into the initial message list
* invokes ``agent.invoke`` (sync) or ``agent.stream`` (streaming)
* extracts the final answer + metadata for callers (CLI, channels, tests)
* propagates run_event/tool_event reporters through the agent's middleware chain
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from superassist.agent.factory import AgentBundle, build_agent
from superassist.agent.prompts import SYSTEM_PROMPT, subagent_section as subagent_prompt_section, team_section as team_prompt_section
from superassist.agent.short_memory import load_short_memory
from superassist.agent.streaming import accumulate_stream_text
from superassist.agent.state import SuperAssistState
from superassist.config import Settings, get_settings
from superassist.llm import is_minimax_model
from superassist.models import AgentRunEvent, AgentRunResult
from superassist.observability import runnable_trace_config, traceable, without_self
from superassist.run_events import run_event_reporter_context
from superassist.teams import set_team_supervisor
from superassist.teams.context import team_thread_context


class AgentRuntime:
    """Convenience wrapper that drives one or more turns through the LangChain agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tool_event_reporter: Callable[[dict[str, Any]], None] | None = None,
        run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._run_event_reporter = run_event_reporter
        self._active_agent_text_seen: set[str] | None = None
        self._tool_event_reporter = tool_event_reporter
        self._bundle: AgentBundle = build_agent(
            self.settings,
            tool_event_reporter=self._report_tool_event,
            run_event_reporter=run_event_reporter,
        )

    @property
    def agent(self) -> Any:
        return self._bundle.agent

    @property
    def model(self) -> Any:
        return self._bundle.model

    @property
    def memory(self) -> Any:
        return self._bundle.memory

    @property
    def memory_queue(self) -> Any:
        return self._bundle.memory_queue

    @property
    def team_supervisor(self) -> Any:
        return self._bundle.team_supervisor

    @property
    def team_config_error(self) -> str | None:
        return self._bundle.team_config_error

    def close(self) -> None:
        if self.team_supervisor is not None:
            self.team_supervisor.close()
            from superassist.teams import get_team_supervisor

            if get_team_supervisor() is self.team_supervisor:
                set_team_supervisor(None)

    def set_run_event_reporter(self, reporter: Callable[[AgentRunEvent], None] | None) -> None:
        self._run_event_reporter = reporter

    # -- Sync turn ---------------------------------------------------------

    def run(self, message: str, *, user_id: str = "local-user", thread_id: str | None = None) -> AgentRunResult:
        return self._run_traced(message, user_id=user_id, thread_id=thread_id)

    @traceable(name="superassist.turn", run_type="chain", process_inputs=without_self)
    def _run_traced(self, message: str, *, user_id: str, thread_id: str | None) -> AgentRunResult:
        state = self._initial_state(message, user_id=user_id, thread_id=thread_id)
        with run_event_reporter_context(self._run_event_reporter):
            self._report_run_event("preparing_context", "Preparing context...", thread_id=state["thread_id"])
            try:
                with team_thread_context(state["thread_id"]):
                    final_state = self.agent.invoke(
                        state,
                        runnable_trace_config(
                            run_name="superassist.lead_agent",
                            user_id=user_id,
                            thread_id=state["thread_id"],
                            tags=["agent", "lead"],
                            metadata={"streaming": False, **self._tool_compatibility_metadata()},
                        ),
                    )
            except Exception as exc:
                return self._error_result(state["thread_id"], exc)
        return self._result_from(state["thread_id"], final_state)

    # -- Streaming turn ----------------------------------------------------

    def run_streaming(self, message: str, *, user_id: str = "local-user", thread_id: str | None = None) -> AgentRunResult:
        return self._run_streaming_traced(message, user_id=user_id, thread_id=thread_id)

    @traceable(name="superassist.turn.streaming", run_type="chain", process_inputs=without_self)
    def _run_streaming_traced(self, message: str, *, user_id: str, thread_id: str | None) -> AgentRunResult:
        state = self._initial_state(message, user_id=user_id, thread_id=thread_id)
        with run_event_reporter_context(self._run_event_reporter):
            self._report_run_event("preparing_context", "Preparing context...", thread_id=state["thread_id"])
            self._report_run_event("thinking", "Thinking...", thread_id=state["thread_id"])
            try:
                final_state = self._stream_agent(state, user_id=user_id, thread_id=state["thread_id"])
            except Exception as exc:
                return self._error_result(state["thread_id"], exc)
        return self._result_from(state["thread_id"], final_state)

    def _stream_agent(self, state: SuperAssistState, *, user_id: str, thread_id: str) -> dict[str, Any]:
        last_values: dict[str, Any] | None = None
        text_buffers: dict[str, str] = {}
        current_message_id: str | None = None
        previous_seen = self._active_agent_text_seen
        self._active_agent_text_seen = set()
        try:
            with team_thread_context(thread_id):
                for item in self.agent.stream(
                    state,
                    runnable_trace_config(
                        run_name="superassist.lead_agent.streaming",
                        user_id=user_id,
                        thread_id=thread_id,
                        tags=["agent", "lead", "streaming"],
                        metadata={"streaming": True, **self._tool_compatibility_metadata()},
                    ),
                    stream_mode=["messages", "values"],
                ):
                    if isinstance(item, tuple) and len(item) == 2:
                        mode, chunk = str(item[0]), item[1]
                    else:
                        mode, chunk = "values", item
                    if mode == "messages":
                        text, current_message_id = accumulate_stream_text(text_buffers, current_message_id, chunk)
                        if text:
                            self._report_agent_text(text, thread_id=thread_id)
                        continue
                    if mode == "values" and isinstance(chunk, dict):
                        last_values = chunk
        finally:
            self._active_agent_text_seen = previous_seen
        return last_values or dict(state)

    # -- Helpers -----------------------------------------------------------

    def _initial_state(self, message: str, *, user_id: str, thread_id: str | None) -> SuperAssistState:
        resolved_thread_id = thread_id or f"thread_{uuid4().hex[:12]}"
        thread_metadata = self._load_thread_metadata(resolved_thread_id)
        history = self._load_history(resolved_thread_id, thread_metadata)
        loaded_skills = sorted({str(item) for item in (thread_metadata.get("loaded_skills") or []) if str(item)})
        metadata: dict[str, Any] = {
            "history_loaded": bool(history.messages),
            "history_message_count": len(history.messages),
            "short_memory_summary_loaded": bool(history.summary),
            "loaded_skills": loaded_skills,
            **self._tool_compatibility_metadata(),
        }
        return {
            "messages": [*history.messages, HumanMessage(content=message)],
            "input": message,
            "user_id": user_id,
            "thread_id": resolved_thread_id,
            "loaded_skills": loaded_skills,
            "tool_events": [],
            "metadata": metadata,
        }

    def _load_history(self, thread_id: str, metadata: dict[str, Any]) -> Any:
        path = self.settings.data_dir / "threads" / thread_id / "messages.jsonl"
        return load_short_memory(path, metadata, token_limit=self.settings.short_memory_token_limit)

    def _load_thread_metadata(self, thread_id: str) -> dict[str, Any]:
        path = self.settings.data_dir / "threads" / thread_id / "thread_meta.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _result_from(self, thread_id: str, final_state: dict[str, Any]) -> AgentRunResult:
        messages = list(final_state.get("messages") or [])
        answer = _last_ai_text(messages)
        metadata = dict(final_state.get("metadata") or {})
        if not metadata.get("final_assistant_text") and answer:
            metadata["final_assistant_text"] = answer
        metadata["loaded_skills"] = sorted(set(final_state.get("loaded_skills") or metadata.get("loaded_skills") or []))
        return AgentRunResult(thread_id=thread_id, answer=answer, metadata=metadata)

    @staticmethod
    def _error_result(thread_id: str, exc: Exception) -> AgentRunResult:
        message = (
            "模型服务拒绝或中断了这次回复，当前对话进程已保留。"
            "你可以换一种问法继续，或者减少敏感/过长的上下文后重试。"
        )
        return AgentRunResult(
            thread_id=thread_id,
            answer=message,
            metadata={
                "model_error": type(exc).__name__,
                "model_error_message": str(exc),
                "final_assistant_text": message,
            },
        )

    def _tool_compatibility_metadata(self) -> dict[str, Any]:
        if self.settings.enable_tools and is_minimax_model(self.settings.model, self.settings.base_url):
            return {"tool_calling_enabled": True, "tool_schema_binding": "openai_compatible_minimax"}
        return {"tool_calling_enabled": self.settings.enable_tools}

    def _report_run_event(self, event_type: str, message: str, **metadata: Any) -> None:
        if self._run_event_reporter is None:
            return
        self._run_event_reporter(AgentRunEvent(type=event_type, message=message, metadata=metadata))

    def _report_agent_text(self, content: str, **metadata: Any) -> None:
        text = content.strip()
        if not text:
            return
        seen = self._active_agent_text_seen
        if seen is not None:
            if text in seen or any(previous.startswith(text) for previous in seen):
                return
            seen.add(text)
        self._report_run_event("agent_text", text, **metadata)

    def _report_tool_event(self, event: dict[str, Any]) -> None:
        if self._tool_event_reporter is not None:
            self._tool_event_reporter(event)
        if event.get("type") != "agent_tool_call":
            return
        content = str(event.get("content") or "").strip()
        if content:
            self._report_agent_text(content)


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = str(message.content or "").strip()
            if text:
                return text
    return ""


__all__ = ["AgentRuntime", "SYSTEM_PROMPT", "subagent_prompt_section", "team_prompt_section"]
