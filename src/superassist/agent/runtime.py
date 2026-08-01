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
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from superassist.agent.factory import AgentBundle, build_agent
from superassist.agent.prompts import SYSTEM_PROMPT
from superassist.agent.prompts import subagent_section as subagent_prompt_section
from superassist.agent.prompts import team_section as team_prompt_section
from superassist.agent.short_memory import format_short_memory_section, load_short_memory, timestamp_user_content
from superassist.agent.state import SuperAssistState
from superassist.agent.streaming import StreamParts, accumulate_stream_parts, clean_answer_text
from superassist.config import Settings, get_settings
from superassist.llm import is_minimax_model
from superassist.models import AgentRunEvent, AgentRunResult
from superassist.observability import runnable_trace_config, traceable, without_self
from superassist.rag.context import rag_turn_context
from superassist.rag.service import LightRAGService
from superassist.run_events import run_event_reporter_context
from superassist.skills import active_skill_activations, active_skill_names
from superassist.teams import set_team_supervisor
from superassist.teams.context import team_thread_context

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Convenience wrapper that drives one or more turns through the LangChain agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tool_event_reporter: Callable[[dict[str, Any]], None] | None = None,
        run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
        rag_mode: bool = False,
        rag_service: LightRAGService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag_mode = rag_mode
        self._run_event_reporter = run_event_reporter
        self._active_agent_text_seen: set[str] | None = None
        self._tool_event_reporter = tool_event_reporter
        self._bundle: AgentBundle = build_agent(
            self.settings,
            tool_event_reporter=self._report_tool_event,
            run_event_reporter=run_event_reporter,
            rag_mode=rag_mode,
            rag_service=rag_service,
        )

    @property
    def agent(self) -> Any:
        return self._bundle.agent

    @property
    def model(self) -> Any:
        return self._bundle.model

    @property
    def memory_model(self) -> Any:
        return self._bundle.memory_model

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
                with rag_turn_context(
                    self._bundle.rag_service,
                    user_id,
                    self.rag_mode,
                    self.settings.rag_max_attempts,
                ):
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

    def run_streaming(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
        message_content: str | list[dict[str, Any]] | None = None,
    ) -> AgentRunResult:
        return self._run_streaming_traced(
            message,
            user_id=user_id,
            thread_id=thread_id,
            message_content=message_content,
        )

    @traceable(name="superassist.turn.streaming", run_type="chain", process_inputs=without_self)
    def _run_streaming_traced(
        self,
        message: str,
        *,
        user_id: str,
        thread_id: str | None,
        message_content: str | list[dict[str, Any]] | None = None,
    ) -> AgentRunResult:
        state = self._initial_state(
            message,
            user_id=user_id,
            thread_id=thread_id,
            message_content=message_content,
        )
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
        stream_buffers: dict[str, StreamParts] = {}
        current_message_id: str | None = None
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}
        seen_usage: set[tuple[str, int, int, int]] = set()
        previous_seen = self._active_agent_text_seen
        self._active_agent_text_seen = set()
        try:
            with rag_turn_context(
                self._bundle.rag_service,
                user_id,
                self.rag_mode,
                self.settings.rag_max_attempts,
            ):
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
                            _accumulate_stream_usage(usage_totals, seen_usage, chunk)
                            reasoning, text, current_message_id = accumulate_stream_parts(
                                stream_buffers, current_message_id, chunk
                            )
                            if reasoning:
                                self._report_run_event("agent_reasoning", reasoning, thread_id=thread_id)
                            if text:
                                self._report_agent_text(text, thread_id=thread_id)
                            continue
                        if mode == "values" and isinstance(chunk, dict):
                            last_values = chunk
        finally:
            self._active_agent_text_seen = previous_seen
        result = last_values or dict(state)
        if usage_totals["input_tokens"]:
            usage_totals["cache_hit_rate"] = round(
                usage_totals["cache_read"] / usage_totals["input_tokens"],
                4,
            )
            metadata = dict(result.get("metadata") or {})
            metadata["model_usage"] = usage_totals
            result = {**result, "metadata": metadata}
            logger.info(
                "Model token usage thread_id=%s input=%d output=%d cache_read=%d cache_hit_rate=%.2f%%",
                thread_id,
                usage_totals["input_tokens"],
                usage_totals["output_tokens"],
                usage_totals["cache_read"],
                usage_totals["cache_hit_rate"] * 100,
            )
        return result

    # -- Helpers -----------------------------------------------------------

    def _initial_state(
        self,
        message: str,
        *,
        user_id: str,
        thread_id: str | None,
        message_content: str | list[dict[str, Any]] | None = None,
    ) -> SuperAssistState:
        resolved_thread_id = thread_id or f"thread_{uuid4().hex[:12]}"
        message_created_at = datetime.now(UTC).isoformat()
        thread_metadata = self._load_thread_metadata(resolved_thread_id)
        history = self._load_history(resolved_thread_id, thread_metadata)
        skill_activations = active_skill_activations(
            thread_metadata.get("skill_activations"),
            self.settings.skill_active_ttl_seconds,
        )
        loaded_skills = active_skill_names(
            skill_activations,
            self.settings.skill_active_ttl_seconds,
        )
        metadata: dict[str, Any] = {
            "history_loaded": bool(history.messages),
            "history_message_count": len(history.messages),
            "short_memory_summary_loaded": bool(history.summary),
            "loaded_skills": loaded_skills,
            "skill_activations": skill_activations,
            "active_skills_at_turn_start": loaded_skills,
            **self._tool_compatibility_metadata(),
        }
        current_content = message if message_content is None else message_content
        return {
            "messages": [
                SystemMessage(content=format_short_memory_section(history.summary)),
                *history.messages,
                HumanMessage(content=timestamp_user_content(current_content, message_created_at)),
            ],
            "input": message,
            "message_created_at": message_created_at,
            "user_id": user_id,
            "thread_id": resolved_thread_id,
            "loaded_skills": loaded_skills,
            "skill_activations": skill_activations,
            "tool_events": [],
            "rag_mode": self.rag_mode,
            "rag_context": "",
            "rag_sources": [],
            "metadata": metadata,
        }

    def _load_history(self, thread_id: str, metadata: dict[str, Any]) -> Any:
        path = self.settings.data_dir / "threads" / thread_id / "messages.jsonl"
        return load_short_memory(
            path,
            metadata,
            keep_recent_turns=self.settings.short_memory_keep_recent_turns,
            token_limit=self.settings.short_memory_token_limit,
        )

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
            text = clean_answer_text(message.content)
            if text:
                return text
    return ""


def _accumulate_stream_usage(
    totals: dict[str, Any],
    seen: set[tuple[str, int, int, int]],
    chunk: Any,
) -> None:
    message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    response_metadata = getattr(message, "response_metadata", None)
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
    usage_id = str(getattr(message, "id", "") or response_metadata.get("id") or id(message))
    fingerprint = (usage_id, input_tokens, output_tokens, total_tokens)
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    input_details = usage.get("input_token_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    totals["input_tokens"] += input_tokens
    totals["output_tokens"] += output_tokens
    totals["cache_read"] += int(input_details.get("cache_read") or 0)


__all__ = ["AgentRuntime", "SYSTEM_PROMPT", "subagent_prompt_section", "team_prompt_section"]
