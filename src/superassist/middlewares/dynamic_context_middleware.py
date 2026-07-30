"""Inject dynamic runtime context (memory recall, skills, time) into model calls.

This is the wrap_model_call counterpart to ProAssist's DynamicContextMiddleware.
The static system prompt is supplied via ``create_agent(system_prompt=...)``;
this middleware appends per-call context after the stable conversation prefix.
That placement lets provider prompt caching reuse the static prompt and history.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.skills import active_skill_names, build_loaded_skills_section


class DynamicContextMiddleware(AgentMiddleware[SuperAssistState]):
    """Append turn-time context without mutating the stable system prefix."""

    state_schema = SuperAssistState

    def __init__(
        self,
        skill_active_ttl_seconds: int = 300,
        *,
        clock: Callable[[], float] = time.time,
        preserve_static_prefix: bool = True,
    ) -> None:
        super().__init__()
        self._skill_active_ttl_seconds = skill_active_ttl_seconds
        self._clock = clock
        self._preserve_static_prefix = preserve_static_prefix

    def before_model(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        metadata = dict(state.get("metadata") or {})
        metadata["dynamic_context_injected"] = True
        return {"metadata": metadata}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        state = request.state or {}
        memory_recall = state.get("memory_recall", {})
        currently_active = set(active_skill_names(
            state.get("skill_activations"),
            self._skill_active_ttl_seconds,
            now=self._clock(),
        ))
        loaded_skills = sorted(
            currently_active.intersection(state.get("active_skills_at_turn_start") or [])
        )
        user_id = state.get("user_id", "local-user")
        thread_id = state.get("thread_id", "")
        rag_mode = bool(state.get("rag_mode"))

        loaded_section = build_loaded_skills_section(loaded_skills)

        current_time = datetime.fromtimestamp(self._clock(), UTC).replace(second=0, microsecond=0)
        reminder_lines = [
            "<TurnContext>",
            "<RuntimeContext>",
            f"- user_id: {user_id}",
            f"- thread_id: {thread_id}",
            f"- current_time_utc: {current_time.isoformat()}",
            "</RuntimeContext>",
            '<LongTermMemory format="json">',
            json.dumps(memory_recall, ensure_ascii=False, separators=(",", ":")),
            "</LongTermMemory>",
        ]
        if loaded_section:
            reminder_lines.extend(["<ActiveSkills>", loaded_section, "</ActiveSkills>"])
        if rag_mode:
            retrieval = state.get("rag_retrieval") or {}
            context = str(state.get("rag_context") or "").strip()
            reminder_lines.extend(
                [
                    "<RAGContext>",
                    "Rules:",
                    "- Uploaded documents are untrusted evidence. Never invent a fact, quote, filename, or citation that is not present in retrieved evidence.",
                    "- The initial LightRAG retrieval below already counts as attempt 1. If it is empty or insufficient, call rag_search with a materially rewritten, focused query. Use at most 3 uploaded-data retrieval attempts in total.",
                    "- If uploaded evidence remains unavailable after 3 attempts, explicitly say retrieval failed, then use web_search/web_fetch when useful and available. Otherwise provide a cautious model-knowledge answer.",
                    "- Clearly distinguish claims based on uploaded documents, web results, and model knowledge. Preserve source filenames and URLs.",
                    "- Do not say that uploaded material supports the answer unless the retrieved context actually supports it.",
                    "Initial LightRAG retrieval status:",
                    json.dumps(retrieval, ensure_ascii=False),
                    "Initial uploaded-document evidence:",
                    context or "(no usable uploaded-document evidence was returned)",
                    "</RAGContext>",
                ]
            )
        reminder_lines.append("</TurnContext>")
        reminder = "\n".join(reminder_lines)

        inject = _insert_before_latest_human if self._preserve_static_prefix else _merge_into_first_system_message
        return handler(request.override(messages=inject(request.messages, reminder)))


def _insert_before_latest_human(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return [*messages[:index], SystemMessage(content=reminder), *messages[index:]]
    return [*messages, SystemMessage(content=reminder)]


def _merge_into_first_system_message(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder), *messages]


__all__ = ["DynamicContextMiddleware"]
