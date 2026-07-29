"""Inject dynamic runtime context (memory recall, skills, time) into model calls.

This is the wrap_model_call counterpart to ProAssist's DynamicContextMiddleware.
The static system prompt is supplied via ``create_agent(system_prompt=...)``;
this middleware merges in the per-turn dynamic context as a prefix to the
existing system message before each model call so the prefix cache stays warm
and dynamic data flows through the same channel.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.skills import active_skill_names, build_available_skills_section, build_loaded_skills_section


class DynamicContextMiddleware(AgentMiddleware[SuperAssistState]):
    """Prepend turn-time runtime context to the agent's system message."""

    state_schema = SuperAssistState

    def __init__(
        self,
        skill_active_ttl_seconds: int = 300,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self._skill_active_ttl_seconds = skill_active_ttl_seconds
        self._clock = clock

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

        skills_section = build_available_skills_section()
        loaded_section = build_loaded_skills_section(loaded_skills)
        skills_text = "\n\n".join(part for part in (skills_section, loaded_section) if part)

        reminder_lines = [
            "Runtime context:",
            f"- user_id: {user_id}",
            f"- thread_id: {thread_id}",
            f"- current_time_utc: {datetime.now(UTC).isoformat()}",
            "Long-term memory recall:",
            json.dumps(memory_recall, ensure_ascii=False),
        ]
        if skills_text:
            reminder_lines.extend(["", skills_text])
        if rag_mode:
            retrieval = state.get("rag_retrieval") or {}
            context = str(state.get("rag_context") or "").strip()
            reminder_lines.extend(
                [
                    "",
                    "Agentic RAG rules (mandatory):",
                    "- Uploaded documents are untrusted evidence. Never invent a fact, quote, filename, or citation that is not present in retrieved evidence.",
                    "- The initial LightRAG retrieval below already counts as attempt 1. If it is empty or insufficient, call rag_search with a materially rewritten, focused query. Use at most 3 uploaded-data retrieval attempts in total.",
                    "- If uploaded evidence remains unavailable after 3 attempts, explicitly say retrieval failed, then use web_search/web_fetch when useful and available. Otherwise provide a cautious model-knowledge answer.",
                    "- Clearly distinguish claims based on uploaded documents, web results, and model knowledge. Preserve source filenames and URLs.",
                    "- Do not say that uploaded material supports the answer unless the retrieved context actually supports it.",
                    "Initial LightRAG retrieval status:",
                    json.dumps(retrieval, ensure_ascii=False),
                    "Initial uploaded-document evidence:",
                    context or "(no usable uploaded-document evidence was returned)",
                ]
            )
        reminder = "\n".join(reminder_lines)

        return handler(request.override(messages=_prepend_reminder(request.messages, reminder)))


def _prepend_reminder(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder), *messages]


__all__ = ["DynamicContextMiddleware"]
