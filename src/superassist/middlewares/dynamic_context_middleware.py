"""Inject dynamic runtime context (memory recall, skills, time) into model calls.

This is the wrap_model_call counterpart to ProAssist's DynamicContextMiddleware.
The static system prompt is supplied via ``create_agent(system_prompt=...)``;
this middleware merges in the per-turn dynamic context as a prefix to the
existing system message before each model call so the prefix cache stays warm
and dynamic data flows through the same channel.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.skills import build_available_skills_section, build_loaded_skills_section


class DynamicContextMiddleware(AgentMiddleware[SuperAssistState]):
    """Prepend turn-time runtime context to the agent's system message."""

    state_schema = SuperAssistState

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
        loaded_skills = state.get("loaded_skills", [])
        user_id = state.get("user_id", "local-user")
        thread_id = state.get("thread_id", "")

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
        reminder = "\n".join(reminder_lines)

        return handler(request.override(messages=_prepend_reminder(request.messages, reminder)))


def _prepend_reminder(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder), *messages]


__all__ = ["DynamicContextMiddleware"]
