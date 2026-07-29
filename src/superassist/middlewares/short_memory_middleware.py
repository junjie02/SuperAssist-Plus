"""Persist completed turns to disk and trigger short-memory compression.

Replaces the standalone ``persist_turn`` graph node from the previous
runtime. Runs after the inner agent has produced its final assistant
message; appends both the user input and assistant answer (plus any tool
events when enabled) to the thread's JSONL log, then asks
Older records remain available for audit and UI history, while model context
loading uses a fixed recent-turn sliding window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.short_memory import append_jsonl, turn_records
from superassist.agent.state import SuperAssistState
from superassist.config import Settings
from superassist.skills import active_skill_activations, active_skill_names


class ShortMemoryMiddleware(AgentMiddleware[SuperAssistState]):
    """Append the latest turn to messages.jsonl and compress when budget exceeded."""

    state_schema = SuperAssistState

    def __init__(self, settings: Settings, _legacy_model: Any | None = None) -> None:
        super().__init__()
        self._settings = settings

    def after_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        thread_id = state.get("thread_id") or ""
        if not thread_id:
            return None
        user_message = state.get("input") or ""
        assistant_answer = _last_ai_text(state.get("messages") or [])
        if not assistant_answer:
            return None

        thread_dir = self._settings.data_dir / "threads" / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        path = thread_dir / "messages.jsonl"
        append_jsonl(
            path,
            turn_records(
                user_message=user_message,
                assistant_answer=assistant_answer,
                tool_events=list(state.get("tool_events") or []),
                include_tool_events=self._settings.short_memory_enable_tool_events,
            ),
        )

        skill_activations = active_skill_activations(
            state.get("skill_activations"),
            self._settings.skill_active_ttl_seconds,
        )
        loaded_skills = active_skill_names(
            skill_activations,
            self._settings.skill_active_ttl_seconds,
        )
        meta_update: dict[str, Any] = {
            "loaded_skills": loaded_skills,
            "skill_activations": skill_activations,
            "user_id": str(state.get("user_id") or ""),
        }
        self._save_thread_metadata(thread_dir, meta_update)

        metadata = dict(state.get("metadata") or {})
        metadata["messages_path"] = str(path)
        metadata["short_memory_window_turns"] = self._settings.short_memory_keep_recent_turns
        return {
            "metadata": metadata,
            "loaded_skills": loaded_skills,
            "skill_activations": skill_activations,
        }

    @staticmethod
    def _load_thread_metadata(thread_dir: Path) -> dict[str, Any]:
        path = thread_dir / "thread_meta.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _save_thread_metadata(cls, thread_dir: Path, update: dict[str, Any]) -> None:
        path = thread_dir / "thread_meta.json"
        existing = cls._load_thread_metadata(thread_dir)
        path.write_text(json.dumps({**existing, **update}, ensure_ascii=False, indent=2), encoding="utf-8")


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = str(message.content or "").strip()
            if text:
                return text
    return ""


__all__ = ["ShortMemoryMiddleware"]
