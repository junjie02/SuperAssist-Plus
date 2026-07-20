"""Persist completed turns to disk and trigger short-memory compression.

Replaces the standalone ``persist_turn`` graph node from the previous
runtime. Runs after the inner agent has produced its final assistant
message; appends both the user input and assistant answer (plus any tool
events when enabled) to the thread's JSONL log, then asks
``maybe_compress_short_memory`` to fold the older portion into a summary if
the token budget is exceeded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.short_memory import append_jsonl, maybe_compress_short_memory, turn_records
from superassist.agent.state import SuperAssistState
from superassist.config import Settings


class ShortMemoryMiddleware(AgentMiddleware[SuperAssistState]):
    """Append the latest turn to messages.jsonl and compress when budget exceeded."""

    state_schema = SuperAssistState

    def __init__(self, settings: Settings, model: BaseChatModel) -> None:
        super().__init__()
        self._settings = settings
        self._model = model

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

        loaded_skills = sorted(set(state.get("loaded_skills") or []))
        thread_metadata = self._load_thread_metadata(thread_dir)
        compression_update = maybe_compress_short_memory(
            messages_path=path,
            metadata=thread_metadata,
            model=self._model,
            token_limit=self._settings.short_memory_token_limit,
            keep_recent_turns=self._settings.short_memory_keep_recent_turns,
            summary_target_tokens=self._settings.short_memory_summary_target_tokens,
            loaded_skills=loaded_skills,
        )

        meta_update: dict[str, Any] = {"loaded_skills": loaded_skills}
        for key in ("summary", "summary_updated_at"):
            if key in compression_update:
                meta_update[key] = compression_update[key]
        self._save_thread_metadata(thread_dir, meta_update)

        metadata = dict(state.get("metadata") or {})
        metadata["messages_path"] = str(path)
        metadata.update(compression_update)
        return {"metadata": metadata, "loaded_skills": loaded_skills}

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
