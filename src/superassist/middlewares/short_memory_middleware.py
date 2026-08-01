"""Persist completed turns to disk and trigger short-memory compression.

Runs after the inner agent has produced its final assistant message. The
durable JSONL remains append-only, while model context advances through
summary checkpoints whenever the active segment reaches its turn/token limit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.short_memory import (
    append_jsonl,
    estimate_tokens,
    maybe_compress_short_memory,
    read_jsonl,
    record_text,
    turn_records,
)
from superassist.agent.state import SuperAssistState
from superassist.agent.streaming import clean_answer_text
from superassist.config import Settings
from superassist.skills import active_skill_activations, active_skill_names


class ShortMemoryMiddleware(AgentMiddleware[SuperAssistState]):
    """Append the latest turn to messages.jsonl and compress when budget exceeded."""

    state_schema = SuperAssistState

    def __init__(self, settings: Settings, _legacy_model: Any | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._model = _legacy_model

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
        assistant_created_at = str(state.get("assistant_message_created_at") or "") or datetime.now(UTC).isoformat()
        append_jsonl(
            path,
            turn_records(
                user_message=user_message,
                assistant_answer=assistant_answer,
                tool_events=[],
                include_tool_events=False,
                user_created_at=str(state.get("message_created_at") or "") or None,
                assistant_created_at=assistant_created_at,
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
        existing_metadata = self._load_thread_metadata(thread_dir)
        if self._model is not None:
            meta_update.update(
                maybe_compress_short_memory(
                    messages_path=path,
                    metadata={**existing_metadata, **meta_update},
                    model=self._model,
                    token_limit=self._settings.short_memory_token_limit,
                    keep_recent_turns=self._settings.short_memory_keep_recent_turns,
                    summary_target_tokens=self._settings.short_memory_summary_target_tokens,
                    loaded_skills=loaded_skills,
                )
            )
        records = read_jsonl(path)
        compacted_count = int(
            meta_update.get(
                "short_memory_compacted_records",
                existing_metadata.get("short_memory_compacted_records", 0),
            )
            or 0
        )
        active_records = records[compacted_count:] if 0 <= compacted_count <= len(records) else records
        meta_update.setdefault(
            "short_memory_active_turns",
            sum(1 for record in active_records if record.get("role") == "user"),
        )
        meta_update.setdefault(
            "short_memory_active_tokens",
            estimate_tokens(str(meta_update.get("summary", existing_metadata.get("summary", "")) or ""))
            + sum(estimate_tokens(record_text(record)) for record in active_records),
        )
        self._save_thread_metadata(thread_dir, meta_update)

        metadata = dict(state.get("metadata") or {})
        metadata["messages_path"] = str(path)
        metadata["short_memory_segment_turn_limit"] = self._settings.short_memory_keep_recent_turns
        metadata["short_memory_summary_version"] = int(
            meta_update.get("summary_version", existing_metadata.get("summary_version", 0)) or 0
        )
        return {
            "metadata": metadata,
            "assistant_message_created_at": assistant_created_at,
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
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**existing, **update}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = clean_answer_text(message.content)
            if text:
                return text
    return ""


__all__ = ["ShortMemoryMiddleware"]
