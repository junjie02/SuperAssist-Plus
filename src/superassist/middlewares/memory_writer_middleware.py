"""Enqueue a memory write payload after the agent finishes.

Replaces the standalone ``enqueue_memory_write`` graph node. Runs after
``ShortMemoryMiddleware`` so the durable JSONL is already on disk before the
write queue fires.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.agent.streaming import clean_answer_text
from superassist.memory.service import MemoryWritePayload
from superassist.memory.writer import MemoryWriteQueue, compact_tool_events


class MemoryWriterMiddleware(AgentMiddleware[SuperAssistState]):
    """Submit the completed turn to the debounced memory write queue."""

    state_schema = SuperAssistState

    def __init__(self, queue: MemoryWriteQueue) -> None:
        super().__init__()
        self._queue = queue

    def after_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if state.get("suppress_memory_write"):
            return None
        event_id = str(state.get("memory_event_id") or "")
        if not event_id:
            return None
        assistant_answer = _last_ai_text(state.get("messages") or [])
        assistant_created_at = str(state.get("assistant_message_created_at") or "") or datetime.now(UTC).isoformat()
        self._queue.add(
            MemoryWritePayload(
                user_id=state["user_id"],
                thread_id=state["thread_id"],
                event_id=event_id,
                user_message=state.get("input") or "",
                user_message_created_at=str(state.get("message_created_at") or ""),
                assistant_answer=assistant_answer,
                assistant_message_created_at=assistant_created_at,
                tool_events=compact_tool_events(list(state.get("tool_events") or [])),
                memory_context=dict(state.get("memory_write_context") or {}),
            )
        )
        return {"assistant_message_created_at": assistant_created_at}


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = clean_answer_text(message.content)
            if text:
                return text
    return ""


__all__ = ["MemoryWriterMiddleware"]
