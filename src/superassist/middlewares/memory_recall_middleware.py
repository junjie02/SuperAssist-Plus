"""Read long-term memory and reserve an optional event id before the agent runs.

This middleware replaces the standalone ``prepare_context`` graph node from
the previous outer-StateGraph design. It runs once at the start of each
agent invocation and writes the recalled context into state so
``DynamicContextMiddleware`` can render it into the model prompt.

Only the id is allocated here. The memory updater creates an event node later
when the completed turn contains durable information.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.memory.service import MemoryService, project_memory_recall, project_memory_write_context


class MemoryRecallMiddleware(AgentMiddleware[SuperAssistState]):
    """Populate recalled memory and reserve the turn's optional event id."""

    state_schema = SuperAssistState

    def __init__(self, memory: MemoryService) -> None:
        super().__init__()
        self._memory = memory

    def before_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if state.get("memory_event_id"):
            return None
        message = state.get("memory_query") or state.get("input") or _last_human_text(state.get("messages") or [])
        if not message:
            return None
        contexts = self._memory.prepare_turn_contexts(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            message=message,
        )
        return {
            "memory_event_id": "" if state.get("suppress_memory_write") else contexts.event_id,
            "memory_recall": project_memory_recall(contexts.read_recall),
            "memory_write_context": project_memory_write_context(contexts.write_recall),
        }


def _last_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""


__all__ = ["MemoryRecallMiddleware"]
