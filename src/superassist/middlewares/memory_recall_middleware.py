"""Read long-term memory and create the user-turn event before the agent runs.

This middleware replaces the standalone ``prepare_context`` graph node from
the previous outer-StateGraph design. It runs once at the start of each
agent invocation and writes the recalled context into state so
``DynamicContextMiddleware`` can render it into the model prompt.

The user-turn event node is created here too — its id is needed by the
memory writer middleware after the turn completes.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.memory.service import MemoryService


class MemoryRecallMiddleware(AgentMiddleware[SuperAssistState]):
    """Populate state.memory_recall and create the turn's event node."""

    state_schema = SuperAssistState

    def __init__(self, memory: MemoryService) -> None:
        super().__init__()
        self._memory = memory

    def before_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if state.get("memory_event_id"):
            return None
        message = state.get("input") or _last_human_text(state.get("messages") or [])
        if not message:
            return None
        contexts = self._memory.prepare_turn_contexts(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            message=message,
        )
        return {
            "memory_event_id": contexts.event_id,
            "memory_recall": contexts.read_recall.model_dump(mode="json"),
            "memory_write_context": contexts.write_recall.model_dump(mode="json"),
        }


def _last_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""


__all__ = ["MemoryRecallMiddleware"]
