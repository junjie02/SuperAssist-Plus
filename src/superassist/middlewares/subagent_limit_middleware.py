"""Truncate excess parallel ``task`` tool calls in a single model response.

When the agent is asked to fan out subagents, we cap the number of parallel
calls at ``max_concurrent`` (1-3). Extra task calls are dropped from the
AIMessage's tool_calls so the executor never sees them.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState


class SubagentLimitMiddleware(AgentMiddleware[SuperAssistState]):
    """Keep at most ``max_concurrent`` task tool calls in any AIMessage."""

    state_schema = SuperAssistState

    def __init__(self, max_concurrent: int) -> None:
        super().__init__()
        self.max_concurrent = max(1, min(3, max_concurrent))

    def after_model(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        message = messages[-1]
        if not isinstance(message, AIMessage):
            return None
        tool_calls = list(getattr(message, "tool_calls", []) or [])
        task_indices = [index for index, call in enumerate(tool_calls) if call.get("name") == "task"]
        if len(task_indices) <= self.max_concurrent:
            return None
        drop_indices = set(task_indices[self.max_concurrent :])
        kept = [call for index, call in enumerate(tool_calls) if index not in drop_indices]
        events = list(state.get("tool_events") or [])
        events.append({"type": "subagent_limit", "max_concurrent": self.max_concurrent, "dropped": len(drop_indices)})
        return {
            "messages": [message.model_copy(update={"tool_calls": kept})],
            "tool_events": events,
        }


__all__ = ["SubagentLimitMiddleware"]
