"""Stop executing tools after a configured per-turn limit."""

from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.agent.state import SuperAssistState


class ToolCallLimitMiddleware(AgentMiddleware[SuperAssistState]):
    """Refuse new tool calls once the turn budget is exhausted."""

    state_schema = SuperAssistState

    def __init__(self, max_tool_calls: int) -> None:
        self.max_tool_calls = max_tool_calls

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        state = request.state if isinstance(request.state, dict) else {}
        events = list(state.get("tool_events") or [])
        completed = sum(1 for event in events if event.get("type") == "tool_result")
        tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "tool")
        if 0 <= self.max_tool_calls <= completed:
            return ToolMessage(
                content=f"Tool call limit reached ({self.max_tool_calls}). Continue with available context.",
                tool_call_id=str(request.tool_call.get("id") or tool_name),
                name=tool_name,
                status="error",
            )
        return handler(request)


__all__ = ["ToolCallLimitMiddleware"]
