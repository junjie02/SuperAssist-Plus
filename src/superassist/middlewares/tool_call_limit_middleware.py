"""Stop executing tools after a configured per-turn limit."""

from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.agent.state import SuperAssistState

_SHENLUN_GRADING_SKILL = "hubei-shenlun-grading"
_SHENLUN_GRADING_TOOL_BUDGET = 64


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
        loaded_skills = set(state.get("loaded_skills") or [])
        active_skills = set(state.get("active_skills_at_turn_start") or [])
        active_skills.update((state.get("skill_activations") or {}).keys())
        effective_limit = self.max_tool_calls
        if effective_limit >= 0 and _SHENLUN_GRADING_SKILL in loaded_skills | active_skills:
            effective_limit = max(effective_limit, _SHENLUN_GRADING_TOOL_BUDGET)
        if 0 <= effective_limit <= completed:
            return ToolMessage(
                content=f"Tool call limit reached ({effective_limit}). Continue with available context.",
                tool_call_id=str(request.tool_call.get("id") or tool_name),
                name=tool_name,
                status="error",
            )
        return handler(request)


__all__ = ["ToolCallLimitMiddleware"]
