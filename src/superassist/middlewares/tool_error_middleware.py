"""Convert tool exceptions into readable ToolMessage responses."""

from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.agent.state import SuperAssistState


class ToolErrorMiddleware(AgentMiddleware[SuperAssistState]):
    """Wrap tool calls so exceptions surface as ToolMessages instead of crashing the graph."""

    state_schema = SuperAssistState

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        try:
            return handler(request)
        except Exception as exc:
            tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "tool")
            return ToolMessage(
                content=f"{tool_name} failed: {exc}",
                tool_call_id=str(request.tool_call.get("id") or tool_name),
                name=tool_name,
                status="error",
            )


__all__ = ["ToolErrorMiddleware"]
