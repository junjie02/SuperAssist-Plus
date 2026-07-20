"""Capture normalized tool start/result events into state.

The agent uses ``state['tool_events']`` for two downstream features:

* the short-memory persistence layer can replay tool calls as JSONL
* the memory writer can summarize tool side-effects when building durable memory

This middleware also auto-populates ``state['loaded_skills']`` whenever the
agent reads a SKILL.md so subsequent turns inject the loaded skill prompt.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.agent.state import SuperAssistState
from superassist.skills import skill_name_from_virtual_path


class ToolEventMiddleware(AgentMiddleware[SuperAssistState]):
    """Record tool start/result events and surface AI tool-call messages."""

    state_schema = SuperAssistState

    def __init__(self, reporter: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__()
        self._reporter = reporter

    def _report(self, event: dict[str, Any]) -> None:
        if self._reporter is not None:
            self._reporter(event)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        response = handler(request)
        for message in response.result:
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                self._report(_agent_tool_call_event(message))
        return response

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        state = request.state if isinstance(request.state, dict) else {}
        events = list(state.get("tool_events") or [])
        tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "")
        args = request.tool_call.get("args") or {}
        events.append({"type": "tool_start", "tool": tool_name, "args": args})
        state["tool_events"] = events
        self._report(events[-1])

        result = handler(request)

        loaded_skills = list(state.get("loaded_skills") or [])
        if tool_name == "read_file":
            skill_name = skill_name_from_virtual_path(str(args.get("path") or ""))
            if skill_name and skill_name not in loaded_skills:
                loaded_skills.append(skill_name)
                state["loaded_skills"] = loaded_skills

        result_event: dict[str, Any] = {
            "type": "tool_result",
            "tool": tool_name,
            "args": args,
            "content": str(getattr(result, "content", "")),
            "status": getattr(result, "status", "success"),
        }
        if result_event["status"] == "error":
            result_event["error"] = result_event["content"]
        if loaded_skills:
            result_event["loaded_skills"] = loaded_skills
        events.append(result_event)
        state["tool_events"] = events
        self._report(result_event)
        return result


def _agent_tool_call_event(message: AIMessage) -> dict[str, Any]:
    tool_calls = list(getattr(message, "tool_calls", []) or [])
    return {
        "type": "agent_tool_call",
        "content": _message_text(message.content),
        "tool_calls": [
            {"name": str(call.get("name") or ""), "args": call.get("args") or {}}
            for call in tool_calls
        ],
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part.strip() for part in parts if part.strip())
    return str(content).strip() if content else ""


__all__ = ["ToolEventMiddleware"]
