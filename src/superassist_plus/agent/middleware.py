from __future__ import annotations

import json
from typing import Any, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime


class SuperAssistAgentState(AgentState):
    """LangChain agent state extended with SuperAssist-Plus metadata."""

    user_id: str
    thread_id: str
    memory_recall: dict[str, Any]
    tool_events: list[dict[str, Any]]
    metadata: dict[str, Any]


class DynamicContextMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Inject dynamic runtime context and recalled memory before model calls."""

    state_schema = SuperAssistAgentState

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt.strip()

    def before_model(self, state: SuperAssistAgentState, runtime: Runtime) -> dict[str, Any] | None:
        metadata = dict(state.get("metadata") or {})
        metadata["dynamic_context_injected"] = True
        return {"metadata": metadata}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        memory_recall = request.state.get("memory_recall", {}) if request.state else {}
        user_id = request.state.get("user_id", "local-user") if request.state else "local-user"
        thread_id = request.state.get("thread_id", "") if request.state else ""
        reminder_text = (
            f"{self.system_prompt}\n\n" if self.system_prompt else ""
        ) + (
            "Runtime context:\n"
            f"- user_id: {user_id}\n"
            f"- thread_id: {thread_id}\n"
            "Long-term memory recall:\n"
            f"{json.dumps(memory_recall, ensure_ascii=False)}"
        )
        return handler(request.override(messages=_merge_runtime_context(request.messages, reminder_text)))


def _merge_runtime_context(messages: list[BaseMessage], reminder_text: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder_text}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder_text), *messages]


class ToolEventMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Capture normalized tool events in state metadata."""

    state_schema = SuperAssistAgentState

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        state = request.state if isinstance(request.state, dict) else {}
        events = list(state.get("tool_events") or [])
        tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "")
        events.append({"type": "tool_start", "tool": tool_name, "args": request.tool_call.get("args") or {}})
        state["tool_events"] = events
        result = handler(request)
        events.append(
            {
                "type": "tool_result",
                "tool": tool_name,
                "content": str(getattr(result, "content", "")),
            }
        )
        state["tool_events"] = events
        return result


class ToolErrorMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Convert tool exceptions into readable tool messages."""

    state_schema = SuperAssistAgentState

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        try:
            return handler(request)
        except Exception as exc:
            tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "tool")
            return ToolMessage(
                content=f"{tool_name} failed: {exc}",
                tool_call_id=str(request.tool_call.get("id") or tool_name),
                name=tool_name,
            )


class MemoryAfterAgentMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Mark the agent output as memory-ready after the LangChain agent finishes."""

    state_schema = SuperAssistAgentState

    def after_agent(self, state: SuperAssistAgentState, runtime: Runtime) -> dict[str, Any] | None:
        metadata = dict(state.get("metadata") or {})
        metadata["memory_ready"] = True
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                metadata["final_assistant_text"] = str(message.content)
                break
        return {"metadata": metadata}


def build_middlewares(system_prompt: str = "") -> list[AgentMiddleware]:
    return [
        DynamicContextMiddleware(system_prompt),
        ToolErrorMiddleware(),
        ToolEventMiddleware(),
        MemoryAfterAgentMiddleware(),
    ]
