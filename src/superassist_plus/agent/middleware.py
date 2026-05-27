from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from superassist_plus.skills import build_available_skills_section, build_loaded_skills_section, skill_name_from_virtual_path


class SuperAssistAgentState(AgentState):
    """LangChain agent state extended with SuperAssist-Plus metadata."""

    user_id: str
    thread_id: str
    memory_recall: dict[str, Any]
    tool_events: list[dict[str, Any]]
    loaded_skills: list[str]
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
        loaded_skills = request.state.get("loaded_skills", []) if request.state else []
        user_id = request.state.get("user_id", "local-user") if request.state else "local-user"
        thread_id = request.state.get("thread_id", "") if request.state else ""
        skills_section = build_available_skills_section()
        loaded_skills_section = build_loaded_skills_section(loaded_skills)
        skills_text = "\n\n".join(part for part in (skills_section, loaded_skills_section) if part)
        reminder_text = (
            f"{self.system_prompt}\n\n" if self.system_prompt else ""
        ) + (
            "Runtime context:\n"
            f"- user_id: {user_id}\n"
            f"- thread_id: {thread_id}\n"
            f"- current_time_utc: {datetime.now(UTC).isoformat()}\n"
            "Long-term memory recall:\n"
            f"{json.dumps(memory_recall, ensure_ascii=False)}"
        )
        if skills_text:
            reminder_text = f"{reminder_text}\n\n{skills_text}"
        return handler(request.override(messages=_merge_runtime_context(request.messages, reminder_text)))


def _merge_runtime_context(messages: list[BaseMessage], reminder_text: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder_text}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder_text), *messages]


class ToolEventMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Capture normalized tool events in state metadata."""

    state_schema = SuperAssistAgentState

    def __init__(self, reporter: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.reporter = reporter

    def _report(self, event: dict[str, Any]) -> None:
        if self.reporter is None:
            return
        self.reporter(event)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
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
        start_event = {"type": "tool_start", "tool": tool_name, "args": args}
        events.append(start_event)
        state["tool_events"] = events
        self._report(start_event)
        result = handler(request)
        loaded_skills = list(state.get("loaded_skills") or [])
        if tool_name == "read_file":
            skill_name = skill_name_from_virtual_path(str(args.get("path") or ""))
            if skill_name and skill_name not in loaded_skills:
                loaded_skills.append(skill_name)
                state["loaded_skills"] = loaded_skills
        result_event = {
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
    content = _message_text(message.content)
    return {
        "type": "agent_tool_call",
        "content": content,
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


class ToolCallLimitMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Stop executing tools after a configured per-turn limit."""

    state_schema = SuperAssistAgentState

    def __init__(self, max_tool_calls: int) -> None:
        self.max_tool_calls = max_tool_calls

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        state = request.state if isinstance(request.state, dict) else {}
        events = list(state.get("tool_events") or [])
        completed_calls = sum(1 for event in events if event.get("type") == "tool_result")
        tool_name = request.tool.name if request.tool is not None else str(request.tool_call.get("name") or "tool")
        if self.max_tool_calls >= 0 and completed_calls >= self.max_tool_calls:
            return ToolMessage(
                content=f"Tool call limit reached ({self.max_tool_calls}). Continue with available context.",
                tool_call_id=str(request.tool_call.get("id") or tool_name),
                name=tool_name,
                status="error",
            )
        return handler(request)


class SubagentLimitMiddleware(AgentMiddleware[SuperAssistAgentState]):
    """Keep at most N task tool calls from a single model response."""

    state_schema = SuperAssistAgentState

    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max(1, min(3, max_concurrent))

    def after_model(self, state: SuperAssistAgentState, runtime: Runtime) -> dict[str, Any] | None:
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
        kept_calls = [call for index, call in enumerate(tool_calls) if index not in drop_indices]
        events = list(state.get("tool_events") or [])
        events.append(
            {
                "type": "subagent_limit",
                "max_concurrent": self.max_concurrent,
                "dropped": len(drop_indices),
            }
        )
        return {
            "messages": [message.model_copy(update={"tool_calls": kept_calls})],
            "tool_events": events,
        }


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


def build_middlewares(
    system_prompt: str = "",
    max_tool_calls: int = 8,
    max_subagent_calls: int = 3,
    subagents_enabled: bool = True,
    tool_event_reporter: Callable[[dict[str, Any]], None] | None = None,
) -> list[AgentMiddleware]:
    middlewares: list[AgentMiddleware] = [
        DynamicContextMiddleware(system_prompt),
        ToolErrorMiddleware(),
        ToolCallLimitMiddleware(max_tool_calls),
    ]
    if subagents_enabled:
        middlewares.append(SubagentLimitMiddleware(max_subagent_calls))
    middlewares.extend(
        [
            ToolEventMiddleware(tool_event_reporter),
            MemoryAfterAgentMiddleware(),
        ]
    )
    return middlewares
