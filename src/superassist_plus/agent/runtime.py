from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from superassist_plus.config import Settings, get_settings
from superassist_plus.llm import create_chat_model, is_minimax_model
from superassist_plus.memory.service import MemoryService, MemoryWritePayload
from superassist_plus.memory.writer import MemoryWriteQueue, MemoryWriter
from superassist_plus.models import AgentRunResult
from superassist_plus.tools import default_tools

from .middleware import SuperAssistAgentState, build_middlewares
from .state import SuperAssistState


SYSTEM_PROMPT = """You are SuperAssist-Plus, a concise and capable assistant.

Use tools when they materially help. Long-term memory may be provided as
structured context; treat it as helpful but not infallible.
"""


class AgentRuntime:
    """LangGraph runtime wrapper for SuperAssist-Plus."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = create_chat_model(self.settings)
        self.memory = MemoryService(self.settings.db_path, self.settings)
        self.memory.preload_embedder()
        self.memory_queue = MemoryWriteQueue(
            MemoryWriter(
                self.memory,
                self.model,
                llm_enabled=self.settings.memory_llm_writer_enabled,
            ),
            debounce_seconds=self.settings.memory_debounce_seconds,
        )
        self.agent = create_agent(
            model=self.model,
            tools=default_tools() if self.settings.enable_tools else [],
            middleware=build_middlewares(SYSTEM_PROMPT),
            state_schema=SuperAssistAgentState,
        )
        self.graph = self._build_graph()

    def run(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
    ) -> AgentRunResult:
        resolved_thread_id = thread_id or f"thread_{uuid4().hex[:12]}"
        history = self._load_thread_history(resolved_thread_id)
        initial_state: SuperAssistState = {
            "messages": [*history, HumanMessage(content=message)],
            "input": message,
            "user_id": user_id,
            "thread_id": resolved_thread_id,
            "metadata": {"history_loaded": bool(history), "history_message_count": len(history)},
        }
        final_state = self.graph.invoke(initial_state)
        return AgentRunResult(
            thread_id=resolved_thread_id,
            answer=str(final_state.get("answer") or ""),
            metadata=dict(final_state.get("metadata") or {}),
        )

    def _build_graph(self):
        graph = StateGraph(SuperAssistState)
        graph.add_node("prepare_context", self._prepare_context)
        graph.add_node("agent", self._run_agent)
        graph.add_node("persist_turn", self._persist_turn)
        graph.add_node("enqueue_memory_write", self._enqueue_memory_write)
        graph.add_edge(START, "prepare_context")
        graph.add_edge("prepare_context", "agent")
        graph.add_edge("agent", "persist_turn")
        graph.add_edge("persist_turn", "enqueue_memory_write")
        graph.add_edge("enqueue_memory_write", END)
        return graph.compile()

    def _prepare_context(self, state: SuperAssistState) -> dict[str, Any]:
        contexts = self.memory.prepare_turn_contexts(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            message=state["input"],
        )
        return {
            "memory_event_id": contexts.event_id,
            "memory_recall": contexts.read_recall.model_dump(mode="json"),
            "memory_write_context": contexts.write_recall.model_dump(mode="json"),
        }

    def _run_agent(self, state: SuperAssistState) -> dict[str, Any]:
        try:
            if not self.settings.enable_tools:
                return self._run_direct_model(state)
            result = self.agent.invoke(
                {
                    "messages": state["messages"],
                    "user_id": state["user_id"],
                    "thread_id": state["thread_id"],
                    "memory_recall": state.get("memory_recall", {}),
                    "tool_events": [],
                    "metadata": state.get("metadata", {}),
                }
            )
        except Exception as exc:
            return self._model_error_response(state, exc)
        messages = list(result.get("messages", []))
        answer = self._last_ai_text(messages)
        metadata = dict(result.get("metadata") or state.get("metadata") or {})
        metadata.update(self._tool_compatibility_metadata())
        return {
            "messages": messages,
            "answer": answer,
            "tool_events": list(result.get("tool_events") or self._tool_events(messages)),
            "metadata": metadata,
        }

    def _run_direct_model(self, state: SuperAssistState) -> dict[str, Any]:
        memory_context = json.dumps(state.get("memory_recall", {}), ensure_ascii=False)
        messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{SYSTEM_PROMPT}\n\n"
                    "Runtime context:\n"
                    f"- user_id: {state['user_id']}\n"
                    f"- thread_id: {state['thread_id']}\n"
                    "Long-term memory recall:\n"
                    f"{memory_context}"
                )
            ),
            *state["messages"],
        ]
        try:
            response = self.model.invoke(messages)
        except Exception as exc:
            return self._model_error_response(state, exc)
        ai_message = response if isinstance(response, AIMessage) else AIMessage(content=str(response.content))
        metadata = dict(state.get("metadata") or {})
        metadata["dynamic_context_injected"] = True
        metadata["memory_ready"] = True
        metadata["final_assistant_text"] = str(ai_message.content)
        metadata["tool_calling_enabled"] = False
        return {
            "messages": [*state["messages"], ai_message],
            "answer": str(ai_message.content),
            "tool_events": [],
            "metadata": metadata,
        }

    @staticmethod
    def _model_error_response(state: SuperAssistState, exc: Exception) -> dict[str, Any]:
        message = (
            "模型服务拒绝或中断了这次回复，当前对话进程已保留。"
            "你可以换一种问法继续，或者减少敏感/过长的上下文后重试。"
        )
        metadata = dict(state.get("metadata") or {})
        metadata["dynamic_context_injected"] = True
        metadata["memory_ready"] = True
        metadata["model_error"] = type(exc).__name__
        metadata["model_error_message"] = str(exc)
        metadata["final_assistant_text"] = message
        return {
            "messages": [*state["messages"], AIMessage(content=message)],
            "answer": message,
            "tool_events": [],
            "metadata": metadata,
        }

    def _tool_compatibility_metadata(self) -> dict[str, Any]:
        if self.settings.enable_tools and is_minimax_model(self.settings.model, self.settings.base_url):
            return {
                "tool_calling_enabled": True,
                "tool_schema_binding": "openai_compatible_minimax",
            }
        return {"tool_calling_enabled": self.settings.enable_tools}

    def _persist_turn(self, state: SuperAssistState) -> dict[str, Any]:
        thread_dir = self.settings.data_dir / "threads" / state["thread_id"]
        thread_dir.mkdir(parents=True, exist_ok=True)
        path = thread_dir / "messages.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": "user", "content": state["input"]}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"role": "assistant", "content": state.get("answer", "")}, ensure_ascii=False) + "\n")
        metadata = dict(state.get("metadata") or {})
        metadata["messages_path"] = str(path)
        return {"metadata": metadata}

    def _enqueue_memory_write(self, state: SuperAssistState) -> dict[str, Any]:
        event_id = str(state.get("memory_event_id") or "")
        if event_id:
            self.memory_queue.add(
                MemoryWritePayload(
                    user_id=state["user_id"],
                    thread_id=state["thread_id"],
                    event_id=event_id,
                    user_message=state["input"],
                    assistant_answer=str(state.get("answer") or ""),
                    tool_events=list(state.get("tool_events") or []),
                    memory_context=dict(state.get("memory_write_context") or {}),
                )
            )
        return {}

    @staticmethod
    def _last_ai_text(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _tool_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for message in messages:
            if getattr(message, "type", "") == "tool":
                events.append({"name": getattr(message, "name", ""), "content": str(message.content)})
        return events

    def _load_thread_history(self, thread_id: str, limit: int = 20) -> list[BaseMessage]:
        path = self.settings.data_dir / "threads" / thread_id / "messages.jsonl"
        if not path.exists():
            return []
        messages: list[BaseMessage] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-limit:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages
