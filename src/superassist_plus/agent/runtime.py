from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from superassist_plus.config import Settings, get_settings
from superassist_plus.llm import create_chat_model, is_minimax_model
from superassist_plus.memory.service import MemoryService, MemoryWritePayload
from superassist_plus.memory.writer import MemoryWriteQueue, MemoryWriter
from superassist_plus.models import AgentRunEvent, AgentRunResult
from superassist_plus.observability import runnable_trace_config, traceable, without_self
from superassist_plus.run_events import run_event_reporter_context
from superassist_plus.tools import default_tools

from .middleware import SuperAssistAgentState, build_middlewares
from .short_memory import append_jsonl, load_short_memory, maybe_compress_short_memory, turn_records
from .state import SuperAssistState


SYSTEM_PROMPT = """
<role>
You are SuperAssist-Plus, a concise and capable assistant.
</role>

<thinking_style>
- Think concisely and strategically before acting.
- Identify what is clear, what is ambiguous, and what information is missing.
- If the user's request is ambiguous, risky, or missing required details, ask a
  short clarification question before doing work.
- After thinking, always provide a visible response to the user.
</thinking_style>

<tool_use>
- Use tools when they materially help.
- Long-term memory may be provided as structured context; treat it as helpful
  but not infallible.
- When you need multiple tool rounds, write human progress notes in assistant
  message content before the next tool call.
- Progress notes should summarize what the previous tool result showed, what is
  still uncertain, and what you will check next.
- Before each tool or `task` call, include one natural-language sentence in
  assistant message content explaining what you are about to do.
- After tools or subagents return, summarize what you learned and your next
  step in assistant message content before deciding whether to call more tools.
</tool_use>

<citations>
- When using web_search, web_fetch, or external sources, cite sourced claims.
- Use inline Markdown citations immediately after the claim:
  [citation:Title](URL)
- For longer research answers, include a "Sources" section with normal Markdown
  links: [Title](URL) - short description.
- Do not invent citations or cite unsourced claims.
</citations>

<response_style>
- Be clear, concise, and natural.
- Prefer prose over bullet lists unless structure helps.
- Focus on delivering the answer or result, not narrating internal process.
- Use the same language as the user.
</response_style>
"""


def subagent_prompt_section(max_concurrent: int) -> str:
    limit = max(1, min(3, max_concurrent))
    return f"""
<subagent_system>
You can delegate complex work to subagents using the `task` tool.

Available subagents:
- general-purpose: Complex multi-step implementation, investigation, and codebase analysis.
- research: Source-backed research and synthesis using web/search tools.

Rules:
- Use subagents only when the request can be split into 2 or more meaningful parallel subtasks.
- Use at most {limit} `task` calls in one response. Extra task calls are discarded.
- For more than {limit} subtasks, run batches across turns.
- Do not wrap simple one-step actions in `task`; use direct tools instead.
- After subagents return, synthesize their results into your own final answer.
</subagent_system>
"""


class AgentRuntime:
    """LangGraph runtime wrapper for SuperAssist-Plus."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tool_event_reporter: Any | None = None,
        run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._tool_event_reporter = tool_event_reporter
        self._run_event_reporter = run_event_reporter
        self._active_agent_text_seen: set[str] | None = None
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
        tools = (
            default_tools(
                include_task=self.settings.subagents_enabled,
                run_event_reporter=self._run_event_reporter,
            )
            if self.settings.enable_tools
            else []
        )
        prompt = SYSTEM_PROMPT
        if self.settings.enable_tools and self.settings.subagents_enabled:
            prompt = f"{prompt}\n\n{subagent_prompt_section(self.settings.subagent_max_concurrent)}"
        self.agent = create_agent(
            model=self.model,
            tools=tools,
            middleware=build_middlewares(
                prompt,
                max_tool_calls=self.settings.max_tool_calls,
                max_subagent_calls=self.settings.subagent_max_concurrent,
                subagents_enabled=self.settings.subagents_enabled,
                tool_event_reporter=self._report_tool_event,
            ),
            state_schema=SuperAssistAgentState,
        )
        self.graph = self._build_graph()

    def set_run_event_reporter(self, reporter: Callable[[AgentRunEvent], None] | None) -> None:
        self._run_event_reporter = reporter

    def run(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
    ) -> AgentRunResult:
        return self._run_traced(message, user_id=user_id, thread_id=thread_id)

    @traceable(name="superassist.turn", run_type="chain", process_inputs=without_self)
    def _run_traced(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
    ) -> AgentRunResult:
        initial_state = self._initial_state(message, user_id=user_id, thread_id=thread_id)
        with run_event_reporter_context(self._run_event_reporter):
            final_state = self.graph.invoke(
                initial_state,
                runnable_trace_config(
                    run_name="superassist.graph",
                    user_id=user_id,
                    thread_id=initial_state["thread_id"],
                    tags=["agent", "graph"],
                    metadata={"streaming": False},
                ),
            )
        return AgentRunResult(
            thread_id=initial_state["thread_id"],
            answer=str(final_state.get("answer") or ""),
            metadata=dict(final_state.get("metadata") or {}),
        )

    def run_streaming(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
    ) -> AgentRunResult:
        return self._run_streaming_traced(message, user_id=user_id, thread_id=thread_id)

    @traceable(name="superassist.turn.streaming", run_type="chain", process_inputs=without_self)
    def _run_streaming_traced(
        self,
        message: str,
        *,
        user_id: str = "local-user",
        thread_id: str | None = None,
    ) -> AgentRunResult:
        """Run one turn while reporting AI text emitted by LangGraph messages mode.

        This mirrors the normal write/persist flow, but the agent step subscribes
        to LangGraph's message stream so channels such as Feishu can patch a
        running card with model-authored text instead of waiting for the final
        answer. Tool-call events remain handled by middleware/reporters.
        """

        state = self._initial_state(message, user_id=user_id, thread_id=thread_id)
        with run_event_reporter_context(self._run_event_reporter):
            state.update(self._prepare_context(state))
            self._report_run_event("thinking", "Thinking...", thread_id=state["thread_id"])
            state.update(self._run_agent_streaming(state))
            state.update(self._persist_turn(state))
            state.update(self._enqueue_memory_write(state))
        return AgentRunResult(
            thread_id=state["thread_id"],
            answer=str(state.get("answer") or ""),
            metadata=dict(state.get("metadata") or {}),
        )

    def _initial_state(self, message: str, *, user_id: str, thread_id: str | None) -> SuperAssistState:
        resolved_thread_id = thread_id or f"thread_{uuid4().hex[:12]}"
        self._report_run_event("preparing_context", "Preparing context...", thread_id=resolved_thread_id)
        thread_metadata = self._load_thread_metadata(resolved_thread_id)
        history_load = self._load_thread_history(resolved_thread_id, thread_metadata)
        history = history_load.messages
        loaded_skills = self._loaded_skills_from_metadata(thread_metadata)
        return {
            "messages": [*history, HumanMessage(content=message)],
            "input": message,
            "user_id": user_id,
            "thread_id": resolved_thread_id,
            "loaded_skills": loaded_skills,
            "metadata": {
                "history_loaded": bool(history),
                "history_message_count": len(history),
                "short_memory_summary_loaded": bool(history_load.summary),
                "loaded_skills": loaded_skills,
            },
        }

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
                self._agent_input(state),
                runnable_trace_config(
                    run_name="superassist.lead_agent",
                    user_id=state["user_id"],
                    thread_id=state["thread_id"],
                    tags=["agent", "lead"],
                    metadata={
                        "enable_tools": self.settings.enable_tools,
                        "subagents_enabled": self.settings.subagents_enabled,
                    },
                ),
            )
        except Exception as exc:
            return self._model_error_response(state, exc)
        return self._finalize_agent_result(state, result)

    def _run_agent_streaming(self, state: SuperAssistState) -> dict[str, Any]:
        try:
            if not self.settings.enable_tools:
                return self._run_direct_model(state)
            last_values: dict[str, Any] | None = None
            text_buffers: dict[str, str] = {}
            current_message_id: str | None = None
            previous_seen = self._active_agent_text_seen
            self._active_agent_text_seen = set()
            try:
                for item in self.agent.stream(
                    self._agent_input(state),
                    runnable_trace_config(
                        run_name="superassist.lead_agent.streaming",
                        user_id=state["user_id"],
                        thread_id=state["thread_id"],
                        tags=["agent", "lead", "streaming"],
                        metadata={
                            "enable_tools": self.settings.enable_tools,
                            "subagents_enabled": self.settings.subagents_enabled,
                        },
                    ),
                    stream_mode=["messages", "values"],
                ):
                    if isinstance(item, tuple) and len(item) == 2:
                        mode, chunk = item
                        mode = str(mode)
                    else:
                        mode, chunk = "values", item
                    if mode == "messages":
                        text, current_message_id = self._accumulate_stream_text(
                            text_buffers,
                            current_message_id,
                            chunk,
                        )
                        if text:
                            self._report_agent_text(text, thread_id=state["thread_id"])
                        continue
                    if mode == "values" and isinstance(chunk, dict):
                        last_values = chunk
            finally:
                self._active_agent_text_seen = previous_seen
            if last_values is None:
                last_values = {"messages": state["messages"], "metadata": state.get("metadata", {})}
            return self._finalize_agent_result(state, last_values)
        except Exception as exc:
            return self._model_error_response(state, exc)

    @staticmethod
    def _agent_input(state: SuperAssistState) -> dict[str, Any]:
        return {
            "messages": state["messages"],
            "user_id": state["user_id"],
            "thread_id": state["thread_id"],
            "memory_recall": state.get("memory_recall", {}),
            "tool_events": [],
            "loaded_skills": list(state.get("loaded_skills") or []),
            "metadata": state.get("metadata", {}),
        }

    def _finalize_agent_result(self, state: SuperAssistState, result: dict[str, Any]) -> dict[str, Any]:
        messages = list(result.get("messages", []))
        answer = self._last_ai_text(messages)
        metadata = dict(result.get("metadata") or state.get("metadata") or {})
        loaded_skills = list(result.get("loaded_skills") or metadata.get("loaded_skills") or state.get("loaded_skills") or [])
        metadata["loaded_skills"] = sorted(set(loaded_skills))
        metadata.update(self._tool_compatibility_metadata())
        return {
            "messages": messages,
            "answer": answer,
            "tool_events": list(result.get("tool_events") or self._tool_events(messages)),
            "loaded_skills": metadata["loaded_skills"],
            "metadata": metadata,
        }

    def _accumulate_stream_text(
        self,
        buffers: dict[str, str],
        current_message_id: str | None,
        chunk: Any,
    ) -> tuple[str | None, str | None]:
        message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            return None, current_message_id
        text = self._message_text(getattr(message, "content", ""))
        if not text:
            return None, current_message_id
        message_id = str(getattr(message, "id", "") or current_message_id or "__default__")
        buffers[message_id] = self._merge_stream_text(buffers.get(message_id, ""), text)
        return buffers[message_id], message_id

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if content and all(isinstance(item, str) for item in content):
                return "".join(content)
            parts: list[str] = []
            pending: list[str] = []
            for item in content:
                if isinstance(item, str):
                    pending.append(item)
                    continue
                if pending:
                    parts.append("".join(pending))
                    pending.clear()
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if pending:
                parts.append("".join(pending))
            return "\n".join(part for part in parts if part)
        return str(content) if content else ""

    @staticmethod
    def _merge_stream_text(existing: str, incoming: str) -> str:
        if not existing:
            return incoming
        if incoming.startswith(existing):
            return incoming
        if existing.endswith(incoming):
            return existing
        return f"{existing}{incoming}"

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
        metadata["loaded_skills"] = list(state.get("loaded_skills") or [])
        return {
            "messages": [*state["messages"], ai_message],
            "answer": str(ai_message.content),
            "tool_events": [],
            "loaded_skills": list(state.get("loaded_skills") or []),
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
        metadata["loaded_skills"] = list(state.get("loaded_skills") or [])
        return {
            "messages": [*state["messages"], AIMessage(content=message)],
            "answer": message,
            "tool_events": [],
            "loaded_skills": list(state.get("loaded_skills") or []),
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
        append_jsonl(
            path,
            turn_records(
                user_message=state["input"],
                assistant_answer=str(state.get("answer") or ""),
                tool_events=list(state.get("tool_events") or []),
                include_tool_events=self.settings.short_memory_enable_tool_events,
            ),
        )
        metadata = dict(state.get("metadata") or {})
        metadata["messages_path"] = str(path)
        metadata["loaded_skills"] = sorted(set(state.get("loaded_skills") or metadata.get("loaded_skills") or []))
        thread_metadata = self._load_thread_metadata(state["thread_id"])
        compression_update = maybe_compress_short_memory(
            messages_path=path,
            metadata=thread_metadata,
            model=self.model,
            token_limit=self.settings.short_memory_token_limit,
            keep_recent_turns=self.settings.short_memory_keep_recent_turns,
            summary_target_tokens=self.settings.short_memory_summary_target_tokens,
            loaded_skills=metadata["loaded_skills"],
        )
        metadata.update(compression_update)
        meta_update = {"loaded_skills": metadata["loaded_skills"]}
        for key in ("summary", "summary_updated_at"):
            if key in compression_update:
                meta_update[key] = compression_update[key]
        self._save_thread_metadata(state["thread_id"], meta_update)
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

    def _report_run_event(self, event_type: str, message: str, **metadata: Any) -> None:
        if self._run_event_reporter is None:
            return
        self._run_event_reporter(AgentRunEvent(type=event_type, message=message, metadata=metadata))

    def _report_agent_text(self, content: str, **metadata: Any) -> None:
        text = content.strip()
        if not text:
            return
        seen = self._active_agent_text_seen
        if seen is not None:
            if text in seen or any(previous.startswith(text) for previous in seen):
                return
            seen.add(text)
        self._report_run_event("agent_text", text, **metadata)

    def _report_tool_event(self, event: dict[str, Any]) -> None:
        if self._tool_event_reporter is not None:
            self._tool_event_reporter(event)
        if event.get("type") != "agent_tool_call":
            return
        content = str(event.get("content") or "").strip()
        if content:
            self._report_agent_text(content)

    @staticmethod
    def _last_ai_text(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = str(message.content or "").strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _tool_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for message in messages:
            if getattr(message, "type", "") == "tool":
                events.append({"name": getattr(message, "name", ""), "content": str(message.content)})
        return events

    def _load_thread_history(self, thread_id: str, metadata: dict[str, Any] | None = None):
        path = self.settings.data_dir / "threads" / thread_id / "messages.jsonl"
        return load_short_memory(
            path,
            metadata or {},
            token_limit=self.settings.short_memory_token_limit,
        )

    def _load_thread_metadata(self, thread_id: str) -> dict[str, Any]:
        path = self.settings.data_dir / "threads" / thread_id / "thread_meta.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_thread_metadata(self, thread_id: str, metadata: dict[str, Any]) -> None:
        thread_dir = self.settings.data_dir / "threads" / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        path = thread_dir / "thread_meta.json"
        existing = self._load_thread_metadata(thread_id)
        path.write_text(json.dumps({**existing, **metadata}, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _loaded_skills_from_metadata(metadata: dict[str, Any]) -> list[str]:
        raw = metadata.get("loaded_skills")
        if not isinstance(raw, list):
            return []
        return sorted({str(item) for item in raw if str(item)})
