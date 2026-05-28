from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError

from superassist_plus.config import Settings, get_settings
from superassist_plus.llm import create_chat_model
from superassist_plus.models import AgentRunEvent
from superassist_plus.observability import runnable_trace_config, trace_extra, traceable, without_self

from .config import SubagentConfig
from .store import TASK_STORE, SubagentResult, SubagentStatus

logger = logging.getLogger(__name__)


class SubagentExecutor:
    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        settings: Settings | None = None,
        run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        self._run_event_reporter = run_event_reporter
        self._reported_subagent_text_seen: set[str] = set()
        self.tools = _filter_tools(tools, config.allowed_tools)
        self.model = create_chat_model(self.settings)
        self.graph = self._build_graph()
        logger.info(
            "SubagentExecutor initialized: subagent_type=%s tools=%s max_turns=%s timeout=%ss",
            self.config.name,
            [tool.name for tool in self.tools],
            self.config.max_turns,
            self.config.timeout_seconds,
        )

    def run(self, prompt: str, *, task_id: str | None = None, description: str = "") -> SubagentResult:
        resolved_task_id = task_id or f"subagent_{uuid4().hex[:12]}"
        return self._run_traced(
            prompt,
            task_id=resolved_task_id,
            description=description,
            **trace_extra(
                metadata={
                    "task_id": resolved_task_id,
                    "description": description,
                    "prompt_preview": prompt,
                    "subagent_type": self.config.name,
                    "allowed_tools": [tool.name for tool in self.tools],
                    "max_turns": self.config.max_turns,
                    "timeout_seconds": self.config.timeout_seconds,
                },
                tags=["subagent", self.config.name],
            ),
        )

    @traceable(name="subagent.run", run_type="chain", process_inputs=without_self)
    def _run_traced(self, prompt: str, *, task_id: str, description: str = "") -> SubagentResult:
        resolved_task_id = task_id
        holder = SubagentResult(
            task_id=resolved_task_id,
            description=description,
            subagent_type=self.config.name,
        )
        TASK_STORE.put(holder)
        logger.info("Subagent task created: task_id=%s description=%s subagent_type=%s", resolved_task_id, description, self.config.name)
        try:
            return _run_coro_sync(self.arun(prompt, result=holder))
        except TimeoutError as exc:
            holder.status = SubagentStatus.TIMED_OUT
            holder.error = str(exc)
            holder.completed_at = datetime.now(UTC)
            TASK_STORE.put(holder)
            logger.warning("Subagent task timed out: task_id=%s error=%s", holder.task_id, holder.error)
            return holder
        except Exception as exc:
            holder.status = SubagentStatus.FAILED
            holder.error = f"{type(exc).__name__}: {exc}"
            holder.completed_at = datetime.now(UTC)
            TASK_STORE.put(holder)
            logger.exception("Subagent task failed: task_id=%s", holder.task_id)
            return holder

    async def arun(self, prompt: str, *, result: SubagentResult | None = None) -> SubagentResult:
        holder = result or SubagentResult(
            task_id=f"subagent_{uuid4().hex[:12]}",
            description="",
            subagent_type=self.config.name,
        )
        holder.status = SubagentStatus.RUNNING
        TASK_STORE.put(holder)
        logger.info("Subagent task running: task_id=%s subagent_type=%s", holder.task_id, self.config.name)
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                output = await asyncio.to_thread(
                    self.graph.invoke,
                    {
                        "prompt": prompt,
                        "messages": [],
                        "result": holder,
                    },
                    runnable_trace_config(
                        run_name="subagent.graph",
                        tags=["subagent", self.config.name, "graph"],
                        metadata={
                            "task_id": holder.task_id,
                            "description": holder.description,
                            "subagent_type": holder.subagent_type,
                        },
                    ),
                )
            updated = output.get("result", holder)
            if isinstance(updated, SubagentResult):
                return updated
            return holder
        except TimeoutError:
            holder.status = SubagentStatus.TIMED_OUT
            holder.error = f"Subagent timed out after {self.config.timeout_seconds}s"
            holder.completed_at = datetime.now(UTC)
            TASK_STORE.put(holder)
            logger.warning("Subagent task timed out: task_id=%s error=%s", holder.task_id, holder.error)
            return holder
        except Exception as exc:
            holder.status = SubagentStatus.FAILED
            holder.error = f"{type(exc).__name__}: {exc}"
            holder.completed_at = datetime.now(UTC)
            TASK_STORE.put(holder)
            logger.exception("Subagent task failed: task_id=%s", holder.task_id)
            return holder

    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("prepare", self._prepare)
        graph.add_node("agent", self._agent)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "agent")
        graph.add_edge("agent", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _prepare(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = str(state.get("prompt") or "")
        logger.info("Subagent prepare: subagent_type=%s prompt_len=%d", self.config.name, len(prompt))
        return {
            "prompt": prompt,
            "result": state["result"],
            "messages": [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=prompt),
            ]
        }

    def _agent(self, state: dict[str, Any]) -> dict[str, Any]:
        holder = state["result"]
        logger.info("Subagent agent invoke: task_id=%s subagent_type=%s", holder.task_id, self.config.name)
        agent = create_agent(
            model=self.model,
            tools=self.tools,
        )
        try:
            result = self._invoke_agent(agent, state["messages"], holder)
        except GraphRecursionError:
            logger.warning("Subagent reached max recursion: task_id=%s max_turns=%s", holder.task_id, self.config.max_turns)
            messages = list(state["messages"])
            summary = self._summarize_after_recursion_limit(messages)
            holder.ai_messages.append(summary)
            TASK_STORE.put(holder)
            return {
                "prompt": state.get("prompt", ""),
                "messages": [*messages, AIMessage(content=summary)],
                "result": holder,
                "recursion_limited": True,
            }
        messages = list(result.get("messages", []))
        for message in messages:
            if isinstance(message, AIMessage):
                text = str(message.content or "").strip()
                if text:
                    holder.ai_messages.append(text)
                    TASK_STORE.put(holder)
                    self._report_subagent_text(text, holder=holder)
                    logger.info("Subagent AI message: task_id=%s chars=%d", holder.task_id, len(text))
        return {"prompt": state.get("prompt", ""), "messages": messages, "result": holder}

    def _invoke_agent(self, agent: Any, messages: list[BaseMessage], holder: SubagentResult) -> dict[str, Any]:
        if not hasattr(agent, "stream"):
            return agent.invoke(
                {"messages": messages},
                runnable_trace_config(
                    run_name="subagent.agent",
                    tags=["subagent", self.config.name, "agent"],
                    metadata={
                        "task_id": holder.task_id,
                        "description": holder.description,
                        "subagent_type": holder.subagent_type,
                        "max_turns": self.config.max_turns,
                    },
                )
                | {"recursion_limit": self.config.max_turns},
            )
        last_values: dict[str, Any] | None = None
        text_buffers: dict[str, str] = {}
        current_message_id: str | None = None
        for item in agent.stream(
            {"messages": messages},
            runnable_trace_config(
                run_name="subagent.agent.streaming",
                tags=["subagent", self.config.name, "agent", "streaming"],
                metadata={
                    "task_id": holder.task_id,
                    "description": holder.description,
                    "subagent_type": holder.subagent_type,
                    "max_turns": self.config.max_turns,
                },
            )
            | {"recursion_limit": self.config.max_turns},
            stream_mode=["messages", "values"],
        ):
            if isinstance(item, tuple) and len(item) == 2:
                mode, chunk = str(item[0]), item[1]
            else:
                mode, chunk = "values", item
            if mode == "messages":
                text, current_message_id = _accumulate_stream_text(text_buffers, current_message_id, chunk)
                if text:
                    self._report_subagent_text(text, holder=holder)
                continue
            if mode == "values" and isinstance(chunk, dict):
                last_values = chunk
        if last_values is None:
            return {"messages": messages}
        return last_values

    def _report_subagent_text(self, text: str, *, holder: SubagentResult) -> None:
        content = text.strip()
        if not content or self._run_event_reporter is None:
            return
        if content in self._reported_subagent_text_seen or any(
            previous.startswith(content) for previous in self._reported_subagent_text_seen
        ):
            return
        self._reported_subagent_text_seen.add(content)
        self._run_event_reporter(
            AgentRunEvent(
                type="subagent_text",
                message=content,
                metadata={
                    "task_id": holder.task_id,
                    "description": holder.description,
                    "subagent_type": holder.subagent_type,
                },
            )
        )

    def _summarize_after_recursion_limit(self, messages: list[BaseMessage]) -> str:
        summary_prompt = SystemMessage(
            content=(
                "The subagent reached its maximum recursion/turn limit before producing a final answer. "
                "Using the full conversation and tool context below, produce one concise final summary now. "
                "Start by explicitly saying that the subagent reached the maximum recursion limit, then summarize "
                "what has been completed, useful findings, remaining uncertainty, and any next steps."
            )
        )
        try:
            response = self.model.invoke([summary_prompt, *messages])
        except Exception as exc:
            logger.exception("Subagent recursion-limit summary failed")
            return (
                "Subagent reached the maximum recursion limit and the final summary call also failed. "
                f"Summary error: {type(exc).__name__}: {exc}"
            )
        text = str(getattr(response, "content", "") or "").strip()
        if not text:
            text = "Subagent reached the maximum recursion limit and returned no additional summary."
        if "maximum recursion" not in text.lower() and "最大递归" not in text:
            text = f"Subagent reached the maximum recursion limit. Summary:\n{text}"
        return text

    def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        holder = state["result"]
        answer = _last_ai_text(state.get("messages", []))
        holder.result = answer
        holder.status = SubagentStatus.COMPLETED
        holder.completed_at = datetime.now(UTC)
        TASK_STORE.put(holder)
        logger.info("Subagent finalized: task_id=%s status=%s result_chars=%d", holder.task_id, holder.status, len(answer))
        return {"result": holder}


def _filter_tools(tools: list[BaseTool], allowed: list[str] | None) -> list[BaseTool]:
    filtered = [tool for tool in tools if tool.name != "task"]
    if allowed is None:
        return filtered
    allowed_set = set(allowed)
    return [tool for tool in filtered if tool.name in allowed_set]


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content or "").strip()
    return ""


def _accumulate_stream_text(
    buffers: dict[str, str],
    current_message_id: str | None,
    chunk: Any,
) -> tuple[str | None, str | None]:
    message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return None, current_message_id
    text = _message_text(getattr(message, "content", ""))
    if not text:
        return None, current_message_id
    message_id = str(getattr(message, "id", "") or current_message_id or "__default__")
    buffers[message_id] = _merge_stream_text(buffers.get(message_id, ""), text)
    return buffers[message_id], message_id


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
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


def _merge_stream_text(existing: str, incoming: str) -> str:
    if not existing:
        return incoming
    if incoming.startswith(existing):
        return incoming
    if existing.endswith(incoming):
        return existing
    return f"{existing}{incoming}"


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with asyncio.Runner() as runner:
        return runner.run(coro)
