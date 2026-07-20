"""Compose the SuperAssist LangChain agent.

This module mirrors ProAssist's ``_make_lead_agent`` factory: there is no
outer LangGraph wrapper around ``create_agent``. Cross-cutting behavior all
lives in middleware. The order below matters; LangChain dispatches
``before_*`` hooks in registration order and ``after_*`` hooks in reverse,
which is documented per-middleware.

Canonical chain (top to bottom):

* ``ToolErrorMiddleware``        — wrap_tool_call: convert exceptions to ToolMessages first
* ``ToolCallLimitMiddleware``    — wrap_tool_call: refuse new tool calls past the per-turn budget
* ``MemoryRecallMiddleware``     — before_agent: read graph memory, create user-turn event
* ``DynamicContextMiddleware``   — wrap_model_call: prepend recall+skills+time to system message
* ``ShortMemoryMiddleware``      — after_agent: persist messages.jsonl, compress when over budget
* ``ToolEventMiddleware``        — wrap_tool_call: collect tool start/result events
* ``SubagentLimitMiddleware``    — after_model: trim parallel ``task`` calls (subagents only)
* ``MemoryWriterMiddleware``     — after_agent: enqueue the durable memory write payload
* ``FinalTextMiddleware``        — after_agent: surface final_assistant_text in metadata
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from superassist.agent.state import SuperAssistState
from superassist.config import Settings, get_settings
from superassist.llm import create_chat_model
from superassist.memory.service import MemoryService
from superassist.memory.writer import MemoryWriteQueue, MemoryWriter
from superassist.middlewares import (
    DynamicContextMiddleware,
    FinalTextMiddleware,
    MemoryRecallMiddleware,
    MemoryWriterMiddleware,
    ShortMemoryMiddleware,
    SubagentLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolEventMiddleware,
)
from superassist.models import AgentRunEvent
from superassist.teams import AgentTeamConfig, TeamSupervisor, set_team_supervisor
from superassist.teams.config import AgentTeamConfigError
from superassist.tools import default_tools


class AgentBundle:
    """The compiled agent plus the collaborators a runtime needs to keep alive."""

    def __init__(
        self,
        *,
        agent: Any,
        settings: Settings,
        model: BaseChatModel,
        memory: MemoryService,
        memory_queue: MemoryWriteQueue,
        team_supervisor: TeamSupervisor | None,
        team_config_error: str | None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.model = model
        self.memory = memory
        self.memory_queue = memory_queue
        self.team_supervisor = team_supervisor
        self.team_config_error = team_config_error


def build_agent(
    settings: Settings | None = None,
    *,
    tool_event_reporter: Callable[[dict[str, Any]], None] | None = None,
    run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
) -> AgentBundle:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    team_supervisor, team_config_error = _build_team_supervisor(settings)
    set_team_supervisor(team_supervisor)

    model = create_chat_model(settings)
    memory = MemoryService(settings=settings)
    memory.preload_embedder()
    memory_queue = MemoryWriteQueue(
        MemoryWriter(memory, model, llm_enabled=settings.memory_llm_writer_enabled),
        debounce_seconds=settings.memory_debounce_seconds,
    )

    tools = (
        default_tools(
            include_task=settings.subagents_enabled,
            include_team_task=bool(team_supervisor and team_supervisor.enabled),
            run_event_reporter=run_event_reporter,
        )
        if settings.enable_tools
        else []
    )

    middlewares = _build_middleware_chain(
        settings,
        memory=memory,
        memory_queue=memory_queue,
        model=model,
        tool_event_reporter=tool_event_reporter,
    )

    if settings.enable_tools:
        from superassist.agent.prompts import compose_system_prompt

        system_prompt = compose_system_prompt(settings, team_supervisor=team_supervisor, team_config_error=team_config_error)
    else:
        from superassist.agent.prompts import SYSTEM_PROMPT

        system_prompt = SYSTEM_PROMPT

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middlewares,
        system_prompt=system_prompt,
        state_schema=SuperAssistState,
    )
    return AgentBundle(
        agent=agent,
        settings=settings,
        model=model,
        memory=memory,
        memory_queue=memory_queue,
        team_supervisor=team_supervisor,
        team_config_error=team_config_error,
    )


def _build_middleware_chain(
    settings: Settings,
    *,
    memory: MemoryService,
    memory_queue: MemoryWriteQueue,
    model: BaseChatModel,
    tool_event_reporter: Callable[[dict[str, Any]], None] | None,
) -> list[AgentMiddleware]:
    chain: list[AgentMiddleware] = [
        ToolErrorMiddleware(),
        ToolCallLimitMiddleware(settings.max_tool_calls),
        MemoryRecallMiddleware(memory),
        DynamicContextMiddleware(),
        ShortMemoryMiddleware(settings, model),
        ToolEventMiddleware(tool_event_reporter),
    ]
    if settings.subagents_enabled:
        chain.append(SubagentLimitMiddleware(settings.subagent_max_concurrent))
    chain.extend(
        [
            MemoryWriterMiddleware(memory_queue),
            FinalTextMiddleware(),
        ]
    )
    return chain


def _build_team_supervisor(settings: Settings) -> tuple[TeamSupervisor | None, str | None]:
    try:
        config = AgentTeamConfig.from_file()
    except AgentTeamConfigError as exc:
        return None, str(exc)
    if not config.enabled or not config.agents:
        return None, None
    return TeamSupervisor(config, settings=settings), None


__all__ = ["AgentBundle", "build_agent"]
