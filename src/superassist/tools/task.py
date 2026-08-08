from __future__ import annotations

import logging
from collections.abc import Callable
from threading import BoundedSemaphore
from typing import Any

from langchain_core.tools import tool

from superassist.config import Settings, get_settings
from superassist.models import AgentRunEvent
from superassist.observability import trace_extra, traceable
from superassist.run_events import current_run_event_reporter
from superassist.teams.context import current_team_thread_id

logger = logging.getLogger(__name__)
_semaphore = BoundedSemaphore(value=3)
SubagentExecutor: Any = None


@tool("task")
def task(
    description: str,
    prompt: str,
    subagent_type: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    """Delegate a complex task to a subagent and wait for its result.

    Args:
        description: Short 3-8 word description for tracking.
        prompt: Full task instructions for the subagent.
        subagent_type: Registered subagent name. Leave blank to use the configured default.
        parameters: Structured task parameters documented by the selected subagent.
    """

    return run_task(description, prompt, subagent_type=subagent_type, parameters=parameters)


def make_task_tool(run_event_reporter: Callable[[AgentRunEvent], None] | None = None):
    @tool("task")
    def bound_task(
        description: str,
        prompt: str,
        subagent_type: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Delegate a complex task to a subagent and wait for its result.

        Args:
            description: Short 3-8 word description for tracking.
            prompt: Full task instructions for the subagent.
            subagent_type: Registered subagent name. Leave blank to use the configured default.
            parameters: Structured task parameters documented by the selected subagent.
        """

        return run_task(
            description,
            prompt,
            subagent_type=subagent_type,
            parameters=parameters,
            run_event_reporter=run_event_reporter,
        )

    return bound_task


def run_task(
    description: str,
    prompt: str,
    *,
    subagent_type: str = "",
    parameters: dict[str, Any] | None = None,
    run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
    parent_thread_id: str | None = None,
    settings: Settings | None = None,
) -> str:
    return _run_task_traced(
        description,
        prompt,
        subagent_type=subagent_type,
        parameters=parameters,
        run_event_reporter=run_event_reporter,
        parent_thread_id=parent_thread_id or current_team_thread_id(),
        settings=settings,
        **trace_extra(
            metadata={
                "description": description,
                "prompt_preview": prompt,
                "subagent_type": subagent_type,
                "parameter_names": sorted((parameters or {}).keys()),
            },
            tags=["tool", "task", "subagent"],
        ),
    )


@traceable(name="task.dispatch", run_type="tool")
def _run_task_traced(
    description: str,
    prompt: str,
    *,
    subagent_type: str = "",
    parameters: dict[str, Any] | None = None,
    run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
    parent_thread_id: str | None = None,
    settings: Settings | None = None,
) -> str:
    from superassist.subagents import SubagentRegistry

    settings = settings or get_settings()
    if not settings.subagents_enabled:
        return "Error: Subagents are disabled by SUPERASSIST_SUBAGENTS_ENABLED=false"
    registry = SubagentRegistry(settings)
    resolved_type = subagent_type.strip() or registry.default().name
    config = registry.get(resolved_type)
    if config is None:
        available = ", ".join(registry.names())
        logger.warning("Task rejected: unknown subagent_type=%s available=%s", resolved_type, available)
        return f"Error: Unknown subagent type '{resolved_type}'. Available: {available}"
    logger.info(
        "Task requested: description=%s subagent_type=%s timeout=%ss",
        description,
        resolved_type,
        config.timeout_seconds,
    )
    acquired = _semaphore.acquire(blocking=True, timeout=config.timeout_seconds)
    if not acquired:
        logger.warning("Task timed out waiting for subagent slot: description=%s subagent_type=%s", description, resolved_type)
        return f"Task timed out. Error: No subagent slot available after {config.timeout_seconds}s"
    try:
        from superassist.tools import default_tools

        task_parameters = dict(parameters or {})
        executor_class = SubagentExecutor
        if executor_class is None:
            from superassist.subagents.executor import SubagentExecutor as executor_class

        subagent_tools = default_tools(include_task=False, include_images=False)
        if config.allowed_tools and "daily_quiz_update" in config.allowed_tools:
            from superassist.channels.daily_quiz import make_daily_quiz_tool

            subagent_tools.append(
                make_daily_quiz_tool(
                    settings,
                    delegated_question_count=task_parameters.get("question_count"),
                )
            )
        executor = executor_class(
            config=config,
            tools=subagent_tools,
            settings=settings,
            run_event_reporter=run_event_reporter or current_run_event_reporter(),
            parent_thread_id=parent_thread_id,
            task_parameters=task_parameters,
        )
        result = executor.run(prompt, description=description)
        logger.info(
            "Task finished: task_id=%s description=%s subagent_type=%s status=%s error=%s",
            result.task_id,
            description,
            resolved_type,
            result.status,
            result.error or "",
        )
    finally:
        _semaphore.release()
    if result.status == "completed":
        return f"Task Succeeded. Result: {result.result}"
    if result.status == "timed_out":
        return f"Task timed out. Error: {result.error}"
    return f"Task failed. Error: {result.error}"
