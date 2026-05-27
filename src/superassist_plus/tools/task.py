from __future__ import annotations

import asyncio
import logging
from threading import BoundedSemaphore

from langchain_core.tools import tool

from superassist_plus.config import get_settings
from superassist_plus.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config

logger = logging.getLogger(__name__)
_semaphore = BoundedSemaphore(value=3)


@tool("task")
def task(description: str, prompt: str, subagent_type: str = "general-purpose") -> str:
    """Delegate a complex task to a subagent and wait for its result.

    Args:
        description: Short 3-8 word description for tracking.
        prompt: Full task instructions for the subagent.
        subagent_type: Subagent type, either general-purpose or research.
    """

    settings = get_settings()
    if not settings.subagents_enabled:
        return "Error: Subagents are disabled by SUPERASSIST_PLUS_SUBAGENTS_ENABLED=false"
    config = get_subagent_config(subagent_type, settings)
    if config is None:
        available = ", ".join(get_available_subagent_names(settings))
        logger.warning("Task rejected: unknown subagent_type=%s available=%s", subagent_type, available)
        return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"
    logger.info(
        "Task requested: description=%s subagent_type=%s timeout=%ss",
        description,
        subagent_type,
        config.timeout_seconds,
    )
    acquired = _semaphore.acquire(blocking=True, timeout=config.timeout_seconds)
    if not acquired:
        logger.warning("Task timed out waiting for subagent slot: description=%s subagent_type=%s", description, subagent_type)
        return f"Task timed out. Error: No subagent slot available after {config.timeout_seconds}s"
    try:
        from superassist_plus.tools import default_tools

        executor = SubagentExecutor(config=config, tools=default_tools(include_task=False), settings=settings)
        result = executor.run(prompt, description=description)
        logger.info(
            "Task finished: task_id=%s description=%s subagent_type=%s status=%s error=%s",
            result.task_id,
            description,
            subagent_type,
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


async def atask(description: str, prompt: str, subagent_type: str = "general-purpose") -> str:
    return await asyncio.to_thread(task.invoke, {"description": description, "prompt": prompt, "subagent_type": subagent_type})
