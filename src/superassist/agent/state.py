"""Shared agent state schema.

SuperAssistState extends LangChain's AgentState (which already provides
``messages`` and ``remaining_steps``) with the SuperAssist-specific runtime
context every middleware reads or writes.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain.agents import AgentState
from typing_extensions import NotRequired


class SuperAssistState(AgentState):
    """LangChain agent state extended with SuperAssist runtime metadata."""

    user_id: str
    thread_id: str
    input: str
    memory_query: NotRequired[str]
    suppress_memory_write: NotRequired[bool]
    suppress_short_memory_write: NotRequired[bool]
    message_created_at: NotRequired[str]
    assistant_message_created_at: NotRequired[str]
    memory_event_id: NotRequired[str]
    memory_recall: NotRequired[dict[str, Any]]
    memory_write_context: NotRequired[dict[str, Any]]
    memory_source_context: NotRequired[dict[str, Any]]
    tool_events: NotRequired[list[dict[str, Any]]]
    loaded_skills: NotRequired[list[str]]
    skill_activations: NotRequired[dict[str, float]]
    active_skills_at_turn_start: NotRequired[list[str]]
    rag_mode: NotRequired[bool]
    rag_context: NotRequired[str]
    rag_sources: NotRequired[list[str]]
    rag_retrieval: NotRequired[dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]
    image_search_results: NotRequired[Annotated[dict[str, dict[str, Any]], _merge_dicts]]
    outbound_images: NotRequired[Annotated[list[dict[str, Any]], _merge_unique_images]]


def _merge_dicts(
    current: dict[str, dict[str, Any]] | None,
    update: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    return {**(current or {}), **(update or {})}


def _merge_unique_images(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*(current or []), *(update or [])]:
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            merged[candidate_id] = item
    return list(merged.values())[:3]


__all__ = ["SuperAssistState"]
