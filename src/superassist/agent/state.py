"""Shared agent state schema.

SuperAssistState extends LangChain's AgentState (which already provides
``messages`` and ``remaining_steps``) with the SuperAssist-specific runtime
context every middleware reads or writes.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentState
from typing_extensions import NotRequired


class SuperAssistState(AgentState):
    """LangChain agent state extended with SuperAssist runtime metadata."""

    user_id: str
    thread_id: str
    input: str
    memory_event_id: NotRequired[str]
    memory_recall: NotRequired[dict[str, Any]]
    memory_write_context: NotRequired[dict[str, Any]]
    tool_events: NotRequired[list[dict[str, Any]]]
    loaded_skills: NotRequired[list[str]]
    skill_activations: NotRequired[dict[str, float]]
    active_skills_at_turn_start: NotRequired[list[str]]
    rag_mode: NotRequired[bool]
    rag_context: NotRequired[str]
    rag_sources: NotRequired[list[str]]
    rag_retrieval: NotRequired[dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]


__all__ = ["SuperAssistState"]
