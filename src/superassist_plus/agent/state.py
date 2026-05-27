from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from typing_extensions import NotRequired


class SuperAssistState(TypedDict):
    messages: list[BaseMessage]
    user_id: str
    thread_id: str
    input: str
    memory_event_id: NotRequired[str]
    memory_recall: NotRequired[dict[str, Any]]
    memory_write_context: NotRequired[dict[str, Any]]
    tool_events: NotRequired[list[dict[str, Any]]]
    answer: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
    history_loaded: NotRequired[bool]
