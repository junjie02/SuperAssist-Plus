"""Shared helpers for accumulating streaming AIMessage chunks.

LangGraph's ``stream_mode=["messages", "values"]`` emits AIMessage chunks
where each chunk's ``content`` may already include earlier text or arrive
as an incremental delta. This module provides the small state machine both
the lead runtime and the subagent executor use to fold those chunks into
the most recent assembled text per message-id.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk


def message_text(content: Any) -> str:
    """Extract plain text from an AIMessage content (string or list-of-blocks)."""

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


def merge_stream_text(existing: str, incoming: str) -> str:
    """Combine an existing accumulated text with the next chunk."""

    if not existing:
        return incoming
    if incoming.startswith(existing):
        return incoming
    if existing.endswith(incoming):
        return existing
    return f"{existing}{incoming}"


def accumulate_stream_text(
    buffers: dict[str, str],
    current_message_id: str | None,
    chunk: Any,
) -> tuple[str | None, str | None]:
    """Update *buffers* with the next streaming chunk.

    Returns the latest text for the active message-id (or None if the chunk
    was not an AIMessage with text) and the message-id we used as the buffer
    key, so the caller can keep calling with consistent state.
    """

    message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return None, current_message_id
    text = message_text(getattr(message, "content", ""))
    if not text:
        return None, current_message_id
    message_id = str(getattr(message, "id", "") or current_message_id or "__default__")
    buffers[message_id] = merge_stream_text(buffers.get(message_id, ""), text)
    return buffers[message_id], message_id


__all__ = ["accumulate_stream_text", "merge_stream_text", "message_text"]
