"""Shared helpers for accumulating streaming AIMessage chunks.

LangGraph's ``stream_mode=["messages", "values"]`` emits AIMessage chunks
where each chunk's ``content`` may already include earlier text or arrive
as an incremental delta. This module provides the small state machine both
the lead runtime and the subagent executor use to fold those chunks into
the most recent assembled text per message-id.
"""

from __future__ import annotations

from dataclasses import dataclass
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
                block_type = str(item.get("type") or "").lower()
                if block_type in {"reasoning", "thinking", "reasoning_content", "chain_of_thought"}:
                    continue
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


@dataclass
class StreamParts:
    raw_text: str = ""
    structured_reasoning: str = ""


def _metadata_reasoning(message: Any) -> str:
    parts: list[str] = []
    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(message, container_name, None)
        if not isinstance(container, dict):
            continue
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = container.get(key)
            if isinstance(value, str) and value and value not in parts:
                parts.append(value)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type not in {"reasoning", "thinking", "reasoning_content", "chain_of_thought"}:
                continue
            value = block.get("text") or block.get("content") or block.get("reasoning")
            if isinstance(value, str) and value and value not in parts:
                parts.append(value)
            summary = block.get("summary")
            if isinstance(summary, list):
                summary_text = "".join(
                    str(item.get("text") or "") for item in summary if isinstance(item, dict)
                )
                if summary_text and summary_text not in parts:
                    parts.append(summary_text)
    return "".join(parts)


def split_inline_thinking(text: str) -> tuple[str, str]:
    """Split streamed ``<think>`` blocks from user-visible answer text."""

    reasoning: list[str] = []
    answer: list[str] = []
    cursor = 0
    lower = text.lower()
    while cursor < len(text):
        start = lower.find("<think>", cursor)
        if start < 0:
            tail = text[cursor:]
            # Do not briefly leak a tag while it is split across chunks.
            for size in range(min(len(tail), len("<think>") - 1), 0, -1):
                if "<think>".startswith(tail[-size:].lower()):
                    answer.append(tail[:-size])
                    return "".join(reasoning), "".join(answer)
            answer.append(tail)
            break
        answer.append(text[cursor:start])
        body_start = start + len("<think>")
        end = lower.find("</think>", body_start)
        if end < 0:
            body = text[body_start:]
            for size in range(min(len(body), len("</think>") - 1), 0, -1):
                if "</think>".startswith(body[-size:].lower()):
                    body = body[:-size]
                    break
            reasoning.append(body)
            break
        reasoning.append(text[body_start:end])
        cursor = end + len("</think>")
    return "".join(reasoning), "".join(answer)


def accumulate_stream_parts(
    buffers: dict[str, StreamParts],
    current_message_id: str | None,
    chunk: Any,
) -> tuple[str | None, str | None, str | None]:
    """Accumulate independent reasoning and answer streams for one AI message."""

    message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return None, None, current_message_id
    raw_text = message_text(getattr(message, "content", ""))
    structured = _metadata_reasoning(message)
    if not raw_text and not structured:
        return None, None, current_message_id
    message_id = str(getattr(message, "id", "") or current_message_id or "__default__")
    state = buffers.setdefault(message_id, StreamParts())
    state.raw_text = merge_stream_text(state.raw_text, raw_text)
    state.structured_reasoning = merge_stream_text(state.structured_reasoning, structured)
    inline_reasoning, answer = split_inline_thinking(state.raw_text)
    reasoning = merge_stream_text(state.structured_reasoning, inline_reasoning)
    return reasoning or None, answer or None, message_id


def clean_answer_text(content: Any) -> str:
    """Return answer text without structured or inline reasoning blocks."""

    _reasoning, answer = split_inline_thinking(message_text(content))
    return answer.strip()


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


__all__ = [
    "StreamParts",
    "accumulate_stream_parts",
    "accumulate_stream_text",
    "clean_answer_text",
    "merge_stream_text",
    "message_text",
    "split_inline_thinking",
]
