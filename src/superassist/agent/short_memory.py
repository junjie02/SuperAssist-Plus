from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from superassist.agent.streaming import clean_answer_text

SUMMARY_SYSTEM_PROMPT = """You compress conversation history for an AI assistant.

Write a concise, structured Markdown summary of the conversation to date.
Preserve durable context that will matter in future turns:
- explicit user preferences, constraints, identity/background, and goals
- current tasks, unfinished work, decisions, and blockers
- important facts learned from tools
- which tools were used, what they checked, and any failures
- question-relevant visual facts from <ImageDescription> blocks when they may matter in later turns
- loaded skill names

Do not preserve long webpage/file contents, repeated greetings, or incidental wording.
Prefer stable facts and task state over chronology unless chronology matters.
"""

USER_TIMESTAMP_LABEL = "系统时间"


@dataclass(frozen=True)
class ShortMemoryLoad:
    messages: list[BaseMessage]
    records: list[dict[str, Any]]
    summary: str


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if not text:
        return 0
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_short_memory(
    messages_path: Path,
    metadata: dict[str, Any],
    *,
    keep_recent_turns: int,
    token_limit: int | None = None,
) -> ShortMemoryLoad:
    """Load the complete active segment after the latest summary checkpoint.

    ``keep_recent_turns`` and ``token_limit`` remain accepted for caller
    compatibility, but limits are enforced only by whole-segment compaction.
    Silently trimming records here would turn the history back into a sliding
    window and invalidate the append-only prompt prefix on every turn.
    """

    records = read_jsonl(messages_path)
    compacted_count = _compacted_record_count(metadata, len(records))
    selected = records[compacted_count:]
    summary = str(metadata.get("summary") or "").strip()
    messages = [record_to_message(record) for record in selected]
    return ShortMemoryLoad(messages=messages, records=selected, summary=summary)


def format_short_memory_section(summary: str) -> str:
    rendered = escape(summary.strip()) if summary.strip() else "(no compressed summary yet)"
    return (
        "<ShortMemory>\n"
        "<Summary format=\"markdown\">\n"
        f"{rendered}\n"
        "</Summary>\n"
        "<RecentConversation>\n"
        "The native user and assistant messages that follow are the uncompressed recent conversation. "
        "Prefer newer explicit user statements over this summary.\n"
        "</RecentConversation>\n"
        "</ShortMemory>"
    )


def turn_records(
    *,
    user_message: str,
    assistant_answer: str,
    user_created_at: str | None = None,
    assistant_created_at: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": user_message,
            "created_at": user_created_at or datetime.now(UTC).isoformat(),
        }
    ]
    records.append(
        {
            "role": "assistant",
            "content": assistant_answer,
            "created_at": assistant_created_at or datetime.now(UTC).isoformat(),
        }
    )
    return records


def maybe_compress_short_memory(
    *,
    messages_path: Path,
    metadata: dict[str, Any],
    model: BaseChatModel,
    token_limit: int,
    keep_recent_turns: int,
    summary_target_tokens: int,
    loaded_skills: list[str],
) -> dict[str, Any]:
    records = read_jsonl(messages_path)
    summary = str(metadata.get("summary") or "").strip()
    compacted_count = _compacted_record_count(metadata, len(records))
    active_records = records[compacted_count:]
    active_turns = sum(1 for record in active_records if record.get("role") == "user")
    total_tokens = estimate_tokens(summary) + sum(estimate_tokens(record_text(record)) for record in active_records)
    over_turn_limit = keep_recent_turns > 0 and active_turns >= keep_recent_turns
    over_token_limit = token_limit > 0 and total_tokens >= token_limit
    if not over_turn_limit and not over_token_limit:
        return {}
    if not active_records:
        return {}

    prompt = build_summary_prompt(
        previous_summary=summary,
        records=active_records,
        summary_target_tokens=summary_target_tokens,
        loaded_skills=loaded_skills,
    )
    try:
        response = model.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        new_summary = clean_answer_text(response.content)
    except Exception as exc:
        return {"short_memory_compression_error": f"{type(exc).__name__}: {exc}"}

    if not new_summary:
        return {"short_memory_compression_error": "empty summary"}

    return {
        "summary": new_summary,
        "summary_updated_at": datetime.now(UTC).isoformat(),
        "summary_version": int(metadata.get("summary_version") or 0) + 1,
        "short_memory_compacted_records": len(records),
        "short_memory_compressed": True,
        "short_memory_compressed_records": len(active_records),
        "short_memory_compaction_trigger": "turns" if over_turn_limit else "tokens",
        "short_memory_active_turns": 0,
        "short_memory_active_tokens": estimate_tokens(new_summary),
    }


def _compacted_record_count(metadata: dict[str, Any], record_count: int) -> int:
    try:
        value = int(metadata.get("short_memory_compacted_records") or 0)
    except (TypeError, ValueError):
        return 0
    return value if 0 <= value <= record_count else 0


def build_summary_prompt(
    *,
    previous_summary: str,
    records: list[dict[str, Any]],
    summary_target_tokens: int,
    loaded_skills: list[str],
) -> str:
    history = "\n".join(record_text(record) for record in records)
    previous = previous_summary or "(none)"
    skills = ", ".join(loaded_skills) if loaded_skills else "(none)"
    return (
        f"Target length: about {summary_target_tokens} tokens or less.\n\n"
        f"Loaded skills: {skills}\n\n"
        "Previous summary:\n"
        f"{previous}\n\n"
        "Older conversation records to merge into the summary:\n"
        f"{history}\n\n"
        "Return only the updated Markdown summary."
    )


def record_to_message(record: dict[str, Any]) -> BaseMessage:
    role = str(record.get("role") or "")
    if role == "assistant":
        return AIMessage(content=str(record.get("content") or ""))
    if role == "tool_event":
        return HumanMessage(content=_tool_event_text(record), name="tool_event")
    return HumanMessage(
        content=timestamp_user_content(
            str(record.get("content") or ""),
            str(record.get("created_at") or ""),
        )
    )


def record_text(record: dict[str, Any]) -> str:
    role = str(record.get("role") or "")
    if role == "tool_event":
        return _tool_event_text(record)
    content = record_to_message(record).content if role == "user" else str(record.get("content") or "")
    return f"{role}: {content}"


def timestamp_user_content(content: Any, created_at: str) -> Any:
    """Append stable receipt-time metadata to a user message without changing its raw input."""

    timestamp = str(created_at or "").strip()
    if not timestamp:
        return content
    marker = f"[{USER_TIMESTAMP_LABEL}: {timestamp}]"
    if isinstance(content, str):
        return _append_timestamp_marker(content, marker)
    if not isinstance(content, list):
        return _append_timestamp_marker(str(content or ""), marker)

    rendered = [dict(item) if isinstance(item, dict) else item for item in content]
    for index in range(len(rendered) - 1, -1, -1):
        item = rendered[index]
        if not isinstance(item, dict) or str(item.get("type") or "").lower() not in {"text", "input_text"}:
            continue
        item["text"] = _append_timestamp_marker(str(item.get("text") or ""), marker)
        return rendered
    rendered.append({"type": "text", "text": marker})
    return rendered


def _append_timestamp_marker(text: str, marker: str) -> str:
    if text.rstrip().endswith(marker):
        return text
    separator = "\n\n" if text.strip() else ""
    return f"{text}{separator}{marker}"


def _tool_event_text(record: dict[str, Any]) -> str:
    return (
        f"Tool event: {record.get('tool') or ''}\n"
        f"Args: {json.dumps(record.get('args') or {}, ensure_ascii=False, default=str)}\n"
        f"Status: {record.get('status') or 'success'}\n"
        f"Error: {record.get('error')}"
    )
