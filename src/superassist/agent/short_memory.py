from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

SUMMARY_SYSTEM_PROMPT = """You compress conversation history for an AI assistant.

Write a concise, structured Markdown summary of the conversation to date.
Preserve durable context that will matter in future turns:
- explicit user preferences, constraints, identity/background, and goals
- current tasks, unfinished work, decisions, and blockers
- important facts learned from tools
- which tools were used, what they checked, and any failures
- loaded skill names

Do not preserve long webpage/file contents, repeated greetings, or incidental wording.
Prefer stable facts and task state over chronology unless chronology matters.
"""


@dataclass(frozen=True)
class ShortMemoryLoad:
    messages: list[BaseMessage]
    records: list[dict[str, Any]]
    summary: str


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars // 2)


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_short_memory(
    messages_path: Path,
    metadata: dict[str, Any],
    *,
    token_limit: int,
) -> ShortMemoryLoad:
    records = read_jsonl(messages_path)
    summary = str(metadata.get("summary") or "").strip()
    budget = max(0, token_limit - estimate_tokens(summary))
    selected: list[dict[str, Any]] = []
    total = 0
    for record in reversed(records):
        cost = estimate_tokens(_record_text(record))
        if selected and total + cost > budget:
            break
        selected.append(record)
        total += cost
    selected.reverse()

    messages: list[BaseMessage] = []
    if summary:
        messages.append(HumanMessage(content=f"Here is a summary of the conversation to date:\n\n{summary}", name="summary"))
    messages.extend(record_to_message(record) for record in selected)
    return ShortMemoryLoad(messages=messages, records=selected, summary=summary)


def turn_records(
    *,
    user_message: str,
    assistant_answer: str,
    tool_events: list[dict[str, Any]],
    include_tool_events: bool,
) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = [{"role": "user", "content": user_message, "created_at": now}]
    if include_tool_events:
        records.extend(_compact_tool_event(event, now) for event in _tool_result_events(tool_events))
    records.append({"role": "assistant", "content": assistant_answer, "created_at": now})
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
    if estimate_tokens(summary) + sum(estimate_tokens(_record_text(record)) for record in records) <= token_limit:
        return {}

    old_records, recent_records = split_records_for_compression(records, keep_recent_turns)
    if not old_records:
        return {}

    prompt = build_summary_prompt(
        previous_summary=summary,
        records=old_records,
        summary_target_tokens=summary_target_tokens,
        loaded_skills=loaded_skills,
    )
    try:
        response = model.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        new_summary = str(response.content).strip()
    except Exception as exc:
        return {"short_memory_compression_error": f"{type(exc).__name__}: {exc}"}

    if not new_summary:
        return {"short_memory_compression_error": "empty summary"}

    write_jsonl(messages_path, recent_records)
    return {
        "summary": new_summary,
        "summary_updated_at": datetime.now(UTC).isoformat(),
        "short_memory_compressed": True,
        "short_memory_compressed_records": len(old_records),
    }


def split_records_for_compression(
    records: list[dict[str, Any]],
    keep_recent_turns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if keep_recent_turns <= 0:
        return records, []

    user_seen = 0
    split_at = 0
    for index in range(len(records) - 1, -1, -1):
        if records[index].get("role") == "user":
            user_seen += 1
            if user_seen == keep_recent_turns:
                split_at = index
                break
    if user_seen < keep_recent_turns:
        return [], records
    return records[:split_at], records[split_at:]


def build_summary_prompt(
    *,
    previous_summary: str,
    records: list[dict[str, Any]],
    summary_target_tokens: int,
    loaded_skills: list[str],
) -> str:
    history = "\n".join(_record_text(record) for record in records)
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
    return HumanMessage(content=str(record.get("content") or ""))


def _record_text(record: dict[str, Any]) -> str:
    role = str(record.get("role") or "")
    if role == "tool_event":
        return _tool_event_text(record)
    return f"{role}: {record.get('content') or ''}"


def _tool_event_text(record: dict[str, Any]) -> str:
    return (
        f"Tool event: {record.get('tool') or ''}\n"
        f"Args: {json.dumps(record.get('args') or {}, ensure_ascii=False, default=str)}\n"
        f"Status: {record.get('status') or 'success'}\n"
        f"Error: {record.get('error')}"
    )


def _tool_result_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == "tool_result"]


def _compact_tool_event(event: dict[str, Any], created_at: str) -> dict[str, Any]:
    status = str(event.get("status") or "success")
    error = event.get("error")
    if status == "error" and error is None:
        error = event.get("content")
    return {
        "role": "tool_event",
        "tool": str(event.get("tool") or event.get("name") or ""),
        "args": event.get("args") or {},
        "status": status,
        "error": None if error is None else str(error)[:1000],
        "created_at": created_at,
    }
