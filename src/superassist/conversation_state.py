"""Deterministic control operations for per-thread conversational state."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from superassist.config import Settings
from superassist.redis_store import RedisRuntimeStore, get_redis_store

CLEAR_COMMAND = "/clear"
CLEAR_CONFIRMATION = (
    "已清除当前会话的短期对话和已激活 Skill。"
    "长期记忆、知识库、错题本、申论文件和账号数据未删除。"
)
CLEAR_GROUP_CONFIRMATION = (
    "已清除本群共享会话的短期对话和已激活 Skill。"
    "长期记忆、知识库、错题本、申论文件和账号数据未删除。"
)
CLEAR_WITH_ATTACHMENTS_MESSAGE = "请单独发送 /clear，不要同时携带图片或文件。"


class ConversationStateBusyError(RuntimeError):
    """Raised when another process owns a thread state lock for too long."""


@dataclass(frozen=True)
class ClearResult:
    thread_id: str
    generation: int
    cleared_message_records: int
    cleared_skills: tuple[str, ...]


_LOCAL_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def parse_clear_command(text: str) -> bool:
    """Match only a standalone /clear command, ignoring case and outer whitespace."""

    return str(text or "").strip().casefold() == CLEAR_COMMAND


def thread_generation(metadata: dict[str, Any] | None) -> int:
    try:
        return max(0, int((metadata or {}).get("conversation_generation") or 0))
    except (TypeError, ValueError):
        return 0


@contextmanager
def conversation_state_lock(
    settings: Settings,
    thread_id: str,
    *,
    redis_store: RedisRuntimeStore | None = None,
    wait_seconds: float = 30.0,
) -> Iterator[None]:
    """Serialize turns and clear operations locally and across runtime processes."""

    lock_key = f"{settings.data_dir.resolve()}::{thread_id}"
    with _LOCAL_LOCKS_GUARD:
        local_lock = _LOCAL_LOCKS.setdefault(lock_key, threading.RLock())
    if not local_lock.acquire(timeout=max(0.1, wait_seconds)):
        raise ConversationStateBusyError(f"Timed out waiting for local thread lock: {thread_id}")
    try:
        store = redis_store or get_redis_store(settings)
        deadline = time.monotonic() + max(0.1, wait_seconds)
        ttl_seconds = max(1200, int(settings.subagent_timeout_seconds) + 300)
        while True:
            with store.lock("thread-state", thread_id, ttl_seconds=ttl_seconds) as acquired:
                if acquired:
                    yield
                    return
            if time.monotonic() >= deadline:
                raise ConversationStateBusyError(f"Timed out waiting for distributed thread lock: {thread_id}")
            time.sleep(0.05)
    finally:
        local_lock.release()


def clear_short_state(
    settings: Settings,
    *,
    thread_id: str,
    user_id: str,
    redis_store: RedisRuntimeStore | None = None,
    acquire_lock: bool = True,
) -> ClearResult:
    """Clear only recent conversation and Skill activation state for one thread."""

    store = redis_store or get_redis_store(settings)
    if acquire_lock:
        with conversation_state_lock(settings, thread_id, redis_store=store):
            return clear_short_state(
                settings,
                thread_id=thread_id,
                user_id=user_id,
                redis_store=store,
                acquire_lock=False,
            )

    thread_dir = _thread_dir(settings, thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = thread_dir / "thread_meta.json"
    messages_path = thread_dir / "messages.jsonl"
    metadata = _read_metadata_for_update(metadata_path)
    existing_owner = str(metadata.get("user_id") or "")
    if existing_owner and existing_owner != user_id:
        raise PermissionError("thread does not belong to the requesting user")

    cleared_records = _nonempty_line_count(messages_path)
    activations = metadata.get("skill_activations")
    cleared_skills = tuple(sorted(str(name) for name in activations)) if isinstance(activations, dict) else ()
    generation = thread_generation(metadata) + 1

    _atomic_write_text(messages_path, "")
    for key in list(metadata):
        if key == "summary" or key == "summary_updated_at" or key.startswith("short_memory_"):
            metadata.pop(key, None)
    metadata.update(
        {
            "user_id": existing_owner or user_id,
            "summary": "",
            "summary_version": 0,
            "short_memory_compacted_records": 0,
            "short_memory_active_turns": 0,
            "short_memory_active_tokens": 0,
            "loaded_skills": [],
            "skill_activations": {},
            "conversation_generation": generation,
            "last_cleared_at": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_write_json(metadata_path, metadata)
    store.clear_thread_short_state(thread_id)
    return ClearResult(thread_id, generation, cleared_records, cleared_skills)


def load_thread_metadata(settings: Settings, thread_id: str) -> dict[str, Any]:
    return _read_metadata(_thread_dir(settings, thread_id) / "thread_meta.json")


def _thread_dir(settings: Settings, thread_id: str) -> Path:
    if not thread_id:
        raise ValueError("thread_id is required")
    root = (settings.data_dir / "threads").resolve()
    candidate = (root / thread_id).resolve()
    if candidate.parent != root:
        raise ValueError("invalid thread_id")
    return candidate


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_metadata_for_update(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("thread metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("thread metadata must be a JSON object")
    return value


def _nonempty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=f".{uuid4().hex}.tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


__all__ = [
    "CLEAR_COMMAND",
    "CLEAR_CONFIRMATION",
    "CLEAR_GROUP_CONFIRMATION",
    "CLEAR_WITH_ATTACHMENTS_MESSAGE",
    "ClearResult",
    "ConversationStateBusyError",
    "clear_short_state",
    "conversation_state_lock",
    "load_thread_metadata",
    "parse_clear_command",
    "thread_generation",
]
