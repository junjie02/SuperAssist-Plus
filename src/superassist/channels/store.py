from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class FeishuThreadStore:
    """JSON-backed mapping from Feishu conversation topics to SuperAssist threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def get_or_create_thread_id(self, *, chat_id: str, topic_id: str, user_id: str) -> str:
        from uuid import uuid4

        key = self._key(chat_id, topic_id)
        with self._lock:
            existing = self._data.get(key)
            if (
                existing
                and isinstance(existing.get("thread_id"), str)
                and existing.get("user_id") == user_id
            ):
                existing["updated_at"] = time.time()
                self._save()
                return existing["thread_id"]
            thread_id = f"feishu_{uuid4().hex[:16]}"
            now = time.time()
            self._data[key] = {
                "thread_id": thread_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "topic_id": topic_id,
                "created_at": now,
                "updated_at": now,
            }
            self._save()
            return thread_id

    def list_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{**entry, "key": key} for key, entry in self._data.items()]

    def get_latest_chat_entry(self, chat_id: str) -> dict[str, Any] | None:
        """Return the most recently used main-agent thread for a Feishu chat."""

        with self._lock:
            matches = [
                {**entry, "key": key}
                for key, entry in self._data.items()
                if entry.get("chat_id") == chat_id
                and isinstance(entry.get("thread_id"), str)
                and isinstance(entry.get("user_id"), str)
            ]
        if not matches:
            return None
        return max(matches, key=lambda item: float(item.get("updated_at") or 0.0))

    def get_reasoning_effort(self, *, chat_id: str, topic_id: str, default: str) -> str:
        key = self._key(chat_id, topic_id)
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                return default
            effort = entry.get("reasoning_effort")
            return effort if isinstance(effort, str) and effort else default

    def set_reasoning_effort(self, *, chat_id: str, topic_id: str, effort: str) -> None:
        key = self._key(chat_id, topic_id)
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                raise KeyError(f"Unknown Feishu conversation: {key}")
            entry["reasoning_effort"] = effort
            entry["updated_at"] = time.time()
            self._save()

    @staticmethod
    def _key(chat_id: str, topic_id: str) -> str:
        return f"feishu:{chat_id}:{topic_id}"

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2)
            temp_name = handle.name
        Path(temp_name).replace(self.path)


class WeComThreadStore:
    """Persistent WeCom conversation mapping and per-chat RAG preference."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def resolve(
        self,
        *,
        chat_id: str,
        sender_user_id: str,
        user_id: str,
        rag_mode_default: bool,
        scope_id: str | None = None,
    ) -> tuple[str, bool]:
        from uuid import uuid4

        resolved_scope_id = scope_id or sender_user_id
        key = self._key(chat_id, resolved_scope_id)
        with self._lock:
            now = time.time()
            entry = self._data.get(key)
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("thread_id"), str)
                or entry.get("user_id") != user_id
            ):
                entry = {
                    "thread_id": f"wecom_{uuid4().hex[:16]}",
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "sender_user_id": sender_user_id,
                    "scope_id": resolved_scope_id,
                    "rag_mode": rag_mode_default,
                    "created_at": now,
                }
                self._data[key] = entry
            entry["updated_at"] = now
            self._save()
            return entry["thread_id"], bool(entry.get("rag_mode", rag_mode_default))

    def set_rag_mode(
        self,
        *,
        chat_id: str,
        sender_user_id: str,
        enabled: bool,
        scope_id: str | None = None,
    ) -> None:
        key = self._key(chat_id, scope_id or sender_user_id)
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                raise KeyError(f"Unknown WeCom conversation: {key}")
            entry["rag_mode"] = enabled
            entry["updated_at"] = time.time()
            self._save()

    def list_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{**entry, "key": key} for key, entry in self._data.items()]

    @staticmethod
    def _key(chat_id: str, scope_id: str) -> str:
        return f"wecom:{chat_id}:{scope_id}"

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2)
            temp_name = handle.name
        Path(temp_name).replace(self.path)
