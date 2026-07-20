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
            if existing and isinstance(existing.get("thread_id"), str):
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
        return [{**entry, "key": key} for key, entry in self._data.items()]

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
