from __future__ import annotations

import json
import hashlib
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeishuStoredMessage:
    seq: int
    message_id: str
    chat_id: str
    sender_open_id: str
    sender_name: str
    text: str
    root_id: str | None
    chat_type: str
    mentions: list[dict[str, Any]]
    files: list[dict[str, str]]
    created_at: str


class FeishuMessageStore:
    """Durable, idempotent inbox for all visible Feishu messages."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def add_message(
        self,
        *,
        message_id: str,
        chat_id: str,
        sender_open_id: str,
        sender_name: str,
        text: str,
        root_id: str | None,
        chat_type: str,
        mentions: list[dict[str, Any]],
        files: list[dict[str, str]],
        created_at: str,
    ) -> tuple[bool, int]:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO feishu_messages (
                    message_id, chat_id, sender_open_id, sender_name, text,
                    root_id, chat_type, mentions_json, files_json, created_at, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    sender_open_id,
                    sender_name,
                    text,
                    root_id,
                    chat_type,
                    json.dumps(mentions, ensure_ascii=False),
                    json.dumps(files, ensure_ascii=False),
                    created_at or now,
                    now,
                ),
            )
            inserted = cursor.rowcount > 0
            row = conn.execute(
                "SELECT seq FROM feishu_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Feishu message was not persisted: {message_id}")
            return inserted, int(row[0])

    def list_unconsumed(self, chat_id: str, *, through_seq: int | None = None) -> list[FeishuStoredMessage]:
        with self._lock, self._connect() as conn:
            cursor_row = conn.execute(
                "SELECT consumed_seq FROM feishu_conversation_cursors WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            consumed_seq = int(cursor_row[0]) if cursor_row else 0
            if through_seq is None:
                rows = conn.execute(
                    "SELECT * FROM feishu_messages WHERE chat_id = ? AND seq > ? ORDER BY seq",
                    (chat_id, consumed_seq),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feishu_messages WHERE chat_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
                    (chat_id, consumed_seq, through_seq),
                ).fetchall()
        return [self._to_message(row) for row in rows]

    def latest_seq(self, chat_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM feishu_messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def commit_consumed(self, chat_id: str, through_seq: int) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feishu_conversation_cursors (chat_id, consumed_seq, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    consumed_seq = MAX(consumed_seq, excluded.consumed_seq),
                    updated_at = excluded.updated_at
                """,
                (chat_id, through_seq, now),
            )
            conn.execute(
                """
                DELETE FROM feishu_image_payloads
                WHERE message_id IN (
                    SELECT message_id FROM feishu_messages
                    WHERE chat_id = ? AND seq <= ?
                )
                """,
                (chat_id, through_seq),
            )

    def save_image(
        self,
        *,
        message_id: str,
        image_key: str,
        data: bytes,
        mime_type: str,
    ) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO feishu_image_payloads (
                        message_id, image_key, sha256, mime_type, data, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, image_key) DO NOTHING
                    """,
                    (
                        message_id,
                        image_key,
                        hashlib.sha256(data).hexdigest(),
                        mime_type,
                        data,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except sqlite3.IntegrityError:
            # Direct helper callers may download before their message has entered the inbox.
            return

    def get_image(self, *, message_id: str, image_key: str) -> tuple[bytes, str] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data, mime_type FROM feishu_image_payloads WHERE message_id = ? AND image_key = ?",
                (message_id, image_key),
            ).fetchone()
        if row is None:
            return None
        return bytes(row["data"]), str(row["mime_type"])

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feishu_messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    chat_id TEXT NOT NULL,
                    sender_open_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    root_id TEXT,
                    chat_type TEXT NOT NULL DEFAULT '',
                    mentions_json TEXT NOT NULL DEFAULT '[]',
                    files_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    received_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feishu_messages_chat_seq
                    ON feishu_messages(chat_id, seq);
                CREATE TABLE IF NOT EXISTS feishu_conversation_cursors (
                    chat_id TEXT PRIMARY KEY,
                    consumed_seq INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feishu_image_payloads (
                    message_id TEXT NOT NULL,
                    image_key TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    data BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(message_id, image_key),
                    FOREIGN KEY(message_id) REFERENCES feishu_messages(message_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_feishu_image_payloads_sha256
                    ON feishu_image_payloads(sha256);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _to_message(row: sqlite3.Row) -> FeishuStoredMessage:
        return FeishuStoredMessage(
            seq=int(row["seq"]),
            message_id=str(row["message_id"]),
            chat_id=str(row["chat_id"]),
            sender_open_id=str(row["sender_open_id"]),
            sender_name=str(row["sender_name"] or ""),
            text=str(row["text"] or ""),
            root_id=str(row["root_id"]) if row["root_id"] else None,
            chat_type=str(row["chat_type"] or ""),
            mentions=_json_list(row["mentions_json"]),
            files=_json_list(row["files_json"]),
            created_at=str(row["created_at"]),
        )


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


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

    def get_entry_by_thread_id(self, thread_id: str) -> dict[str, Any] | None:
        """Resolve the Feishu conversation that owns a main-agent thread."""

        with self._lock:
            for key, entry in self._data.items():
                if entry.get("thread_id") == thread_id:
                    return {**entry, "key": key}
        return None

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
