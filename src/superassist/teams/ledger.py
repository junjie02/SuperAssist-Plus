from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock


GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    pass


class LedgerTamperError(LedgerError):
    pass


class TeamLedger:
    """Durable append-only JSONL ledger for a team thread.

    Every stream uses the same record envelope: a monotonic sequence, a
    previous-record hash, a content hash, and an HMAC.  ``FileLock`` protects
    the read/validate/append transaction across threads and processes.
    """

    def __init__(self, root: Path, *, key_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.key_path = (key_path or self.root / "supervisor.key").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def thread_dir(self, thread_id: str) -> Path:
        path = self.root / "threads" / _safe_name(thread_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ledger_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "ledger.jsonl"

    def inbox_path(self, thread_id: str, agent: str) -> Path:
        path = self.thread_dir(thread_id) / "inbox"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{_safe_name(agent)}.jsonl"

    def outbox_path(self, thread_id: str, agent: str) -> Path:
        path = self.thread_dir(thread_id) / "outbox"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{_safe_name(agent)}.raw.jsonl"

    def workspace_dir(self, thread_id: str, agent: str) -> Path:
        path = self.thread_dir(thread_id) / "workspaces" / _safe_name(agent)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_message(
        self,
        thread_id: str,
        *,
        sender: str,
        recipient: str,
        kind: str,
        body: str,
        artifact_paths: list[str] | None = None,
        parent_ids: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self.ledger_path(thread_id)
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = self._read_records_unlocked(path)
            self._validate_records(records)
            previous = records[-1] if records else None
            record: dict[str, Any] = {
                "id": f"msg_{uuid4().hex}",
                "seq": len(records) + 1,
                "thread_id": thread_id,
                "sender": sender,
                "recipient": recipient,
                "kind": kind,
                "body": body,
                "artifact_paths": artifact_paths or [],
                "parent_ids": parent_ids or [],
                "created_at": datetime.now(UTC).isoformat(),
                "prev_hash": previous["hash"] if previous else GENESIS_HASH,
            }
            if extra:
                record["extra"] = extra
            record["hash"] = _record_hash(record)
            record["sig"] = self._sign(record)
            _durable_append(path, record)
            return record

    def append_inbox(self, thread_id: str, agent: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._append_side_channel(self.inbox_path(thread_id, agent), thread_id, agent, payload)

    def append_raw(self, thread_id: str, agent: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._append_side_channel(self.outbox_path(thread_id, agent), thread_id, agent, payload)

    def _append_side_channel(self, path: Path, thread_id: str, agent: str, payload: dict[str, Any]) -> dict[str, Any]:
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = self._read_records_unlocked(path)
            self._validate_records(records)
            previous = records[-1] if records else None
            record = {
                "id": f"raw_{uuid4().hex}",
                "seq": len(records) + 1,
                "thread_id": thread_id,
                "agent": agent,
                "channel": "inbox" if path.parent.name == "inbox" else "outbox",
                "created_at": datetime.now(UTC).isoformat(),
                "payload": payload,
                "prev_hash": previous["hash"] if previous else GENESIS_HASH,
            }
            record["hash"] = _record_hash(record)
            record["sig"] = self._sign(record)
            _durable_append(path, record)
        return record

    def read_ledger(self, thread_id: str) -> list[dict[str, Any]]:
        path = self.ledger_path(thread_id)
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = self._read_records_unlocked(path)
            self._validate_records(records)
            return records

    def validate_thread(self, thread_id: str) -> None:
        self.read_ledger(thread_id)
        thread = self.thread_dir(thread_id)
        for channel_name in ("inbox", "outbox"):
            channel_dir = thread / channel_name
            if not channel_dir.exists():
                continue
            for path in channel_dir.glob("*.jsonl"):
                self._read_side_channel(path)

    def read_inbox(self, thread_id: str, agent: str) -> list[dict[str, Any]]:
        return self._read_side_channel(self.inbox_path(thread_id, agent))

    def read_raw(self, thread_id: str, agent: str) -> list[dict[str, Any]]:
        return self._read_side_channel(self.outbox_path(thread_id, agent))

    def _read_side_channel(self, path: Path) -> list[dict[str, Any]]:
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = self._read_records_unlocked(path)
            self._validate_records(records)
            return records

    def _read_records_unlocked(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerTamperError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(data, dict):
                    raise LedgerTamperError(f"Invalid ledger record at {path}:{line_number}")
                records.append(data)
        return records

    def _validate_records(self, records: list[dict[str, Any]]) -> None:
        previous_hash = GENESIS_HASH
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise LedgerTamperError(f"Invalid ledger record at index {index}")
            if record.get("seq") != index:
                raise LedgerTamperError(f"Ledger sequence mismatch at record {index}")
            if record.get("prev_hash") != previous_hash:
                raise LedgerTamperError(f"Ledger hash chain mismatch at record {index}")
            expected_hash = _record_hash(record)
            if record.get("hash") != expected_hash:
                raise LedgerTamperError(f"Ledger hash mismatch at record {index}")
            expected_sig = self._sign(record)
            if not hmac.compare_digest(str(record.get("sig") or ""), expected_sig):
                raise LedgerTamperError(f"Ledger signature mismatch at record {index}")
            previous_hash = expected_hash

    def _key(self) -> bytes:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.key_path) + ".lock")
        with lock:
            if self.key_path.exists():
                try:
                    key = bytes.fromhex(self.key_path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError) as exc:
                    raise LedgerError(f"Invalid HMAC key file: {self.key_path}") from exc
                if len(key) != 32:
                    raise LedgerError(f"Invalid HMAC key length in {self.key_path}")
                return key
            key = secrets.token_bytes(32)
            temporary = self.key_path.with_name(f".{self.key_path.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(key.hex())
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    temporary.chmod(0o600)
                except OSError:
                    pass
                os.replace(temporary, self.key_path)
                _fsync_directory(self.key_path.parent)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            return key

    def _sign(self, record: dict[str, Any]) -> str:
        return hmac.new(self._key(), _canonical_bytes(record, exclude={"sig"}), hashlib.sha256).hexdigest()


def _record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(record, exclude={"hash", "sig"})).hexdigest()


def _canonical_bytes(record: dict[str, Any], *, exclude: set[str]) -> bytes:
    payload = {key: value for key, value in record.items() if key not in exclude}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    name = "".join(keep).strip("._")
    return name or "default"


def _durable_append(path: Path, record: dict[str, Any]) -> None:
    """Append one complete JSON line and force it through the OS buffers."""
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; opening directories is unsupported on Windows."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
