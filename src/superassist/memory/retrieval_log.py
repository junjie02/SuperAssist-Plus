"""Structured, privacy-conscious timing log for long-term memory retrieval."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_LOG_LOCK = threading.Lock()
_LOG_BACKUPS = 3


def append_memory_retrieval_log(path: Path, payload: dict[str, Any], *, max_bytes: int) -> None:
    """Append one JSONL timing record without allowing logging to break recall."""

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "retrieval_id": f"memory_retrieval_{uuid4().hex}",
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    encoded_size = len(line.encode("utf-8"))

    try:
        with _LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size + encoded_size > max(1024, max_bytes):
                _rotate_log(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except OSError as exc:
        logger.warning("Unable to append memory retrieval timing log: %s", exc)


def _rotate_log(path: Path) -> None:
    for index in range(_LOG_BACKUPS, 0, -1):
        source = path if index == 1 else Path(f"{path}.{index - 1}")
        target = Path(f"{path}.{index}")
        if not source.exists():
            continue
        if target.exists():
            target.unlink()
        source.replace(target)


__all__ = ["append_memory_retrieval_log"]
