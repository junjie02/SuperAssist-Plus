"""Trajectory collector — thread-safe JSONL writer with resume support."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TrajectoryCollector:
    """Collects LLM trajectories and writes them as JSONL.

    Thread-safe via a lock. Supports resume by checking existing
    trajectory IDs in the output file on initialization.

    Usage::

        collector = TrajectoryCollector(Path("output/trajectories.jsonl"))
        collector.record(trajectory_dict)
        collector.close()
    """

    def __init__(self, output_path: Path, flush_every: int = 10):
        self._output_path = output_path
        self._flush_every = flush_every
        self._lock = threading.Lock()
        self._count: int = 0
        self._written_ids: set[str] = set()
        self._fh: Any = None  # file handle, opened lazily
        self._total_written: int = 0  # includes already-existing lines

    # ---- public API ----------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of trajectories recorded in this session."""
        return self._count

    @property
    def total_count(self) -> int:
        """Total trajectories in file (pre-existing + new)."""
        return self._total_written + self._count

    def should_collect(self, trajectory_id: str) -> bool:
        """Return True if a trajectory with this ID has NOT been written yet."""
        with self._lock:
            self._ensure_open()
            return trajectory_id not in self._written_ids

    def record(self, trajectory: dict[str, Any]) -> bool:
        """Record one trajectory. Deduplicates by ``trajectory_id``.

        Returns True if the trajectory was written, False if skipped (duplicate).
        """
        tid = trajectory.get("trajectory_id")
        if not tid:
            raise ValueError("Trajectory must have a 'trajectory_id' field")

        with self._lock:
            self._ensure_open()
            if tid in self._written_ids:
                return False
            self._fh.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            self._written_ids.add(tid)
            self._count += 1
            if self._count % self._flush_every == 0:
                self._fh.flush()
            return True

    def flush(self) -> None:
        """Flush buffered writes to disk."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        """Flush and close the output file."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    # ---- internals -----------------------------------------------------------

    def _ensure_open(self) -> None:
        """Open output file if not already open. Scan existing IDs for resume."""
        if self._fh is not None:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._output_path.exists():
            with open(self._output_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        tid = obj.get("trajectory_id")
                        if tid:
                            self._written_ids.add(tid)
                            self._total_written += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

        self._fh = open(self._output_path, "a", encoding="utf-8")
