from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock


class SubagentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class SubagentResult:
    task_id: str
    description: str
    subagent_type: str
    status: SubagentStatus = SubagentStatus.PENDING
    result: str = ""
    error: str = ""
    ai_messages: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return data


class SubagentTaskStore:
    def __init__(self, max_items: int = 200) -> None:
        self.max_items = max_items
        self._items: dict[str, SubagentResult] = {}
        self._order: deque[str] = deque()
        self._lock = Lock()

    def put(self, result: SubagentResult) -> None:
        with self._lock:
            if result.task_id not in self._items:
                self._order.appendleft(result.task_id)
            self._items[result.task_id] = result
            while len(self._order) > self.max_items:
                old_id = self._order.pop()
                self._items.pop(old_id, None)

    def get(self, task_id: str) -> SubagentResult | None:
        with self._lock:
            return self._items.get(task_id)

    def list(self, limit: int = 50) -> list[SubagentResult]:
        with self._lock:
            task_ids = list(self._order)[: max(1, min(limit, self.max_items))]
            return [self._items[task_id] for task_id in task_ids if task_id in self._items]

    def delete(self, task_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(task_id, None) is not None
            if existed:
                self._order = deque(item for item in self._order if item != task_id)
            return existed


TASK_STORE = SubagentTaskStore()
