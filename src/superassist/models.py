from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class NodeType(str, Enum):
    EVENT = "event"
    CONCEPT = "concept"
    INTENT = "intent"
    TIME = "time"


class EdgeType(str, Enum):
    GROUNDS = "GROUNDS"
    CAUSES = "CAUSES"
    TRIGGERS = "TRIGGERS"
    REINFORCES = "REINFORCES"
    PART_OF = "PART_OF"
    DERIVED_FROM = "DERIVED_FROM"
    DEADLINE_FOR = "DEADLINE_FOR"
    RELATED_TO = "RELATED_TO"
    USER_FEEDBACK = "USER_FEEDBACK"


EDGE_TYPE_DEFAULT_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.GROUNDS: 0.9,
    EdgeType.CAUSES: 0.9,
    EdgeType.TRIGGERS: 0.8,
    EdgeType.USER_FEEDBACK: 0.8,
    EdgeType.REINFORCES: 0.7,
    EdgeType.PART_OF: 0.7,
    EdgeType.DERIVED_FROM: 0.6,
    EdgeType.DEADLINE_FOR: 0.6,
    EdgeType.RELATED_TO: 0.5,
}


EDGE_TYPE_CONSTRAINTS: dict[EdgeType, tuple[set[NodeType], set[NodeType]]] = {
    EdgeType.GROUNDS: ({NodeType.EVENT}, {NodeType.CONCEPT, NodeType.INTENT}),
    EdgeType.CAUSES: ({NodeType.EVENT}, {NodeType.EVENT}),
    EdgeType.TRIGGERS: ({NodeType.EVENT, NodeType.CONCEPT}, {NodeType.INTENT}),
    EdgeType.REINFORCES: ({NodeType.EVENT}, {NodeType.CONCEPT}),
    EdgeType.PART_OF: ({NodeType.CONCEPT}, {NodeType.CONCEPT}),
    EdgeType.DERIVED_FROM: ({NodeType.CONCEPT}, {NodeType.CONCEPT}),
    EdgeType.DEADLINE_FOR: ({NodeType.TIME}, {NodeType.EVENT, NodeType.CONCEPT, NodeType.INTENT}),
    EdgeType.RELATED_TO: ({NodeType.CONCEPT}, {NodeType.CONCEPT}),
    EdgeType.USER_FEEDBACK: ({NodeType.EVENT, NodeType.CONCEPT}, {NodeType.INTENT}),
}


class MemoryNode(BaseModel):
    id: str
    user_id: str
    type: NodeType
    title: str
    description: str
    importance: float = 0.5
    access_count: int = 0
    embedding: list[float] | None = None
    reasoning: str = ""
    grounded_in: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None


class MemoryEdge(BaseModel):
    id: str
    user_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_activated_at: datetime | None = None


class MemoryRecall(BaseModel):
    immediate: list[MemoryNode] = Field(default_factory=list)
    working: list[MemoryNode] = Field(default_factory=list)
    background: list[MemoryNode] = Field(default_factory=list)
    buffer: list[MemoryNode] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    thread_id: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunEvent(BaseModel):
    type: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
