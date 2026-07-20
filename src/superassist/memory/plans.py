"""Pydantic schema for memory UpdatePlan operations.

The memory writer (LLM or deterministic) produces a structured plan describing
how the typed memory graph should change in response to a completed turn.
This module defines that plan as a discriminated union of operation models so
the dispatch logic in operations.py can switch on `op.op` with type safety
instead of dict-string introspection.

The legacy `{nodes: [...], edges: [...]}` shape produced by the deterministic
fallback writer is converted to an UpdatePlan via `from_legacy()` so a single
dispatcher handles both formats.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from superassist.models import EdgeType, NodeType


class NodeData(BaseModel):
    title: str = ""
    description: str = ""
    importance: float = 0.5
    reasoning: str = ""
    ref: str | None = None
    node_id: str | None = None
    grounded_in: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_description(self) -> NodeData:
        if not self.description:
            self.description = self.title
        return self


class AddNodeOp(BaseModel):
    op: Literal["ADD_NODE"]
    node_type: NodeType
    data: NodeData = Field(default_factory=NodeData)
    grounded_in: list[str] = Field(default_factory=list)
    reasoning: str = ""

    def effective_grounded_in(self) -> list[str]:
        return list(dict.fromkeys([*self.grounded_in, *self.data.grounded_in]))


class AddEdgeOp(BaseModel):
    op: Literal["ADD_EDGE"]
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float | None = None
    reasoning: str = ""


class UpdateNodeOp(BaseModel):
    op: Literal["UPDATE_NODE"]
    node_id: str
    data: NodeData = Field(default_factory=NodeData)
    update_reasoning: str = ""


class RemoveNodeOp(BaseModel):
    op: Literal["REMOVE_NODE"]
    node_id: str


class RemoveEdgeOp(BaseModel):
    op: Literal["REMOVE_EDGE"]
    source_id: str
    target_id: str
    edge_type: EdgeType | None = None


class MergedNodeData(BaseModel):
    title: str = ""
    description: str = ""


class MergeNodesOp(BaseModel):
    op: Literal["MERGE_NODES"]
    node_ids: list[str]
    merged_data: MergedNodeData = Field(default_factory=MergedNodeData)
    reasoning: str = ""

    @model_validator(mode="after")
    def _check_min_two_nodes(self) -> MergeNodesOp:
        if len(self.node_ids) < 2:
            raise ValueError("MERGE_NODES requires at least two node_ids.")
        return self


Operation = Annotated[
    AddNodeOp | AddEdgeOp | UpdateNodeOp | RemoveNodeOp | RemoveEdgeOp | MergeNodesOp,
    Field(discriminator="op"),
]


class UpdatePlan(BaseModel):
    """Structured memory update plan."""

    reasoning: str = ""
    operations: list[Operation] = Field(default_factory=list)
    symbolic_actions: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_legacy(cls, raw: dict[str, Any]) -> UpdatePlan:
        """Convert the historical `{nodes, edges}` shape into operations."""

        operations: list[dict[str, Any]] = []
        ref_to_event_grounding = "current_event"

        for raw_node in _as_list(raw.get("nodes")):
            if not isinstance(raw_node, dict):
                continue
            node_type = str(raw_node.get("type") or "concept").lower()
            description = str(raw_node.get("description") or raw_node.get("title") or "").strip()
            if not description:
                continue
            ref = str(raw_node.get("ref") or "").strip() or None
            data: dict[str, Any] = {
                "title": str(raw_node.get("title") or ""),
                "description": description,
                "reasoning": str(raw_node.get("reasoning") or ""),
            }
            if ref:
                data["ref"] = ref
            operations.append(
                {
                    "op": "ADD_NODE",
                    "node_type": node_type,
                    "data": data,
                    "grounded_in": [ref_to_event_grounding],
                    "reasoning": str(raw_node.get("reasoning") or ""),
                }
            )
            if ref:
                operations.append(
                    {
                        "op": "ADD_EDGE",
                        "source_id": ref_to_event_grounding,
                        "target_id": ref,
                        "edge_type": EdgeType.GROUNDS.value,
                    }
                )

        for raw_edge in _as_list(raw.get("edges")):
            if not isinstance(raw_edge, dict):
                continue
            source = str(raw_edge.get("source") or "").strip()
            target = str(raw_edge.get("target") or "").strip()
            if not source or not target:
                continue
            operations.append(
                {
                    "op": "ADD_EDGE",
                    "source_id": source,
                    "target_id": target,
                    "edge_type": str(raw_edge.get("edge_type") or EdgeType.RELATED_TO.value),
                }
            )

        return cls.model_validate({"reasoning": str(raw.get("reasoning") or ""), "operations": operations})

    @classmethod
    def parse(cls, raw: Any) -> UpdatePlan:
        """Best-effort plan parser accepting both new and legacy shapes."""

        if not isinstance(raw, dict):
            return cls()
        if raw.get("operations"):
            return cls.model_validate(raw)
        if raw.get("nodes") or raw.get("edges"):
            return cls.from_legacy(raw)
        return cls.model_validate(raw)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


__all__ = [
    "AddEdgeOp",
    "AddNodeOp",
    "MergeNodesOp",
    "MergedNodeData",
    "NodeData",
    "Operation",
    "RemoveEdgeOp",
    "RemoveNodeOp",
    "UpdateNodeOp",
    "UpdatePlan",
]
