"""Memory plan operation handlers.

Each operation in an UpdatePlan maps to a pure function below. All state
mutation goes through MemoryGraphStore; the embedder is passed in so test code
can swap in deterministic vectors.

ApplyResult carries running counts of (added, updated, removed, merged) so
MemoryService can report a single combined summary back to the writer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from superassist.memory.plans import (
    AddEdgeOp,
    AddNodeOp,
    MergeNodesOp,
    Operation,
    RemoveEdgeOp,
    RemoveNodeOp,
    UpdateNodeOp,
    UpdatePlan,
)
from superassist.memory.storage import MemoryGraphStore
from superassist.models import EdgeType, MemoryNode, NodeType

logger = logging.getLogger(__name__)

Embedder = Callable[[str], list[float]]


@dataclass
class ApplyContext:
    store: MemoryGraphStore
    user_id: str
    thread_id: str
    event_id: str
    embed: Embedder
    ref_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ApplyResult:
    nodes: int = 0
    edges: int = 0
    updated: int = 0
    merged: int = 0
    removed_nodes: int = 0
    removed_edges: int = 0

    def to_summary(self) -> dict[str, int]:
        return {"nodes": self.nodes, "edges": self.edges, "updated": self.updated, "merged": self.merged}


class _Handler(Protocol):
    def __call__(self, op: Operation, ctx: ApplyContext, result: ApplyResult) -> None: ...


def apply_plan(plan: UpdatePlan, ctx: ApplyContext) -> ApplyResult:
    """Dispatch every operation in *plan* through its registered handler.

    ADD_NODE operations run first so subsequent operations can reference the
    newly created refs in their source_id / target_id fields.
    """

    result = ApplyResult()
    add_nodes = [op for op in plan.operations if isinstance(op, AddNodeOp)]
    other_ops = [op for op in plan.operations if not isinstance(op, AddNodeOp)]
    for op in [*add_nodes, *other_ops]:
        handler = _DISPATCH.get(type(op))
        if handler is None:
            logger.warning("No handler for operation type %s", type(op).__name__)
            continue
        try:
            handler(op, ctx, result)
        except (KeyError, ValueError) as exc:
            logger.warning("Operation %s skipped: %s", type(op).__name__, exc)
    return result


def _resolve(ctx: ApplyContext, raw_id: str | None) -> str:
    """Translate an UpdatePlan ref ('current_event' or a custom ref) to a real node id."""

    if not raw_id:
        return ""
    return ctx.ref_map.get(raw_id, raw_id)


def _embed_for(node_type: NodeType, *, embed: Embedder, title: str, description: str) -> list[float] | None:
    if node_type == NodeType.TIME:
        return None
    return embed(f"{title}\n{description}")


def _apply_add_node(op: AddNodeOp, ctx: ApplyContext, result: ApplyResult) -> None:
    description = op.data.description.strip()
    if not description:
        return
    title = op.data.title.strip() or _title_from_text(description, op.node_type.value)
    grounded_in = [_resolve(ctx, raw) for raw in op.effective_grounded_in()]
    grounded_in = [node_id for node_id in grounded_in if node_id]
    if not grounded_in:
        grounded_in = [ctx.event_id]

    metadata = {
        "thread_id": ctx.thread_id,
        "source": "memory_writer",
        "plan_format": "operations",
        **(op.data.metadata or {}),
    }
    node = ctx.store.add_node(
        user_id=ctx.user_id,
        node_type=op.node_type,
        title=title,
        description=description,
        node_id=op.data.node_id or None,
        embedding=_embed_for(op.node_type, embed=ctx.embed, title=title, description=description),
        reasoning=op.reasoning or op.data.reasoning or "Created from memory update plan.",
        grounded_in=grounded_in,
        metadata=metadata,
        importance=op.data.importance,
    )
    result.nodes += 1

    for ref_key in (op.data.ref, op.data.node_id, f"{op.node_type.value}_id"):
        if ref_key:
            ctx.ref_map[ref_key] = node.id

    result.edges += _add_grounding_edges(node, grounded_in, ctx)


def _apply_add_edge(op: AddEdgeOp, ctx: ApplyContext, result: ApplyResult) -> None:
    source_id = _resolve(ctx, op.source_id)
    target_id = _resolve(ctx, op.target_id)
    if not source_id or not target_id:
        return
    ctx.store.add_or_boost_edge(
        user_id=ctx.user_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=op.edge_type,
        weight=op.weight,
        metadata={"source": "memory_writer", "plan_format": "operations", "reasoning": op.reasoning},
    )
    result.edges += 1


def _apply_update_node(op: UpdateNodeOp, ctx: ApplyContext, result: ApplyResult) -> None:
    node_id = _resolve(ctx, op.node_id)
    node = ctx.store.get_node(ctx.user_id, node_id)
    if node is None:
        return
    if op.data.title:
        node.title = op.data.title.strip() or node.title
    if op.data.description:
        node.description = op.data.description.strip() or node.description
    node.importance = op.data.importance if op.data.importance != 0.5 else node.importance
    node.reasoning = op.update_reasoning or op.data.reasoning or node.reasoning
    extra_grounding = [_resolve(ctx, raw) for raw in op.data.grounded_in if raw]
    if extra_grounding:
        node.grounded_in = sorted(set(node.grounded_in + extra_grounding))
    node.metadata = {**node.metadata, "source": "memory_writer", "plan_format": "operations"}
    node.embedding = _embed_for(node.type, embed=ctx.embed, title=node.title, description=node.description)
    ctx.store.update_node(node)
    result.updated += 1


def _apply_remove_node(op: RemoveNodeOp, ctx: ApplyContext, result: ApplyResult) -> None:
    node_id = _resolve(ctx, op.node_id)
    if node_id and ctx.store.get_node(ctx.user_id, node_id) is not None:
        ctx.store.delete_node(ctx.user_id, node_id)
        result.removed_nodes += 1


def _apply_remove_edge(op: RemoveEdgeOp, ctx: ApplyContext, result: ApplyResult) -> None:
    source_id = _resolve(ctx, op.source_id)
    target_id = _resolve(ctx, op.target_id)
    if not source_id or not target_id:
        return
    edge_type_value = op.edge_type.value if op.edge_type else None
    matches = [
        edge.id
        for edge in ctx.store.list_edges(ctx.user_id)
        if edge.source_id == source_id
        and edge.target_id == target_id
        and (edge_type_value is None or edge.edge_type.value == edge_type_value)
    ]
    if matches:
        ctx.store.delete_edges(matches)
        result.removed_edges += len(matches)


def _apply_merge_nodes(op: MergeNodesOp, ctx: ApplyContext, result: ApplyResult) -> None:
    node_ids = [_resolve(ctx, raw) for raw in op.node_ids]
    nodes = [node for node_id in node_ids if node_id for node in [ctx.store.get_node(ctx.user_id, node_id)] if node]
    if len(nodes) < 2:
        return
    keeper = max(nodes, key=lambda node: (node.access_count, node.importance))
    descriptions = [node.description for node in nodes if node.description]
    keeper.title = (op.merged_data.title or keeper.title).strip() or keeper.title
    keeper.description = (op.merged_data.description or "\n".join(dict.fromkeys(descriptions))).strip()
    keeper.importance = max(keeper.importance, *(node.importance for node in nodes))
    keeper.reasoning = op.reasoning or keeper.reasoning
    keeper.grounded_in = sorted({grounded for node in nodes for grounded in node.grounded_in})
    keeper.embedding = _embed_for(keeper.type, embed=ctx.embed, title=keeper.title, description=keeper.description)
    keeper.metadata = {**keeper.metadata, "source": "memory_writer", "plan_format": "operations", "merged": True}
    ctx.store.update_node(keeper)
    for node in nodes:
        if node.id == keeper.id:
            continue
        ctx.store.replace_edge_endpoint(ctx.user_id, node.id, keeper.id)
        ctx.store.delete_node(ctx.user_id, node.id)
        for ref, resolved in list(ctx.ref_map.items()):
            if resolved == node.id:
                ctx.ref_map[ref] = keeper.id
    result.merged += len(nodes) - 1


_GROUNDING_RULES: list[tuple[NodeType, NodeType, EdgeType, bool]] = [
    (NodeType.EVENT, NodeType.CONCEPT, EdgeType.GROUNDS, False),
    (NodeType.EVENT, NodeType.INTENT, EdgeType.GROUNDS, False),
    (NodeType.EVENT, NodeType.TIME, EdgeType.DEADLINE_FOR, True),
    (NodeType.CONCEPT, NodeType.INTENT, EdgeType.TRIGGERS, False),
    (NodeType.CONCEPT, NodeType.CONCEPT, EdgeType.RELATED_TO, False),
]


def _add_grounding_edges(node: MemoryNode, grounded_in: list[str], ctx: ApplyContext) -> int:
    """Materialize every grounded_in reference as a typed edge."""

    added = 0
    for raw_id in grounded_in:
        ground = ctx.store.get_node(ctx.user_id, raw_id)
        if ground is None:
            continue
        edge = _grounding_edge(ground.type, node.type)
        if edge is None:
            continue
        edge_type, swap = edge
        source_id, target_id = (node.id, ground.id) if swap else (ground.id, node.id)
        try:
            ctx.store.add_or_boost_edge(
                user_id=ctx.user_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                metadata={"source": "memory_writer", "mechanic": "grounded_in"},
            )
            added += 1
        except (KeyError, ValueError):
            continue
    return added


def _grounding_edge(ground_type: NodeType, node_type: NodeType) -> tuple[EdgeType, bool] | None:
    for source, target, edge_type, swap in _GROUNDING_RULES:
        if source == ground_type and target == node_type:
            return edge_type, swap
    return None


def _title_from_text(text: str, fallback: str) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:80] if cleaned else fallback


_DISPATCH: dict[type, _Handler] = {
    AddNodeOp: _apply_add_node,
    AddEdgeOp: _apply_add_edge,
    UpdateNodeOp: _apply_update_node,
    RemoveNodeOp: _apply_remove_node,
    RemoveEdgeOp: _apply_remove_edge,
    MergeNodesOp: _apply_merge_nodes,
}


__all__ = ["ApplyContext", "ApplyResult", "apply_plan"]
