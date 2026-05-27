from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superassist_plus.config import Settings, get_settings
from superassist_plus.memory.embedding import Embedder, cosine_similarity, get_embedder
from superassist_plus.memory.scoring import EventProbe, MemoryContextRanker
from superassist_plus.memory.storage import MemoryGraphStore
from superassist_plus.memory.vector_index import PersistentFaissIndex
from superassist_plus.models import EdgeType, MemoryNode, MemoryRecall, NodeType


@dataclass(frozen=True)
class MemoryWritePayload:
    user_id: str
    thread_id: str
    event_id: str
    user_message: str
    assistant_answer: str
    tool_events: list[dict[str, Any]]
    memory_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class TurnMemoryContexts:
    event_id: str
    read_recall: MemoryRecall
    write_recall: MemoryRecall


class MemoryService:
    """High-level CogniFold-style memory operations."""

    def __init__(self, db_path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = MemoryGraphStore(db_path or self.settings.db_path)
        self.embedder: Embedder = get_embedder(self.settings)
        self._faiss_indexes: dict[str, PersistentFaissIndex] = {}
        self._ranker = MemoryContextRanker(self.store, self.settings)

    def prepare_turn(self, user_id: str, thread_id: str, message: str) -> tuple[str, MemoryRecall]:
        contexts = self.prepare_turn_contexts(user_id, thread_id, message)
        return contexts.event_id, contexts.write_recall

    def prepare_turn_contexts(self, user_id: str, thread_id: str, message: str) -> TurnMemoryContexts:
        probe = EventProbe(
            user_id=user_id,
            text=message,
            embedding=self.embed(message),
        )
        self.rebuild_vector_index(user_id)
        entry_matches = self.vector_index(user_id).search(probe.embedding, self.settings.memory_read_entry_points)
        read_context = self._ranker.assemble_read_context(probe, entry_matches, limit=self.settings.memory_top_k)
        write_context = self._ranker.assemble_context(probe, limit=self.settings.memory_top_k)
        self.store.replace_recall_snapshot(user_id, self._recall_snapshot_items(read_context))
        selected_ids = [*read_context.ordered_node_ids(), *write_context.ordered_node_ids()]
        self.store.touch_nodes(user_id, selected_ids)

        event = self.store.add_node(
            user_id=user_id,
            node_type=NodeType.EVENT,
            title=self._title_from_text(message, "User turn"),
            description=message,
            embedding=probe.embedding,
            metadata={"thread_id": thread_id, "source": "user_turn"},
        )

        concept, score = self.best_concept_match(user_id, message)
        if concept and score >= self.settings.memory_reinforce_similarity:
            self.store.add_or_boost_edge(
                user_id=user_id,
                source_id=event.id,
                target_id=concept.id,
                edge_type=EdgeType.REINFORCES,
                metadata={"similarity": round(score, 4), "mechanic": "accumulation"},
            )
        return TurnMemoryContexts(
            event_id=event.id,
            read_recall=self._to_recall(read_context),
            write_recall=self._to_recall(write_context),
        )

    def recall(self, user_id: str, query: str, limit: int = 12) -> MemoryRecall:
        probe = EventProbe(user_id=user_id, text=query, embedding=self.embed(query))
        self.rebuild_vector_index(user_id)
        entry_matches = self.vector_index(user_id).search(probe.embedding, self.settings.memory_read_entry_points)
        context = self._ranker.assemble_read_context(probe, entry_matches, limit=limit)
        selected = context.ordered_nodes()
        self.store.touch_nodes(user_id, [node.id for node in selected])
        return self._to_recall(context)

    @staticmethod
    def _to_recall(context: Any) -> MemoryRecall:
        return MemoryRecall(
            immediate=context.immediate,
            working=context.working,
            background=context.background,
            buffer=context.buffer,
        )

    @staticmethod
    def _recall_snapshot_items(context: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        tiers = (
            ("immediate", context.immediate),
            ("working", context.working),
            ("background", context.background),
            ("buffer", context.buffer),
        )
        for tier, nodes in tiers:
            for node in nodes:
                score = context.scores.get(node.id)
                if score is None:
                    continue
                items.append(
                    {
                        "node_id": node.id,
                        "tier": tier,
                        "score": score.score,
                        "pagerank": score.pagerank,
                        "recency": score.recency,
                        "access": score.access,
                        "urgency": score.urgency,
                        "semantic_affinity": score.semantic_affinity,
                    }
                )
        return items

    def best_concept_match(self, user_id: str, text: str) -> tuple[MemoryNode | None, float]:
        query_embedding = self.embed(text)
        self.rebuild_vector_index(user_id)
        concept_ids = {node.id for node in self.store.list_nodes(user_id, NodeType.CONCEPT)}
        matches = self.vector_index(user_id).search(query_embedding, max(len(concept_ids), 1))
        for match in matches:
            if match.node_id not in concept_ids:
                continue
            node = self.store.get_node(user_id, match.node_id)
            if node is not None:
                return node, match.score
        return None, 0.0

    def apply_structured_memory(
        self,
        payload: MemoryWritePayload,
        plan: dict[str, Any],
    ) -> dict[str, int]:
        """Apply a CogniFold-style memory update plan.

        Preferred plans use an `operations` array with ADD_NODE, UPDATE_NODE,
        ADD_EDGE, REMOVE_NODE, REMOVE_EDGE, and MERGE_NODES operations. The
        previous simple `nodes`/`edges` shape remains supported for the
        deterministic fallback writer and older tests.
        """

        ref_map = {"event": payload.event_id, "current_event": payload.event_id}
        added_nodes = 0
        added_edges = 0
        updated_nodes = 0
        merged_nodes = 0

        operations = self._as_list(plan.get("operations"))
        if operations:
            removed_nodes = 0
            for raw_operation in operations:
                if not isinstance(raw_operation, dict):
                    continue
                if str(raw_operation.get("op") or "").upper() != "ADD_NODE":
                    continue
                node = self._apply_add_node_operation(payload, raw_operation, ref_map)
                if node is None:
                    continue
                added_nodes += 1
                added_edges += self._add_grounding_edges(payload.user_id, node.id, raw_operation, ref_map)

            for raw_operation in operations:
                if not isinstance(raw_operation, dict):
                    continue
                op = str(raw_operation.get("op") or "").upper()
                if op == "ADD_NODE":
                    continue
                if op == "ADD_EDGE":
                    added_edges += self._apply_add_edge_operation(payload.user_id, raw_operation, ref_map)
                elif op == "UPDATE_NODE":
                    updated_nodes += self._apply_update_node_operation(payload.user_id, raw_operation, ref_map)
                elif op == "REMOVE_NODE":
                    removed_nodes += self._apply_remove_node_operation(payload.user_id, raw_operation, ref_map)
                elif op == "REMOVE_EDGE":
                    self._apply_remove_edge_operation(payload.user_id, raw_operation, ref_map)
                elif op == "MERGE_NODES":
                    merged_nodes += self._apply_merge_nodes_operation(payload.user_id, raw_operation, ref_map)

            if added_nodes or updated_nodes or merged_nodes or removed_nodes:
                self.rebuild_vector_index(payload.user_id)
            return {
                "nodes": added_nodes,
                "edges": added_edges,
                "updated": updated_nodes,
                "merged": merged_nodes,
            }

        for raw_node in self._as_list(plan.get("nodes")):
            if not isinstance(raw_node, dict):
                continue
            node_type = NodeType(str(raw_node.get("type") or "concept").lower())
            if node_type == NodeType.EVENT:
                continue
            description = str(raw_node.get("description") or raw_node.get("title") or "").strip()
            title = str(raw_node.get("title") or self._title_from_text(description, node_type.value))
            if not description:
                continue
            node = self.store.add_node(
                user_id=payload.user_id,
                node_type=node_type,
                title=title,
                description=description,
                embedding=self.embed(f"{title}\n{description}") if node_type != NodeType.TIME else None,
                reasoning=str(raw_node.get("reasoning") or "Created from durable conversation memory."),
                grounded_in=[payload.event_id],
                metadata={"thread_id": payload.thread_id, "source": "memory_writer"},
            )
            added_nodes += 1
            ref = str(raw_node.get("ref") or "")
            if ref:
                ref_map[ref] = node.id
            self.store.add_or_boost_edge(
                user_id=payload.user_id,
                source_id=payload.event_id,
                target_id=node.id,
                edge_type=EdgeType.GROUNDS,
            )
            added_edges += 1
        if added_nodes:
            self.rebuild_vector_index(payload.user_id)

        for raw_edge in self._as_list(plan.get("edges")):
            if not isinstance(raw_edge, dict):
                continue
            source_id = ref_map.get(str(raw_edge.get("source")), str(raw_edge.get("source") or ""))
            target_id = ref_map.get(str(raw_edge.get("target")), str(raw_edge.get("target") or ""))
            if not source_id or not target_id:
                continue
            edge_type = self._edge_type(raw_edge.get("edge_type"))
            self.store.add_or_boost_edge(
                user_id=payload.user_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                metadata={"source": "memory_writer"},
            )
            added_edges += 1

        return {"nodes": added_nodes, "edges": added_edges}

    def vector_index(self, user_id: str) -> PersistentFaissIndex:
        if user_id not in self._faiss_indexes:
            safe_user_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in user_id)
            self._faiss_indexes[user_id] = PersistentFaissIndex(
                self.settings.faiss_dir / f"{safe_user_id}.index",
                self.settings.faiss_dir / f"{safe_user_id}.mapping.json",
            )
        return self._faiss_indexes[user_id]

    def rebuild_vector_index(self, user_id: str) -> None:
        self.vector_index(user_id).rebuild(self.store.list_nodes(user_id))

    def _apply_add_node_operation(
        self,
        payload: MemoryWritePayload,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> MemoryNode | None:
        try:
            node_type = NodeType(str(operation.get("node_type") or "").lower())
        except ValueError:
            return None
        if node_type == NodeType.EVENT:
            return None

        data = operation.get("data")
        if not isinstance(data, dict):
            data = {}
        description = str(
            data.get("description")
            or data.get("content")
            or data.get("summary")
            or data.get("title")
            or ""
        ).strip()
        title = str(data.get("title") or self._title_from_text(description, node_type.value)).strip()
        if not description:
            return None

        grounded_in = [
            ref_map.get(str(item), str(item))
            for item in self._as_list(operation.get("grounded_in") or data.get("grounded_in"))
            if str(item)
        ]
        if not grounded_in:
            grounded_in = [payload.event_id]

        node = self.store.add_node(
            user_id=payload.user_id,
            node_type=node_type,
            title=title,
            description=description,
            node_id=self._optional_ref(data.get("node_id") or data.get(f"{node_type.value}_id")),
            embedding=self.embed(f"{title}\n{description}") if node_type != NodeType.TIME else None,
            reasoning=str(operation.get("reasoning") or data.get("reasoning") or "Created from memory update plan."),
            grounded_in=grounded_in,
            metadata={
                "thread_id": payload.thread_id,
                "source": "memory_writer",
                "plan_format": "operations",
            },
            importance=self._coerce_float(data.get("importance"), 0.5),
        )
        for key in ("ref", "node_id", f"{node_type.value}_id"):
            ref = self._optional_ref(data.get(key))
            if ref:
                ref_map[ref] = node.id
        return node

    def _apply_add_edge_operation(
        self,
        user_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> int:
        source_id = self._resolve_ref(operation.get("source_id") or operation.get("source"), ref_map)
        target_id = self._resolve_ref(operation.get("target_id") or operation.get("target"), ref_map)
        if not source_id or not target_id:
            return 0
        try:
            edge_type = self._edge_type(operation.get("edge_type"))
            self.store.add_or_boost_edge(
                user_id=user_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=self._coerce_optional_float(operation.get("weight")),
                metadata={
                    "source": "memory_writer",
                    "plan_format": "operations",
                    "reasoning": str(operation.get("reasoning") or ""),
                },
            )
        except (KeyError, ValueError):
            return 0
        return 1

    def _apply_update_node_operation(
        self,
        user_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> int:
        node_id = self._resolve_ref(operation.get("node_id"), ref_map)
        data = operation.get("data")
        if not node_id or not isinstance(data, dict):
            return 0
        node = self.store.get_node(user_id, node_id)
        if node is None:
            return 0
        if "title" in data:
            node.title = str(data["title"]).strip() or node.title
        if "description" in data:
            node.description = str(data["description"]).strip() or node.description
        if "importance" in data:
            node.importance = self._coerce_float(data["importance"], node.importance)
        node.reasoning = str(operation.get("update_reasoning") or data.get("reasoning") or node.reasoning)
        grounded_in = [
            ref_map.get(str(item), str(item))
            for item in self._as_list(operation.get("grounded_in") or data.get("grounded_in"))
            if str(item)
        ]
        if grounded_in:
            node.grounded_in = sorted(set(node.grounded_in + grounded_in))
        node.metadata = {**node.metadata, "source": "memory_writer", "plan_format": "operations"}
        node.embedding = self.embed(f"{node.title}\n{node.description}") if node.type != NodeType.TIME else None
        self.store.update_node(node)
        return 1

    def _apply_remove_node_operation(
        self,
        user_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> int:
        node_id = self._resolve_ref(operation.get("node_id"), ref_map)
        if not node_id or self.store.get_node(user_id, node_id) is None:
            return 0
        self.store.delete_node(user_id, node_id)
        return 1

    def _apply_remove_edge_operation(
        self,
        user_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> None:
        source_id = self._resolve_ref(operation.get("source_id") or operation.get("source"), ref_map)
        target_id = self._resolve_ref(operation.get("target_id") or operation.get("target"), ref_map)
        if not source_id or not target_id:
            return
        edge_type = str(operation.get("edge_type") or "").upper()
        delete_ids = [
            edge.id
            for edge in self.store.list_edges(user_id)
            if edge.source_id == source_id
            and edge.target_id == target_id
            and (not edge_type or edge.edge_type.value == edge_type)
        ]
        self.store.delete_edges(delete_ids)

    def _apply_merge_nodes_operation(
        self,
        user_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> int:
        node_ids = [self._resolve_ref(item, ref_map) for item in self._as_list(operation.get("node_ids"))]
        nodes = [node for node_id in node_ids if node_id for node in [self.store.get_node(user_id, node_id)] if node]
        if len(nodes) < 2:
            return 0
        merged_data = operation.get("merged_data")
        if not isinstance(merged_data, dict):
            merged_data = {}
        keeper = max(nodes, key=lambda node: (node.access_count, node.importance))
        descriptions = [node.description for node in nodes if node.description]
        keeper.title = str(merged_data.get("title") or keeper.title).strip() or keeper.title
        keeper.description = str(merged_data.get("description") or "\n".join(dict.fromkeys(descriptions))).strip()
        keeper.importance = max([keeper.importance, *[node.importance for node in nodes]])
        keeper.reasoning = str(operation.get("reasoning") or keeper.reasoning)
        grounded: set[str] = set()
        for node in nodes:
            grounded.update(node.grounded_in)
        keeper.grounded_in = sorted(grounded)
        keeper.embedding = self.embed(f"{keeper.title}\n{keeper.description}") if keeper.type != NodeType.TIME else None
        keeper.metadata = {**keeper.metadata, "source": "memory_writer", "plan_format": "operations", "merged": True}
        self.store.update_node(keeper)
        for node in nodes:
            if node.id == keeper.id:
                continue
            self.store.replace_edge_endpoint(user_id, node.id, keeper.id)
            self.store.delete_node(user_id, node.id)
            for ref, resolved in list(ref_map.items()):
                if resolved == node.id:
                    ref_map[ref] = keeper.id
        return len(nodes) - 1

    def _add_grounding_edges(
        self,
        user_id: str,
        node_id: str,
        operation: dict[str, Any],
        ref_map: dict[str, str],
    ) -> int:
        node = self.store.get_node(user_id, node_id)
        if node is None:
            return 0
        added = 0
        node = self.store.get_node(user_id, node_id)
        if node is None:
            return 0
        grounds = self._as_list(operation.get("grounded_in")) or node.grounded_in
        for raw_ground in grounds:
            ground_id = self._resolve_ref(raw_ground, ref_map)
            if not ground_id:
                continue
            ground = self.store.get_node(user_id, ground_id)
            if ground is None:
                continue
            try:
                if ground.type == NodeType.EVENT and node.type in {NodeType.CONCEPT, NodeType.INTENT}:
                    self.store.add_or_boost_edge(
                        user_id=user_id,
                        source_id=ground.id,
                        target_id=node.id,
                        edge_type=EdgeType.GROUNDS,
                        metadata={"source": "memory_writer", "mechanic": "grounded_in"},
                    )
                    added += 1
                elif ground.type == NodeType.EVENT and node.type == NodeType.TIME:
                    self.store.add_or_boost_edge(
                        user_id=user_id,
                        source_id=node.id,
                        target_id=ground.id,
                        edge_type=EdgeType.DEADLINE_FOR,
                        metadata={"source": "memory_writer", "mechanic": "grounded_in"},
                    )
                    added += 1
                elif ground.type == NodeType.CONCEPT and node.type == NodeType.INTENT:
                    self.store.add_or_boost_edge(
                        user_id=user_id,
                        source_id=ground.id,
                        target_id=node.id,
                        edge_type=EdgeType.TRIGGERS,
                        metadata={"source": "memory_writer", "mechanic": "grounded_in"},
                    )
                    added += 1
                elif ground.type == NodeType.CONCEPT and node.type == NodeType.CONCEPT:
                    self.store.add_or_boost_edge(
                        user_id=user_id,
                        source_id=ground.id,
                        target_id=node.id,
                        edge_type=EdgeType.RELATED_TO,
                        metadata={"source": "memory_writer", "mechanic": "grounded_in"},
                    )
                    added += 1
            except (KeyError, ValueError):
                continue
        return added

    @staticmethod
    def _resolve_ref(value: Any, ref_map: dict[str, str]) -> str:
        ref = str(value or "")
        return ref_map.get(ref, ref)

    @staticmethod
    def _optional_ref(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _edge_type(value: Any) -> EdgeType:
        raw = str(value or EdgeType.RELATED_TO.value).strip().upper()
        return EdgeType(raw)

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def merge_similar_concepts(self, user_id: str, threshold: float | None = None) -> int:
        threshold = threshold if threshold is not None else self.settings.memory_concept_merge_similarity
        concepts = self.store.list_nodes(user_id, NodeType.CONCEPT)
        merged = 0
        removed: set[str] = set()
        for index, left in enumerate(concepts):
            if left.id in removed:
                continue
            for right in concepts[index + 1 :]:
                if right.id in removed:
                    continue
                score = cosine_similarity(left.embedding, right.embedding)
                if score < threshold:
                    continue
                keeper, dropped = (left, right) if left.access_count >= right.access_count else (right, left)
                keeper.description = self._merge_text(keeper.description, dropped.description)
                keeper.importance = max(keeper.importance, dropped.importance)
                keeper.grounded_in = sorted(set(keeper.grounded_in + dropped.grounded_in))
                self.store.update_node(keeper)
                self.store.replace_edge_endpoint(user_id, dropped.id, keeper.id)
                self.store.delete_node(user_id, dropped.id)
                removed.add(dropped.id)
                merged += 1
        return merged

    def decay_edges(self, user_id: str) -> int:
        now = datetime.now(UTC)
        delete_ids: list[str] = []
        updated = 0
        for edge in self.store.list_edges(user_id):
            age_days = max(0.0, (now - edge.updated_at).total_seconds() / 86400)
            decayed = edge.weight * math.exp(-self.settings.memory_decay_lambda * age_days)
            if decayed < self.settings.memory_edge_delete_threshold:
                delete_ids.append(edge.id)
            elif abs(decayed - edge.weight) > 0.001:
                self.store.update_edge_weight(edge.id, decayed)
                updated += 1
        self.store.delete_edges(delete_ids)
        return updated + len(delete_ids)

    def complete_orphans(self, user_id: str) -> int:
        edges = self.store.list_edges(user_id)
        connected_targets = {edge.target_id for edge in edges}
        concepts = self.store.list_nodes(user_id, NodeType.CONCEPT)
        events = self.store.list_nodes(user_id, NodeType.EVENT)
        added = 0
        for concept in concepts:
            if concept.id in connected_targets:
                continue
            ranked = sorted(
                (
                    (cosine_similarity(concept.embedding, event.embedding), event)
                    for event in events
                    if event.embedding and concept.embedding
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            for score, event in ranked[: self.settings.memory_completion_top_k]:
                if score < self.settings.memory_completion_similarity:
                    continue
                self.store.add_or_boost_edge(
                    user_id=user_id,
                    source_id=event.id,
                    target_id=concept.id,
                    edge_type=EdgeType.GROUNDS,
                    metadata={"similarity": round(score, 4), "mechanic": "completion"},
                )
                added += 1
                break
        return added

    def consolidate(self, user_id: str) -> dict[str, int]:
        return {
            "merged": self.merge_similar_concepts(user_id),
            "decayed": self.decay_edges(user_id),
            "completed": self.complete_orphans(user_id),
        }

    def preload_embedder(self) -> None:
        self.embedder.preload()

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    @staticmethod
    def _title_from_text(text: str, fallback: str) -> str:
        cleaned = " ".join(str(text or "").split())
        return cleaned[:80] if cleaned else fallback

    @staticmethod
    def _merge_text(left: str, right: str) -> str:
        if right in left:
            return left
        if left in right:
            return right
        return f"{left}\n{right}".strip()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []
