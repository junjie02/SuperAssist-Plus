"""High-level memory operations.

MemoryService is the only public entry point used by middleware. It owns:

* the SQLite-backed graph store
* the per-user FAISS index
* the embedder
* the read-path ranker (vector entry points + bidirectional BFS / PPR)
* turn-time read+write context assembly
* consolidation (concept merge, edge decay, orphan completion)

Structured plan application is delegated to memory.operations.apply_plan;
this module no longer hand-rolls the per-op dispatch logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine

from superassist.config import Settings, get_settings
from superassist.memory.embedding import Embedder, cosine_similarity, get_embedder
from superassist.memory.operations import ApplyContext, ApplyResult, apply_plan
from superassist.memory.plans import UpdatePlan
from superassist.memory.scoring import EventProbe, MemoryContextRanker
from superassist.memory.storage import MemoryGraphStore, create_engine_from_settings
from superassist.memory.vector_index import PersistentFaissIndex
from superassist.models import EdgeType, MemoryNode, MemoryRecall, NodeType


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

    def __init__(self, engine: Engine | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = MemoryGraphStore(engine or create_engine_from_settings(self.settings))
        self.embedder: Embedder = get_embedder(self.settings)
        self._faiss_indexes: dict[str, PersistentFaissIndex] = {}
        self._ranker = MemoryContextRanker(self.store, self.settings)

    # -- Turn lifecycle ----------------------------------------------------

    def prepare_turn(self, user_id: str, thread_id: str, message: str) -> tuple[str, MemoryRecall]:
        contexts = self.prepare_turn_contexts(user_id, thread_id, message)
        return contexts.event_id, contexts.write_recall

    def prepare_turn_contexts(self, user_id: str, thread_id: str, message: str) -> TurnMemoryContexts:
        probe = EventProbe(user_id=user_id, text=message, embedding=self.embed(message))
        self.rebuild_vector_index(user_id)
        entry_matches = self.vector_index(user_id).search(probe.embedding, self.settings.memory_read_entry_points)
        read_context = self._ranker.assemble_read_context(probe, entry_matches, limit=self.settings.memory_top_k)
        write_context = self._ranker.assemble_context(probe, limit=self.settings.memory_top_k)
        self.store.replace_recall_snapshot(user_id, _recall_snapshot_items(read_context))
        selected_ids = [*read_context.ordered_node_ids(), *write_context.ordered_node_ids()]
        self.store.touch_nodes(user_id, selected_ids)

        # Pre-allocate an event id — the actual event node is created later
        # by the LLM memory writer (or fallback writer) with a proper summary.
        from superassist.memory.storage import new_id

        event_id = new_id("event")

        return TurnMemoryContexts(
            event_id=event_id,
            read_recall=_to_recall(read_context),
            write_recall=_to_recall(write_context),
        )

    def recall(self, user_id: str, query: str, limit: int = 12) -> MemoryRecall:
        probe = EventProbe(user_id=user_id, text=query, embedding=self.embed(query))
        self.rebuild_vector_index(user_id)
        entry_matches = self.vector_index(user_id).search(probe.embedding, self.settings.memory_read_entry_points)
        context = self._ranker.assemble_read_context(probe, entry_matches, limit=limit)
        self.store.touch_nodes(user_id, [node.id for node in context.ordered_nodes()])
        return _to_recall(context)

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

    # -- Plan application --------------------------------------------------

    def apply_structured_memory(
        self,
        payload: MemoryWritePayload,
        plan: dict[str, Any] | UpdatePlan,
    ) -> dict[str, int]:
        update_plan = plan if isinstance(plan, UpdatePlan) else UpdatePlan.parse(plan)
        context = ApplyContext(
            store=self.store,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            event_id=payload.event_id,
            embed=self.embed,
            ref_map={"event": payload.event_id, "current_event": payload.event_id},
        )
        result: ApplyResult = apply_plan(update_plan, context)
        if result.nodes or result.updated or result.merged or result.removed_nodes:
            self.rebuild_vector_index(payload.user_id)
        return result.to_summary()

    # -- Vector index ------------------------------------------------------

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

    # -- Consolidation -----------------------------------------------------

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
                keeper.description = _merge_text(keeper.description, dropped.description)
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

    # -- Embedding helpers -------------------------------------------------

    def preload_embedder(self) -> None:
        self.embedder.preload()

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)


def _to_recall(context: Any) -> MemoryRecall:
    return MemoryRecall(
        immediate=context.immediate,
        working=context.working,
        background=context.background,
        buffer=context.buffer,
    )


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


def _title_from_text(text: str, fallback: str) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:80] if cleaned else fallback


def _merge_text(left: str, right: str) -> str:
    if right in left:
        return left
    if left in right:
        return right
    return f"{left}\n{right}".strip()


__all__ = ["MemoryService", "MemoryWritePayload", "TurnMemoryContexts"]
