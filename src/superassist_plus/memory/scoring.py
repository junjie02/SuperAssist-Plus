from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections import deque

import networkx as nx

from superassist_plus.config import Settings
from superassist_plus.memory.embedding import cosine_similarity
from superassist_plus.memory.vector_index import VectorMatch
from superassist_plus.memory.storage import MemoryGraphStore
from superassist_plus.models import EdgeType, MemoryNode, NodeType


@dataclass(frozen=True)
class EventProbe:
    """Transient event/query representation; never persisted."""

    user_id: str
    text: str
    embedding: list[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class MemoryNodeScore:
    node_id: str
    pagerank: float
    recency: float
    access: float
    urgency: float
    score: float
    semantic_affinity: float = 0.0


@dataclass(frozen=True)
class TieredMemoryContext:
    immediate: list[MemoryNode] = field(default_factory=list)
    working: list[MemoryNode] = field(default_factory=list)
    background: list[MemoryNode] = field(default_factory=list)
    buffer: list[MemoryNode] = field(default_factory=list)
    scores: dict[str, MemoryNodeScore] = field(default_factory=dict)

    def ordered_nodes(self) -> list[MemoryNode]:
        return [*self.immediate, *self.working, *self.background, *self.buffer]

    def ordered_node_ids(self) -> list[str]:
        return [node.id for node in self.ordered_nodes()]


class MemoryContextRanker:
    """CogniFold-style proactive write-path context ranker."""

    damping = 0.85
    node_decay_lambda_per_hour = 0.01
    urgency_window_hours = 24.0

    def __init__(self, store: MemoryGraphStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def assemble_context(self, probe: EventProbe, limit: int) -> TieredMemoryContext:
        nodes = self.store.list_nodes(probe.user_id)
        if not nodes or limit <= 0:
            return TieredMemoryContext()

        pagerank = self.compute_pagerank(probe.user_id)
        scores = self.score_nodes(probe, nodes, pagerank)
        ranked_scores = sorted(scores.values(), key=lambda item: (item.score, item.semantic_affinity), reverse=True)
        pool = ranked_scores[: min(len(ranked_scores), self.settings.memory_candidate_pool_size)]
        context = self._select_tiers(probe.user_id, nodes, pool, limit)
        return TieredMemoryContext(
            immediate=context.immediate,
            working=context.working,
            background=context.background,
            buffer=context.buffer,
            scores=scores,
        )

    def assemble_read_context(
        self,
        probe: EventProbe,
        entry_matches: list[VectorMatch],
        limit: int,
    ) -> TieredMemoryContext:
        nodes = self.store.list_nodes(probe.user_id)
        if not nodes or limit <= 0:
            return TieredMemoryContext()

        node_ids = {node.id for node in nodes}
        entry_scores: dict[str, float] = {}
        for match in entry_matches[: self.settings.memory_read_entry_points]:
            if match.node_id in node_ids and match.score > entry_scores.get(match.node_id, -1.0):
                entry_scores[match.node_id] = max(0.0, match.score)
        if not entry_scores:
            return self.assemble_context(probe, limit)

        bfs_scores = self._bfs_scores(probe.user_id, entry_scores)
        if self.settings.memory_read_use_ppr:
            ppr_scores = self.compute_personalized_pagerank(probe.user_id, list(entry_scores))
            read_rank = self._blend_read_scores(bfs_scores, ppr_scores)
        else:
            read_rank = bfs_scores

        base_scores = self.score_nodes(probe, nodes, read_rank)
        ranked_scores = sorted(base_scores.values(), key=lambda item: (item.score, item.semantic_affinity), reverse=True)
        pool = ranked_scores[: min(len(ranked_scores), self.settings.memory_candidate_pool_size)]
        context = self._select_tiers(probe.user_id, nodes, pool, limit)
        return TieredMemoryContext(
            immediate=context.immediate,
            working=context.working,
            background=context.background,
            buffer=context.buffer,
            scores=base_scores,
        )

    def compute_pagerank(self, user_id: str) -> dict[str, float]:
        graph = self._build_graph(user_id)
        if graph.number_of_nodes() == 0:
            return {}
        if graph.number_of_edges() == 0:
            return dict.fromkeys(graph.nodes(), 1.0)
        scores = nx.pagerank(graph, alpha=self.damping, weight="effective_weight")
        return self._normalize(scores)

    def compute_personalized_pagerank(self, user_id: str, entry_points: list[str]) -> dict[str, float]:
        graph = self._build_graph(user_id)
        if graph.number_of_nodes() == 0 or not entry_points:
            return {}

        personalization = dict.fromkeys(graph.nodes(), 0.0)
        valid_seeds = [node_id for node_id in entry_points if node_id in personalization]
        if not valid_seeds:
            return self.compute_pagerank(user_id)

        seed_weight = 1.0 / len(valid_seeds)
        for node_id in valid_seeds:
            personalization[node_id] = seed_weight

        if graph.number_of_edges() == 0:
            return self._normalize(personalization)
        scores = nx.pagerank(
            graph,
            alpha=self.damping,
            personalization=personalization,
            weight="effective_weight",
        )
        return self._normalize(scores)

    def score_nodes(
        self,
        probe: EventProbe,
        nodes: list[MemoryNode],
        pagerank: dict[str, float],
    ) -> dict[str, MemoryNodeScore]:
        max_access = max((node.access_count for node in nodes), default=0)
        scores: dict[str, MemoryNodeScore] = {}
        for node in nodes:
            pr = pagerank.get(node.id, 0.0)
            recency = self._recency(node, probe.timestamp)
            access = node.access_count / max_access if max_access > 0 else 0.0
            urgency = self._urgency(node, probe.timestamp)
            score = (0.4 * pr + 0.4 * recency + 0.2 * access) * urgency
            semantic_affinity = cosine_similarity(probe.embedding, node.embedding)
            scores[node.id] = MemoryNodeScore(
                node_id=node.id,
                pagerank=pr,
                recency=recency,
                access=access,
                urgency=urgency,
                score=score,
                semantic_affinity=semantic_affinity,
            )
        return scores

    def _build_graph(self, user_id: str) -> nx.MultiDiGraph:
        now = datetime.now(UTC)
        graph = nx.MultiDiGraph()
        for node in self.store.list_nodes(user_id):
            graph.add_node(node.id)
        for edge in self.store.list_edges(user_id):
            effective = self._effective_edge_weight(edge_updated_at=edge.updated_at, weight=edge.weight, now=now)
            graph.add_edge(edge.source_id, edge.target_id, effective_weight=effective, edge_type=edge.edge_type.value)
        return graph

    def _bfs_scores(self, user_id: str, entry_scores: dict[str, float]) -> dict[str, float]:
        adjacency: dict[str, set[str]] = {}
        for node in self.store.list_nodes(user_id):
            adjacency[node.id] = set()
        for edge in self.store.list_edges(user_id):
            adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
            adjacency.setdefault(edge.target_id, set()).add(edge.source_id)

        visited: dict[str, tuple[int, float]] = {}
        queue: deque[tuple[str, int, float]] = deque()
        for node_id, score in entry_scores.items():
            queue.append((node_id, 0, score))
            visited[node_id] = (0, score)

        while queue:
            node_id, depth, score = queue.popleft()
            if depth >= self.settings.memory_read_max_depth:
                continue
            for neighbor_id in adjacency.get(node_id, set()):
                neighbor_score = score * self.settings.memory_read_bfs_decay
                if neighbor_id not in visited or visited[neighbor_id][1] < neighbor_score:
                    visited[neighbor_id] = (depth + 1, neighbor_score)
                    queue.append((neighbor_id, depth + 1, neighbor_score))

        return self._normalize({node_id: score for node_id, (_depth, score) in visited.items()})

    def _blend_read_scores(
        self,
        bfs_scores: dict[str, float],
        ppr_scores: dict[str, float],
    ) -> dict[str, float]:
        node_ids = set(bfs_scores) | set(ppr_scores)
        total_weight = self.settings.memory_read_bfs_weight + self.settings.memory_read_ppr_weight
        if total_weight <= 0:
            bfs_weight = 1.0
            ppr_weight = 0.0
        else:
            bfs_weight = self.settings.memory_read_bfs_weight / total_weight
            ppr_weight = self.settings.memory_read_ppr_weight / total_weight
        return self._normalize(
            {
                node_id: bfs_weight * bfs_scores.get(node_id, 0.0) + ppr_weight * ppr_scores.get(node_id, 0.0)
                for node_id in node_ids
            }
        )

    def _select_tiers(
        self,
        user_id: str,
        nodes: list[MemoryNode],
        pool: list[MemoryNodeScore],
        limit: int,
    ) -> TieredMemoryContext:
        node_by_id = {node.id: node for node in nodes}
        selected: set[str] = set()
        immediate_size = 1 if limit > 0 else 0
        working_size = max(0, int(limit * 0.30))
        background_size = max(0, int(limit * 0.50))

        immediate = self._select(
            sorted(
                pool,
                key=lambda score: (
                    round(0.7 * score.recency + 0.3 * (score.urgency - 1.0), 6),
                    score.semantic_affinity,
                ),
                reverse=True,
            ),
            node_by_id,
            selected,
            immediate_size,
        )
        working = self._select(
            sorted(
                pool,
                key=lambda score: (
                    (
                        0.5 * score.pagerank
                        + 0.3 * score.recency
                        + 0.2 * self._type_bonus(node_by_id[score.node_id])
                    ),
                    score.semantic_affinity,
                ),
                reverse=True,
            ),
            node_by_id,
            selected,
            working_size,
        )
        background = self._select(
            self._background_ranked(pool, node_by_id),
            node_by_id,
            selected,
            background_size,
        )
        remaining_slots = max(0, limit - len(selected))
        buffer = self._select(
            sorted(pool, key=lambda score: (score.score, score.semantic_affinity), reverse=True),
            node_by_id,
            selected,
            remaining_slots,
        )
        return TieredMemoryContext(
            immediate=immediate,
            working=working,
            background=background,
            buffer=buffer,
            scores={score.node_id: score for score in pool},
        )

    def _background_ranked(
        self,
        pool: list[MemoryNodeScore],
        node_by_id: dict[str, MemoryNode],
    ) -> list[MemoryNodeScore]:
        type_counts: dict[NodeType, int] = {}
        ranked: list[tuple[float, MemoryNodeScore]] = []
        for score in pool:
            node = node_by_id[score.node_id]
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
            diversity = 1.0 / type_counts[node.type]
            ranked.append((0.8 * score.pagerank + 0.2 * diversity, score))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [score for _rank, score in ranked]

    @staticmethod
    def _select(
        ranked: list[MemoryNodeScore],
        node_by_id: dict[str, MemoryNode],
        selected: set[str],
        size: int,
    ) -> list[MemoryNode]:
        result: list[MemoryNode] = []
        for score in ranked:
            if len(result) >= size:
                break
            if score.node_id in selected:
                continue
            node = node_by_id.get(score.node_id)
            if node is None:
                continue
            result.append(node)
            selected.add(score.node_id)
        return result

    def _urgency(self, node: MemoryNode, probe_time: datetime) -> float:
        if node.type != NodeType.INTENT:
            return 1.0
        best = 1.0
        for edge in self.store.list_edges(node.user_id):
            if edge.edge_type != EdgeType.DEADLINE_FOR or edge.target_id != node.id:
                continue
            time_node = self.store.get_node(node.user_id, edge.source_id)
            if time_node is None or time_node.type != NodeType.TIME:
                continue
            deadline = self._deadline_from_node(time_node)
            if deadline is None:
                continue
            hours_until = (self._as_utc(deadline) - self._as_utc(probe_time)).total_seconds() / 3600
            if hours_until < 0 or hours_until >= self.urgency_window_hours:
                continue
            best = max(best, 1.0 + (1.0 - hours_until / self.urgency_window_hours))
        return min(2.0, max(1.0, best))

    @staticmethod
    def _deadline_from_node(node: MemoryNode) -> datetime | None:
        for key in ("scheduled_time", "deadline", "datetime", "time"):
            raw = node.metadata.get(key)
            if isinstance(raw, str) and raw:
                try:
                    return datetime.fromisoformat(raw)
                except ValueError:
                    continue
        for raw in (node.description, node.title):
            try:
                return datetime.fromisoformat(raw.strip())
            except ValueError:
                continue
        return None

    def _recency(self, node: MemoryNode, probe_time: datetime) -> float:
        hours = self._age_hours(node, probe_time)
        return math.exp(-self.node_decay_lambda_per_hour * hours)

    def _age_hours(self, node: MemoryNode, probe_time: datetime) -> float:
        anchor = node.last_accessed_at or node.updated_at or node.created_at
        return max(0.0, (self._as_utc(probe_time) - self._as_utc(anchor)).total_seconds() / 3600)

    def _effective_edge_weight(self, *, edge_updated_at: datetime, weight: float, now: datetime) -> float:
        age_days = max(0.0, (self._as_utc(now) - self._as_utc(edge_updated_at)).total_seconds() / 86400)
        return weight * math.exp(-self.settings.memory_decay_lambda * age_days)

    @staticmethod
    def _type_bonus(node: MemoryNode) -> float:
        if node.type == NodeType.CONCEPT:
            return 1.0
        if node.type == NodeType.INTENT:
            return 0.8
        if node.type == NodeType.TIME:
            return 0.4
        return 0.2

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values()) or 0.0
        if max_score <= 0:
            return dict.fromkeys(scores, 0.0)
        return {node_id: score / max_score for node_id, score in scores.items()}

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
