from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from superassist_plus.config import Settings
from superassist_plus.memory.scoring import EventProbe, MemoryContextRanker
from superassist_plus.memory.service import MemoryService
from superassist_plus.models import EdgeType, NodeType


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        SUPERASSIST_PLUS_DATA_DIR=str(tmp_path),
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(tmp_path / "memory.sqlite3", settings=_settings(tmp_path))


def _probe(service: MemoryService, user_id: str, text: str = "concise direct answers") -> EventProbe:
    return EventProbe(user_id=user_id, text=text, embedding=service.embed(text))


def test_write_path_pagerank_uses_real_directed_graph_only(tmp_path: Path) -> None:
    service = _service(tmp_path)
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Evidence",
        description="Evidence",
        embedding=service.embed("Evidence"),
    )
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concept",
        description="Concept",
        embedding=service.embed("Concept"),
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=event.id,
        target_id=concept.id,
        edge_type=EdgeType.GROUNDS,
    )

    ranker = MemoryContextRanker(service.store, service.settings)
    scores = ranker.compute_pagerank("u")

    assert scores[concept.id] == 1.0
    assert scores[event.id] < scores[concept.id]
    assert not any(edge.source_id == concept.id and edge.target_id == event.id for edge in service.store.list_edges("u"))


def test_score_nodes_uses_pagerank_recency_access_and_urgency(tmp_path: Path) -> None:
    service = _service(tmp_path)
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concept",
        description="Concept",
        embedding=service.embed("Concept"),
    )
    intent = service.store.add_node(
        user_id="u",
        node_type=NodeType.INTENT,
        title="Submit report",
        description="Submit report",
        embedding=service.embed("Submit report"),
    )
    service.store.touch_nodes("u", [concept.id])
    concept = service.store.get_node("u", concept.id)
    intent = service.store.get_node("u", intent.id)
    assert concept is not None and intent is not None

    ranker = MemoryContextRanker(service.store, service.settings)
    scores = ranker.score_nodes(_probe(service, "u"), [concept, intent], {concept.id: 0.5, intent.id: 1.0})

    assert scores[concept.id].access == 1.0
    assert scores[intent.id].access == 0.0
    assert math.isclose(
        scores[concept.id].score,
        0.4 * scores[concept.id].pagerank + 0.4 * scores[concept.id].recency + 0.2,
        rel_tol=0.001,
    )


def test_deadline_for_only_boosts_time_to_intent_target(tmp_path: Path) -> None:
    service = _service(tmp_path)
    deadline = datetime.now(UTC) + timedelta(hours=12)
    time_node = service.store.add_node(
        user_id="u",
        node_type=NodeType.TIME,
        title="Deadline",
        description=deadline.isoformat(),
        metadata={"scheduled_time": deadline.isoformat()},
    )
    intent = service.store.add_node(
        user_id="u",
        node_type=NodeType.INTENT,
        title="Submit report",
        description="Submit report",
        embedding=service.embed("Submit report"),
    )
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Related",
        description="Related",
        embedding=service.embed("Related"),
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=time_node.id,
        target_id=intent.id,
        edge_type=EdgeType.DEADLINE_FOR,
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=time_node.id,
        target_id=concept.id,
        edge_type=EdgeType.DEADLINE_FOR,
    )

    ranker = MemoryContextRanker(service.store, service.settings)
    urgency = ranker.score_nodes(_probe(service, "u"), service.store.list_nodes("u"), {intent.id: 1.0, concept.id: 1.0})

    assert math.isclose(urgency[intent.id].urgency, 1.5, rel_tol=0.05)
    assert urgency[concept.id].urgency == 1.0


def test_hierarchical_context_and_prepare_turn_order(tmp_path: Path) -> None:
    service = _service(tmp_path)
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concise answers",
        description="concise direct answers",
        embedding=service.embed("concise direct answers"),
    )
    service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Old evidence",
        description="concise direct answers",
        embedding=service.embed("concise direct answers"),
    )

    event_id, recall = service.prepare_turn("u", "thread", "concise direct answers")
    touched = service.store.get_node("u", concept.id)
    event = service.store.get_node("u", event_id)

    assert event is not None
    assert touched is not None and touched.access_count >= 1
    assert recall.immediate or recall.working or recall.background or recall.buffer
    assert event.id not in {
        node.id for node in [*recall.immediate, *recall.working, *recall.background, *recall.buffer]
    }


def test_read_path_uses_bidirectional_bfs_and_optional_ppr(tmp_path: Path) -> None:
    service = _service(tmp_path)
    source = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Source",
        description="semantic anchor",
        embedding=service.embed("semantic anchor"),
    )
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concept",
        description="connected concept",
        embedding=service.embed("connected concept"),
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=source.id,
        target_id=concept.id,
        edge_type=EdgeType.GROUNDS,
    )

    service.rebuild_vector_index("u")
    probe = EventProbe(user_id="u", text="semantic anchor", embedding=service.embed("semantic anchor"))
    matches = service.vector_index("u").search(probe.embedding, 1)
    ranker = MemoryContextRanker(service.store, service.settings)
    with_ppr = ranker.assemble_read_context(probe, matches, limit=2)

    service.settings.memory_read_use_ppr = False
    without_ppr = ranker.assemble_read_context(probe, matches, limit=2)

    with_ids = {node.id for node in with_ppr.ordered_nodes()}
    without_ids = {node.id for node in without_ppr.ordered_nodes()}
    assert concept.id in with_ids
    assert concept.id in without_ids
    assert with_ppr.scores[concept.id].pagerank != without_ppr.scores[concept.id].pagerank


def test_candidate_pool_size_is_configurable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.settings.memory_candidate_pool_size = 2
    for index in range(5):
        service.store.add_node(
            user_id="u",
            node_type=NodeType.CONCEPT,
            title=f"Concept {index}",
            description=f"candidate concept {index}",
            embedding=service.embed(f"candidate concept {index}"),
        )

    probe = _probe(service, "u")
    context = MemoryContextRanker(service.store, service.settings).assemble_context(probe, limit=5)

    assert len(context.ordered_nodes()) <= 2


def test_tier_selection_never_exceeds_limit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for index in range(20):
        service.store.add_node(
            user_id="u",
            node_type=NodeType.CONCEPT,
            title=f"Concept {index}",
            description=f"limit concept {index}",
            embedding=service.embed(f"limit concept {index}"),
        )

    probe = _probe(service, "u")
    context = MemoryContextRanker(service.store, service.settings).assemble_context(probe, limit=12)

    assert len(context.ordered_nodes()) == 12
