import json
from pathlib import Path

import pytest

from superassist.config import Settings
from superassist.memory.service import MemoryService
from superassist.models import EdgeType, NodeType


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )


def test_edge_type_constraints_reject_invalid_relationship(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concept",
        description="Reusable concept",
        embedding=service.embed("Reusable concept"),
    )
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Event",
        description="Observed event",
        embedding=service.embed("Observed event"),
    )

    with pytest.raises(ValueError):
        service.store.add_or_boost_edge(
            user_id="u",
            source_id=concept.id,
            target_id=event.id,
            edge_type=EdgeType.GROUNDS,
        )


def test_event_can_trigger_intent(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Event",
        description="User asks to remember a goal",
        embedding=service.embed("User asks to remember a goal"),
    )
    intent = service.store.add_node(
        user_id="u",
        node_type=NodeType.INTENT,
        title="Remember goal",
        description="User wants follow-up memory support",
        embedding=service.embed("User wants follow-up memory support"),
    )

    edge = service.store.add_or_boost_edge(
        user_id="u",
        source_id=event.id,
        target_id=intent.id,
        edge_type=EdgeType.TRIGGERS,
    )

    assert edge.edge_type == EdgeType.TRIGGERS


def test_event_can_record_occurrence_time(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Event",
        description="A durable event",
        embedding=service.embed("A durable event"),
    )
    time_node = service.store.add_node(
        user_id="u",
        node_type=NodeType.TIME,
        title="Occurred at",
        description="2026-08-01T08:00:00+00:00",
    )

    edge = service.store.add_or_boost_edge(
        user_id="u",
        source_id=event.id,
        target_id=time_node.id,
        edge_type=EdgeType.OCCURRED_AT,
    )

    assert edge.edge_type == EdgeType.OCCURRED_AT


def test_new_memory_node_always_has_initial_timestamps(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))

    node = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Timestamped node",
        description="Every node has storage timestamps.",
        embedding=service.embed("Every node has storage timestamps."),
    )
    reloaded = service.store.get_node("u", node.id)

    assert node.created_at == node.updated_at
    assert reloaded is not None
    assert reloaded.created_at == node.created_at
    assert reloaded.updated_at == node.updated_at


def test_prepare_turn_returns_pending_event_id(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Prefers concise answers",
        description="User prefers concise answers",
        embedding=service.embed("User prefers concise answers"),
    )

    event_id, recall = service.prepare_turn("u", "t", "User prefers concise answers")

    nodes = service.store.list_nodes("u")
    assert event_id
    assert event_id.startswith("event_")
    assert recall.immediate
    # Event node is NOT created by prepare_turn — it is now created by the
    # memory writer (LLM or fallback) with a proper AI-generated summary.
    assert not any(n.id == event_id for n in nodes)
    # REINFORCES edges are also the memory writer's responsibility.
    edges = service.store.list_edges("u")
    assert len(edges) == 0


def test_recall_uses_dense_vector_index(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    concise = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concise answers",
        description="User prefers concise direct answers",
        embedding=service.embed("User prefers concise direct answers"),
    )
    service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Travel plans",
        description="Flights and hotel booking details",
        embedding=service.embed("Flights and hotel booking details"),
    )

    recall = service.recall("u", "concise direct answer preference", limit=1)

    assert recall.immediate[0].id == concise.id


def test_memory_retrieval_timing_log_records_stages_without_query_text(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_MEMORY_RETRIEVAL_LOG_ENABLED=True,
    )
    service = MemoryService(settings=settings)
    service.store.add_node(
        user_id="private-user-id",
        node_type=NodeType.CONCEPT,
        title="Concise answers",
        description="User prefers concise direct answers",
        embedding=service.embed("User prefers concise direct answers"),
    )

    service.recall("private-user-id", "private retrieval query", limit=1)

    log_text = settings.memory_retrieval_log_path.read_text(encoding="utf-8")
    record = json.loads(log_text)
    assert record["operation"] == "recall"
    assert record["status"] == "success"
    assert record["cache_hit"] is False
    assert record["indexed_node_count"] == 1
    assert record["entry_match_count"] == 1
    assert record["read_selected_count"] == 1
    assert record["total_ms"] >= 0
    assert record["faiss_rebuild_ms"] >= 0
    assert record["faiss_search_ms"] >= 0
    assert record["read_context_ms"] >= 0
    assert "private retrieval query" not in log_text
    assert "private-user-id" not in log_text


def test_memory_retrieval_timing_log_can_be_disabled(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_MEMORY_RETRIEVAL_LOG_ENABLED=False,
    )
    service = MemoryService(settings=settings)

    service.recall("u", "query", limit=1)

    assert not settings.memory_retrieval_log_path.exists()


def test_faiss_index_is_persisted_with_node_mapping(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = MemoryService(settings=settings)
    concise = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concise answers",
        description="User prefers concise direct answers",
        embedding=service.embed("User prefers concise direct answers"),
    )
    service.rebuild_vector_index("u")

    index_path = settings.faiss_dir / "u.index"
    mapping_path = settings.faiss_dir / "u.mapping.json"

    assert index_path.exists()
    assert mapping_path.exists()
    assert json.loads(mapping_path.read_text(encoding="utf-8"))["ids"] == [concise.id]

    reloaded = MemoryService(settings=settings)
    recall = reloaded.recall("u", "concise direct answer preference", limit=1)

    assert recall.immediate[0].id == concise.id


def test_merge_similar_concepts_transfers_edges(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Turn",
        description="The user prefers brief direct responses",
        embedding=service.embed("The user prefers brief direct responses"),
    )
    first = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concise replies",
        description="User prefers concise replies",
        embedding=service.embed("User prefers concise replies"),
    )
    second = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Brief answers",
        description="User prefers concise replies",
        embedding=service.embed("User prefers concise replies"),
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=event.id,
        target_id=second.id,
        edge_type=EdgeType.GROUNDS,
    )

    merged = service.merge_similar_concepts("u", threshold=0.99)

    concepts = service.store.list_nodes("u", NodeType.CONCEPT)
    edges = service.store.list_edges("u")
    assert merged == 1
    assert len(concepts) == 1
    assert edges[0].target_id == first.id or edges[0].target_id == concepts[0].id
