from pathlib import Path
import json

import pytest

from superassist_plus.config import Settings
from superassist_plus.memory.service import MemoryService
from superassist_plus.models import EdgeType, NodeType


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS=0.01,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
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


def test_prepare_turn_reinforces_existing_concept(tmp_path: Path) -> None:
    service = MemoryService(settings=make_settings(tmp_path))
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Prefers concise answers",
        description="User prefers concise answers",
        embedding=service.embed("User prefers concise answers"),
    )

    event_id, recall = service.prepare_turn("u", "t", "User prefers concise answers")

    edges = service.store.list_edges("u")
    assert event_id
    assert recall.immediate
    assert any(edge.source_id == event_id and edge.target_id == concept.id and edge.edge_type == EdgeType.REINFORCES for edge in edges)


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
