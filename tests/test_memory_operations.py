from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event

from superassist.memory.operations import ApplyContext, apply_plan
from superassist.memory.plans import UpdatePlan
from superassist.memory.storage import MemoryGraphStore
from superassist.models import EdgeType, NodeType


def _sqlite_engine(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def _pragma(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


@pytest.fixture
def graph(tmp_path: Path) -> MemoryGraphStore:
    return MemoryGraphStore(_sqlite_engine(tmp_path / "graph.sqlite3"))


def _embed(text: str) -> list[float]:
    return [float(len(text) % 7), 1.0, 0.0]


def _bootstrap_event(graph: MemoryGraphStore) -> str:
    event = graph.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="root event",
        description="user said something",
        embedding=_embed("root"),
        metadata={"source": "user_turn"},
    )
    return event.id


def _ctx(graph: MemoryGraphStore, event_id: str) -> ApplyContext:
    return ApplyContext(
        store=graph,
        user_id="u",
        thread_id="t",
        event_id=event_id,
        embed=_embed,
        ref_map={"current_event": event_id, "event": event_id},
    )


def test_add_node_creates_concept_and_grounds_from_event(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    ctx = _ctx(graph, event_id)

    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {
                    "op": "ADD_NODE",
                    "node_type": "concept",
                    "data": {"ref": "c1", "title": "Likes brevity", "description": "Prefers short answers."},
                    "grounded_in": ["current_event"],
                }
            ]
        }
    )

    result = apply_plan(plan, ctx)

    assert result.nodes == 1
    assert result.edges == 1
    new_id = ctx.ref_map["c1"]
    edges = graph.list_edges("u")
    assert any(edge.source_id == event_id and edge.target_id == new_id and edge.edge_type == EdgeType.GROUNDS for edge in edges)


def test_add_edge_uses_ref_map_for_freshly_created_nodes(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    ctx = _ctx(graph, event_id)
    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {
                    "op": "ADD_NODE",
                    "node_type": "concept",
                    "data": {"ref": "concept_a", "title": "A", "description": "A"},
                    "grounded_in": ["current_event"],
                },
                {
                    "op": "ADD_NODE",
                    "node_type": "intent",
                    "data": {"ref": "intent_a", "title": "Goal", "description": "Achieve A"},
                    "grounded_in": ["concept_a"],
                },
                {
                    "op": "ADD_EDGE",
                    "source_id": "concept_a",
                    "target_id": "intent_a",
                    "edge_type": "TRIGGERS",
                },
            ]
        }
    )

    result = apply_plan(plan, ctx)
    assert result.nodes == 2
    triggers = [edge for edge in graph.list_edges("u") if edge.edge_type == EdgeType.TRIGGERS]
    assert any(
        edge.source_id == ctx.ref_map["concept_a"] and edge.target_id == ctx.ref_map["intent_a"]
        for edge in triggers
    )


def test_update_node_changes_description(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    seed = graph.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="seed",
        description="original",
        embedding=_embed("original"),
    )
    ctx = _ctx(graph, event_id)
    plan = UpdatePlan.model_validate(
        {"operations": [{"op": "UPDATE_NODE", "node_id": seed.id, "data": {"description": "revised", "importance": 0.8}}]}
    )

    result = apply_plan(plan, ctx)

    assert result.updated == 1
    refreshed = graph.get_node("u", seed.id)
    assert refreshed is not None
    assert refreshed.description == "revised"
    assert refreshed.importance == 0.8


def test_remove_node_deletes(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    seed = graph.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="dropme",
        description="drop me",
        embedding=_embed("drop"),
    )
    ctx = _ctx(graph, event_id)

    plan = UpdatePlan.model_validate(
        {"operations": [{"op": "REMOVE_NODE", "node_id": seed.id}]}
    )

    result = apply_plan(plan, ctx)

    assert result.removed_nodes == 1
    assert graph.get_node("u", seed.id) is None


def test_remove_edge_deletes_specific_typed_edge(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    concept = graph.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="seed",
        description="seed",
        embedding=_embed("seed"),
    )
    graph.add_or_boost_edge(
        user_id="u",
        source_id=event_id,
        target_id=concept.id,
        edge_type=EdgeType.GROUNDS,
    )
    ctx = _ctx(graph, event_id)

    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {
                    "op": "REMOVE_EDGE",
                    "source_id": event_id,
                    "target_id": concept.id,
                    "edge_type": "GROUNDS",
                }
            ]
        }
    )

    result = apply_plan(plan, ctx)

    assert result.removed_edges == 1
    assert graph.list_edges("u") == []


def test_merge_nodes_collapses_duplicates(graph: MemoryGraphStore) -> None:
    event_id = _bootstrap_event(graph)
    keeper = graph.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="A",
        description="seed A",
        embedding=_embed("A"),
    )
    duplicate = graph.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="A duplicate",
        description="seed A copy",
        embedding=_embed("Acopy"),
    )
    ctx = _ctx(graph, event_id)

    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {
                    "op": "MERGE_NODES",
                    "node_ids": [keeper.id, duplicate.id],
                    "merged_data": {"title": "merged", "description": "combined"},
                }
            ]
        }
    )

    result = apply_plan(plan, ctx)

    assert result.merged == 1
    survivors = {node.id for node in graph.list_nodes("u", NodeType.CONCEPT)}
    assert duplicate.id not in survivors
    assert keeper.id in survivors


def test_dispatch_runs_add_nodes_before_edges(graph: MemoryGraphStore) -> None:
    """ADD_EDGE referencing a freshly created ref must succeed even when listed first."""

    event_id = _bootstrap_event(graph)
    ctx = _ctx(graph, event_id)
    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {
                    "op": "ADD_EDGE",
                    "source_id": "current_event",
                    "target_id": "future_concept",
                    "edge_type": "GROUNDS",
                },
                {
                    "op": "ADD_NODE",
                    "node_type": "concept",
                    "data": {"ref": "future_concept", "title": "Late", "description": "late"},
                    "grounded_in": ["current_event"],
                },
            ]
        }
    )

    result = apply_plan(plan, ctx)
    assert result.nodes == 1
    assert result.edges >= 1
