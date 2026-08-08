from __future__ import annotations

import json

import pytest

from superassist.memory.plans import (
    AddEdgeOp,
    AddNodeOp,
    MergeNodesOp,
    RemoveNodeOp,
    UpdateNodeOp,
    UpdatePlan,
)


def test_update_plan_round_trips_operations_payload() -> None:
    raw = {
        "reasoning": "demo",
        "operations": [
            {
                "op": "ADD_NODE",
                "node_type": "concept",
                "data": {"ref": "c1", "title": "title", "description": "body"},
                "grounded_in": ["current_event"],
            },
            {
                "op": "ADD_EDGE",
                "source_id": "current_event",
                "target_id": "c1",
                "edge_type": "GROUNDS",
                "weight": 0.8,
            },
        ],
    }

    plan = UpdatePlan.parse(raw)

    assert plan.reasoning == "demo"
    assert isinstance(plan.operations[0], AddNodeOp)
    assert plan.operations[0].data.ref == "c1"
    assert plan.operations[0].effective_grounded_in() == ["current_event"]
    assert isinstance(plan.operations[1], AddEdgeOp)
    assert plan.operations[1].edge_type.value == "GROUNDS"
    assert plan.operations[1].weight == 0.8


def test_update_plan_rejects_event_in_add_node() -> None:
    with pytest.raises(Exception):
        UpdatePlan.model_validate(
            {
                "operations": [
                    {
                        "op": "ADD_NODE",
                        "node_type": "event",
                        "data": {"description": "x"},
                    }
                ]
            }
        )


def test_update_plan_rejects_unknown_edge_type() -> None:
    with pytest.raises(Exception):
        UpdatePlan.model_validate(
            {
                "operations": [
                    {
                        "op": "ADD_EDGE",
                        "source_id": "a",
                        "target_id": "b",
                        "edge_type": "NOT_A_TYPE",
                    }
                ]
            }
        )


def test_update_plan_merge_requires_two_node_ids() -> None:
    with pytest.raises(Exception):
        UpdatePlan.model_validate(
            {"operations": [{"op": "MERGE_NODES", "node_ids": ["only_one"]}]}
        )


def test_update_plan_legacy_nodes_become_add_node_and_grounds_edge() -> None:
    legacy = {
        "nodes": [
            {
                "ref": "c1",
                "type": "concept",
                "title": "User likes brevity",
                "description": "User prefers concise answers.",
            }
        ],
        "edges": [],
    }

    plan = UpdatePlan.parse(legacy)
    op_kinds = [type(op).__name__ for op in plan.operations]

    assert op_kinds == ["AddNodeOp", "AddEdgeOp"]
    add_node = plan.operations[0]
    assert isinstance(add_node, AddNodeOp)
    assert add_node.data.ref == "c1"
    assert add_node.effective_grounded_in() == ["current_event"]
    add_edge = plan.operations[1]
    assert isinstance(add_edge, AddEdgeOp)
    assert add_edge.target_id == "c1"
    assert add_edge.edge_type.value == "GROUNDS"


def test_update_plan_legacy_skips_event_nodes() -> None:
    plan = UpdatePlan.parse(
        {"nodes": [{"type": "event", "title": "ignored", "description": "ignored"}]}
    )
    assert plan.operations == []


def test_update_plan_handles_empty_input() -> None:
    assert UpdatePlan.parse(None).operations == []
    assert UpdatePlan.parse({}).operations == []


def test_update_plan_serializes_back_to_dict() -> None:
    plan = UpdatePlan.parse(
        {
            "operations": [
                {
                    "op": "REMOVE_EDGE",
                    "source_id": "a",
                    "target_id": "b",
                }
            ]
        }
    )
    payload = plan.model_dump()
    assert payload["operations"][0]["op"] == "REMOVE_EDGE"
    json.dumps(payload)


def test_update_plan_other_operation_types_round_trip() -> None:
    plan = UpdatePlan.model_validate(
        {
            "operations": [
                {"op": "UPDATE_NODE", "node_id": "n1", "data": {"importance": 0.7}},
                {"op": "REMOVE_NODE", "node_id": "n2"},
                {
                    "op": "MERGE_NODES",
                    "node_ids": ["a", "b", "c"],
                    "merged_data": {"title": "merged"},
                },
            ]
        }
    )
    assert isinstance(plan.operations[0], UpdateNodeOp)
    assert isinstance(plan.operations[1], RemoveNodeOp)
    assert isinstance(plan.operations[2], MergeNodesOp)
