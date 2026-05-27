from superassist_plus.models import EdgeType, NodeType


def test_memory_ontology_is_stable() -> None:
    assert {node.value for node in NodeType} == {"event", "concept", "intent", "time"}
    assert {edge.value for edge in EdgeType} == {
        "GROUNDS",
        "CAUSES",
        "TRIGGERS",
        "REINFORCES",
        "PART_OF",
        "DERIVED_FROM",
        "DEADLINE_FOR",
        "RELATED_TO",
        "USER_FEEDBACK",
    }

