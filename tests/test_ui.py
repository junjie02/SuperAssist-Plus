from fastapi.testclient import TestClient

from superassist_plus.config import Settings
from superassist_plus.memory.service import MemoryService
from superassist_plus.models import EdgeType, NodeType
from superassist_plus.ui.server import create_app, graph_payload


def test_graph_payload_contains_nodes_edges_updates_and_stats(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    service = MemoryService(settings=settings)
    event = service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Turn",
        description="User likes concise answers.",
        embedding=service.embed("User likes concise answers."),
    )
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Prefers concise answers",
        description="User prefers concise answers.",
        embedding=service.embed("User prefers concise answers."),
    )
    service.store.add_or_boost_edge(
        user_id="u",
        source_id=event.id,
        target_id=concept.id,
        edge_type=EdgeType.GROUNDS,
    )

    payload = graph_payload(service, "u")

    assert payload["stats"]["nodes"] == 2
    assert payload["stats"]["edges"] == 1
    assert payload["stats"]["by_type"]["concept"] == 1
    assert payload["nodes"][0]["id"]
    assert payload["edges"][0]["edge_type"] == "GROUNDS"
    assert payload["updates"]


def test_fastapi_graph_endpoint_returns_payload(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    service = MemoryService(settings=settings)
    service.store.add_node(
        user_id="u",
        node_type=NodeType.EVENT,
        title="Turn",
        description="Hello",
        embedding=service.embed("Hello"),
    )
    app = create_app(settings=settings, default_user_id="u")
    client = TestClient(app)

    response = client.get("/api/graph?user_id=u")

    assert response.status_code == 200
    assert response.json()["stats"]["nodes"] == 1


def test_graph_payload_marks_latest_read_recall_scores(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    service = MemoryService(settings=settings)
    concept = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Prefers concise answers",
        description="User prefers concise answers.",
        embedding=service.embed("User prefers concise answers."),
    )

    service.prepare_turn_contexts("u", "t", "Please remember I prefer concise answers.")
    payload = graph_payload(service, "u")
    node = next(item for item in payload["nodes"] if item["id"] == concept.id)

    assert node["active_recall"] is True
    assert node["recall_tier"] in {"immediate", "working", "background", "buffer"}
    assert isinstance(node["recall_score"], float)
    assert set(node["recall_components"]) == {
        "pagerank",
        "recency",
        "access",
        "urgency",
        "semantic_affinity",
    }


def test_graph_payload_caps_active_recall_to_memory_top_k(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_PLUS_MEMORY_TOP_K=2,
    )
    service = MemoryService(settings=settings)
    nodes = [
        service.store.add_node(
            user_id="u",
            node_type=NodeType.CONCEPT,
            title=f"Concept {index}",
            description=f"Concept {index}",
            embedding=service.embed(f"Concept {index}"),
        )
        for index in range(4)
    ]
    service.store.replace_recall_snapshot(
        "u",
        [
            {
                "node_id": node.id,
                "tier": "buffer",
                "score": float(index),
            }
            for index, node in enumerate(nodes)
        ],
    )

    payload = graph_payload(service, "u")

    assert len([node for node in payload["nodes"] if node["active_recall"]]) == 2
