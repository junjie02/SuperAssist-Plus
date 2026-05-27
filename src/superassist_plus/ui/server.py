from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from superassist_plus.config import PROJECT_ROOT, Settings
from superassist_plus.memory.service import MemoryService
from superassist_plus.models import MemoryEdge, MemoryNode, NodeType

FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app(settings: Settings | None = None, default_user_id: str = "local-user") -> FastAPI:
    service = MemoryService(settings=settings or Settings())
    app = FastAPI(title="SuperAssist-Plus Memory Graph", version="0.1.0")

    @app.get("/api/graph")
    def get_graph(user_id: str = Query(default_user_id), update_limit: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
        return graph_payload(service, user_id, update_limit=update_limit)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    return app


def graph_payload(service: MemoryService, user_id: str, update_limit: int = 80) -> dict[str, Any]:
    nodes = service.store.list_nodes(user_id)
    edges = service.store.list_edges(user_id)
    recall_snapshot = _limit_recall_snapshot(
        service.store.list_recall_snapshot(user_id),
        service.settings.memory_top_k,
    )
    by_type = {node_type.value: 0 for node_type in NodeType}
    for node in nodes:
        by_type[node.type.value] += 1
    updates = sorted(
        [*_node_updates(nodes), *_edge_updates(edges)],
        key=lambda item: item["updated_at"],
        reverse=True,
    )[:update_limit]
    return {
        "nodes": [_node_payload(node, recall_snapshot.get(node.id)) for node in nodes],
        "edges": [_edge_payload(edge) for edge in edges],
        "updates": updates,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "by_type": by_type,
        },
    }


def _limit_recall_snapshot(recall_snapshot: dict[str, dict[str, Any]], limit: int) -> dict[str, dict[str, Any]]:
    if limit <= 0:
        return {}
    if len(recall_snapshot) <= limit:
        return recall_snapshot
    tier_order = {"immediate": 0, "working": 1, "background": 2, "buffer": 3}
    ranked = sorted(
        recall_snapshot.items(),
        key=lambda item: (tier_order.get(str(item[1].get("tier")), 99), -float(item[1].get("score", 0.0))),
    )
    return dict(ranked[:limit])


def run_server(host: str = "127.0.0.1", port: int = 8765, user_id: str = "local-user") -> None:
    print(f"SuperAssist-Plus memory graph UI: http://{host}:{port}/?user_id={user_id}")
    uvicorn.run(create_app(default_user_id=user_id), host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SuperAssist-Plus memory graph UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user-id", default="local-user")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, user_id=args.user_id)


def _node_payload(node: MemoryNode, recall: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "id": node.id,
        "type": node.type.value,
        "title": node.title,
        "description": node.description,
        "importance": node.importance,
        "access_count": node.access_count,
        "reasoning": node.reasoning,
        "grounded_in": node.grounded_in,
        "metadata": node.metadata,
        "created_at": node.created_at.isoformat(),
        "updated_at": node.updated_at.isoformat(),
        "last_accessed_at": node.last_accessed_at.isoformat() if node.last_accessed_at else None,
    }
    if recall is None:
        payload.update(
            {
                "active_recall": False,
                "recall_tier": None,
                "recall_score": None,
                "recall_components": None,
                "recall_updated_at": None,
            }
        )
        return payload
    payload.update(
        {
            "active_recall": True,
            "recall_tier": recall["tier"],
            "recall_score": recall["score"],
            "recall_components": {
                "pagerank": recall["pagerank"],
                "recency": recall["recency"],
                "access": recall["access"],
                "urgency": recall["urgency"],
                "semantic_affinity": recall["semantic_affinity"],
            },
            "recall_updated_at": recall["updated_at"],
        }
    )
    return payload


def _edge_payload(edge: MemoryEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "edge_type": edge.edge_type.value,
        "weight": edge.weight,
        "metadata": edge.metadata,
        "created_at": edge.created_at.isoformat(),
        "updated_at": edge.updated_at.isoformat(),
        "last_activated_at": edge.last_activated_at.isoformat() if edge.last_activated_at else None,
    }


def _node_updates(nodes: list[MemoryNode]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "node",
            "id": node.id,
            "title": f"{node.type.value}: {node.title}",
            "description": node.reasoning or node.description,
            "updated_at": node.updated_at.isoformat(),
        }
        for node in nodes
    ]


def _edge_updates(edges: list[MemoryEdge]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "edge",
            "id": edge.id,
            "title": f"{edge.edge_type.value} · {edge.weight:.2f}",
            "description": f"{edge.source_id} -> {edge.target_id}",
            "updated_at": edge.updated_at.isoformat(),
        }
        for edge in edges
    ]


if __name__ == "__main__":
    main()
