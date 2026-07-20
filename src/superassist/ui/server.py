from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from superassist.agent.runtime import AgentRuntime
from superassist.config import PROJECT_ROOT, Settings, get_settings
from superassist.memory.service import MemoryService
from superassist.models import AgentRunEvent, AgentRunResult, MemoryEdge, MemoryNode, NodeType
from superassist.subagents import TASK_STORE

logger = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ---------------------------------------------------------------------------
# Shared helpers (used by both apps)
# ---------------------------------------------------------------------------


def _build_memory_service(settings: Settings | None = None) -> MemoryService:
    return MemoryService(settings=settings or Settings())


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str | None = None


# ---------------------------------------------------------------------------
# SSE chat endpoint (internal – called by Go server)
# ---------------------------------------------------------------------------


async def _sse_chat_stream(request: ChatRequest, settings: Settings) -> StreamingResponse:
    """Handle POST /internal/chat — return a truly-streaming SSE response.

    Events are pushed through an ``asyncio.Queue`` so that the browser sees
    ``agent_text`` chunks *as they are generated*, not all at the end.
    """

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        resolved = settings or get_settings()

        logger.info(
            "Building agent runtime: provider=%s model=%s base_url=%s api_key=%s",
            resolved.model_provider, resolved.model,
            resolved.base_url, "***" if resolved.api_key else "(empty)",
        )

        # ---- callbacks that push into the async queue from the agent thread ----
        def run_reporter(event: AgentRunEvent) -> None:
            payload: dict[str, Any] = {"type": event.type, "content": event.message}
            if event.metadata:
                payload["metadata"] = event.metadata
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        def tool_reporter(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        runtime = AgentRuntime(
            settings=resolved,
            tool_event_reporter=tool_reporter,
            run_event_reporter=run_reporter,
        )

        # ---- run the agent in a background thread so we can stream ----
        result_container: list[AgentRunResult | None] = [None]
        error_container: list[str | None] = [None]

        def _run() -> None:
            try:
                logger.info("run_streaming start: user=%s msg=%s...",
                            request.user_id, request.message[:60])
                result_container[0] = runtime.run_streaming(
                    request.message,
                    user_id=request.user_id,
                    thread_id=request.thread_id,
                )
                runtime.memory_queue.flush()
                logger.info("run_streaming done: thread_id=%s answer=%s",
                            result_container[0].thread_id, (result_container[0].answer or "")[:80])
            except Exception as exc:
                error_container[0] = f"{type(exc).__name__}: {exc}"
                logger.exception("Chat streaming failed for user=%s", request.user_id)
            finally:
                runtime.close()
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # ---- stream events as they arrive ----
        while True:
            evt = await queue.get()
            if evt is None:  # sentinel — agent thread finished
                break
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        # ---- terminal message ----
        error_msg = error_container[0]
        result = result_container[0]
        if error_msg:
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
        elif result is not None:
            yield f"data: {json.dumps({'type': 'done', 'thread_id': result.thread_id, 'answer': result.answer or ''}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No response from AI engine'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# App factory: legacy memory-graph UI (backward compatible)
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None, default_user_id: str = "local-user") -> FastAPI:
    service = _build_memory_service(settings)
    app = FastAPI(title="SuperAssist Memory Graph", version="0.1.0")

    @app.get("/api/graph")
    def get_graph(user_id: str = Query(default_user_id), update_limit: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
        return graph_payload(service, user_id, update_limit=update_limit)

    @app.get("/api/subagents/tasks")
    def list_subagent_tasks(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        return {"tasks": [task.to_dict() for task in TASK_STORE.list(limit)]}

    @app.get("/api/subagents/tasks/{task_id}")
    def get_subagent_task(task_id: str) -> dict[str, Any]:
        task = TASK_STORE.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Subagent task not found")
        return task.to_dict()

    @app.delete("/api/subagents/tasks/{task_id}")
    def delete_subagent_task(task_id: str) -> dict[str, Any]:
        if not TASK_STORE.delete(task_id):
            raise HTTPException(status_code=404, detail="Subagent task not found")
        return {"deleted": True}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    return app


# ---------------------------------------------------------------------------
# App factory: AI engine (internal, for Go server)
# ---------------------------------------------------------------------------


def create_ai_engine_app(settings: Settings | None = None) -> FastAPI:
    """Create the AI-engine FastAPI app.

    This app exposes internal endpoints consumed by the Go web server.
    It listens on ``127.0.0.1`` only — never public-facing.
    """
    resolved = settings or get_settings()
    service = _build_memory_service(resolved)

    # Preload the embedding model at startup so the first chat request
    # doesn't block for 30-60s downloading from HuggingFace.
    logger.info("Preloading embedding model (%s)...", resolved.embedding_model)
    service.preload_embedder()
    logger.info("Embedding model ready.")

    app = FastAPI(title="SuperAssist AI Engine", version="0.1.0")

    # -- Health ----------------------------------------------------------

    @app.get("/internal/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # -- Chat (SSE streaming) --------------------------------------------

    @app.post("/internal/chat")
    async def internal_chat(request: ChatRequest) -> StreamingResponse:
        return await _sse_chat_stream(request, resolved)

    # -- Memory graph (proxied by Go) ------------------------------------

    @app.get("/internal/graph")
    def internal_graph(user_id: str = Query(...), update_limit: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
        return graph_payload(service, user_id, update_limit=update_limit)

    return app


# ---------------------------------------------------------------------------
# Graph payload (shared)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def run_server(host: str = "127.0.0.1", port: int = 8765, user_id: str = "local-user") -> None:
    print(f"SuperAssist memory graph UI: http://{host}:{port}/?user_id={user_id}")
    uvicorn.run(create_app(default_user_id=user_id), host=host, port=port, log_level="info")


def main() -> None:
    """``superassist-memory-ui`` entry point (legacy, backward compatible)."""
    parser = argparse.ArgumentParser(description="Serve the SuperAssist memory graph UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user-id", default="local-user")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, user_id=args.user_id)


def serve_ai_engine(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the AI-engine server (internal endpoints)."""
    print(f"SuperAssist AI engine: http://{host}:{port}")
    uvicorn.run(create_ai_engine_app(), host=host, port=port, log_level="info")


def serve_main() -> None:
    """``superassist-ai-engine`` entry point."""
    parser = argparse.ArgumentParser(description="Serve the SuperAssist AI engine (internal API).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve_ai_engine(host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Node / edge payload helpers
# ---------------------------------------------------------------------------


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
