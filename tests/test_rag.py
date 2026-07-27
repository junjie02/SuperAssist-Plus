from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from superassist.agent.runtime import AgentRuntime
from superassist.config import Settings
from superassist.middlewares.rag_attribution_middleware import RagAttributionMiddleware
from superassist.middlewares.rag_retrieval_middleware import RagRetrievalMiddleware
from superassist.rag.context import RagRetrievalResult, RagTurnSession, rag_turn_context
from superassist.rag.documents import extract_document, safe_filename
from superassist.rag.service import LightRAGService, _retrieval_result
from superassist.ui.rag import register_rag_routes

_ = AgentRuntime  # Initialize the agent package before importing middleware modules directly.


class FakeRetriever:
    def __init__(self, success_on: int | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.success_on = success_on

    def retrieve(self, user_id: str, query: str, mode: str = "mix") -> RagRetrievalResult:
        self.calls.append((user_id, query, mode))
        success = len(self.calls) == self.success_on
        return RagRetrievalResult(
            query=query,
            mode=mode,
            context="supported evidence" if success else "",
            sources=["guide.md"] if success else [],
            success=success,
            message="found" if success else "not found",
        )


def test_document_extractors_and_filename_safety(tmp_path: Path) -> None:
    markdown = tmp_path / "notes.md"
    markdown.write_text("# LightRAG\nGraph retrieval", encoding="utf-8")
    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("name,value\nalpha,1", encoding="utf-8")

    assert "Graph retrieval" in extract_document(markdown)
    assert "alpha | 1" in extract_document(csv_file)
    assert safe_filename("../bad:name?.txt") == "bad_name_.txt"


def test_rag_turn_session_enforces_three_attempts_and_recovers_errors() -> None:
    retriever = FakeRetriever(success_on=2)
    session = RagTurnSession(retriever=retriever, user_id="user-a", enabled=True, max_attempts=3)

    assert not session.search("first").success
    assert session.search("second", "local").success
    assert not session.search("third").success
    limited = session.search("fourth")

    assert "limit reached" in limited.message
    assert session.trace() == {
        "enabled": True,
        "attempts": 3,
        "max_attempts": 3,
        "queries": ["first", "second", "third"],
        "sources": ["guide.md"],
        "uploaded_evidence_found": True,
    }

    class BrokenRetriever:
        def retrieve(self, user_id: str, query: str, mode: str = "mix") -> RagRetrievalResult:
            raise RuntimeError("index offline")

    failed = RagTurnSession(BrokenRetriever(), "user-a", True).search("query")
    assert not failed.success
    assert "index offline" in failed.message


def test_retrieval_result_preserves_light_rag_sources() -> None:
    result = _retrieval_result(
        "what",
        "mix",
        {
            "status": "success",
            "message": "ok",
            "data": {
                "chunks": [{"file_path": "manual.pdf", "content": "verified statement"}],
                "entities": [],
                "relationships": [],
                "references": [{"file_path": "manual.pdf"}],
            },
        },
        5000,
    )

    assert result.success
    assert result.sources == ["manual.pdf"]
    assert "verified statement" in result.context


def test_rag_middlewares_seed_context_and_append_provenance() -> None:
    retriever = FakeRetriever(success_on=1)
    state = {
        "messages": [AIMessage(content="Answer")],
        "input": "question",
        "user_id": "user-a",
        "thread_id": "thread-a",
        "rag_mode": True,
        "tool_events": [],
        "metadata": {},
    }
    with rag_turn_context(retriever, "user-a", True, 3):
        update = RagRetrievalMiddleware().before_agent(state, runtime=None)
        state.update(update or {})
        attributed = RagAttributionMiddleware().after_agent(state, runtime=None)

    assert update and update["rag_context"] == "supported evidence"
    assert attributed
    assert "回答依据" in attributed["messages"][0].content
    assert "guide.md" in attributed["messages"][0].content


def test_manifest_is_isolated_by_hashed_user_directory(tmp_path: Path) -> None:
    service = object.__new__(LightRAGService)
    service.base_dir = tmp_path
    service._manifest_lock = threading.RLock()
    document = {
        "id": "doc-a",
        "name": "a.txt",
        "storage_name": "doc-a.txt",
        "size": 1,
        "status": "ready",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    service._write_manifest("user-a", [document])

    assert service.list_documents("user-a")[0]["id"] == "doc-a"
    assert service.list_documents("user-b") == []
    manifests = list(tmp_path.glob("*/documents.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))[0]["name"] == "a.txt"


def test_graph_payload_normalizes_lightrag_storage_records() -> None:
    class FakeGraphStorage:
        async def get_all_nodes(self) -> list[dict]:
            return [
                {"id": "A", "entity_id": "Alpha", "entity_type": "method", "description": "one<SEP>two"},
                {"id": "B", "entity_id": "Beta", "entity_type": "concept", "description": "three"},
            ]

        async def get_all_edges(self) -> list[dict]:
            return [{"source": "A", "target": "B", "weight": 4, "keywords": "uses", "description": "link"}]

    class FakeRAG:
        chunk_entity_relation_graph = FakeGraphStorage()

    service = object.__new__(LightRAGService)

    async def get_rag(_user_id: str) -> FakeRAG:
        return FakeRAG()

    service._get_rag = get_rag
    payload = asyncio.run(service._graph_payload("user-a", [{"updated_at": "2026-01-01"}]))

    assert payload["stats"] == {"nodes": 2, "edges": 1, "documents": 1}
    assert payload["nodes"][0]["importance"] == 1
    assert payload["nodes"][0]["description"] == "one | two"
    assert payload["edges"][0]["raw_weight"] == 4
    assert payload["edges"][0]["weight"] == 1


class FakeDocumentService:
    supported_extensions = [".txt"]

    def __init__(self) -> None:
        self.documents: dict[str, list[dict]] = {}
        self.graph_users: list[str] = []

    def graph_payload(self, user_id: str) -> dict:
        self.graph_users.append(user_id)
        return {
            "nodes": [{"id": user_id, "type": "entity", "title": user_id, "importance": 1.0}],
            "edges": [],
            "stats": {"nodes": 1, "edges": 0, "documents": 1},
            "updated_at": "",
        }

    def list_documents(self, user_id: str) -> list[dict]:
        return self.documents.get(user_id, [])

    def add_document(self, user_id: str, filename: str, content: bytes) -> dict:
        document = {"id": f"doc-{len(content)}", "name": filename, "size": len(content), "status": "queued"}
        self.documents.setdefault(user_id, []).append(document)
        return document

    def delete_document(self, user_id: str, document_id: str) -> dict:
        current = self.documents.get(user_id, [])
        if not any(item["id"] == document_id for item in current):
            raise KeyError(document_id)
        self.documents[user_id] = [item for item in current if item["id"] != document_id]
        return {"id": document_id, "status": "deleting"}


def test_internal_rag_routes_upload_list_and_isolate_users(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_RAG_MAX_FILE_SIZE_MB=1,
    )
    service = FakeDocumentService()
    app = FastAPI()
    register_rag_routes(app, service, settings)
    client = TestClient(app)

    uploaded = client.post(
        "/internal/rag/documents?user_id=user-a",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert uploaded.status_code == 202
    assert uploaded.json()["documents"][0]["status"] == "queued"
    assert client.get("/internal/rag/documents?user_id=user-a").json()["documents"]
    assert client.get("/internal/rag/documents?user_id=user-b").json()["documents"] == []

    graph = client.get("/internal/rag/graph?user_id=user-b")
    assert graph.status_code == 200
    assert graph.json()["nodes"][0]["id"] == "user-b"
    assert service.graph_users == ["user-b"]

    deleted = client.delete("/internal/rag/documents/doc-5?user_id=user-a")
    assert deleted.status_code == 202
