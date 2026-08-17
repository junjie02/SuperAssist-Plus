from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import ClassVar

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from superassist.agent.runtime import AgentRuntime
from superassist.config import Settings
from superassist.middlewares.rag_attribution_middleware import RagAttributionMiddleware
from superassist.middlewares.rag_retrieval_middleware import RagRetrievalMiddleware
from superassist.rag.chunking import chunk_document, lexical_terms
from superassist.rag.context import RagRetrievalResult, RagTurnSession, rag_turn_context
from superassist.rag.documents import extract_document, safe_filename
from superassist.rag.service import HybridRAGService, _rrf_fuse
from superassist.ui.rag import register_rag_routes

_ = AgentRuntime  # Initialize the agent package before importing middleware modules directly.


class FakeRetriever:
    def __init__(self, chunk_ids_by_call: list[list[str]]) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.chunk_ids_by_call = chunk_ids_by_call

    def retrieve(self, user_id: str, query: str, mode: str = "hybrid") -> RagRetrievalResult:
        self.calls.append((user_id, query, mode))
        index = min(len(self.calls) - 1, len(self.chunk_ids_by_call) - 1)
        chunk_ids = self.chunk_ids_by_call[index] if self.chunk_ids_by_call else []
        hits = [
            {
                "chunk_id": chunk_id,
                "document_name": "guide.md",
                "heading": "Guide",
                "text": f"evidence {chunk_id}",
                "token_count": 20,
            }
            for chunk_id in chunk_ids
        ]
        return RagRetrievalResult(
            query=query,
            mode=mode,
            context="unused raw context",
            sources=["guide.md"] if hits else [],
            success=bool(hits),
            message="found" if hits else "not found",
            hits=hits,
            evidence_tokens=sum(int(item["token_count"]) for item in hits),
        )


def test_document_extractors_and_filename_safety(tmp_path: Path) -> None:
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Hybrid RAG\nDense and BM25 retrieval", encoding="utf-8")
    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("name,value\nalpha,1", encoding="utf-8")

    assert "Dense and BM25" in extract_document(markdown)
    assert "alpha | 1" in extract_document(csv_file)
    assert safe_filename("../bad:name?.txt") == "bad_name_.txt"


def test_structure_aware_chunking_and_chinese_lexical_terms() -> None:
    text = "# 第一章\n\n这是混合检索系统。" * 120 + "\n\n# 第二章\n\n系统支持向量检索和关键词检索。"
    chunks = chunk_document(
        text,
        document_id="doc-a",
        document_name="manual.md",
        target_tokens=80,
        max_tokens=100,
        overlap_tokens=10,
    )

    assert len(chunks) > 2
    assert all(item.token_count <= 100 for item in chunks)
    assert all(item.id.startswith("chunk-") for item in chunks)
    assert "混合" in lexical_terms("混合检索 BM25")
    assert "bm25" in lexical_terms("混合检索 BM25")


def test_rrf_fusion_preserves_dense_and_bm25_ranks() -> None:
    fused = _rrf_fuse([("dense-only", 0.9), ("both", 0.8)], [("both", 5.0), ("bm25-only", 4.0)], 60, "hybrid")

    assert fused[0][0] == "both"
    assert fused[0][2:] == (2, 1)
    assert {item[0] for item in fused} == {"dense-only", "both", "bm25-only"}


def test_rag_turn_session_deduplicates_and_stops_after_stagnation() -> None:
    retriever = FakeRetriever([["chunk-a"], ["chunk-a"], ["chunk-a"], ["chunk-b"]])
    session = RagTurnSession(
        retriever=retriever,
        user_id="user-a",
        enabled=True,
        evidence_max_tokens=100,
        stagnant_search_limit=2,
    )

    first = session.search("first")
    second = session.search("second", "bm25")
    third = session.search("third")
    stopped = session.search("fourth")

    assert first.success and first.new_hits == 1
    assert not second.success and second.duplicate_hits == 1
    assert not third.success
    assert "No new" in stopped.message
    assert len(retriever.calls) == 3
    assert session.trace() == {
        "enabled": True,
        "attempts": 3,
        "queries": ["first", "second", "third"],
        "sources": ["guide.md"],
        "uploaded_evidence_found": True,
        "unique_chunks": 1,
        "evidence_tokens": 20,
        "evidence_max_tokens": 100,
        "stagnant_searches": 2,
        "stopped_reason": "No new uploaded-data chunks were found in consecutive searches",
    }


def test_rag_turn_session_recovers_retrieval_errors() -> None:
    class BrokenRetriever:
        def retrieve(self, user_id: str, query: str, mode: str = "hybrid") -> RagRetrievalResult:
            raise RuntimeError("index offline")

    failed = RagTurnSession(BrokenRetriever(), "user-a", True).search("query")
    assert not failed.success
    assert "index offline" in failed.message


def test_rag_middlewares_seed_context_and_append_provenance() -> None:
    retriever = FakeRetriever([["chunk-a"]])
    state = {
        "messages": [AIMessage(content="Answer")],
        "input": "question",
        "user_id": "user-a",
        "thread_id": "thread-a",
        "rag_mode": True,
        "tool_events": [],
        "metadata": {},
    }
    with rag_turn_context(retriever, "user-a", True):
        update = RagRetrievalMiddleware().before_agent(state, runtime=None)
        state.update(update or {})
        attributed = RagAttributionMiddleware().after_agent(state, runtime=None)

    assert update and "chunk:chunk-a" in update["rag_context"]
    assert attributed
    assert "回答依据" in attributed["messages"][0].content
    assert "guide.md" in attributed["messages"][0].content


def test_manifest_is_isolated_by_hashed_user_directory(tmp_path: Path) -> None:
    service = object.__new__(HybridRAGService)
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


def test_hybrid_service_indexes_and_returns_only_original_chunks(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_RAG_CHUNK_TARGET_TOKENS=80,
        SUPERASSIST_RAG_CHUNK_MAX_TOKENS=100,
        SUPERASSIST_RAG_CHUNK_OVERLAP_TOKENS=10,
    )
    service = HybridRAGService(settings)
    try:
        document = service.add_document(
            "user-a",
            "manual.md",
            "# 部署协议\n\n系统使用蓝鲸协议进行服务发现。\n\n# 检索\n\n系统组合向量检索与BM25关键词检索。".encode(),
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = next(item for item in service.list_documents("user-a") if item["id"] == document["id"])
            if current["status"] in {"ready", "failed"}:
                break
            time.sleep(0.02)
        assert current["status"] == "ready", current.get("error")

        result = service.retrieve("user-a", "蓝鲸协议", "hybrid")

        assert result.success
        assert result.hits
        assert all(item["chunk_id"].startswith("chunk-") for item in result.hits)
        assert "蓝鲸协议" in result.context
        assert "实体证据" not in result.context
        assert result.hits[0]["bm25_rank"] is not None
    finally:
        service.close()


class FakeDocumentService:
    supported_extensions: ClassVar[list[str]] = [".txt"]

    def __init__(self) -> None:
        self.documents: dict[str, list[dict]] = {}
        self.graph_users: list[str] = []

    def graph_payload(self, user_id: str) -> dict:
        self.graph_users.append(user_id)
        return {
            "nodes": [],
            "edges": [],
            "stats": {"nodes": 0, "edges": 0, "documents": 1, "chunks": 3},
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
    assert graph.json()["stats"]["chunks"] == 3
    assert service.graph_users == ["user-b"]

    deleted = client.delete("/internal/rag/documents/doc-5?user_id=user-a")
    assert deleted.status_code == 202
