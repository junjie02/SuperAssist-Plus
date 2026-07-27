from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from superassist.config import Settings
from superassist.llm import create_chat_model
from superassist.memory.embedding import Embedder, get_embedder
from superassist.rag.context import RagRetrievalResult
from superassist.rag.documents import SUPPORTED_EXTENSIONS, extract_document, safe_filename

logger = logging.getLogger(__name__)


class LightRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_dir = settings.rag_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._model = create_chat_model(settings)
        self._embedder: Embedder = get_embedder(settings)
        self._embedding_dim = len(self._embedder.embed("LightRAG embedding dimension probe"))
        self._manifest_lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="superassist-lightrag", daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=10)
        self._rags: dict[str, LightRAG] = {}
        self._rag_locks: dict[str, asyncio.Lock] = {}

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(SUPPORTED_EXTENSIONS)

    def add_document(self, user_id: str, filename: str, content: bytes) -> dict[str, Any]:
        if len(content) > self.settings.rag_max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File exceeds {self.settings.rag_max_file_size_mb} MB limit")
        filename = safe_filename(filename)
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {Path(filename).suffix or '(none)'}")
        if not content:
            raise ValueError("File is empty")

        document_id = f"doc-{uuid4().hex}"
        user_dir = self._user_dir(user_id)
        raw_dir = user_dir / "files"
        raw_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"{document_id}{Path(filename).suffix.lower()}"
        (raw_dir / storage_name).write_bytes(content)
        now = _utc_now()
        document = {
            "id": document_id,
            "name": filename,
            "storage_name": storage_name,
            "size": len(content),
            "status": "queued",
            "error": None,
            "characters": None,
            "created_at": now,
            "updated_at": now,
        }
        self._put_document(user_id, document)
        future = self._submit(self._index_document(user_id, document_id))
        future.add_done_callback(lambda completed: self._log_background_error(completed, document_id))
        return self._public_document(document)

    def list_documents(self, user_id: str) -> list[dict[str, Any]]:
        documents = self._read_manifest(user_id)
        return [self._public_document(item) for item in sorted(documents, key=lambda item: item["created_at"], reverse=True)]

    def graph_payload(self, user_id: str) -> dict[str, Any]:
        ready_documents = [item for item in self._read_manifest(user_id) if item.get("status") == "ready"]
        if not ready_documents:
            return _empty_graph_payload()
        timeout = max(30, min(self.settings.subagent_timeout_seconds, 300))
        return self._submit(self._graph_payload(user_id, ready_documents)).result(timeout=timeout)

    def delete_document(self, user_id: str, document_id: str) -> dict[str, Any]:
        document = self._get_document(user_id, document_id)
        if document is None:
            raise KeyError(document_id)
        if document["status"] in {"queued", "parsing", "indexing", "deleting"}:
            raise ValueError("Document is still being processed")
        self._update_document(user_id, document_id, status="deleting", error=None)
        future = self._submit(self._delete_document(user_id, document_id))
        future.add_done_callback(lambda completed: self._log_background_error(completed, document_id))
        return {"id": document_id, "status": "deleting"}

    def retrieve(self, user_id: str, query: str, mode: str = "mix") -> RagRetrievalResult:
        timeout = max(30, min(self.settings.subagent_timeout_seconds, 900))
        return self._submit(self._retrieve(user_id, query, mode)).result(timeout=timeout)

    def close(self) -> None:
        if not self._loop.is_running():
            return
        try:
            self._submit(self._finalize()).result(timeout=30)
        except Exception:
            logger.exception("Failed to finalize LightRAG storages")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _get_rag(self, user_id: str) -> LightRAG:
        key = self._user_key(user_id)
        if key in self._rags:
            return self._rags[key]
        lock = self._rag_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._rags:
                return self._rags[key]
            index_dir = self._user_dir(user_id) / "index"
            index_dir.mkdir(parents=True, exist_ok=True)

            # LightRAG snapshots its dataclass config with ``asdict``. Plain
            # closures avoid deepcopying this service's thread locks through
            # bound-method ``__self__`` references.
            async def llm_model_func(*args: Any, **kwargs: Any) -> str:
                return await self._llm_model_func(*args, **kwargs)

            async def embedding_func(texts: list[str]) -> np.ndarray:
                return await self._embedding_func(texts)

            rag = LightRAG(
                working_dir=str(index_dir),
                workspace="default",
                llm_model_func=llm_model_func,
                llm_model_name=self.settings.model,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self._embedding_dim,
                    max_token_size=8192,
                    model_name=self.settings.embedding_model,
                    func=embedding_func,
                ),
                top_k=self.settings.rag_top_k,
                chunk_top_k=self.settings.rag_chunk_top_k,
                addon_params={"language": "Chinese"},
                enable_llm_cache=True,
                enable_llm_cache_for_entity_extract=True,
            )
            await rag.initialize_storages()
            self._rags[key] = rag
            return rag

    async def _index_document(self, user_id: str, document_id: str) -> None:
        try:
            document = self._get_document(user_id, document_id)
            if document is None:
                return
            self._update_document(user_id, document_id, status="parsing", error=None)
            path = self._user_dir(user_id) / "files" / document["storage_name"]
            text = (await asyncio.to_thread(extract_document, path)).strip()
            if len(text) < 10:
                raise ValueError("No usable text could be extracted from this file")
            self._update_document(user_id, document_id, status="indexing", characters=len(text))
            rag = await self._get_rag(user_id)
            await rag.ainsert(text, ids=[document_id], file_paths=[document["name"]])
            self._update_document(user_id, document_id, status="ready", error=None, characters=len(text))
        except Exception as exc:
            logger.exception("LightRAG indexing failed for %s", document_id)
            self._update_document(user_id, document_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    async def _delete_document(self, user_id: str, document_id: str) -> None:
        document = self._get_document(user_id, document_id)
        if document is None:
            return
        try:
            if document.get("status") != "failed":
                rag = await self._get_rag(user_id)
                result = await rag.adelete_by_doc_id(document_id)
                if result.status not in {"success", "not_found"}:
                    raise RuntimeError(result.message)
            path = self._user_dir(user_id) / "files" / document["storage_name"]
            path.unlink(missing_ok=True)
            self._remove_document(user_id, document_id)
        except Exception as exc:
            logger.exception("LightRAG deletion failed for %s", document_id)
            self._update_document(user_id, document_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    async def _retrieve(self, user_id: str, query: str, mode: str) -> RagRetrievalResult:
        if mode not in {"mix", "hybrid", "local", "global", "naive"}:
            mode = "mix"
        ready_documents = [item for item in self._read_manifest(user_id) if item.get("status") == "ready"]
        if not ready_documents:
            return RagRetrievalResult(query=query, mode=mode, context="", sources=[], success=False, message="No indexed documents are ready")
        rag = await self._get_rag(user_id)
        raw = await rag.aquery_data(
            query,
            QueryParam(
                mode=mode,
                top_k=self.settings.rag_top_k,
                chunk_top_k=self.settings.rag_chunk_top_k,
                max_total_tokens=12000,
                enable_rerank=False,
                include_references=True,
            ),
        )
        return _retrieval_result(query, mode, raw, self.settings.rag_context_max_chars)

    async def _graph_payload(
        self,
        user_id: str,
        ready_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rag = await self._get_rag(user_id)
        raw_nodes, raw_edges = await asyncio.gather(
            rag.chunk_entity_relation_graph.get_all_nodes(),
            rag.chunk_entity_relation_graph.get_all_edges(),
        )

        node_ids = {str(item.get("id") or "").strip() for item in raw_nodes}
        node_ids.discard("")
        valid_edges = [
            item
            for item in raw_edges
            if str(item.get("source") or "").strip() in node_ids
            and str(item.get("target") or "").strip() in node_ids
        ]
        degree = {node_id: 0 for node_id in node_ids}
        for edge in valid_edges:
            degree[str(edge["source"]).strip()] += 1
            degree[str(edge["target"]).strip()] += 1
        max_degree = max(degree.values(), default=1) or 1
        max_weight = max((_as_float(item.get("weight"), 1.0) for item in valid_edges), default=1.0) or 1.0

        nodes = []
        for item in raw_nodes:
            node_id = str(item.get("id") or "").strip()
            if not node_id:
                continue
            nodes.append(
                {
                    "id": node_id,
                    "type": "entity",
                    "title": str(item.get("entity_id") or node_id),
                    "description": _clean_lightrag_text(item.get("description")),
                    "entity_type": str(item.get("entity_type") or "unknown"),
                    "file_path": _clean_lightrag_text(item.get("file_path")),
                    "degree": degree[node_id],
                    "importance": degree[node_id] / max_degree,
                }
            )
        nodes.sort(key=lambda item: (-item["degree"], item["title"].casefold()))

        edges = []
        for item in valid_edges:
            source = str(item["source"]).strip()
            target = str(item["target"]).strip()
            raw_weight = _as_float(item.get("weight"), 1.0)
            edge_key = hashlib.sha1(f"{source}\0{target}".encode("utf-8")).hexdigest()[:16]
            edges.append(
                {
                    "id": f"rag-edge-{edge_key}",
                    "source_id": source,
                    "target_id": target,
                    "edge_type": _clean_lightrag_text(item.get("keywords")) or "related",
                    "description": _clean_lightrag_text(item.get("description")),
                    "file_path": _clean_lightrag_text(item.get("file_path")),
                    "raw_weight": raw_weight,
                    "weight": raw_weight / max_weight,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "documents": len(ready_documents),
            },
            "updated_at": max((item.get("updated_at") or "" for item in ready_documents), default=""),
        }

    async def _embedding_func(self, texts: list[str]) -> np.ndarray:
        vectors = await asyncio.to_thread(self._embedder.embed_many, list(texts))
        return np.asarray(vectors, dtype=np.float32)

    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        for item in history_messages or []:
            content = str(item.get("content") or "")
            messages.append(AIMessage(content=content) if item.get("role") == "assistant" else HumanMessage(content=content))
        messages.append(HumanMessage(content=prompt))
        response = await self._model.ainvoke(messages)
        return _message_text(response.content)

    async def _finalize(self) -> None:
        for rag in list(self._rags.values()):
            await _shutdown_lightrag_workers(rag)
            await rag.finalize_storages()
        self._rags.clear()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _submit(self, coroutine) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _user_key(self, user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]

    def _user_dir(self, user_id: str) -> Path:
        path = self.base_dir / self._user_key(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "documents.json"

    def _read_manifest(self, user_id: str) -> list[dict[str, Any]]:
        with self._manifest_lock:
            path = self._manifest_path(user_id)
            if not path.exists():
                return []
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.exception("Failed to read RAG manifest for %s", user_id)
                return []
            return value if isinstance(value, list) else []

    def _write_manifest(self, user_id: str, documents: list[dict[str, Any]]) -> None:
        path = self._manifest_path(user_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _put_document(self, user_id: str, document: dict[str, Any]) -> None:
        with self._manifest_lock:
            documents = self._read_manifest(user_id)
            documents.append(document)
            self._write_manifest(user_id, documents)

    def _get_document(self, user_id: str, document_id: str) -> dict[str, Any] | None:
        return next((item for item in self._read_manifest(user_id) if item.get("id") == document_id), None)

    def _update_document(self, user_id: str, document_id: str, **updates: Any) -> None:
        with self._manifest_lock:
            documents = self._read_manifest(user_id)
            for document in documents:
                if document.get("id") == document_id:
                    document.update(updates)
                    document["updated_at"] = _utc_now()
                    self._write_manifest(user_id, documents)
                    return

    def _remove_document(self, user_id: str, document_id: str) -> None:
        with self._manifest_lock:
            documents = [item for item in self._read_manifest(user_id) if item.get("id") != document_id]
            self._write_manifest(user_id, documents)

    @staticmethod
    def _public_document(document: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in document.items() if key != "storage_name"}

    @staticmethod
    def _log_background_error(future: Future, document_id: str) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Unhandled LightRAG background error for %s", document_id)


def _retrieval_result(query: str, mode: str, raw: dict[str, Any], max_chars: int) -> RagRetrievalResult:
    data = raw.get("data") if isinstance(raw, dict) else None
    data = data if isinstance(data, dict) else {}
    chunks = list(data.get("chunks") or [])
    entities = list(data.get("entities") or [])
    relationships = list(data.get("relationships") or [])
    references = list(data.get("references") or [])
    sources = sorted(
        {
            str(item.get("file_path") or "").strip()
            for item in [*references, *chunks, *entities, *relationships]
            if str(item.get("file_path") or "").strip()
        }
    )
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[上传资料:{chunk.get('file_path') or 'unknown'}]\n{chunk.get('content') or ''}")
    if entities:
        parts.append("实体证据:\n" + "\n".join(f"- {item.get('entity_name')}: {item.get('description')}" for item in entities))
    if relationships:
        parts.append(
            "关系证据:\n"
            + "\n".join(
                f"- {item.get('src_id')} -> {item.get('tgt_id')}: {item.get('description')}" for item in relationships
            )
        )
    context = "\n\n".join(parts).strip()[:max_chars]
    success = bool(context and (chunks or entities or relationships)) and raw.get("status") != "failure"
    return RagRetrievalResult(
        query=query,
        mode=mode,
        context=context,
        sources=sources,
        success=success,
        message=str(raw.get("message") or ("Evidence found" if success else "No relevant uploaded evidence found")),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _empty_graph_payload() -> dict[str, Any]:
    return {
        "nodes": [],
        "edges": [],
        "stats": {"nodes": 0, "edges": 0, "documents": 0},
        "updated_at": "",
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_lightrag_text(value: Any) -> str:
    return str(value or "").replace("<SEP>", " | ").strip()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return str(content) if content else ""


async def _shutdown_lightrag_workers(rag: LightRAG) -> None:
    candidates = list(getattr(rag, "role_llm_funcs", {}).values())
    embedding_func = getattr(getattr(rag, "embedding_func", None), "func", None)
    candidates.extend([embedding_func, getattr(rag, "rerank_model_func", None)])
    seen: set[int] = set()
    for function in candidates:
        if function is None or id(function) in seen:
            continue
        seen.add(id(function))
        shutdown = getattr(function, "shutdown", None)
        if callable(shutdown):
            try:
                await shutdown(graceful=True)
            except Exception:
                logger.exception("Failed to stop a LightRAG worker queue")
