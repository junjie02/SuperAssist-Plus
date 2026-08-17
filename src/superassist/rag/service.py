"""Local hybrid RAG service backed by SQLite FTS5 and FAISS."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from superassist.config import Settings
from superassist.memory.embedding import Embedder, get_embedder
from superassist.rag.chunking import chunk_document, lexical_terms, lexical_text, truncate_tokens
from superassist.rag.context import RagRetrievalResult
from superassist.rag.documents import SUPPORTED_EXTENSIONS, extract_document, safe_filename

logger = logging.getLogger(__name__)


class HybridRAGService:
    """Index uploaded files as original chunks and retrieve them with Dense + BM25 RRF."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_dir = settings.rag_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Embedder = get_embedder(settings)
        self._embedder.embed("Hybrid RAG embedding dimension probe")
        self._manifest_lock = threading.RLock()
        self._user_locks_guard = threading.Lock()
        self._user_locks: dict[str, threading.RLock] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="superassist-rag-index")

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

        content_hash = hashlib.sha256(content).hexdigest()
        duplicate = next(
            (
                item
                for item in self._read_manifest(user_id)
                if item.get("content_hash") == content_hash and item.get("status") != "failed"
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"The same file content is already indexed as {duplicate.get('name') or duplicate['id']}")

        document_id = f"doc-{uuid4().hex}"
        raw_dir = self._user_dir(user_id) / "files"
        raw_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"{document_id}{Path(filename).suffix.lower()}"
        (raw_dir / storage_name).write_bytes(content)
        now = _utc_now()
        document = {
            "id": document_id,
            "name": filename,
            "storage_name": storage_name,
            "content_hash": content_hash,
            "size": len(content),
            "status": "queued",
            "error": None,
            "characters": None,
            "chunks": None,
            "created_at": now,
            "updated_at": now,
        }
        self._put_document(user_id, document)
        future = self._executor.submit(self._index_document, user_id, document_id)
        future.add_done_callback(lambda completed: self._log_background_error(completed, document_id))
        return self._public_document(document)

    def list_documents(self, user_id: str) -> list[dict[str, Any]]:
        documents = self._read_manifest(user_id)
        return [self._public_document(item) for item in sorted(documents, key=lambda item: item["created_at"], reverse=True)]

    def graph_payload(self, user_id: str) -> dict[str, Any]:
        documents = [item for item in self._read_manifest(user_id) if item.get("status") == "ready"]
        chunks = 0
        database = self._database_path(user_id)
        if database.exists():
            with self._connect(user_id) as connection:
                chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return {
            "nodes": [],
            "edges": [],
            "stats": {"nodes": 0, "edges": 0, "documents": len(documents), "chunks": chunks},
            "updated_at": max((item.get("updated_at") or "" for item in documents), default=""),
        }

    def delete_document(self, user_id: str, document_id: str) -> dict[str, Any]:
        document = self._get_document(user_id, document_id)
        if document is None:
            raise KeyError(document_id)
        if document["status"] in {"queued", "parsing", "indexing", "deleting"}:
            raise ValueError("Document is still being processed")
        self._update_document(user_id, document_id, status="deleting", error=None)
        future = self._executor.submit(self._delete_document, user_id, document_id)
        future.add_done_callback(lambda completed: self._log_background_error(completed, document_id))
        return {"id": document_id, "status": "deleting"}

    def retrieve(self, user_id: str, query: str, mode: str = "hybrid") -> RagRetrievalResult:
        mode = mode if mode in {"hybrid", "dense", "bm25"} else "hybrid"
        query = str(query or "").strip()
        if not query:
            return RagRetrievalResult(query, mode, "", [], False, "Search query is empty")
        if not any(item.get("status") == "ready" for item in self._read_manifest(user_id)):
            return RagRetrievalResult(query, mode, "", [], False, "No indexed documents are ready")

        with self._user_lock(user_id):
            self._ensure_store(user_id)
            dense = self._dense_search(user_id, query, self.settings.rag_candidate_top_k) if mode != "bm25" else []
            sparse = self._bm25_search(user_id, query, self.settings.rag_candidate_top_k) if mode != "dense" else []
            fused = _rrf_fuse(dense, sparse, self.settings.rag_rrf_k, mode)
            candidate_ids = [chunk_id for chunk_id, _score, _dense_rank, _bm25_rank in fused]
            rows = self._load_chunks(user_id, candidate_ids)

        by_id = {str(row["id"]): row for row in rows}
        hits: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        context_parts: list[str] = []
        remaining = self.settings.rag_context_max_tokens
        for chunk_id, score, dense_rank, bm25_rank in fused:
            if len(hits) >= self.settings.rag_chunk_top_k:
                break
            row = by_id.get(chunk_id)
            if row is None or row["content_hash"] in seen_hashes:
                continue
            header = f"[上传资料:{row['document_name']} | chunk:{chunk_id}"
            if row["heading"]:
                header += f" | section:{row['heading']}"
            header += "]"
            header_tokens = _token_count(header) + 1
            if remaining <= header_tokens + 16:
                break
            text = truncate_tokens(str(row["text"]), remaining - header_tokens)
            if not text:
                continue
            used = header_tokens + _token_count(text)
            context_parts.append(f"{header}\n{text}")
            remaining -= used
            seen_hashes.add(str(row["content_hash"]))
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": row["document_id"],
                    "document_name": row["document_name"],
                    "ordinal": int(row["ordinal"]),
                    "heading": row["heading"],
                    "text": text,
                    "token_count": used,
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "rrf_score": score,
                }
            )

        context = "\n\n".join(context_parts)
        sources = sorted({str(item["document_name"]) for item in hits}, key=str.casefold)
        return RagRetrievalResult(
            query=query,
            mode=mode,
            context=context,
            sources=sources,
            success=bool(hits),
            message=f"Retrieved {len(hits)} original chunks" if hits else "No relevant uploaded evidence found",
            hits=hits,
            evidence_tokens=self.settings.rag_context_max_tokens - remaining,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _index_document(self, user_id: str, document_id: str) -> None:
        try:
            document = self._get_document(user_id, document_id)
            if document is None:
                return
            self._update_document(user_id, document_id, status="parsing", error=None)
            path = self._user_dir(user_id) / "files" / document["storage_name"]
            text = extract_document(path).strip()
            if len(text) < 10:
                raise ValueError("No usable text could be extracted from this file")
            chunks = chunk_document(
                text,
                document_id=document_id,
                document_name=str(document["name"]),
                target_tokens=self.settings.rag_chunk_target_tokens,
                max_tokens=self.settings.rag_chunk_max_tokens,
                overlap_tokens=self.settings.rag_chunk_overlap_tokens,
            )
            if not chunks:
                raise ValueError("No usable chunks could be produced from this file")
            self._update_document(user_id, document_id, status="indexing", characters=len(text), chunks=len(chunks))
            vectors = self._embedder.embed_many([item.searchable_text for item in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("Embedding provider returned an unexpected vector count")

            with self._user_lock(user_id):
                self._ensure_store(user_id)
                with self._connect(user_id) as connection:
                    old_ids = [
                        str(row[0])
                        for row in connection.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
                    ]
                    connection.executemany("DELETE FROM chunk_fts WHERE chunk_id = ?", [(item,) for item in old_ids])
                    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        array = np.asarray(vector, dtype="float32")
                        connection.execute(
                            """
                            INSERT INTO chunks (
                                id, document_id, document_name, ordinal, parent_id, heading,
                                text, token_count, content_hash, embedding, embedding_dim
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chunk.id,
                                chunk.document_id,
                                chunk.document_name,
                                chunk.ordinal,
                                chunk.parent_id,
                                chunk.heading,
                                chunk.text,
                                chunk.token_count,
                                chunk.content_hash,
                                array.tobytes(),
                                len(array),
                            ),
                        )
                        connection.execute(
                            "INSERT INTO chunk_fts (chunk_id, document_name, heading, body) VALUES (?, ?, ?, ?)",
                            (
                                chunk.id,
                                lexical_text(chunk.document_name),
                                lexical_text(chunk.heading),
                                lexical_text(chunk.text),
                            ),
                        )
                    connection.commit()
                    self._rebuild_dense_index(user_id, connection)
            self._update_document(
                user_id,
                document_id,
                status="ready",
                error=None,
                characters=len(text),
                chunks=len(chunks),
            )
        except Exception as exc:
            logger.exception("Hybrid RAG indexing failed for %s", document_id)
            self._update_document(user_id, document_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _delete_document(self, user_id: str, document_id: str) -> None:
        document = self._get_document(user_id, document_id)
        if document is None:
            return
        try:
            with self._user_lock(user_id):
                self._ensure_store(user_id)
                with self._connect(user_id) as connection:
                    chunk_ids = [
                        str(row[0])
                        for row in connection.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
                    ]
                    connection.executemany("DELETE FROM chunk_fts WHERE chunk_id = ?", [(item,) for item in chunk_ids])
                    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                    connection.commit()
                    self._rebuild_dense_index(user_id, connection)
                (self._user_dir(user_id) / "files" / document["storage_name"]).unlink(missing_ok=True)
                self._remove_document(user_id, document_id)
        except Exception as exc:
            logger.exception("Hybrid RAG deletion failed for %s", document_id)
            self._update_document(user_id, document_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _ensure_store(self, user_id: str) -> None:
        with self._connect(user_id) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    parent_id TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_name,
                    heading,
                    body,
                    tokenize='unicode61'
                );
                """
            )

    def _dense_search(self, user_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        index_path, mapping_path = self._dense_paths(user_id)
        if not index_path.exists() or not mapping_path.exists():
            return []
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        chunk_ids = list(mapping.get("ids") or [])
        dimension = int(mapping.get("dimension") or 0)
        vector = self._embedder.embed(query)
        if not chunk_ids or len(vector) != dimension:
            return []
        import faiss

        index = faiss.read_index(str(index_path))
        query_matrix = np.asarray([vector], dtype="float32")
        faiss.normalize_L2(query_matrix)
        scores, ids = index.search(query_matrix, max(1, min(limit, len(chunk_ids))))
        return [
            (str(chunk_ids[int(index_id)]), float(score))
            for score, index_id in zip(scores[0], ids[0], strict=True)
            if 0 <= index_id < len(chunk_ids)
        ]

    def _bm25_search(self, user_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(lexical_terms(query)))
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms[:64])
        with self._connect(user_id) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, bm25(chunk_fts, 0.0, 2.5, 1.8, 1.0) AS score
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        return [(str(row["chunk_id"]), -float(row["score"])) for row in rows]

    def _load_chunks(self, user_id: str, chunk_ids: list[str]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect(user_id) as connection:
            return list(connection.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids))

    def _rebuild_dense_index(self, user_id: str, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT id, embedding, embedding_dim FROM chunks ORDER BY document_id, ordinal").fetchall()
        index_path, mapping_path = self._dense_paths(user_id)
        if not rows:
            index_path.unlink(missing_ok=True)
            mapping_path.unlink(missing_ok=True)
            return
        dimension = int(rows[0]["embedding_dim"])
        valid = [row for row in rows if int(row["embedding_dim"]) == dimension]
        matrix = np.vstack([np.frombuffer(row["embedding"], dtype="float32", count=dimension) for row in valid])
        import faiss

        faiss.normalize_L2(matrix)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        index.add_with_ids(matrix, np.arange(len(valid), dtype="int64"))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        temporary = mapping_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"dimension": dimension, "ids": [str(row["id"]) for row in valid]}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(mapping_path)

    @contextmanager
    def _connect(self, user_id: str) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path(user_id), timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    def _database_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "hybrid.sqlite3"

    def _dense_paths(self, user_id: str) -> tuple[Path, Path]:
        user_dir = self._user_dir(user_id)
        return user_dir / "chunks.faiss", user_dir / "chunks.mapping.json"

    def _user_lock(self, user_id: str) -> threading.RLock:
        key = self._user_key(user_id)
        with self._user_locks_guard:
            return self._user_locks.setdefault(key, threading.RLock())

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
        return {key: value for key, value in document.items() if key not in {"storage_name", "content_hash"}}

    @staticmethod
    def _log_background_error(future: Future, document_id: str) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Unhandled Hybrid RAG background error for %s", document_id)


def _rrf_fuse(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    rank_constant: int,
    mode: str,
) -> list[tuple[str, float, int | None, int | None]]:
    candidates: dict[str, dict[str, float | int | None]] = {}
    dense_weight = 1.0 if mode == "dense" else 0.55
    sparse_weight = 1.0 if mode == "bm25" else 0.45
    for rank, (chunk_id, _score) in enumerate(dense, start=1):
        item = candidates.setdefault(chunk_id, {"score": 0.0, "dense_rank": None, "bm25_rank": None})
        item["score"] = float(item["score"] or 0.0) + dense_weight / (rank_constant + rank)
        item["dense_rank"] = rank
    for rank, (chunk_id, _score) in enumerate(sparse, start=1):
        item = candidates.setdefault(chunk_id, {"score": 0.0, "dense_rank": None, "bm25_rank": None})
        item["score"] = float(item["score"] or 0.0) + sparse_weight / (rank_constant + rank)
        item["bm25_rank"] = rank
    ranked = sorted(candidates.items(), key=lambda item: (-float(item[1]["score"] or 0.0), item[0]))
    return [
        (chunk_id, float(value["score"] or 0.0), value["dense_rank"], value["bm25_rank"])
        for chunk_id, value in ranked
    ]


def _token_count(text: str) -> int:
    from superassist.rag.chunking import count_tokens

    return count_tokens(text)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["HybridRAGService"]
