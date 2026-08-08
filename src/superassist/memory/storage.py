from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, Connection, create_engine, event, text

from superassist.models import (
    EDGE_TYPE_CONSTRAINTS,
    EDGE_TYPE_DEFAULT_WEIGHTS,
    EdgeType,
    MemoryEdge,
    MemoryNode,
    NodeType,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def create_engine_from_settings(settings: "Settings") -> Engine:  # type: ignore[name-defined]  # noqa: F821
    """Build a SQLAlchemy Engine from settings.

    When ``SUPERASSIST_DB_URL`` is empty, creates a SQLite engine at the
    configured data path (backward compatible).  When set, creates a MySQL
    engine with connection pooling.
    """
    db_url = settings.db_url.strip()

    if not db_url:
        # --- SQLite fallback (development / testing) -----------------------
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{settings.db_path}"
        engine = create_engine(url)

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()

        return engine

    # --- MySQL ----------------------------------------------------------
    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
    )


class MemoryGraphStore:
    """SQLAlchemy-backed typed memory graph store.

    Supports SQLite (default) and MySQL via the ``Engine`` injected at
    construction time.  The public method signatures are unchanged from the
    previous sqlite3-only implementation.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.init_schema()

    def connect(self) -> Connection:
        """Return a transactional connection context manager.

        Usage: ``with self.connect() as conn: ...``

        The transaction is committed when the context exits normally,
        rolled back on exception.
        """
        return self._engine.begin()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        statements: list[str] = [
            # -- memory_nodes -------------------------------------------------
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                embedding_json TEXT,
                reasoning TEXT NOT NULL DEFAULT '',
                grounded_in_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_user_type
                ON memory_nodes(user_id, type)
            """,
            # -- memory_edges -------------------------------------------------
            """
            CREATE TABLE IF NOT EXISTS memory_edges (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_activated_at TEXT,
                UNIQUE(user_id, source_id, target_id, edge_type),
                FOREIGN KEY(source_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_edges_user_source
                ON memory_edges(user_id, source_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_edges_user_target
                ON memory_edges(user_id, target_id)
            """,
            # -- memory_jobs --------------------------------------------------
            """
            CREATE TABLE IF NOT EXISTS memory_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_jobs_status
                ON memory_jobs(status, updated_at)
            """,
            # -- memory_recall_snapshots --------------------------------------
            """
            CREATE TABLE IF NOT EXISTS memory_recall_snapshots (
                user_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                score REAL NOT NULL,
                pagerank REAL NOT NULL DEFAULT 0,
                recency REAL NOT NULL DEFAULT 0,
                access REAL NOT NULL DEFAULT 0,
                urgency REAL NOT NULL DEFAULT 1,
                semantic_affinity REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, node_id),
                FOREIGN KEY(node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_recall_snapshots_user
                ON memory_recall_snapshots(user_id, updated_at)
            """,
            # -- users (shared with Go server) --------------------------------
            """
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(64) PRIMARY KEY,
                username VARCHAR(128) NOT NULL UNIQUE,
                password_hash VARCHAR(256) NOT NULL,
                created_at VARCHAR(64) NOT NULL,
                updated_at VARCHAR(64) NOT NULL
            )
            """,
        ]

        # MySQL needs a separate index for the UNIQUE username column
        # because "UNIQUE" in the column definition creates the index
        # automatically.  We add an explicit IF NOT EXISTS for safety.
        statements.append(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                ON users(username)
            """
        )

        with self.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(
        self,
        *,
        user_id: str,
        node_type: NodeType,
        title: str,
        description: str,
        node_id: str | None = None,
        embedding: list[float] | None = None,
        reasoning: str = "",
        grounded_in: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryNode:
        now = utc_now_iso()
        node = MemoryNode(
            id=node_id or new_id(node_type.value),
            user_id=user_id,
            type=node_type,
            title=title.strip() or node_type.value,
            description=description.strip(),
            embedding=embedding,
            reasoning=reasoning,
            grounded_in=grounded_in or [],
            metadata=metadata or {},
            importance=importance,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO memory_nodes (
                        id, user_id, type, title, description, importance, access_count,
                        embedding_json, reasoning, grounded_in_json, metadata_json,
                        created_at, updated_at, last_accessed_at
                    )
                    VALUES (
                        :id, :user_id, :type, :title, :description, :importance, :access_count,
                        :embedding_json, :reasoning, :grounded_in_json, :metadata_json,
                        :created_at, :updated_at, :last_accessed_at
                    )
                    """
                ),
                {
                    "id": node.id,
                    "user_id": user_id,
                    "type": node.type.value,
                    "title": node.title,
                    "description": node.description,
                    "importance": node.importance,
                    "access_count": node.access_count,
                    "embedding_json": json.dumps(node.embedding) if node.embedding is not None else None,
                    "reasoning": node.reasoning,
                    "grounded_in_json": json.dumps(node.grounded_in, ensure_ascii=False),
                    "metadata_json": json.dumps(node.metadata, ensure_ascii=False),
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed_at": None,
                },
            )
        return node

    def update_node(self, node: MemoryNode) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE memory_nodes
                    SET title = :title, description = :description,
                        importance = :importance, access_count = :access_count,
                        embedding_json = :embedding_json, reasoning = :reasoning,
                        grounded_in_json = :grounded_in_json,
                        metadata_json = :metadata_json, updated_at = :updated_at,
                        last_accessed_at = :last_accessed_at
                    WHERE id = :id AND user_id = :user_id
                    """
                ),
                {
                    "title": node.title,
                    "description": node.description,
                    "importance": node.importance,
                    "access_count": node.access_count,
                    "embedding_json": json.dumps(node.embedding) if node.embedding is not None else None,
                    "reasoning": node.reasoning,
                    "grounded_in_json": json.dumps(node.grounded_in, ensure_ascii=False),
                    "metadata_json": json.dumps(node.metadata, ensure_ascii=False),
                    "updated_at": now,
                    "last_accessed_at": node.last_accessed_at.isoformat() if node.last_accessed_at else None,
                    "id": node.id,
                    "user_id": node.user_id,
                },
            )

    def get_node(self, user_id: str, node_id: str) -> MemoryNode | None:
        with self.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM memory_nodes WHERE user_id = :user_id AND id = :id"),
                {"user_id": user_id, "id": node_id},
            ).mappings().fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, user_id: str, node_type: NodeType | None = None) -> list[MemoryNode]:
        if node_type is not None:
            with self.connect() as conn:
                rows = conn.execute(
                    text("SELECT * FROM memory_nodes WHERE user_id = :user_id AND type = :type"),
                    {"user_id": user_id, "type": node_type.value},
                ).mappings().fetchall()
        else:
            with self.connect() as conn:
                rows = conn.execute(
                    text("SELECT * FROM memory_nodes WHERE user_id = :user_id"),
                    {"user_id": user_id},
                ).mappings().fetchall()
        return [self._row_to_node(row) for row in rows]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_or_boost_edge(
        self,
        *,
        user_id: str,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float | None = None,
        metadata: dict[str, Any] | None = None,
        boost: float = 0.05,
    ) -> MemoryEdge:
        source = self.get_node(user_id, source_id)
        target = self.get_node(user_id, target_id)
        if source is None or target is None:
            raise KeyError("source or target node not found")
        self._validate_edge(edge_type, source.type, target.type)

        now = utc_now_iso()
        default_weight = EDGE_TYPE_DEFAULT_WEIGHTS[edge_type] if weight is None else weight
        with self.connect() as conn:
            existing = conn.execute(
                text(
                    """
                    SELECT * FROM memory_edges
                    WHERE user_id = :user_id AND source_id = :source_id
                      AND target_id = :target_id AND edge_type = :edge_type
                    """
                ),
                {
                    "user_id": user_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_type": edge_type.value,
                },
            ).mappings().fetchone()

            if existing is not None:
                new_weight = min(1.0, float(existing["weight"]) + boost)
                conn.execute(
                    text(
                        """
                        UPDATE memory_edges
                        SET weight = :weight, metadata_json = :metadata_json,
                            updated_at = :updated_at, last_activated_at = :last_activated_at
                        WHERE id = :id
                        """
                    ),
                    {
                        "weight": new_weight,
                        "metadata_json": json.dumps(
                            {**self._safe_json(existing["metadata_json"]), **(metadata or {})},
                            ensure_ascii=False,
                        ),
                        "updated_at": now,
                        "last_activated_at": now,
                        "id": existing["id"],
                    },
                )
                row = conn.execute(
                    text("SELECT * FROM memory_edges WHERE id = :id"),
                    {"id": existing["id"]},
                ).mappings().fetchone()
            else:
                edge_id = new_id("edge")
                conn.execute(
                    text(
                        """
                        INSERT INTO memory_edges (
                            id, user_id, source_id, target_id, edge_type, weight,
                            metadata_json, created_at, updated_at, last_activated_at
                        )
                        VALUES (
                            :id, :user_id, :source_id, :target_id, :edge_type, :weight,
                            :metadata_json, :created_at, :updated_at, :last_activated_at
                        )
                        """
                    ),
                    {
                        "id": edge_id,
                        "user_id": user_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type.value,
                        "weight": max(0.0, min(1.0, default_weight)),
                        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
                        "created_at": now,
                        "updated_at": now,
                        "last_activated_at": now,
                    },
                )
                row = conn.execute(
                    text("SELECT * FROM memory_edges WHERE id = :id"),
                    {"id": edge_id},
                ).mappings().fetchone()

        assert row is not None
        return self._row_to_edge(row)

    def list_edges(self, user_id: str) -> list[MemoryEdge]:
        with self.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM memory_edges WHERE user_id = :user_id"),
                {"user_id": user_id},
            ).mappings().fetchall()
        return [self._row_to_edge(row) for row in rows]

    def replace_edge_endpoint(self, user_id: str, old_id: str, new_id: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE OR IGNORE memory_edges
                    SET source_id = :new_id, updated_at = :now
                    WHERE user_id = :user_id AND source_id = :old_id
                    """
                ),
                {"new_id": new_id, "now": now, "user_id": user_id, "old_id": old_id},
            )
            conn.execute(
                text(
                    """
                    UPDATE OR IGNORE memory_edges
                    SET target_id = :new_id, updated_at = :now
                    WHERE user_id = :user_id AND target_id = :old_id
                    """
                ),
                {"new_id": new_id, "now": now, "user_id": user_id, "old_id": old_id},
            )

    def delete_node(self, user_id: str, node_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                text("DELETE FROM memory_nodes WHERE user_id = :user_id AND id = :id"),
                {"user_id": user_id, "id": node_id},
            )

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        with self.connect() as conn:
            conn.execute(
                text("UPDATE memory_edges SET weight = :weight, updated_at = :updated_at WHERE id = :id"),
                {
                    "weight": max(0.0, min(1.0, weight)),
                    "updated_at": utc_now_iso(),
                    "id": edge_id,
                },
            )

    def delete_edges(self, edge_ids: Iterable[str]) -> None:
        params = [{"id": eid} for eid in edge_ids]
        if not params:
            return
        with self.connect() as conn:
            conn.execute(text("DELETE FROM memory_edges WHERE id = :id"), params)

    # ------------------------------------------------------------------
    # Touch & recall snapshot
    # ------------------------------------------------------------------

    def touch_nodes(self, user_id: str, node_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE memory_nodes
                    SET access_count = access_count + 1,
                        last_accessed_at = :now,
                        updated_at = :now
                    WHERE user_id = :user_id AND id = :id
                    """
                ),
                [{"now": now, "user_id": user_id, "id": node_id} for node_id in ids],
            )

    def replace_recall_snapshot(self, user_id: str, items: Iterable[dict[str, Any]]) -> None:
        now = utc_now_iso()
        rows = [
            {
                "user_id": user_id,
                "node_id": str(item["node_id"]),
                "tier": str(item["tier"]),
                "score": float(item["score"]),
                "pagerank": float(item.get("pagerank", 0.0)),
                "recency": float(item.get("recency", 0.0)),
                "access": float(item.get("access", 0.0)),
                "urgency": float(item.get("urgency", 1.0)),
                "semantic_affinity": float(item.get("semantic_affinity", 0.0)),
                "updated_at": now,
            }
            for item in items
            if item.get("node_id")
        ]
        with self.connect() as conn:
            conn.execute(
                text("DELETE FROM memory_recall_snapshots WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            if rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO memory_recall_snapshots (
                            user_id, node_id, tier, score, pagerank, recency, access,
                            urgency, semantic_affinity, updated_at
                        )
                        VALUES (
                            :user_id, :node_id, :tier, :score, :pagerank, :recency,
                            :access, :urgency, :semantic_affinity, :updated_at
                        )
                        """
                    ),
                    rows,
                )

    def enqueue_memory_job(self, job_id: str, user_id: str, payload: dict[str, Any]) -> bool:
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                text("SELECT status FROM memory_jobs WHERE id = :id"),
                {"id": job_id},
            ).mappings().fetchone()
            if existing is not None:
                return str(existing["status"]) != "completed"
            conn.execute(
                text(
                    """
                    INSERT INTO memory_jobs (
                        id, user_id, status, payload_json, attempts, error, created_at, updated_at
                    ) VALUES (
                        :id, :user_id, 'pending', :payload_json, 0, '', :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": job_id,
                    "user_id": user_id,
                    "payload_json": json.dumps(payload, ensure_ascii=False),
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return True

    def list_memory_jobs(self, *, statuses: tuple[str, ...] = ("pending", "failed")) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join(f":status_{index}" for index in range(len(statuses)))
        parameters = {f"status_{index}": status for index, status in enumerate(statuses)}
        with self.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT id, user_id, status, payload_json, attempts, error, created_at, updated_at
                    FROM memory_jobs
                    WHERE status IN ({placeholders})
                    ORDER BY created_at
                    """
                ),
                parameters,
            ).mappings().fetchall()
        return [dict(row) for row in rows]

    def requeue_stale_memory_jobs(self, stale_before: str) -> int:
        with self.connect() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE memory_jobs
                    SET status = 'failed', error = 'Recovered stale running job', updated_at = :updated_at
                    WHERE status = 'running' AND updated_at < :stale_before
                    """
                ),
                {"stale_before": stale_before, "updated_at": utc_now_iso()},
            )
        return int(result.rowcount or 0)

    def mark_memory_job_running(self, job_id: str) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE memory_jobs
                    SET status = 'running', attempts = attempts + 1, error = '', updated_at = :updated_at
                    WHERE id = :id AND status IN ('pending', 'failed')
                    """
                ),
                {"id": job_id, "updated_at": now},
            )
            row = conn.execute(
                text("SELECT attempts FROM memory_jobs WHERE id = :id AND status = 'running' AND updated_at = :updated_at"),
                {"id": job_id, "updated_at": now},
            ).fetchone()
        return int(row[0]) if row else 0

    def finish_memory_job(self, job_id: str, *, error: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE memory_jobs
                    SET status = :status, error = :error, updated_at = :updated_at
                    WHERE id = :id
                    """
                ),
                {
                    "id": job_id,
                    "status": "failed" if error else "completed",
                    "error": error[:2000],
                    "updated_at": utc_now_iso(),
                },
            )

    def list_recall_snapshot(self, user_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT node_id, tier, score, pagerank, recency, access,
                           urgency, semantic_affinity, updated_at
                    FROM memory_recall_snapshots
                    WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            ).mappings().fetchall()
        return {
            str(row["node_id"]): {
                "tier": row["tier"],
                "score": float(row["score"]),
                "pagerank": float(row["pagerank"]),
                "recency": float(row["recency"]),
                "access": float(row["access"]),
                "urgency": float(row["urgency"]),
                "semantic_affinity": float(row["semantic_affinity"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_edge(self, edge_type: EdgeType, source_type: NodeType, target_type: NodeType) -> None:
        allowed_source, allowed_target = EDGE_TYPE_CONSTRAINTS[edge_type]
        if source_type not in allowed_source or target_type not in allowed_target:
            raise ValueError(
                f"{edge_type.value} cannot connect {source_type.value} -> {target_type.value}"
            )

    # Row mapping access: .mappings() returns dict-like RowMapping objects
    # that support row["col"] syntax, so the bodies are unchanged from sqlite3.

    def _row_to_node(self, row: Any) -> MemoryNode:
        embedding_raw = row["embedding_json"]
        return MemoryNode(
            id=row["id"],
            user_id=row["user_id"],
            type=NodeType(row["type"]),
            title=row["title"],
            description=row["description"],
            importance=float(row["importance"]),
            access_count=int(row["access_count"]),
            embedding=json.loads(embedding_raw) if embedding_raw else None,
            reasoning=row["reasoning"],
            grounded_in=json.loads(row["grounded_in_json"] or "[]"),
            metadata=self._safe_json(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=_parse_dt(row["last_accessed_at"]),
        )

    def _row_to_edge(self, row: Any) -> MemoryEdge:
        return MemoryEdge(
            id=row["id"],
            user_id=row["user_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=float(row["weight"]),
            metadata=self._safe_json(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_activated_at=_parse_dt(row["last_activated_at"]),
        )

    @staticmethod
    def _safe_json(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
