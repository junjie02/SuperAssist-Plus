from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superassist_plus.models import (
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


class MemoryGraphStore:
    """SQLite-backed typed memory graph store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
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
                );

                CREATE INDEX IF NOT EXISTS idx_memory_nodes_user_type
                    ON memory_nodes(user_id, type);

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
                );

                CREATE INDEX IF NOT EXISTS idx_memory_edges_user_source
                    ON memory_edges(user_id, source_id);
                CREATE INDEX IF NOT EXISTS idx_memory_edges_user_target
                    ON memory_edges(user_id, target_id);

                CREATE TABLE IF NOT EXISTS memory_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_jobs_status
                    ON memory_jobs(status, updated_at);

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
                );

                CREATE INDEX IF NOT EXISTS idx_memory_recall_snapshots_user
                    ON memory_recall_snapshots(user_id, updated_at);
                """
            )

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
                """
                INSERT INTO memory_nodes (
                    id, user_id, type, title, description, importance, access_count,
                    embedding_json, reasoning, grounded_in_json, metadata_json,
                    created_at, updated_at, last_accessed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    user_id,
                    node.type.value,
                    node.title,
                    node.description,
                    node.importance,
                    node.access_count,
                    json.dumps(node.embedding) if node.embedding is not None else None,
                    node.reasoning,
                    json.dumps(node.grounded_in, ensure_ascii=False),
                    json.dumps(node.metadata, ensure_ascii=False),
                    now,
                    now,
                    None,
                ),
            )
        return node

    def update_node(self, node: MemoryNode) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE memory_nodes
                SET title = ?, description = ?, importance = ?, access_count = ?,
                    embedding_json = ?, reasoning = ?, grounded_in_json = ?,
                    metadata_json = ?, updated_at = ?, last_accessed_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    node.title,
                    node.description,
                    node.importance,
                    node.access_count,
                    json.dumps(node.embedding) if node.embedding is not None else None,
                    node.reasoning,
                    json.dumps(node.grounded_in, ensure_ascii=False),
                    json.dumps(node.metadata, ensure_ascii=False),
                    now,
                    node.last_accessed_at.isoformat() if node.last_accessed_at else None,
                    node.id,
                    node.user_id,
                ),
            )

    def get_node(self, user_id: str, node_id: str) -> MemoryNode | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_nodes WHERE user_id = ? AND id = ?",
                (user_id, node_id),
            ).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, user_id: str, node_type: NodeType | None = None) -> list[MemoryNode]:
        sql = "SELECT * FROM memory_nodes WHERE user_id = ?"
        params: list[Any] = [user_id]
        if node_type is not None:
            sql += " AND type = ?"
            params.append(node_type.value)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_node(row) for row in rows]

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
                """
                SELECT * FROM memory_edges
                WHERE user_id = ? AND source_id = ? AND target_id = ? AND edge_type = ?
                """,
                (user_id, source_id, target_id, edge_type.value),
            ).fetchone()
            if existing:
                new_weight = min(1.0, float(existing["weight"]) + boost)
                conn.execute(
                    """
                    UPDATE memory_edges
                    SET weight = ?, metadata_json = ?, updated_at = ?, last_activated_at = ?
                    WHERE id = ?
                    """,
                    (
                        new_weight,
                        json.dumps({**self._safe_json(existing["metadata_json"]), **(metadata or {})}, ensure_ascii=False),
                        now,
                        now,
                        existing["id"],
                    ),
                )
                row = conn.execute("SELECT * FROM memory_edges WHERE id = ?", (existing["id"],)).fetchone()
            else:
                edge_id = new_id("edge")
                conn.execute(
                    """
                    INSERT INTO memory_edges (
                        id, user_id, source_id, target_id, edge_type, weight, metadata_json,
                        created_at, updated_at, last_activated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        user_id,
                        source_id,
                        target_id,
                        edge_type.value,
                        max(0.0, min(1.0, default_weight)),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        now,
                        now,
                    ),
                )
                row = conn.execute("SELECT * FROM memory_edges WHERE id = ?", (edge_id,)).fetchone()
        return self._row_to_edge(row)

    def list_edges(self, user_id: str) -> list[MemoryEdge]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM memory_edges WHERE user_id = ?", (user_id,)).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def replace_edge_endpoint(self, user_id: str, old_id: str, new_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE OR IGNORE memory_edges SET source_id = ?, updated_at = ? WHERE user_id = ? AND source_id = ?",
                (new_id, utc_now_iso(), user_id, old_id),
            )
            conn.execute(
                "UPDATE OR IGNORE memory_edges SET target_id = ?, updated_at = ? WHERE user_id = ? AND target_id = ?",
                (new_id, utc_now_iso(), user_id, old_id),
            )

    def delete_node(self, user_id: str, node_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM memory_nodes WHERE user_id = ? AND id = ?", (user_id, node_id))

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE memory_edges SET weight = ?, updated_at = ? WHERE id = ?",
                (max(0.0, min(1.0, weight)), utc_now_iso(), edge_id),
            )

    def delete_edges(self, edge_ids: Iterable[str]) -> None:
        ids = list(edge_ids)
        if not ids:
            return
        with self.connect() as conn:
            conn.executemany("DELETE FROM memory_edges WHERE id = ?", [(edge_id,) for edge_id in ids])

    def touch_nodes(self, user_id: str, node_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return
        now = utc_now_iso()
        with self.connect() as conn:
            conn.executemany(
                """
                UPDATE memory_nodes
                SET access_count = access_count + 1, last_accessed_at = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                [(now, now, user_id, node_id) for node_id in ids],
            )

    def replace_recall_snapshot(self, user_id: str, items: Iterable[dict[str, Any]]) -> None:
        now = utc_now_iso()
        rows = [
            (
                user_id,
                str(item["node_id"]),
                str(item["tier"]),
                float(item["score"]),
                float(item.get("pagerank", 0.0)),
                float(item.get("recency", 0.0)),
                float(item.get("access", 0.0)),
                float(item.get("urgency", 1.0)),
                float(item.get("semantic_affinity", 0.0)),
                now,
            )
            for item in items
            if item.get("node_id")
        ]
        with self.connect() as conn:
            conn.execute("DELETE FROM memory_recall_snapshots WHERE user_id = ?", (user_id,))
            conn.executemany(
                """
                INSERT INTO memory_recall_snapshots (
                    user_id, node_id, tier, score, pagerank, recency, access,
                    urgency, semantic_affinity, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def list_recall_snapshot(self, user_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT node_id, tier, score, pagerank, recency, access,
                       urgency, semantic_affinity, updated_at
                FROM memory_recall_snapshots
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()
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

    def _validate_edge(self, edge_type: EdgeType, source_type: NodeType, target_type: NodeType) -> None:
        allowed_source, allowed_target = EDGE_TYPE_CONSTRAINTS[edge_type]
        if source_type not in allowed_source or target_type not in allowed_target:
            raise ValueError(
                f"{edge_type.value} cannot connect {source_type.value} -> {target_type.value}"
            )

    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
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

    def _row_to_edge(self, row: sqlite3.Row) -> MemoryEdge:
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
