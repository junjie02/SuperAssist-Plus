from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable

import numpy as np

from superassist_plus.models import MemoryNode


@dataclass(frozen=True)
class VectorMatch:
    node_id: str
    score: float


class PersistentFaissIndex:
    """Persistent FAISS index plus node-id mapping for memory embeddings."""

    def __init__(self, index_path: Path, mapping_path: Path) -> None:
        self.index_path = index_path
        self.mapping_path = mapping_path
        self._lock = RLock()

    def rebuild(self, nodes: Iterable[MemoryNode]) -> None:
        vectors: list[list[float]] = []
        node_ids: list[str] = []
        dimension = 0
        for node in nodes:
            if not node.embedding:
                continue
            if dimension == 0:
                dimension = len(node.embedding)
            if len(node.embedding) != dimension:
                continue
            vectors.append(node.embedding)
            node_ids.append(node.id)

        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            if not vectors:
                self._delete_files()
                return

            import faiss

            matrix = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            ids = np.arange(len(node_ids), dtype="int64")
            index.add_with_ids(matrix, ids)
            faiss.write_index(index, str(self.index_path))
            self.mapping_path.write_text(
                json.dumps(
                    {
                        "dimension": dimension,
                        "ids": node_ids,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def search(self, query_embedding: list[float], limit: int) -> list[VectorMatch]:
        if not query_embedding or not self.index_path.exists() or not self.mapping_path.exists():
            return []
        with self._lock:
            mapping = self._load_mapping()
            dimension = int(mapping.get("dimension") or 0)
            node_ids = list(mapping.get("ids") or [])
            if not dimension or len(query_embedding) != dimension or not node_ids:
                return []

            import faiss

            index = faiss.read_index(str(self.index_path))
            query = np.asarray([query_embedding], dtype="float32")
            faiss.normalize_L2(query)
            scores, ids = index.search(query, max(1, min(limit, len(node_ids))))

        matches: list[VectorMatch] = []
        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            if faiss_id < 0 or faiss_id >= len(node_ids):
                continue
            matches.append(VectorMatch(node_id=node_ids[int(faiss_id)], score=float(score)))
        return matches

    def remove(self, node_ids: Iterable[str], current_nodes: Iterable[MemoryNode]) -> None:
        removed = set(node_ids)
        if not removed:
            return
        self.rebuild(node for node in current_nodes if node.id not in removed)

    def _load_mapping(self) -> dict:
        try:
            value = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _delete_files(self) -> None:
        for path in (self.index_path, self.mapping_path):
            try:
                path.unlink()
            except FileNotFoundError:
                continue


class EmptyVectorIndex:
    """Fallback index that simply returns no matches."""

    def rebuild(self, nodes: Iterable[MemoryNode]) -> None:
        return None

    def search(self, query_embedding: list[float], limit: int) -> list[VectorMatch]:
        return []

    def remove(self, node_ids: Iterable[str], current_nodes: Iterable[MemoryNode]) -> None:
        return None
