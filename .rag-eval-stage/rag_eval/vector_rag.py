from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from rag_eval.common import Chunk, VECTOR_INDEX_PATH, get_settings


@dataclass
class Retrieval:
    context: str
    chunk_ids: list[str]
    scores: list[float]
    elapsed_seconds: float
    retrieval_usage: dict[str, int]


class VectorRAG:
    """A conventional dense-vector RAG retriever over the exact LightRAG chunks."""

    def __init__(self, chunks: list[Chunk], top_k: int = 5) -> None:
        from superassist.memory.embedding import get_embedder

        self.chunks = chunks
        self.top_k = top_k
        self.settings = get_settings()
        self.embedder = get_embedder(self.settings)
        self.matrix: np.ndarray | None = None

    def build(self, force: bool = False) -> dict:
        started = time.perf_counter()
        fingerprint = self._fingerprint()
        if VECTOR_INDEX_PATH.exists() and not force:
            stored = np.load(VECTOR_INDEX_PATH, allow_pickle=False)
            if str(stored["fingerprint"].item()) == fingerprint:
                self.matrix = stored["matrix"].astype(np.float32)
                return {
                    "cache_hit": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "chunks": len(self.chunks),
                    "embedding_input_tokens": sum(item.tokens for item in self.chunks),
                    "llm_tokens": 0,
                }

        vectors = self.embedder.embed_many([item.content for item in self.chunks])
        self.matrix = np.asarray(vectors, dtype=np.float32)
        VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(VECTOR_INDEX_PATH, matrix=self.matrix, fingerprint=fingerprint)
        return {
            "cache_hit": False,
            "elapsed_seconds": time.perf_counter() - started,
            "chunks": len(self.chunks),
            "embedding_input_tokens": sum(item.tokens for item in self.chunks),
            "llm_tokens": 0,
        }

    def retrieve(self, query: str) -> Retrieval:
        if self.matrix is None:
            self.build()
        assert self.matrix is not None
        started = time.perf_counter()
        query_vector = np.asarray(self.embedder.embed(query), dtype=np.float32)
        scores = self.matrix @ query_vector
        count = min(self.top_k, len(self.chunks))
        indices = np.argsort(-scores)[:count]
        selected = [self.chunks[int(index)] for index in indices]
        context = "\n\n".join(
            f"[SOURCE {chunk.file_path} | CHUNK {chunk.chunk_id}]\n{chunk.content}"
            for chunk in selected
        )
        return Retrieval(
            context=context,
            chunk_ids=[item.chunk_id for item in selected],
            scores=[float(scores[int(index)]) for index in indices],
            elapsed_seconds=time.perf_counter() - started,
            retrieval_usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "measured_calls": 0,
            },
        )

    def _fingerprint(self) -> str:
        payload = {
            "embedding_model": self.settings.embedding_model,
            "chunks": [asdict(item) for item in self.chunks],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

