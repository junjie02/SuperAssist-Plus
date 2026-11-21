"""Contracts and deterministic building blocks for Agentic RAG.

The graph deliberately talks to these small interfaces instead of knowing
about FAISS, OpenSearch, or a particular web provider.  The current local
HybridRAGService is adapted here; production backends can replace it without
changing graph control flow.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from superassist.rag.chunking import lexical_terms
from superassist.rag.context import RagRetrievalResult


@dataclass(frozen=True)
class RetrievalRequest:
    user_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = 10
    run_id: str = ""


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: str
    text: str
    source: str
    locator: str = ""
    chunk_id: str = ""
    document_id: str = ""
    document_name: str = ""
    score: float = 0.0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "text": self.text,
            "source": self.source,
            "locator": self.locator,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "score": self.score,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalResult:
    run_id: str
    query: str
    mode: str
    hits: list[Evidence]
    success: bool
    message: str
    latency_ms: float = 0.0


class Retriever(Protocol):
    def search(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class QueryRewriter(Protocol):
    def rewrite(self, question: str, attempt: int) -> tuple[str, str]:
        ...


def evidence_id_for(source_type: str, source_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_type}:{source_id}:{text}".encode()).hexdigest()[:16]
    return f"ev-{digest}"


class HybridRetrieverAdapter:
    """Adapt the existing local HybridRAGService to the graph contract."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        result: RagRetrievalResult = self.service.retrieve(request.user_id, request.query, request.mode)
        hits: list[Evidence] = []
        for hit in result.hits[: request.top_k]:
            chunk_id = str(hit.get("chunk_id") or "")
            text = str(hit.get("text") or "")
            document_name = str(hit.get("document_name") or "")
            hits.append(
                Evidence(
                    evidence_id=evidence_id_for("kb", chunk_id, text),
                    source_type="kb",
                    text=text,
                    source=document_name,
                    locator=str(hit.get("heading") or ""),
                    chunk_id=chunk_id,
                    document_id=str(hit.get("document_id") or ""),
                    document_name=document_name,
                    score=float(hit.get("rrf_score") or 0.0),
                    token_count=int(hit.get("token_count") or 0),
                    metadata={
                        "dense_rank": hit.get("dense_rank"),
                        "bm25_rank": hit.get("bm25_rank"),
                        "reranker_score": hit.get("reranker_score"),
                    },
                )
            )
        return RetrievalResult(
            run_id=request.run_id,
            query=result.query,
            mode=result.mode,
            hits=hits,
            success=bool(hits),
            message=result.message,
        )


class DeterministicQueryRewriter:
    """Produce materially different, safe query variants without an LLM call."""

    def rewrite(self, question: str, attempt: int) -> tuple[str, str]:
        terms = list(dict.fromkeys(lexical_terms(question)))
        if attempt % 3 == 1:
            return " ".join(terms[:24]) or question, "keyword_extract"
        if attempt % 3 == 2:
            return f"{question} 具体定义 原因 影响 方法", "aspect_expand"
        return f"{question} 相关规定 流程 条件", "constraint_expand"


_CITATION_RE = re.compile(r"\[(ev-[0-9a-f]{16})\]")


def parse_answer_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(_CITATION_RE.findall(answer)))


def evidence_context(evidence: list[Evidence], *, max_items: int = 8) -> str:
    return "\n\n".join(
        f"[{item.evidence_id}] {item.source}\n{item.text}"
        for item in evidence[:max_items]
        if item.text.strip()
    )


def json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "DeterministicQueryRewriter",
    "Evidence",
    "HybridRetrieverAdapter",
    "QueryRewriter",
    "RetrievalRequest",
    "RetrievalResult",
    "Retriever",
    "evidence_context",
    "evidence_id_for",
    "parse_answer_citations",
]
