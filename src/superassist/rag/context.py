from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Protocol


@dataclass
class RagRetrievalResult:
    query: str
    mode: str
    context: str
    sources: list[str]
    success: bool
    message: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    evidence_tokens: int = 0
    new_hits: int = 0
    duplicate_hits: int = 0


class Retriever(Protocol):
    settings: object

    def retrieve(self, user_id: str, query: str, mode: str = "hybrid") -> RagRetrievalResult:
        ...


@dataclass
class RagTurnSession:
    retriever: Retriever | None
    user_id: str
    enabled: bool
    evidence_max_tokens: int = 8000
    stagnant_search_limit: int = 2
    attempts: int = 0
    sources: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    successful: bool = False
    evidence_tokens: int = 0
    stagnant_searches: int = 0
    stopped_reason: str = ""
    _seen_chunk_ids: set[str] = field(default_factory=set, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def search(self, query: str, mode: str = "hybrid") -> RagRetrievalResult:
        with self._lock:
            if not self.enabled or self.retriever is None:
                return RagRetrievalResult(query, mode, "", [], False, "RAG mode is not enabled")
            if self.stopped_reason:
                return RagRetrievalResult(query, mode, "", [], False, self.stopped_reason)
            self.attempts += 1
            self.queries.append(query)
            try:
                result = self.retriever.retrieve(self.user_id, query, mode)
            except Exception as exc:  # noqa: BLE001 - retrieval failures must not abort the Agent turn
                return RagRetrievalResult(
                    query,
                    mode,
                    "",
                    [],
                    False,
                    f"Hybrid RAG retrieval error: {type(exc).__name__}: {exc}",
                )

            duplicate_hits = 0
            selected: list[dict[str, Any]] = []
            remaining = max(0, self.evidence_max_tokens - self.evidence_tokens)
            for hit in result.hits:
                chunk_id = str(hit.get("chunk_id") or "")
                if not chunk_id or chunk_id in self._seen_chunk_ids:
                    duplicate_hits += 1
                    continue
                token_count = max(0, int(hit.get("token_count") or 0))
                if token_count > remaining:
                    continue
                self._seen_chunk_ids.add(chunk_id)
                selected.append(hit)
                self.evidence_tokens += token_count
                remaining -= token_count

            if selected:
                self.stagnant_searches = 0
                self.successful = True
                self.sources.update(str(item.get("document_name") or "") for item in selected)
                self.sources.discard("")
            else:
                self.stagnant_searches += 1

            if self.evidence_tokens >= self.evidence_max_tokens:
                self.stopped_reason = "Uploaded-data evidence token budget reached"
            elif self.stagnant_searches >= self.stagnant_search_limit:
                self.stopped_reason = "No new uploaded-data chunks were found in consecutive searches"

            context = _format_hits(selected)
            message = (
                f"Retrieved {len(selected)} new original chunks"
                if selected
                else self.stopped_reason or "No new relevant uploaded-data chunks were found"
            )
            return replace(
                result,
                context=context,
                sources=sorted({str(item.get("document_name") or "") for item in selected} - {""}),
                success=bool(selected),
                message=message,
                hits=selected,
                evidence_tokens=sum(int(item.get("token_count") or 0) for item in selected),
                new_hits=len(selected),
                duplicate_hits=duplicate_hits,
            )

    def trace(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "attempts": self.attempts,
            "queries": list(self.queries),
            "sources": sorted(self.sources),
            "uploaded_evidence_found": self.successful,
            "unique_chunks": len(self._seen_chunk_ids),
            "evidence_tokens": self.evidence_tokens,
            "evidence_max_tokens": self.evidence_max_tokens,
            "stagnant_searches": self.stagnant_searches,
            "stopped_reason": self.stopped_reason,
        }


def _format_hits(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for hit in hits:
        header = f"[上传资料:{hit.get('document_name') or 'unknown'} | chunk:{hit.get('chunk_id') or 'unknown'}"
        if hit.get("heading"):
            header += f" | section:{hit['heading']}"
        parts.append(f"{header}]\n{hit.get('text') or ''}")
    return "\n\n".join(parts)


_current_session: ContextVar[RagTurnSession | None] = ContextVar("superassist_rag_session", default=None)


def current_rag_session() -> RagTurnSession | None:
    return _current_session.get()


@contextmanager
def rag_turn_context(
    retriever: Retriever | None,
    user_id: str,
    enabled: bool,
) -> Iterator[RagTurnSession]:
    settings = getattr(retriever, "settings", None)
    session = RagTurnSession(
        retriever=retriever,
        user_id=user_id,
        enabled=enabled,
        evidence_max_tokens=int(getattr(settings, "rag_accumulated_evidence_max_tokens", 8000)),
        stagnant_search_limit=int(getattr(settings, "rag_stagnant_search_limit", 2)),
    )
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)


__all__ = ["RagRetrievalResult", "RagTurnSession", "current_rag_session", "rag_turn_context"]
