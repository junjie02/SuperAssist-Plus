from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, Protocol


@dataclass
class RagRetrievalResult:
    query: str
    mode: str
    context: str
    sources: list[str]
    success: bool
    message: str


class Retriever(Protocol):
    settings: object

    def retrieve(self, user_id: str, query: str, mode: str = "mix") -> RagRetrievalResult:
        ...


@dataclass
class RagTurnSession:
    retriever: Retriever | None
    user_id: str
    enabled: bool
    max_attempts: int = 3
    attempts: int = 0
    sources: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    successful: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False)

    def search(self, query: str, mode: str = "mix") -> RagRetrievalResult:
        with self._lock:
            if not self.enabled or self.retriever is None:
                return RagRetrievalResult(query, mode, "", [], False, "RAG mode is not enabled")
            if self.attempts >= self.max_attempts:
                return RagRetrievalResult(query, mode, "", [], False, "Uploaded-data retrieval limit reached")
            self.attempts += 1
            self.queries.append(query)
            try:
                result = self.retriever.retrieve(self.user_id, query, mode)
            except Exception as exc:
                return RagRetrievalResult(
                    query,
                    mode,
                    "",
                    [],
                    False,
                    f"LightRAG retrieval error: {type(exc).__name__}: {exc}",
                )
            self.sources.update(result.sources)
            self.successful = self.successful or result.success
            return result

    def trace(self) -> dict:
        return {
            "enabled": self.enabled,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "queries": list(self.queries),
            "sources": sorted(self.sources),
            "uploaded_evidence_found": self.successful,
        }


_current_session: ContextVar[RagTurnSession | None] = ContextVar("superassist_rag_session", default=None)


def current_rag_session() -> RagTurnSession | None:
    return _current_session.get()


@contextmanager
def rag_turn_context(
    retriever: Retriever | None,
    user_id: str,
    enabled: bool,
    max_attempts: int,
) -> Iterator[RagTurnSession]:
    session = RagTurnSession(retriever=retriever, user_id=user_id, enabled=enabled, max_attempts=max_attempts)
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)
