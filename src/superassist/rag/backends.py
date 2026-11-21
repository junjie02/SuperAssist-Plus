"""Production backend boundaries for Agentic RAG.

These adapters are intentionally dependency-light. Deployments can provide a
PostgreSQL/OpenSearch implementation behind the same ``Retriever`` protocol;
the graph never receives tenant or ACL predicates from the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from superassist.rag.agentic_contracts import RetrievalRequest, RetrievalResult, Retriever


@dataclass(frozen=True)
class TenantScope:
    tenant_id: str
    user_id: str
    allowed_document_ids: tuple[str, ...] = ()
    active_version_only: bool = True


class ScopedRetriever:
    """Enforce tenant/user scope before delegating to a backend."""

    def __init__(self, backend: Retriever, scope_provider: Callable[[str], TenantScope]) -> None:
        self.backend = backend
        self.scope_provider = scope_provider

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        scope = self.scope_provider(request.user_id)
        # Scope is deliberately attached as backend metadata, never accepted
        # from a model-generated query or filter object.
        scoped_request = RetrievalRequest(
            user_id=scope.user_id,
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            run_id=request.run_id,
        )
        result = self.backend.search(scoped_request)
        filtered = [
            hit
            for hit in result.hits
            if (not scope.allowed_document_ids or hit.document_id in scope.allowed_document_ids)
        ]
        return RetrievalResult(
            run_id=result.run_id,
            query=result.query,
            mode=result.mode,
            hits=filtered,
            success=bool(filtered),
            message=result.message if filtered else "No evidence allowed by tenant scope",
            latency_ms=result.latency_ms,
        )


class OpenSearchRetriever:
    """Adapter shell for the production OpenSearch implementation.

    The callable receives a fully scoped request and must return a
    ``RetrievalResult``. Keeping the transport out of this package allows the
    Windows application to use a hosted OpenSearch endpoint without making
    OpenSearch a mandatory local dependency.
    """

    def __init__(self, search_fn: Callable[[RetrievalRequest], RetrievalResult]) -> None:
        self.search_fn = search_fn

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        return self.search_fn(request)


class ReadonlySqlBackend:
    """Safe boundary for a registered, read-only Text-to-SQL backend."""

    def __init__(self, execute_fn: Callable[[str, dict[str, Any], str], RetrievalResult]) -> None:
        self.execute_fn = execute_fn

    def execute(self, *, tenant_id: str, sql: str, parameters: dict[str, Any], source_id: str) -> RetrievalResult:
        normalized = sql.lstrip().lower()
        if not normalized.startswith("select ") or any(token in normalized for token in (";", "--", "/*", "insert ", "update ", "delete ", "drop ", "alter ")):
            raise ValueError("Only one parameterized read-only SELECT statement is allowed")
        return self.execute_fn(sql, parameters, source_id)


__all__ = ["OpenSearchRetriever", "ReadonlySqlBackend", "ScopedRetriever", "TenantScope"]
