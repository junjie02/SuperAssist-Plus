# Agentic RAG Runtime Contract

RAG mode can run through `AgenticRagGraph` when
`SUPERASSIST_AGENTIC_RAG_ENABLED=true`. Ordinary chat and the legacy RAG
middleware path remain available behind the feature flag.

The graph owns route, plan, retrieval, empty-result retry, rewrite, fallback,
synthesis, and finalization. The router supports `uploaded_kb`, optional
`sql`, `web`, and explicit `direct` decisions; SQL is enabled by injecting a
`Retriever` adapter, so an unavailable SQL backend never becomes a false route.
Retrieval results are converted to stable
`evidence_id` values for citation rendering. Retrieval results and generated
claims are not graded or verified by the control plane.

Current local backend:

```text
HybridRAGService -> HybridRetrieverAdapter -> AgenticRagGraph
```

Production adapters should implement `Retriever.search()` and enforce tenant,
user, ACL, and document-version filters in the backend. The model must never
provide those predicates. `OpenSearchRetriever`, `ScopedRetriever`, and
`ReadonlySqlBackend` define the integration boundaries without adding a hard
dependency on external infrastructure to the Windows development runtime.

The graph uses an in-memory LangGraph checkpointer for one-turn replay and
tests. Production cross-restart recovery requires a real LangGraph checkpointer
such as PostgreSQL; the existing Redis runtime cache is not a checkpointer.

SSE-compatible run events include `rag_route`, `rag_plan`, `rag_retrieval`,
`rag_rewrite`, `rag_backtrack`, and `rag_fallback`.
