from __future__ import annotations

from langchain_core.tools import tool

from superassist.rag.context import current_rag_session


@tool("rag_search")
def rag_search(query: str, mode: str = "hybrid") -> str:
    """Search uploaded documents with hybrid Dense + BM25 retrieval.

    Rewrite the query and call this tool again when the returned original chunks
    do not provide enough evidence. Repeated chunks are removed across calls.

    Args:
        query: Focused search query for the uploaded knowledge base.
        mode: Retrieval mode: hybrid, dense, or bm25.
    """

    session = current_rag_session()
    if session is None:
        return "RAG search is unavailable outside an active chat turn."
    result = session.search(query, mode)
    if not result.success:
        return f"RAG_RETRIEVAL_FAILED: {result.message}. Searches: {session.attempts}."
    sources = ", ".join(result.sources) or "unknown uploaded document"
    return (
        f"RAG_RETRIEVAL_SUCCESS\nSources: {sources}\n"
        f"New chunks: {result.new_hits}; duplicates removed: {result.duplicate_hits}; "
        f"accumulated evidence tokens: {session.evidence_tokens}/{session.evidence_max_tokens}\n\n"
        f"{result.context}"
    )


__all__ = ["rag_search"]
