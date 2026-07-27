from __future__ import annotations

from langchain_core.tools import tool

from superassist.rag.context import current_rag_session


@tool("rag_search")
def rag_search(query: str, mode: str = "mix") -> str:
    """Search uploaded documents with LightRAG.

    Rewrite the query and call this tool again when evidence is insufficient.
    At most three uploaded-data searches are allowed per user turn.

    Args:
        query: Focused search query for the uploaded knowledge base.
        mode: LightRAG mode: mix, hybrid, local, global, or naive.
    """

    session = current_rag_session()
    if session is None:
        return "RAG search is unavailable outside an active chat turn."
    result = session.search(query, mode)
    if not result.success:
        return f"RAG_RETRIEVAL_FAILED: {result.message}. Attempts: {session.attempts}/{session.max_attempts}."
    sources = ", ".join(result.sources) or "unknown uploaded document"
    return f"RAG_RETRIEVAL_SUCCESS\nSources: {sources}\n\n{result.context}"


__all__ = ["rag_search"]
