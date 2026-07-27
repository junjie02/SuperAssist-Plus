"""Run the first uploaded-document retrieval before the model is called."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.rag.context import current_rag_session


class RagRetrievalMiddleware(AgentMiddleware[SuperAssistState]):
    """Seed an Agentic RAG turn with one LightRAG ``mix`` retrieval."""

    state_schema = SuperAssistState

    def before_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if not state.get("rag_mode"):
            return None
        session = current_rag_session()
        if session is None:
            return {
                "rag_context": "",
                "rag_sources": [],
                "rag_retrieval": {"success": False, "message": "RAG session is unavailable"},
            }
        result = session.search(str(state.get("input") or ""), "mix")
        return {
            "rag_context": result.context,
            "rag_sources": result.sources,
            "rag_retrieval": {
                "success": result.success,
                "message": result.message,
                "query": result.query,
                "mode": result.mode,
                "attempt": session.attempts,
                "max_attempts": session.max_attempts,
            },
        }


__all__ = ["RagRetrievalMiddleware"]
