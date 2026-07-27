"""Guarantee retrieval retries before a RAG-mode answer falls back."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.rag.context import current_rag_session


class RagRetryMiddleware(AgentMiddleware[SuperAssistState]):
    """Insert a focused LightRAG retry when the model tries to stop too early."""

    state_schema = SuperAssistState

    def after_model(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if not state.get("rag_mode"):
            return None
        session = current_rag_session()
        if session is None or session.successful or session.attempts >= session.max_attempts:
            return None
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        if getattr(message, "tool_calls", None):
            return None

        attempt = session.attempts + 1
        original = str(state.get("input") or "").strip()
        if attempt == 2:
            query = f"{original}\n检索重点：原文中的关键术语、定义、事实和直接证据。"
            mode = "naive"
        else:
            query = f"{original}\n检索重点：相关实体、别名、关系和更广泛的上下文。"
            mode = "global"
        tool_call = {
            "name": "rag_search",
            "args": {"query": query, "mode": mode},
            "id": f"rag_retry_{uuid4().hex}",
            "type": "tool_call",
        }
        return {"messages": [message.model_copy(update={"tool_calls": [tool_call]})]}


__all__ = ["RagRetryMiddleware"]
