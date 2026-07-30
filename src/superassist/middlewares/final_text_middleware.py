"""Capture the agent's final assistant text into state metadata.

The runtime exposes the answer to channels (Feishu cards, CLI) by reading
``metadata['final_assistant_text']``. This middleware sets that key once the
inner agent's last turn has produced an AIMessage.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.agent.streaming import clean_answer_text


class FinalTextMiddleware(AgentMiddleware[SuperAssistState]):
    """Promote the last AIMessage text into ``metadata.final_assistant_text``."""

    state_schema = SuperAssistState

    def after_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                text = clean_answer_text(message.content)
                if text:
                    metadata = dict(state.get("metadata") or {})
                    metadata["memory_ready"] = True
                    metadata["final_assistant_text"] = text
                    return {"metadata": metadata}
                break
        return None


__all__ = ["FinalTextMiddleware"]
