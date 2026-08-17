"""Append deterministic provenance to answers produced in RAG mode."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.agent.streaming import clean_answer_text
from superassist.rag.context import current_rag_session

_URL_PATTERN = re.compile(r"https?://[^\s<>()\]}'\"]+")


class RagAttributionMiddleware(AgentMiddleware[SuperAssistState]):
    """Make the answer's uploaded/web/model basis visible and machine-stable."""

    state_schema = SuperAssistState

    def after_agent(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        if not state.get("rag_mode"):
            return None
        answer = _last_ai_text(state.get("messages") or [])
        if not answer or "\n\n---\n回答依据\n" in answer:
            return None

        session = current_rag_session()
        trace = session.trace() if session is not None else {
            "enabled": True,
            "attempts": 0,
            "queries": [],
            "sources": list(state.get("rag_sources") or []),
            "uploaded_evidence_found": bool(state.get("rag_context")),
        }
        uploaded_sources = sorted(set(trace.get("sources") or state.get("rag_sources") or []))
        web_sources = _web_sources(state.get("tool_events") or [])

        lines = ["", "---", "回答依据"]
        uploaded_evidence_found = bool(trace.get("uploaded_evidence_found"))
        if uploaded_evidence_found:
            if uploaded_sources:
                lines.extend(f"- 上传资料：{source}" for source in uploaded_sources)
            else:
                lines.append("- 上传资料：已检索的知识库（来源文件名不可用）")
        if web_sources:
            lines.extend(f"- 联网检索：{source}" for source in web_sources)
        if not uploaded_evidence_found and not web_sources:
            attempts = int(trace.get("attempts") or 0)
            lines.append(f"- 模型自身知识（上传资料检索 {attempts} 次未获得可用证据）")

        attributed = answer.rstrip() + "\n" + "\n".join(lines)
        metadata = dict(state.get("metadata") or {})
        metadata["rag_trace"] = trace
        metadata["answer_provenance"] = {
            "uploaded_documents": uploaded_sources if uploaded_evidence_found else [],
            "web": web_sources,
            "model_knowledge": not bool(uploaded_evidence_found or web_sources),
        }
        return {"messages": [AIMessage(content=attributed)], "metadata": metadata}


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = clean_answer_text(message.content)
            if text:
                return text
    return ""


def _web_sources(events: list[dict[str, Any]]) -> list[str]:
    sources: set[str] = set()
    for event in events:
        if event.get("type") != "tool_result" or event.get("tool") not in {"web_search", "web_fetch"}:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        content = str(event.get("content") or "").strip()
        if content.lower().startswith(("error:", "network tools are disabled", "no search results")):
            continue
        if event.get("tool") == "web_fetch" and str(args.get("url") or "").strip():
            sources.add(str(args["url"]).strip())
        sources.update(_URL_PATTERN.findall(content))
    return sorted(sources)


__all__ = ["RagAttributionMiddleware"]
