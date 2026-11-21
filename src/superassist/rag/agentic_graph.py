"""Explicit LangGraph control plane for Agentic RAG turns."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from superassist.config import Settings
from superassist.models import AgentRunEvent
from superassist.rag.agentic_contracts import (
    DeterministicQueryRewriter,
    Evidence,
    HybridRetrieverAdapter,
    RetrievalRequest,
    Retriever,
    evidence_context,
    evidence_id_for,
    parse_answer_citations,
)

logger = logging.getLogger(__name__)


class AgenticRagState(TypedDict, total=False):
    messages: list[BaseMessage]
    input: str
    user_id: str
    thread_id: str
    route: str
    route_confidence: float
    plan: list[dict[str, Any]]
    active_subquestion: int
    query: str
    mode: str
    query_attempts: list[dict[str, Any]]
    evidence_pool: list[dict[str, Any]]
    retrieval_runs: list[dict[str, Any]]
    answer: str
    draft_answer: str
    citations: list[str]
    decision_stack: list[dict[str, Any]]
    step_count: int
    rewrite_count: int
    backtrack_count: int
    retrieval_count: int
    last_retrieval_hit_count: int
    stagnant_count: int
    final_status: str
    stop_reason: str
    fallback_used: bool
    started_at: float
    metadata: dict[str, Any]


class AgenticRagGraph:
    """Build and run the deterministic control plane around a chat model."""

    def __init__(
        self,
        *,
        model: Any,
        retriever: Retriever | Any,
        settings: Settings,
        sql_retriever: Retriever | None = None,
        reporter: Callable[[AgentRunEvent], None] | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.retriever = retriever if hasattr(retriever, "search") else HybridRetrieverAdapter(retriever)
        self.sql_retriever = sql_retriever
        self.rewriter = DeterministicQueryRewriter()
        self.reporter = reporter
        self.graph = self._compile()

    def _compile(self):
        graph = StateGraph(AgenticRagState)
        graph.add_node("route", self._route)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("next_subquestion", self._next_subquestion)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("backtrack", self._backtrack)
        graph.add_node("web_fallback", self._web_fallback)
        graph.add_node("direct", self._direct)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "route")
        graph.add_edge("route", "plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"retrieve": "retrieve", "web_fallback": "web_fallback", "direct": "direct", "finalize": "finalize"},
        )
        retrieval_routes = {
            "next_subquestion": "next_subquestion",
            "synthesize": "synthesize",
            "rewrite": "rewrite",
            "backtrack": "backtrack",
            "web_fallback": "web_fallback",
            "finalize": "finalize",
        }
        graph.add_conditional_edges("retrieve", self._after_retrieve, retrieval_routes)
        graph.add_edge("next_subquestion", "retrieve")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("backtrack", "retrieve")
        graph.add_conditional_edges("web_fallback", self._after_retrieve, retrieval_routes)
        graph.add_edge("direct", "finalize")
        graph.add_edge("synthesize", "finalize")
        graph.add_edge("finalize", END)
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return graph.compile(checkpointer=MemorySaver())
        except ImportError:  # pragma: no cover - optional checkpoint package
            return graph.compile()

    def initial_state(
        self,
        *,
        messages: list[BaseMessage],
        input_text: str,
        user_id: str,
        thread_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgenticRagState:
        return {
            "messages": messages,
            "input": input_text,
            "user_id": user_id,
            "thread_id": thread_id,
            "route": "uploaded_kb",
            "route_confidence": 1.0,
            "plan": [],
            "active_subquestion": 0,
            "query": input_text,
            "mode": "hybrid",
            "query_attempts": [],
            "evidence_pool": [],
            "retrieval_runs": [],
            "answer": "",
            "draft_answer": "",
            "citations": [],
            "decision_stack": [],
            "step_count": 0,
            "rewrite_count": 0,
            "backtrack_count": 0,
            "retrieval_count": 0,
            "last_retrieval_hit_count": 0,
            "stagnant_count": 0,
            "final_status": "running",
            "stop_reason": "",
            "fallback_used": False,
            "started_at": time.perf_counter(),
            "metadata": dict(metadata or {}),
        }

    def invoke(self, state: AgenticRagState) -> dict[str, Any]:
        config = {"configurable": {"thread_id": state.get("thread_id") or f"rag-{uuid4().hex}"}}
        return dict(self.graph.invoke(state, config))

    def stream(self, state: AgenticRagState):
        config = {"configurable": {"thread_id": state.get("thread_id") or f"rag-{uuid4().hex}"}}
        return self.graph.stream(state, config, stream_mode=["messages", "values", "updates"])

    def _emit(self, event_type: str, message: str, **metadata: Any) -> None:
        if self.reporter is not None:
            self.reporter(AgentRunEvent(type=event_type, message=message, metadata=metadata))

    def _step(self, state: AgenticRagState) -> int:
        return int(state.get("step_count") or 0) + 1

    def _control_budget_reason(self, state: AgenticRagState) -> str:
        if int(state.get("step_count") or 0) >= self.settings.agentic_rag_max_steps:
            return "step_budget_exhausted"
        started_at = float(state.get("started_at") or 0.0)
        if started_at and time.perf_counter() - started_at >= self.settings.agentic_rag_wall_clock_seconds:
            return "wall_clock_budget_exhausted"
        return ""

    def _route(self, state: AgenticRagState) -> dict[str, Any]:
        question = str(state.get("input") or "")
        lowered = question.lower()
        direct = ("直接回答", "无需检索", "不用检索", "不查资料", "based on your knowledge")
        sql = ("sql", "数据库", "订单", "销售额", "统计报表", "聚合查询")
        web = ("latest", "today", "最新", "今天", "实时", "新闻")
        if any(token in lowered for token in direct):
            route = "direct"
            confidence = 0.90
        elif self.sql_retriever is not None and any(token in lowered for token in sql):
            route = "sql"
            confidence = 0.82
        elif any(token in lowered for token in web):
            route = "web"
            confidence = 0.82
        else:
            route = "uploaded_kb"
            confidence = 0.70
        self._emit("rag_route", f"route={route}", route=route)
        return {"route": route, "route_confidence": confidence, "mode": "sql" if route == "sql" else state.get("mode") or "hybrid", "step_count": self._step(state)}

    def _plan(self, state: AgenticRagState) -> dict[str, Any]:
        question = str(state.get("input") or "").strip()
        pieces = [item.strip(" ，,。；;") for item in re.split(r"(?:然后|并且|以及|同时|并|且|\?|？)", question) if item.strip()]
        if len(pieces) <= 1:
            pieces = [question]
        pieces = pieces[: self.settings.agentic_rag_max_subquestions]
        plan = [{"id": index, "question": item, "status": "pending" if index else "active", "depends_on": []} for index, item in enumerate(pieces)]
        self._emit("rag_plan", f"planned {len(plan)} subquestion(s)", subquestions=len(plan))
        step_count = self._step(state)
        update: dict[str, Any] = {
            "plan": plan,
            "active_subquestion": 0,
            "query": pieces[0],
            "decision_stack": [{"node": "plan", "plan": plan}],
            "step_count": step_count,
        }
        budget_reason = self._control_budget_reason({**state, **update})
        if budget_reason:
            update["stop_reason"] = budget_reason
        return update

    def _after_plan(self, state: AgenticRagState) -> str:
        if state.get("stop_reason") in {"step_budget_exhausted", "wall_clock_budget_exhausted"}:
            return "finalize"
        if state.get("route") == "direct":
            return "direct"
        return "web_fallback" if state.get("route") == "web" else "retrieve"

    def _retrieve(self, state: AgenticRagState) -> dict[str, Any]:
        budget_reason = self._control_budget_reason(state)
        if budget_reason:
            return {"stop_reason": budget_reason, "final_status": "abstain", "step_count": self._step(state)}
        if int(state.get("retrieval_count") or 0) >= self.settings.agentic_rag_max_retrieval_calls:
            return {
                "stop_reason": "retrieval_budget_exhausted",
                "final_status": "abstain",
                "step_count": self._step(state),
            }
        query = str(state.get("query") or state.get("input") or "").strip()
        mode = str(state.get("mode") or "hybrid")
        canonical = f"{mode}:" + " ".join(query.lower().split())
        previous = {
            f"{item.get('mode') or 'hybrid'}:" + " ".join(str(item.get("query") or "").lower().split())
            for item in state.get("query_attempts") or []
        }
        if canonical in previous:
            return {
                "last_retrieval_hit_count": 0,
                "stagnant_count": int(state.get("stagnant_count") or 0) + 1,
                "step_count": self._step(state),
            }
        run_id = f"rr-{uuid4().hex[:12]}"
        started = time.perf_counter()
        backend = self.sql_retriever if state.get("route") == "sql" and self.sql_retriever is not None else self.retriever
        result = backend.search(RetrievalRequest(str(state.get("user_id") or ""), query, mode, self.settings.rag_chunk_top_k, run_id))
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        existing = {str(item.get("evidence_id")) for item in state.get("evidence_pool") or []}
        evidence = [item.to_dict() for item in result.hits if item.evidence_id not in existing]
        attempts = [*(state.get("query_attempts") or []), {"query": query, "mode": mode, "run_id": run_id, "new_hits": len(evidence)}]
        runs = [*(state.get("retrieval_runs") or []), {"run_id": run_id, "query": query, "mode": mode, "latency_ms": latency_ms, "hit_count": len(evidence), "success": result.success}]
        self._emit("rag_retrieval", f"retrieved {len(evidence)} new evidence item(s)", query=query, mode=mode, run_id=run_id, hit_count=len(evidence))
        return {"evidence_pool": [*(state.get("evidence_pool") or []), *evidence], "query_attempts": attempts, "retrieval_runs": runs, "retrieval_count": int(state.get("retrieval_count") or 0) + 1, "last_retrieval_hit_count": len(evidence), "stagnant_count": 0 if evidence else int(state.get("stagnant_count") or 0) + 1, "step_count": self._step(state)}

    def _after_retrieve(self, state: AgenticRagState) -> str:
        if state.get("stop_reason") in {
            "retrieval_budget_exhausted",
            "step_budget_exhausted",
            "wall_clock_budget_exhausted",
            "web_disabled",
        }:
            return "synthesize" if state.get("evidence_pool") else "finalize"
        if int(state.get("last_retrieval_hit_count") or 0) > 0:
            if int(state.get("active_subquestion") or 0) + 1 < len(state.get("plan") or []):
                return "next_subquestion"
            return "synthesize"
        if (
            state.get("route") != "sql"
            and
            int(state.get("stagnant_count") or 0) >= self.settings.rag_stagnant_search_limit
            and int(state.get("backtrack_count") or 0) < self.settings.agentic_rag_max_backtracks
        ):
            return "backtrack"
        if int(state.get("rewrite_count") or 0) < self.settings.agentic_rag_max_rewrites and int(state.get("step_count") or 0) < self.settings.agentic_rag_max_steps:
            return "rewrite"
        if self.settings.agentic_rag_web_fallback_enabled and not state.get("fallback_used"):
            return "web_fallback"
        if state.get("evidence_pool"):
            return "synthesize"
        return "finalize"

    def _next_subquestion(self, state: AgenticRagState) -> dict[str, Any]:
        plan = [dict(item) for item in state.get("plan") or []]
        current = int(state.get("active_subquestion") or 0)
        if current < len(plan):
            plan[current]["status"] = "completed"
        next_index = min(current + 1, max(len(plan) - 1, 0))
        if plan:
            plan[next_index]["status"] = "active"
            query = str(plan[next_index].get("question") or state.get("input") or "")
        else:
            query = str(state.get("input") or "")
        self._emit("rag_plan", f"executing subquestion {next_index + 1}/{max(len(plan), 1)}", active_subquestion=next_index)
        return {
            "plan": plan,
            "active_subquestion": next_index,
            "query": query,
            "mode": "sql" if state.get("route") == "sql" else "hybrid",
            "rewrite_count": 0,
            "stagnant_count": 0,
            "step_count": self._step(state),
        }

    def _rewrite(self, state: AgenticRagState) -> dict[str, Any]:
        attempt = int(state.get("rewrite_count") or 0) + 1
        plan = state.get("plan") or []
        active = int(state.get("active_subquestion") or 0)
        base_question = str(
            plan[active].get("question")
            if active < len(plan)
            else state.get("query") or state.get("input") or ""
        )
        query, strategy = self.rewriter.rewrite(base_question, attempt)
        self._emit("rag_rewrite", query, strategy=strategy, attempt=attempt)
        return {
            "query": query,
            "mode": "sql" if state.get("route") == "sql" else ("bm25" if attempt % 2 == 0 else "hybrid"),
            "rewrite_count": attempt,
            "decision_stack": [
                *(state.get("decision_stack") or []),
                {"node": "rewrite", "strategy": strategy, "query": query},
            ],
            "step_count": self._step(state),
        }

    def _backtrack(self, state: AgenticRagState) -> dict[str, Any]:
        count = int(state.get("backtrack_count") or 0) + 1
        modes = ("dense", "bm25", "hybrid")
        current_mode = str(state.get("mode") or "hybrid")
        try:
            next_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
        except ValueError:
            next_mode = "dense"
        plan = state.get("plan") or []
        active = int(state.get("active_subquestion") or 0)
        query = str(plan[active].get("question") if active < len(plan) else state.get("input") or "")
        self._emit("rag_backtrack", f"backtracked to alternate retrieval mode {next_mode}", mode=next_mode, count=count)
        return {
            "query": query,
            "mode": next_mode,
            "backtrack_count": count,
            "stagnant_count": 0,
            "decision_stack": [*(state.get("decision_stack") or []), {"node": "backtrack", "mode": next_mode, "query": query}],
            "step_count": self._step(state),
        }

    def _web_fallback(self, state: AgenticRagState) -> dict[str, Any]:
        # Web is intentionally an optional backend. It is loaded lazily so
        # offline tests and deployments with networking disabled stay local.
        budget_reason = self._control_budget_reason(state)
        if budget_reason:
            return {"fallback_used": True, "stop_reason": budget_reason, "step_count": self._step(state)}
        if not self.settings.tool_network_enabled:
            return {"fallback_used": True, "stop_reason": "web_disabled", "step_count": self._step(state)}
        try:
            from superassist.tools.web import web_search

            raw = web_search.invoke({"query": str(state.get("input") or ""), "max_results": 5})
            rows = json.loads(raw) if isinstance(raw, str) and raw.lstrip().startswith("[") else []
        except Exception as exc:  # noqa: BLE001 - fallback must not fail the turn
            logger.info("Agentic RAG web fallback failed: %s", exc)
            rows = []
        evidence = [
            {
                "evidence_id": evidence_id_for(
                    "web",
                    str(row.get("url") or row.get("title") or ""),
                    str(row.get("snippet") or ""),
                ),
                "source_type": "web",
                "text": f"{row.get('title', '')}: {row.get('snippet', '')}",
                "source": str(row.get("url") or ""),
                "locator": str(row.get("url") or ""),
                "chunk_id": "",
                "document_id": "",
                "document_name": str(row.get("title") or ""),
                "score": 0.5,
                "token_count": max(1, len(str(row.get("snippet") or "").split())),
                "metadata": {"title": row.get("title", "")},
            }
            for row in rows
            if isinstance(row, dict)
        ]
        self._emit("rag_fallback", f"web fallback returned {len(evidence)} item(s)", hit_count=len(evidence))
        return {"evidence_pool": [*(state.get("evidence_pool") or []), *evidence], "fallback_used": True, "query": str(state.get("input") or ""), "last_retrieval_hit_count": len(evidence), "step_count": self._step(state)}

    def _direct(self, state: AgenticRagState) -> dict[str, Any]:
        """Answer from model prior only when the router explicitly selected it."""
        try:
            response = self.model.invoke(list(state.get("messages") or []))
            answer = str(getattr(response, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001 - direct mode still must finalize safely
            logger.info("Agentic RAG direct answer failed: %s", exc)
            answer = ""
        if not answer:
            answer = "当前无法生成直接回答。"
        self._emit("rag_direct", "answered from model prior")
        return {
            "draft_answer": answer,
            "citations": [],
            "step_count": self._step(state),
        }

    def _synthesize(self, state: AgenticRagState) -> dict[str, Any]:
        evidence = [Evidence(**item) for item in state.get("evidence_pool") or []]
        context = evidence_context(evidence)
        prompt = SystemMessage(content=(
            "Answer only from the evidence below. Every factual claim must cite one or more evidence IDs "
            "exactly in the form [ev-...]. If evidence is insufficient, say so explicitly.\n\n"
            f"Evidence:\n{context}"
        ))
        try:
            response = self.model.invoke([*list(state.get("messages") or []), prompt])
            answer = str(getattr(response, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001 - generation failure is a controlled abstention
            logger.info("Agentic RAG synthesis failed: %s", exc)
            answer = ""
        citations = parse_answer_citations(answer)
        if not answer or not citations:
            if evidence:
                answer = "根据检索到的资料：\n" + "\n".join(f"- {item.text[:500]} [{item.evidence_id}]" for item in evidence[:3])
                citations = [item.evidence_id for item in evidence[:3]]
            else:
                answer = "当前检索未获得足够证据，无法可靠回答该问题。"
                citations = []
        return {"draft_answer": answer, "citations": citations, "step_count": self._step(state)}

    def _finalize(self, state: AgenticRagState) -> dict[str, Any]:
        answer = str(state.get("draft_answer") or "")
        if not answer:
            answer = "当前检索未获得足够证据，无法可靠回答该问题。"
        answer = _render_citations(answer, state.get("evidence_pool") or [])
        status = "direct" if state.get("route") == "direct" else ("completed" if state.get("evidence_pool") else "abstain")
        metadata = dict(state.get("metadata") or {})
        metadata["rag_trace"] = {
            "route": state.get("route"),
            "plan": state.get("plan") or [],
            "query_attempts": state.get("query_attempts") or [],
            "retrieval_runs": state.get("retrieval_runs") or [],
            "fallback_used": bool(state.get("fallback_used")),
            "rewrite_count": int(state.get("rewrite_count") or 0),
            "backtrack_count": int(state.get("backtrack_count") or 0),
            "step_count": int(state.get("step_count") or 0),
            "elapsed_ms": round(
                max(0.0, time.perf_counter() - float(state.get("started_at") or time.perf_counter())) * 1000,
                2,
            ),
            "stop_reason": state.get("stop_reason") or ("completed" if status == "completed" else "no_results"),
        }
        metadata["answer_provenance"] = {
            "evidence_ids": list(state.get("citations") or []),
            "sources": sorted({str(item.get("source") or "") for item in state.get("evidence_pool") or [] if item.get("source")}),
        }
        messages = list(state.get("messages") or [])
        if answer:
            messages.append(AIMessage(content=answer))
        metadata["final_assistant_text"] = answer
        metadata["memory_ready"] = bool(answer)
        return {"answer": answer, "messages": messages, "final_status": status, "metadata": metadata, "step_count": self._step(state)}


def _render_citations(answer: str, evidence_pool: list[dict[str, Any]]) -> str:
    rendered = answer
    for item in evidence_pool:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id:
            continue
        if item.get("source_type") == "web":
            label = str(item.get("source") or evidence_id)
        else:
            label = str(item.get("document_name") or item.get("source") or "uploaded-document")
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id:
                label += f"#chunk={chunk_id}"
        rendered = rendered.replace(f"[{evidence_id}]", f"[{label}]")
    return rendered


__all__ = ["AgenticRagGraph", "AgenticRagState"]
