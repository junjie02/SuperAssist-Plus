from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from superassist.config import Settings
from superassist.rag.agentic_contracts import Evidence, RetrievalResult
from superassist.rag.agentic_graph import AgenticRagGraph


@dataclass
class SequencedRetriever:
    batches: list[list[Evidence]]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def search(self, request):
        self.calls.append((request.query, request.mode))
        index = min(len(self.calls) - 1, len(self.batches) - 1)
        hits = self.batches[index] if self.batches else []
        return RetrievalResult(request.run_id, request.query, request.mode, hits, bool(hits), "ok" if hits else "empty")


class EchoModel:
    def invoke(self, messages):
        return type("Response", (), {"content": "模型草稿没有引用"})()


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "SUPERASSIST_EMBEDDING_PROVIDER": "hash",
        "SUPERASSIST_AGENTIC_RAG_WEB_FALLBACK_ENABLED": False,
        "SUPERASSIST_AGENTIC_RAG_MAX_REWRITES": 2,
        "SUPERASSIST_AGENTIC_RAG_MAX_BACKTRACKS": 1,
    }
    values.update(overrides)
    return Settings(**values)


def evidence(text: str, suffix: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{suffix:0>16}",
        source_type="kb",
        text=text,
        source="guide.md",
        chunk_id=f"chunk-{suffix}",
        document_name="guide.md",
        token_count=12,
    )


def test_empty_result_rewrites_then_accepts_next_retrieval() -> None:
    retriever = SequencedRetriever([[], [evidence("蓝鲸协议用于服务发现", "a1b2c3d4e5f60718")]])
    graph = AgenticRagGraph(model=EchoModel(), retriever=retriever, settings=settings())

    result = graph.invoke(
        graph.initial_state(
            messages=[HumanMessage(content="蓝鲸协议是什么")],
            input_text="蓝鲸协议是什么",
            user_id="user-a",
            thread_id="agentic-test-1",
        )
    )

    assert len(retriever.calls) == 2
    assert result["answer"]
    assert result["final_status"] == "completed"
    assert result["metadata"]["rag_trace"]["rewrite_count"] == 1
    assert "retrieval_grade" not in result["metadata"]["rag_trace"]
    assert "verification" not in result["metadata"]["rag_trace"]
    assert result["metadata"]["answer_provenance"]["evidence_ids"]


def test_dead_end_switches_retrieval_mode_and_records_backtrack() -> None:
    retriever = SequencedRetriever([[], [evidence("系统支持关键词检索", "1122334455667788")]])
    graph = AgenticRagGraph(
        model=EchoModel(),
        retriever=retriever,
        settings=settings(SUPERASSIST_AGENTIC_RAG_MAX_REWRITES=0, SUPERASSIST_RAG_STAGNANT_SEARCH_LIMIT=1),
    )

    result = graph.invoke(
        graph.initial_state(
            messages=[],
            input_text="关键词检索如何工作",
            user_id="user-a",
            thread_id="agentic-test-2",
        )
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[1][1] != retriever.calls[0][1]
    assert result["metadata"]["rag_trace"]["backtrack_count"] == 1


def test_insufficient_evidence_abstains_without_citations() -> None:
    retriever = SequencedRetriever([[], [], []])
    graph = AgenticRagGraph(model=EchoModel(), retriever=retriever, settings=settings(SUPERASSIST_AGENTIC_RAG_MAX_REWRITES=1))

    result = graph.invoke(
        graph.initial_state(
            messages=[],
            input_text="不存在的资料问题",
            user_id="user-a",
            thread_id="agentic-test-3",
        )
    )

    assert result["final_status"] == "abstain"
    assert "证据" in result["answer"]
    assert result["metadata"]["answer_provenance"]["evidence_ids"] == []


def test_step_budget_stops_retrieval_loop() -> None:
    retriever = SequencedRetriever([[], [], []])
    graph = AgenticRagGraph(
        model=EchoModel(),
        retriever=retriever,
        settings=settings(
            SUPERASSIST_AGENTIC_RAG_MAX_STEPS=4,
            SUPERASSIST_AGENTIC_RAG_MAX_REWRITES=3,
        ),
    )

    result = graph.invoke(
        graph.initial_state(
            messages=[],
            input_text="没有命中的问题",
            user_id="user-a",
            thread_id="agentic-test-budget",
        )
    )

    assert len(retriever.calls) == 1
    assert result["metadata"]["rag_trace"]["stop_reason"] == "step_budget_exhausted"


def test_rewrite_uses_active_subquestion() -> None:
    graph = AgenticRagGraph(model=EchoModel(), retriever=SequencedRetriever([]), settings=settings())
    seen_questions: list[str] = []

    class CapturingRewriter:
        def rewrite(self, question: str, attempt: int) -> tuple[str, str]:
            seen_questions.append(question)
            return question, f"attempt-{attempt}"

    graph.rewriter = CapturingRewriter()
    state = graph.initial_state(
        messages=[],
        input_text="总问题",
        user_id="user-a",
        thread_id="agentic-test-subquestion",
    )
    state.update(
        {
            "plan": [
                {"id": 0, "question": "第一个问题", "status": "completed", "depends_on": []},
                {"id": 1, "question": "第二个问题", "status": "active", "depends_on": []},
            ],
            "active_subquestion": 1,
        }
    )

    update = graph._rewrite(state)

    assert update["query"] == "第二个问题"
    assert seen_questions == ["第二个问题"]


def test_wall_clock_budget_stops_before_backend_call() -> None:
    retriever = SequencedRetriever([evidence("命中", "deadbeefdeadbeef")])
    graph = AgenticRagGraph(
        model=EchoModel(),
        retriever=retriever,
        settings=settings(SUPERASSIST_AGENTIC_RAG_WALL_CLOCK_SECONDS=1),
    )
    state = graph.initial_state(
        messages=[],
        input_text="超时问题",
        user_id="user-a",
        thread_id="agentic-test-wall-clock",
    )
    state["started_at"] = time.perf_counter() - 2

    result = graph.invoke(state)

    assert retriever.calls == []
    assert result["metadata"]["rag_trace"]["stop_reason"] == "wall_clock_budget_exhausted"


def test_router_can_select_sql_backend_when_configured() -> None:
    primary = SequencedRetriever([[]])
    sql = SequencedRetriever([[evidence("订单总额为 10", "0011223344556677")]])
    graph = AgenticRagGraph(
        model=EchoModel(),
        retriever=primary,
        sql_retriever=sql,
        settings=settings(SUPERASSIST_AGENTIC_RAG_MAX_REWRITES=0),
    )

    result = graph.invoke(
        graph.initial_state(
            messages=[],
            input_text="查询数据库订单总额",
            user_id="user-a",
            thread_id="agentic-test-sql-route",
        )
    )

    assert primary.calls == []
    assert sql.calls[0][1] == "sql"
    assert result["metadata"]["rag_trace"]["route"] == "sql"


def test_router_direct_mode_skips_retrieval() -> None:
    retriever = SequencedRetriever([[]])
    graph = AgenticRagGraph(model=EchoModel(), retriever=retriever, settings=settings())

    result = graph.invoke(
        graph.initial_state(
            messages=[],
            input_text="请直接回答这个常识问题",
            user_id="user-a",
            thread_id="agentic-test-direct-route",
        )
    )

    assert retriever.calls == []
    assert result["final_status"] == "direct"
