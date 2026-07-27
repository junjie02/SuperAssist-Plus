from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.common import (
    QUESTION_PATH,
    RESULTS_PATH,
    SUMMARY_JSON_PATH,
    SUMMARY_MD_PATH,
    TokenUsage,
    atomic_write_json,
    document_names,
    get_settings,
    load_chunks,
    message_text,
    parse_json_response,
)
from rag_eval.lightrag_rag import ExistingLightRAG
from rag_eval.metrics import aggregate_system, retrieval_scores
from rag_eval.vector_rag import Retrieval, VectorRAG


ANSWER_SYSTEM_PROMPT = """你是一个封闭知识库问答系统。你没有联网搜索功能，也不得使用训练数据、常识或任何检索上下文之外的信息。
只能根据用户消息中的 RETRIEVED_CONTEXT 回答。上下文证据不足时，必须回答“根据检索内容无法确定”，禁止猜测、补全或编造。
答案应简洁、直接，并在关键结论后标注提供的 SOURCE/CHUNK。"""

JUDGE_SYSTEM_PROMPT = """你是严格的 RAG 评测裁判。只根据标准答案和论文证据判断候选答案，不得联网搜索或引入外部知识。
分别独立评分，不因表达风格、长度或候选顺序偏袒任何系统。"""


def _sum_usage(*items: dict[str, int]) -> dict[str, int]:
    keys = {key for item in items for key in item}
    return {key: sum(int(item.get(key, 0)) for item in items) for key in keys}


async def _answer(model: Any, question: str, retrieval: Retrieval) -> tuple[str, dict[str, int], float]:
    usage = TokenUsage()
    started = time.perf_counter()
    response = await model.ainvoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED_CONTEXT:\n{retrieval.context or '[EMPTY]'}"
                )
            ),
        ]
    )
    usage.add_message(response)
    return message_text(response).strip(), usage.snapshot(), time.perf_counter() - started


async def _judge(
    model: Any,
    question: dict[str, Any],
    vector_answer: str,
    light_answer: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    # Alternate ordering to reduce position bias while scoring both candidates in one call.
    swapped = int(str(question["id"])[1:]) % 2 == 0
    candidates = (
        {"A": light_answer, "B": vector_answer}
        if swapped
        else {"A": vector_answer, "B": light_answer}
    )
    evidence = "\n".join(
        f"- [{item['document']} | {item['chunk_id']}] {item['quote']}"
        for item in question["evidence"]
    )
    prompt = f"""问题：{question['question']}
标准答案：{question['reference_answer']}
标准论文证据：
{evidence}

候选 A：{candidates['A']}
候选 B：{candidates['B']}

请只输出以下 JSON：
{{
  "A": {{"score": 0, "correct": false, "reason": "..."}},
  "B": {{"score": 0, "correct": false, "reason": "..."}}
}}

评分：2=正确且关键点完整；1=部分正确但缺少关键点；0=错误、矛盾、编造或证据不足。
只有 score=2 时 correct 才能为 true。"""
    usage = TokenUsage()
    last_error: Exception | None = None
    for _attempt in range(2):
        response = await model.ainvoke(
            [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        usage.add_message(response)
        try:
            raw = parse_json_response(message_text(response))
            a = raw["A"]
            b = raw["B"]
            mapped = (
                {"lightrag": a, "vector": b}
                if swapped
                else {"vector": a, "lightrag": b}
            )
            for value in mapped.values():
                value["score"] = int(value.get("score") or 0)
                value["correct"] = value["score"] == 2
            return mapped, usage.snapshot()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Judge returned invalid JSON twice: {last_error}")


def _result_record(
    retrieval: Retrieval,
    answer: str,
    answer_usage: dict[str, int],
    answer_seconds: float,
    gold_chunk_ids: list[str],
) -> dict[str, Any]:
    return {
        "answer": answer,
        "retrieved_chunk_ids": retrieval.chunk_ids,
        "retrieval_metrics": retrieval_scores(gold_chunk_ids, retrieval.chunk_ids),
        "retrieval_seconds": retrieval.elapsed_seconds,
        "answer_seconds": answer_seconds,
        "end_to_end_seconds": retrieval.elapsed_seconds + answer_seconds,
        "retrieval_usage": retrieval.retrieval_usage,
        "answer_usage": answer_usage,
        "system_usage": _sum_usage(retrieval.retrieval_usage, answer_usage),
    }


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows, {str(item["id"]) for item in rows}


def _append_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _markdown(summary: dict[str, Any]) -> str:
    vector = summary["systems"]["vector"]
    light = summary["systems"]["lightrag"]
    return f"""# Vector RAG vs LightRAG Evaluation

## Setup

- Papers: {', '.join(summary['documents'])}
- Questions: {summary['questions']}
- Chunking: 1200 tokens, 100-token overlap (reusing exact LightRAG chunks)
- Vector RAG: BGE cosine retrieval, top-5 chunks
- LightRAG: mix mode, entity/relation top-20, chunk top-5, 6000-token retrieval budget
- Answer policy: retrieved context only; web search and model-knowledge fallback disabled
- Model: {summary['model']}

## Results

| Metric | Vector RAG | LightRAG |
| --- | ---: | ---: |
| Answer accuracy | {vector['answer_accuracy']:.2%} | {light['answer_accuracy']:.2%} |
| Average answer score (0-2) | {vector['average_answer_score']:.3f} | {light['average_answer_score']:.3f} |
| Retrieval hit rate | {vector['retrieval_hit_rate']:.2%} | {light['retrieval_hit_rate']:.2%} |
| All-evidence hit rate | {vector['all_evidence_hit_rate']:.2%} | {light['all_evidence_hit_rate']:.2%} |
| Mean evidence recall | {vector['mean_retrieval_recall']:.2%} | {light['mean_retrieval_recall']:.2%} |
| Mean evidence precision | {vector['mean_retrieval_precision']:.2%} | {light['mean_retrieval_precision']:.2%} |
| Retrieval latency P50 | {vector['retrieval_latency_p50_seconds']:.3f}s | {light['retrieval_latency_p50_seconds']:.3f}s |
| Retrieval latency P95 | {vector['retrieval_latency_p95_seconds']:.3f}s | {light['retrieval_latency_p95_seconds']:.3f}s |
| End-to-end latency P50 | {vector['end_to_end_latency_p50_seconds']:.3f}s | {light['end_to_end_latency_p50_seconds']:.3f}s |
| End-to-end latency P95 | {vector['end_to_end_latency_p95_seconds']:.3f}s | {light['end_to_end_latency_p95_seconds']:.3f}s |
| Avg. input tokens / question | {vector['average_input_tokens']:.1f} | {light['average_input_tokens']:.1f} |
| Avg. output tokens / question | {vector['average_output_tokens']:.1f} | {light['average_output_tokens']:.1f} |
| Avg. total tokens / question | {vector['average_total_tokens']:.1f} | {light['average_total_tokens']:.1f} |
| Token usage coverage | {vector['token_usage_coverage']:.2%} | {light['token_usage_coverage']:.2%} |

Judge token usage is reported separately and is not included in either system's cost.
"""


async def evaluate(limit: int | None = None, results_path: Path = RESULTS_PATH) -> None:
    from superassist.llm import create_chat_model

    if not QUESTION_PATH.exists():
        raise FileNotFoundError(f"Generate questions first: {QUESTION_PATH}")
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]

    chunks = load_chunks()
    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError("SUPERASSIST_API_KEY is required to run answer generation and judging")

    vector = VectorRAG(chunks, top_k=5)
    vector_index = vector.build()
    light = ExistingLightRAG(chunks, top_k=20, chunk_top_k=5)
    await light.initialize()
    answer_model = create_chat_model(settings).bind(max_tokens=1024)
    judge_model = create_chat_model(settings).bind(max_tokens=1200)
    rows, completed = _load_completed(results_path)
    judge_usage = TokenUsage()

    try:
        for index, question in enumerate(questions, start=1):
            if str(question["id"]) in completed:
                continue
            gold = sorted({str(item["chunk_id"]) for item in question["evidence"]})

            vector_retrieval = vector.retrieve(question["question"])
            light_retrieval = await light.retrieve(question["question"])
            vector_answer, vector_usage, vector_seconds = await _answer(
                answer_model, question["question"], vector_retrieval
            )
            light_answer, light_usage, light_seconds = await _answer(
                answer_model, question["question"], light_retrieval
            )
            judgement, one_judge_usage = await _judge(
                judge_model, question, vector_answer, light_answer
            )
            judge_usage.add_usage(one_judge_usage)

            row = {
                "id": question["id"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "scope": question.get("scope"),
                "documents": question.get("documents"),
                "gold_chunk_ids": gold,
                "vector": _result_record(
                    vector_retrieval, vector_answer, vector_usage, vector_seconds, gold
                ),
                "lightrag": _result_record(
                    light_retrieval, light_answer, light_usage, light_seconds, gold
                ),
                "judge_usage": one_judge_usage,
            }
            for system in ("vector", "lightrag"):
                row[system]["answer_score"] = judgement[system]["score"]
                row[system]["correct"] = judgement[system]["correct"]
                row[system]["judge_reason"] = judgement[system].get("reason", "")
            _append_result(results_path, row)
            rows.append(row)
            completed.add(str(question["id"]))
            print(
                f"[{index}/{len(questions)}] {question['id']} "
                f"vector={row['vector']['answer_score']} light={row['lightrag']['answer_score']}"
            )
    finally:
        await light.close()

    selected_ids = {str(item["id"]) for item in questions}
    selected_rows = [row for row in rows if str(row["id"]) in selected_ids]
    summary = {
        "questions": len(selected_rows),
        "documents": document_names(chunks),
        "chunks": len(chunks),
        "model": settings.model,
        "embedding_model": settings.embedding_model,
        "vector_index": vector_index,
        "judge_usage": judge_usage.snapshot(),
        "systems": {
            "vector": aggregate_system(selected_rows, "vector"),
            "lightrag": aggregate_system(selected_rows, "lightrag"),
        },
    }
    atomic_write_json(SUMMARY_JSON_PATH, summary)
    SUMMARY_MD_PATH.write_text(_markdown(summary), encoding="utf-8")
    print(f"Summary: {SUMMARY_MD_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vector RAG against existing LightRAG")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    asyncio.run(evaluate(args.limit, args.results))


if __name__ == "__main__":
    main()
