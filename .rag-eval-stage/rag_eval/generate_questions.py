from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.common import (
    ARTIFACTS_DIR,
    QUESTION_PATH,
    Chunk,
    TokenUsage,
    atomic_write_json,
    document_names,
    get_settings,
    load_chunks,
    message_text,
    normalize_text,
    parse_json_response,
)


SYSTEM_PROMPT = """你是 RAG 评测题库构造器。只能使用用户提供的论文片段，不得使用联网搜索、工具、训练知识或常识补充。
每个问题必须能被给定片段明确回答；标准答案必须忠于原文。不要提出需要论文外部知识才能回答的问题。"""


def _even_sample(chunks: list[Chunk], count: int, offset: int = 0) -> list[Chunk]:
    if len(chunks) <= count:
        return chunks
    step = len(chunks) / count
    return [chunks[min(len(chunks) - 1, int(offset + index * step) % len(chunks))] for index in range(count)]


def _build_batches(chunks: list[Chunk]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.file_path].append(chunk)

    batches: list[dict[str, Any]] = []
    # Up to 90 single-document questions: 30 per paper in small, stable batches.
    # Generation stops at the requested global count, so validated cross-section
    # questions from an earlier/resumed run remain part of the final 100.
    for name in sorted(by_doc, key=str.casefold):
        for batch_number, offset in enumerate(range(6), start=1):
            batches.append(
                {
                    "kind": "single_document",
                    "label": name,
                    "chunks": _even_sample(by_doc[name], 4, offset),
                    "count": 5,
                    "target_total": batch_number * 5,
                }
            )

    # 15 cross-section questions: five per paper.
    for index, name in enumerate(sorted(by_doc, key=str.casefold)):
        batches.append(
            {
                "kind": "cross_section",
                "label": name,
                "chunks": _even_sample(by_doc[name], 6, index),
                "count": 5,
                "target_total": 5,
            }
        )

    # 10 cross-document comparison questions, with evidence from at least two papers.
    names = sorted(by_doc, key=str.casefold)
    for batch_number, offset in enumerate((0, 2), start=1):
        selected: list[Chunk] = []
        for name in names:
            selected.extend(_even_sample(by_doc[name], 2, offset))
        batches.append(
            {
                "kind": "cross_document",
                "label": " + ".join(names),
                "chunks": selected,
                "count": 5,
                "target_total": batch_number * 5,
            }
        )
    return batches


def _prompt(batch: dict[str, Any], existing_questions: list[str]) -> str:
    chunks: list[Chunk] = batch["chunks"]
    context = "\n\n".join(
        f"<chunk id=\"{item.chunk_id}\" document=\"{item.file_path}\">\n{item.content}\n</chunk>"
        for item in chunks
    )
    kind_rules = {
        "single_document": "问题只针对这一篇论文，可以包含事实、方法、实验结果和局限性。",
        "cross_section": "每道题应综合同一篇论文至少两个不同 chunk 的信息。",
        "cross_document": "每道题必须比较或综合至少两篇论文，并引用至少两个不同 document 的证据 chunk。",
    }
    return f"""请生成 {batch['count']} 道中文问答题，用于比较传统向量 RAG 与 LightRAG。

题型：{batch['kind']}
范围：{batch['label']}
要求：{kind_rules[batch['kind']]}

约束：
1. 问题、答案都只能依据下方论文片段，禁止外部知识。
2. 答案应简洁、可判定，避免开放式观点题。
3. 每题提供 evidence 数组；chunk_id 必须原样取自输入，quote 必须是对应 chunk 中的连续原文。
4. 不得生成与已有题目重复或近似的问题。
5. 只输出 JSON 数组，不要 Markdown。

JSON 格式：
[
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "single_fact|cross_section|cross_document|comparison|numeric",
    "evidence": [{{"chunk_id": "...", "quote": "..."}}]
  }}
]

已有问题：
{existing_questions[-30:]}

论文片段：
{context}
"""


def _validate_items(
    raw_items: Any,
    batch: dict[str, Any],
    chunks_by_id: dict[str, Chunk],
    seen: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    allowed = {item.chunk_id for item in batch["chunks"]}
    output: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("reference_answer") or "").strip()
        evidence = item.get("evidence") or []
        if not question or not answer or question.casefold() in seen or not isinstance(evidence, list):
            continue
        cleaned_evidence = []
        evidence_documents: set[str] = set()
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            chunk_id = str(entry.get("chunk_id") or "")
            quote = str(entry.get("quote") or "").strip()
            if chunk_id not in allowed or chunk_id not in chunks_by_id or not quote:
                continue
            chunk = chunks_by_id[chunk_id]
            if normalize_text(quote) not in normalize_text(chunk.content):
                continue
            evidence_documents.add(chunk.file_path)
            cleaned_evidence.append(
                {"chunk_id": chunk_id, "document": chunk.file_path, "quote": quote}
            )
        if not cleaned_evidence:
            continue
        if batch["kind"] == "cross_document" and len(evidence_documents) < 2:
            continue
        if batch["kind"] == "cross_section" and len({x["chunk_id"] for x in cleaned_evidence}) < 2:
            continue
        output.append(
            {
                "question": question,
                "reference_answer": answer,
                "question_type": str(item.get("question_type") or batch["kind"]),
                "scope": batch["kind"],
                "documents": sorted(evidence_documents),
                "evidence": cleaned_evidence,
            }
        )
        seen.add(question.casefold())
    return output


async def generate(count: int = 100, force: bool = False) -> None:
    from superassist.llm import create_chat_model

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    names = document_names(chunks)
    if len(names) < 3:
        raise RuntimeError(f"Expected at least 3 indexed papers, found: {names}")

    questions: list[dict[str, Any]] = []
    if QUESTION_PATH.exists() and not force:
        questions = list(__import__("json").loads(QUESTION_PATH.read_text(encoding="utf-8")))
    if len(questions) >= count:
        print(f"Question bank already contains {len(questions)} questions: {QUESTION_PATH}")
        return

    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError("SUPERASSIST_API_KEY is required to generate the question bank")
    model = create_chat_model(settings).bind(max_tokens=3072)
    usage = TokenUsage()
    chunks_by_id = {item.chunk_id: item for item in chunks}
    seen = {str(item.get("question") or "").casefold() for item in questions}

    for batch_index, batch in enumerate(_build_batches(chunks), start=1):
        if len(questions) >= count:
            break
        if batch["kind"] == "cross_document":
            existing_for_category = sum(
                item.get("scope") == "cross_document" for item in questions
            )
        else:
            existing_for_category = sum(
                item.get("scope") == batch["kind"]
                and item.get("documents") == [batch["label"]]
                for item in questions
            )
        needed_for_category = max(0, int(batch["target_total"]) - existing_for_category)
        needed = min(needed_for_category, count - len(questions))
        if needed == 0:
            continue
        batch = {**batch, "count": needed}
        accepted: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            prompt = _prompt(batch, [item["question"] for item in questions + accepted])
            response = await model.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            usage.add_message(response)
            try:
                raw_items = parse_json_response(message_text(response))
            except (ValueError, __import__("json").JSONDecodeError) as exc:
                print(f"Batch {batch_index} attempt {attempt}: invalid JSON ({exc})")
                continue
            accepted.extend(_validate_items(raw_items, batch, chunks_by_id, seen))
            if len(accepted) >= needed:
                break
            batch["count"] = needed - len(accepted)

        for item in accepted[:needed]:
            item["id"] = f"q{len(questions) + 1:03d}"
            questions.append(item)
        atomic_write_json(QUESTION_PATH, questions)
        atomic_write_json(
            ARTIFACTS_DIR / "question_generation_usage.json",
            {"questions": len(questions), "documents": names, "usage": usage.snapshot()},
        )
        print(f"Batch {batch_index}: +{min(len(accepted), needed)}, total={len(questions)}")

    if len(questions) < count:
        raise RuntimeError(f"Only generated {len(questions)} valid questions; rerun to continue")
    print(f"Generated {len(questions)} questions across {names}: {QUESTION_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded three-paper RAG evaluation set")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Replace the existing question bank")
    args = parser.parse_args()
    asyncio.run(generate(args.count, args.force))


if __name__ == "__main__":
    main()
