from __future__ import annotations

import math
from statistics import mean
from typing import Any

import numpy as np


def evaluate_rankings(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Evaluate ordered document IDs against BEIR qrels."""

    metrics: dict[str, list[float]] = {}
    for query_id, relevance in qrels.items():
        relevant = {doc_id: int(score) for doc_id, score in relevance.items() if int(score) > 0}
        ranked = list(dict.fromkeys(rankings.get(query_id, [])))
        for k in k_values:
            prefix = ranked[:k]
            hits = [doc_id for doc_id in prefix if doc_id in relevant]
            metrics.setdefault(f"hit@{k}", []).append(float(bool(hits)))
            metrics.setdefault(f"recall@{k}", []).append(len(hits) / len(relevant) if relevant else 0.0)
            metrics.setdefault(f"precision@{k}", []).append(len(hits) / k)
            metrics.setdefault(f"mrr@{k}", []).append(
                next((1.0 / rank for rank, doc_id in enumerate(prefix, start=1) if doc_id in relevant), 0.0)
            )
            metrics.setdefault(f"map@{k}", []).append(_average_precision(prefix, relevant))
            metrics.setdefault(f"ndcg@{k}", []).append(_ndcg(prefix, relevant, k))
    return {name: mean(values) if values else 0.0 for name, values in metrics.items()}


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": mean(values) if values else 0.0,
        "p50_seconds": float(np.percentile(values, 50)) if values else 0.0,
        "p95_seconds": float(np.percentile(values, 95)) if values else 0.0,
        "p99_seconds": float(np.percentile(values, 99)) if values else 0.0,
    }


def _average_precision(ranked: list[str], relevant: dict[str, int]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), len(ranked)) if ranked else 0.0


def _ndcg(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(
        (2 ** relevant.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def markdown_report(summary: dict[str, Any]) -> str:
    modes = list(summary["modes"])
    lines = [
        f"# BEIR {summary['dataset']} Retrieval Ablation",
        "",
        "## Setup",
        "",
        f"- Model: `{summary['model']}`",
        f"- Split: `{summary['split']}`",
        f"- Corpus documents: {summary['corpus_documents']}",
        f"- Evaluated queries: {summary['queries']}",
        f"- Candidate Top-K: {summary['candidate_top_k']}",
        f"- Returned Top-K: {summary['top_k']}",
        "- Retrieval implementation: SuperAssist `HybridRAGService` (SQLite FTS5 + FAISS + RRF)",
        "",
        "## Results",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "| --- | " + " | ".join("---:" for _ in modes) + " |",
    ]
    if summary.get("sample_seed") is not None:
        lines.insert(7, f"- Random sample seed: {summary['sample_seed']}")
    if summary.get("reranker_model"):
        lines.insert(10, f"- Reranker: `{summary['reranker_model']}` over Top-{summary['reranker_candidate_top_k']} candidates")
        reranked = summary.get("modes", {}).get("hybrid_rerank", {}).get("reranked_queries")
        if reranked is not None:
            lines.insert(11, f"- Successfully reranked queries: {reranked}/{summary['queries']}")
    selected = ("ndcg@10", "map@10", "mrr@10", "hit@10", "recall@10", "recall@20", "precision@10")
    for metric in selected:
        values = [float(summary["modes"][mode]["metrics"].get(metric, 0.0)) for mode in modes]
        lines.append(f"| {metric} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    for metric in ("p50_seconds", "p95_seconds", "p99_seconds"):
        values = [float(summary["modes"][mode]["latency"].get(metric, 0.0)) for mode in modes]
        lines.append(f"| latency {metric} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "The benchmark evaluates retrieval only. It does not measure answer generation, citation faithfulness,",
            "or Agentic query-rewrite quality. BEIR SciFact is English; use a separately labelled Chinese dataset",
            "before selecting the production model for Chinese political-theory documents.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["evaluate_rankings", "latency_summary", "markdown_report"]
