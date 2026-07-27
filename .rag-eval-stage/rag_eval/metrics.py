from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np


def retrieval_scores(gold: list[str], retrieved: list[str]) -> dict[str, float | bool]:
    gold_set = set(gold)
    retrieved_set = set(retrieved)
    hits = len(gold_set & retrieved_set)
    return {
        "hit": hits > 0,
        "all_evidence_hit": bool(gold_set) and gold_set <= retrieved_set,
        "recall": hits / len(gold_set) if gold_set else 0.0,
        "precision": hits / len(retrieved_set) if retrieved_set else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else 0.0


def aggregate_system(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    records = [row[system] for row in rows if system in row]
    if not records:
        return {}
    retrieval = [item["retrieval_metrics"] for item in records]
    retrieval_times = [float(item["retrieval_seconds"]) for item in records]
    end_to_end = [float(item["end_to_end_seconds"]) for item in records]
    usage = [item["system_usage"] for item in records]
    return {
        "questions": len(records),
        "answer_accuracy": mean(bool(item.get("correct")) for item in records),
        "average_answer_score": mean(float(item.get("answer_score") or 0) for item in records),
        "retrieval_hit_rate": mean(bool(item["hit"]) for item in retrieval),
        "all_evidence_hit_rate": mean(bool(item["all_evidence_hit"]) for item in retrieval),
        "mean_retrieval_recall": mean(float(item["recall"]) for item in retrieval),
        "mean_retrieval_precision": mean(float(item["precision"]) for item in retrieval),
        "retrieval_latency_p50_seconds": percentile(retrieval_times, 50),
        "retrieval_latency_p95_seconds": percentile(retrieval_times, 95),
        "end_to_end_latency_p50_seconds": percentile(end_to_end, 50),
        "end_to_end_latency_p95_seconds": percentile(end_to_end, 95),
        "average_input_tokens": mean(int(item.get("input_tokens") or 0) for item in usage),
        "average_output_tokens": mean(int(item.get("output_tokens") or 0) for item in usage),
        "average_total_tokens": mean(int(item.get("total_tokens") or 0) for item in usage),
        "token_usage_coverage": sum(int(item.get("measured_calls") or 0) for item in usage)
        / max(1, sum(int(item.get("calls") or 0) for item in usage)),
    }

