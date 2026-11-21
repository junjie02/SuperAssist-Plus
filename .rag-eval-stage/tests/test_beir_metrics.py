from rag_eval.beir_metrics import evaluate_rankings, latency_summary, markdown_report


def test_evaluate_rankings_uses_ordered_document_ids() -> None:
    metrics = evaluate_rankings(
        {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}},
        {"q1": ["noise", "d1", "d2"], "q2": ["d3", "noise"]},
        k_values=(1, 2, 3),
    )

    assert metrics["hit@1"] == 0.5
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@2"] == 0.75
    assert 0.0 < metrics["ndcg@3"] <= 1.0


def test_latency_summary_reports_percentiles() -> None:
    summary = latency_summary([0.1, 0.2, 0.3])

    assert summary["mean_seconds"] == 0.2
    assert summary["p50_seconds"] == 0.2


def test_markdown_report_records_successful_reranking() -> None:
    report = markdown_report(
        {
            "dataset": "scifact",
            "model": "embedding-model",
            "split": "test",
            "corpus_documents": 10,
            "queries": 2,
            "sample_seed": 42,
            "candidate_top_k": 10,
            "top_k": 5,
            "reranker_model": "reranker-model",
            "reranker_candidate_top_k": 8,
            "modes": {
                "hybrid_rerank": {
                    "metrics": {},
                    "latency": {},
                    "reranked_queries": 2,
                }
            },
        }
    )

    assert "Reranker: `reranker-model` over Top-8 candidates" in report
    assert "Successfully reranked queries: 2/2" in report
    assert "Random sample seed: 42" in report
