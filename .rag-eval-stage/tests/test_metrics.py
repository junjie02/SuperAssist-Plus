from rag_eval.common import TokenUsage, parse_json_response
from rag_eval.metrics import aggregate_system, retrieval_scores


def test_parse_json_response_handles_fences_and_think() -> None:
    value = parse_json_response('<think>hidden</think>```json\n[{"ok": true}]\n```')
    assert value == [{"ok": True}]


def test_retrieval_scores() -> None:
    result = retrieval_scores(["a", "b"], ["b", "c"])
    assert result == {
        "hit": True,
        "all_evidence_hit": False,
        "recall": 0.5,
        "precision": 0.5,
    }


def test_token_usage_fallback_metadata() -> None:
    class Message:
        usage_metadata = {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}

    usage = TokenUsage()
    usage.add_message(Message())
    assert usage.snapshot() == {
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
        "calls": 1,
        "measured_calls": 1,
    }

