import sys
import types

import numpy as np

from superassist.rag.reranking import CrossEncoderReranker


def test_cross_encoder_reranker_scores_query_document_pairs(monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(
            self,
            model_name: str,
            *,
            device: str,
            cache_folder: str | None,
            max_length: int,
        ) -> None:
            assert model_name == "reranker-model"
            assert device == "cpu"
            assert cache_folder == "cache"
            assert max_length == 256

        def predict(
            self,
            pairs,
            *,
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: bool,
        ):
            assert pairs == [("query", "doc-a"), ("query", "doc-b")]
            assert batch_size == 8
            assert show_progress_bar is False
            assert convert_to_numpy is True
            return np.asarray([0.25, 0.75], dtype="float32")

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    reranker = CrossEncoderReranker(
        "reranker-model",
        device="cpu",
        cache_dir="cache",
        batch_size=8,
        max_length=256,
    )

    assert reranker.score("query", ["doc-a", "doc-b"]) == [0.25, 0.75]
