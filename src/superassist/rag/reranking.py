from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np

from superassist.config import Settings


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder wrapper."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        cache_dir: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires sentence-transformers. Install project dependencies first."
                ) from exc
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                max_length=self.max_length,
            )
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(scores, dtype="float32")
        if array.ndim != 1 or len(array) != len(documents):
            raise RuntimeError(
                f"Reranker returned shape {array.shape}; expected one score for each of {len(documents)} documents"
            )
        return [float(value) for value in array]


@lru_cache(maxsize=4)
def _cached_reranker(
    model_name: str,
    device: str,
    cache_dir: str,
    batch_size: int,
    max_length: int,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name,
        device=device,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_length=max_length,
    )


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    return _cached_reranker(
        settings.rag_reranker_model,
        settings.embedding_device,
        str(settings.huggingface_cache_dir),
        settings.rag_reranker_batch_size,
        settings.rag_reranker_max_length,
    )


__all__ = ["CrossEncoderReranker", "Reranker", "get_reranker"]
