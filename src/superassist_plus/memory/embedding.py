from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from superassist_plus.config import Settings, get_settings


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def preload(self) -> None:
        ...


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text or "")]


class HashEmbedder:
    """Deterministic local fallback embedding provider."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return normalize(vector)

    def preload(self) -> None:
        return None


class BGEEmbedder:
    """BGE embedding provider backed by sentence-transformers."""

    def __init__(self, model_name: str, device: str = "cpu", cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "BGE embeddings require sentence-transformers. Install project dependencies in the CF environment."
                ) from exc
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
            self._model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_dir)
        return self._model

    def preload(self) -> None:
        _ = self.model

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text or ""],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(value) for value in vector.tolist()]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def create_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower().strip()
    if provider == "bge":
        return BGEEmbedder(
            settings.embedding_model,
            settings.embedding_device,
            str(settings.huggingface_cache_dir),
        )
    if provider in {"hash", "local", "fallback"}:
        return HashEmbedder()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


@lru_cache(maxsize=8)
def _cached_embedder(provider: str, model_name: str, device: str, cache_dir: str) -> Embedder:
    settings = Settings(
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER=provider,
        SUPERASSIST_PLUS_EMBEDDING_MODEL=model_name,
        SUPERASSIST_PLUS_EMBEDDING_DEVICE=device,
        SUPERASSIST_PLUS_DATA_DIR=str(Path(cache_dir).parent) if cache_dir else ".superassist-plus",
    )
    return create_embedder(settings)


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    return _cached_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_device,
        str(settings.huggingface_cache_dir),
    )


def embed_text(text: str, settings: Settings | None = None) -> list[float]:
    return get_embedder(settings).embed(text)


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))
