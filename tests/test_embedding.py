import sys
import types

from superassist_plus.config import Settings
from superassist_plus.memory.embedding import BGEEmbedder, HashEmbedder, create_embedder


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder()

    assert embedder.embed("same text") == embedder.embed("same text")


def test_create_embedder_uses_bge_provider() -> None:
    settings = Settings(
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="bge",
        SUPERASSIST_PLUS_EMBEDDING_MODEL="BAAI/bge-base-zh-v1.5",
        SUPERASSIST_PLUS_EMBEDDING_DEVICE="cpu",
    )

    embedder = create_embedder(settings)

    assert isinstance(embedder, BGEEmbedder)
    assert embedder.model_name == "BAAI/bge-base-zh-v1.5"


def test_bge_embedder_uses_sentence_transformer(monkeypatch) -> None:
    class FakeVector:
        def tolist(self):
            return [0.0, 1.0]

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str, cache_folder: str | None = None) -> None:
            self.model_name = model_name
            self.device = device
            self.cache_folder = cache_folder

        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):
            assert texts == ["hello"]
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [FakeVector()]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = BGEEmbedder("BAAI/bge-base-zh-v1.5", "cpu")

    assert embedder.embed("hello") == [0.0, 1.0]


def test_bge_preload_keeps_model_instance(monkeypatch) -> None:
    class FakeSentenceTransformer:
        instance_count = 0

        def __init__(self, model_name: str, device: str, cache_folder: str | None = None) -> None:
            FakeSentenceTransformer.instance_count += 1

        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):
            class FakeVector:
                def tolist(self):
                    return [1.0, 0.0]

            return [FakeVector()]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = BGEEmbedder("BAAI/bge-base-zh-v1.5", "cpu")
    embedder.preload()
    embedder.embed("hello")

    assert FakeSentenceTransformer.instance_count == 1
