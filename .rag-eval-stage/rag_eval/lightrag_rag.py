from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from rag_eval.common import Chunk, TokenUsage, find_chunk_store, get_settings, message_text
from rag_eval.vector_rag import Retrieval


class ExistingLightRAG:
    """Read the existing SuperAssist LightRAG index without rebuilding it."""

    def __init__(self, chunks: list[Chunk], top_k: int = 20, chunk_top_k: int = 5) -> None:
        from superassist.llm import create_chat_model
        from superassist.memory.embedding import get_embedder

        self.chunks = chunks
        self.chunk_by_id = {item.chunk_id: item for item in chunks}
        self.settings = get_settings()
        self.model = create_chat_model(self.settings)
        self.embedder = get_embedder(self.settings)
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.usage = TokenUsage()
        self.rag: LightRAG | None = None

    async def initialize(self) -> None:
        embedding_dim = len(self.embedder.embed("LightRAG evaluation dimension probe"))
        index_dir = find_chunk_store().parents[1]

        # LightRAG snapshots dataclass configuration with asdict(). Plain
        # closures avoid deepcopying this instance and its model thread locks.
        async def llm_model_func(*args: Any, **kwargs: Any) -> str:
            return await self._llm_model_func(*args, **kwargs)

        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await self._embedding_func(texts)

        self.rag = LightRAG(
            working_dir=str(index_dir),
            workspace="default",
            llm_model_func=llm_model_func,
            llm_model_name=self.settings.model,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                model_name=self.settings.embedding_model,
                func=embedding_func,
            ),
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
            addon_params={"language": "Chinese"},
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
        )
        await self.rag.initialize_storages()

    async def retrieve(self, query: str) -> Retrieval:
        if self.rag is None:
            await self.initialize()
        assert self.rag is not None
        before = self.usage.snapshot()
        started = time.perf_counter()
        raw = await self.rag.aquery_data(
            query,
            QueryParam(
                mode="mix",
                top_k=self.top_k,
                chunk_top_k=self.chunk_top_k,
                max_entity_tokens=1500,
                max_relation_tokens=1500,
                max_total_tokens=6000,
                enable_rerank=False,
                include_references=True,
            ),
        )
        elapsed = time.perf_counter() - started
        data = raw.get("data") if isinstance(raw, dict) else {}
        data = data if isinstance(data, dict) else {}
        chunks = list(data.get("chunks") or [])
        entities = list(data.get("entities") or [])
        relationships = list(data.get("relationships") or [])
        chunk_ids = self._evidence_chunk_ids(chunks, entities, relationships)
        parts: list[str] = []
        for item in chunks:
            parts.append(
                f"[SOURCE {item.get('file_path') or 'unknown'}]\n{item.get('content') or ''}"
            )
        if entities:
            parts.append(
                "ENTITIES:\n"
                + "\n".join(
                    f"- {item.get('entity_name')}: {item.get('description')}" for item in entities
                )
            )
        if relationships:
            parts.append(
                "RELATIONSHIPS:\n"
                + "\n".join(
                    f"- {item.get('src_id')} -> {item.get('tgt_id')}: {item.get('description')}"
                    for item in relationships
                )
            )
        return Retrieval(
            context="\n\n".join(parts)[:24000],
            chunk_ids=chunk_ids,
            scores=[],
            elapsed_seconds=elapsed,
            retrieval_usage=self.usage.delta(before),
        )

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None

    async def _embedding_func(self, texts: list[str]) -> np.ndarray:
        vectors = await asyncio.to_thread(self.embedder.embed_many, list(texts))
        return np.asarray(vectors, dtype=np.float32)

    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        for item in history_messages or []:
            content = str(item.get("content") or "")
            messages.append(
                AIMessage(content=content)
                if item.get("role") == "assistant"
                else HumanMessage(content=content)
            )
        messages.append(HumanMessage(content=prompt))
        response = await self.model.ainvoke(messages)
        self.usage.add_message(response)
        return message_text(response)

    def _evidence_chunk_ids(self, *groups: list[dict[str, Any]]) -> list[str]:
        found: set[str] = set()
        for group in groups:
            for item in group:
                for key in ("chunk_id", "id", "_id", "source_id"):
                    raw_value = item.get(key) or ""
                    values = raw_value if isinstance(raw_value, list) else [raw_value]
                    for value in values:
                        for candidate in re.split(r"<SEP>|[,;\s]+", str(value)):
                            if candidate in self.chunk_by_id:
                                found.add(candidate)
                content = str(item.get("content") or "").strip()
                if content:
                    for chunk in self.chunks:
                        if content == chunk.content or content[:200] in chunk.content:
                            found.add(chunk.chunk_id)
        return sorted(found)
