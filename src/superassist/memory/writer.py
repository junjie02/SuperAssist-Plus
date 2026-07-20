from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any

from langchain_core.language_models import BaseChatModel

from superassist.memory.plans import UpdatePlan
from superassist.memory.prompts import format_memory_writer_prompt
from superassist.memory.service import MemoryService, MemoryWritePayload

logger = logging.getLogger(__name__)

MEMORY_WRITER_PROMPT = format_memory_writer_prompt(mode="quick")


class MemoryWriter:
    """LLM-assisted memory writer with deterministic fallback."""

    def __init__(
        self,
        service: MemoryService,
        model: BaseChatModel | None = None,
        *,
        llm_enabled: bool = False,
    ) -> None:
        self.service = service
        self.model = model
        self.llm_enabled = llm_enabled

    def write(self, payload: MemoryWritePayload) -> dict[str, int]:
        plan = self._build_plan(payload)
        result = self.service.apply_structured_memory(payload, plan)
        result.update(self.service.consolidate(payload.user_id))
        return result

    def _build_plan(self, payload: MemoryWritePayload) -> UpdatePlan:
        if self.llm_enabled and self.model is not None and getattr(self.model, "_llm_type", "") != "superassist-fallback":
            try:
                response = self.model.invoke(
                    [
                        ("system", MEMORY_WRITER_PROMPT),
                        (
                            "human",
                            json.dumps(
                                {
                                    "user_message": payload.user_message,
                                    "assistant_answer": payload.assistant_answer,
                                    "tool_events": _compact_tool_events(payload.tool_events),
                                    "memory_context": _compact_memory_context(payload.memory_context or {}),
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ]
                )
                return UpdatePlan.parse(_extract_json(str(response.content)))
            except Exception as exc:
                logger.warning("LLM memory writer failed; using fallback plan: %s", exc)
        return self._fallback_plan(payload)

    @staticmethod
    def _fallback_plan(payload: MemoryWritePayload) -> UpdatePlan:
        text = payload.user_message.strip()
        if not text or len(text) < 12:
            return UpdatePlan()
        answer = (payload.assistant_answer or "").strip()
        return UpdatePlan.from_legacy(
            {
                "nodes": [
                    {
                        "ref": "current_event",
                        "type": "event",
                        "title": text[:80],
                        "description": f"User: {text[:300]}\nAssistant: {answer[:300]}",
                        "reasoning": "Conversation turn (fallback event).",
                    },
                    {
                        "ref": "turn_concept",
                        "type": "concept",
                        "title": text[:80],
                        "description": f"User discussed: {text[:500]}",
                        "reasoning": "Fallback durable concept from the user turn.",
                    },
                ],
            }
        )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    return value if isinstance(value, dict) else {}


def _compact_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for event in events[:20]:
        compact.append(
            {
                "name": event.get("name") or event.get("tool") or "",
                "content": _preview(str(event.get("content") or event.get("error") or ""), 1000),
                "status": event.get("status", "success"),
            }
        )
    return compact


def _compact_memory_context(memory_context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    compact: dict[str, list[dict[str, Any]]] = {}
    for tier in ("immediate", "working", "background", "buffer"):
        nodes = memory_context.get(tier) or []
        if not isinstance(nodes, list):
            compact[tier] = []
            continue
        compact[tier] = [_compact_memory_node(node) for node in nodes[:5]]
    return compact


def _compact_memory_node(node: Any) -> dict[str, Any]:
    if hasattr(node, "model_dump"):
        raw = node.model_dump(mode="json")
    elif isinstance(node, dict):
        raw = node
    else:
        raw = {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return {
        "id": raw.get("id", ""),
        "type": raw.get("type", ""),
        "title": _preview(str(raw.get("title") or ""), 160),
        "description": _preview(str(raw.get("description") or ""), 1200),
        "importance": raw.get("importance", 0.5),
        "access_count": raw.get("access_count", 0),
        "reasoning": _preview(str(raw.get("reasoning") or ""), 500),
        "grounded_in": list(raw.get("grounded_in") or [])[:10],
        "source": metadata.get("source", ""),
        "thread_id": metadata.get("thread_id", ""),
    }


def _preview(text: str, limit: int) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


class MemoryWriteQueue:
    """Debounced in-process queue for background memory writes."""

    def __init__(self, writer: MemoryWriter, debounce_seconds: float = 30.0) -> None:
        self.writer = writer
        self.debounce_seconds = debounce_seconds
        self._queue: deque[MemoryWritePayload] = deque()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def add(self, payload: MemoryWritePayload) -> None:
        with self._lock:
            self._queue.append(payload)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.flush)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> None:
        with self._lock:
            payloads = list(self._queue)
            self._queue.clear()
            self._timer = None
        for payload in payloads:
            try:
                self.writer.write(payload)
            except Exception:
                logger.exception("Memory write failed for thread %s", payload.thread_id)
