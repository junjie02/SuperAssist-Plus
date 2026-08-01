from __future__ import annotations

import json
import logging
import re
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
                            "<MemoryWriteInput format=\"json\">\n"
                            + json.dumps(
                                {
                                    "user_message": payload.user_message,
                                    "user_message_created_at": payload.user_message_created_at,
                                    "assistant_answer": payload.assistant_answer,
                                    "assistant_message_created_at": payload.assistant_message_created_at,
                                    "tool_events": compact_tool_events(payload.tool_events),
                                    "memory_context": _compact_memory_context(payload.memory_context or {}),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n</MemoryWriteInput>",
                        ),
                    ]
                )
                return UpdatePlan.parse(_extract_json(_response_text(response.content)))
            except Exception as exc:
                logger.warning("LLM memory writer failed; using fallback plan: %s", exc)
        return self._fallback_plan(payload)

    @staticmethod
    def _fallback_plan(payload: MemoryWritePayload) -> UpdatePlan:
        text = payload.user_message.strip()
        if not text or _is_pure_greeting(text):
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


def _is_pure_greeting(text: str) -> bool:
    normalized = re.sub(r"[\W_]+", "", text.casefold())
    return normalized in {
        "hi",
        "hello",
        "hey",
        "goodmorning",
        "goodafternoon",
        "goodevening",
        "thanks",
        "thankyou",
        "你好",
        "你好呀",
        "你好啊",
        "您好",
        "嗨",
        "哈喽",
        "哈啰",
        "早上好",
        "上午好",
        "中午好",
        "下午好",
        "晚上好",
        "在吗",
        "谢谢",
        "多谢",
        "辛苦了",
    }


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


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "").lower() not in {
                "reasoning",
                "thinking",
                "reasoning_content",
                "chain_of_thought",
            }:
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = "\n".join(parts)
    else:
        text = str(content or "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def compact_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project completed tool calls to the only fields memory work may see."""

    compact: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type not in {None, "tool_result"}:
            continue
        status = str(event.get("status") or "success")
        error = event.get("error_summary", event.get("error"))
        if status == "error" and error is None:
            error = event.get("content")
        compact.append(
            {
                "name": event.get("tool") or event.get("name") or "",
                "status": status,
                "error_summary": _preview(str(error or ""), 500),
            }
        )
        if len(compact) >= 20:
            break
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
    return {
        "tier": raw.get("tier", ""),
        "id": raw.get("id", ""),
        "type": raw.get("type", ""),
        "title": _preview(str(raw.get("title") or ""), 160),
        "description": _preview(str(raw.get("description") or ""), 1200),
        "user_id": raw.get("user_id", ""),
        "importance": raw.get("importance", 0.5),
        "grounded_in": list(raw.get("grounded_in") or [])[:10],
        "source": raw.get("source", ""),
        "created_at": raw.get("created_at", ""),
        "updated_at": raw.get("updated_at", ""),
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
