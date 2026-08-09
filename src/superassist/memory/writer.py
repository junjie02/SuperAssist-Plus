from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel

from superassist.memory.plans import AddNodeOp, UpdatePlan
from superassist.memory.prompts import format_memory_writer_prompt
from superassist.memory.service import MemoryService, MemoryWritePayload
from superassist.redis_store import get_redis_store

logger = logging.getLogger(__name__)

MEMORY_WRITER_PROMPT = format_memory_writer_prompt()


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
        _attach_source_context(plan, payload.source_context or {})
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
                                    "source_context": payload.source_context or {},
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n</MemoryWriteInput>",
                        ),
                    ]
                )
                return UpdatePlan.parse(_extract_json(_response_text(response.content)))
            except Exception as exc:  # noqa: BLE001 - model/provider failures use the deterministic writer
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


def _attach_source_context(plan: UpdatePlan, source_context: dict[str, Any]) -> None:
    if not source_context:
        return
    allowed = {
        "source_batch_id": str(source_context.get("batch_id") or ""),
        "source_message_ids": [str(item) for item in source_context.get("message_ids") or []][:100],
        "asserted_by": [str(item) for item in source_context.get("sender_ids") or []][:50],
        "source_channel": str(source_context.get("channel") or ""),
    }
    metadata = {key: value for key, value in allowed.items() if value}
    for operation in plan.operations:
        if isinstance(operation, AddNodeOp):
            operation.data.metadata = {**operation.data.metadata, **metadata}


class MemoryWriteQueue:
    """Debounced in-process queue for background memory writes."""

    def __init__(self, writer: MemoryWriter, debounce_seconds: float = 30.0) -> None:
        self.writer = writer
        self.debounce_seconds = debounce_seconds
        self._queue: deque[tuple[str, MemoryWritePayload]] = deque()
        self._queued_job_ids: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._redis = get_redis_store(writer.service.settings)
        self._recover_jobs()

    def add(self, payload: MemoryWritePayload) -> None:
        job_id = _memory_job_id(payload)
        should_run = self.writer.service.store.enqueue_memory_job(
            job_id,
            payload.user_id,
            asdict(payload),
        )
        if not should_run:
            return
        if self._redis.enabled:
            self._redis.schedule_memory_job(
                job_id,
                asdict(payload),
                due_at=time.time() + max(0.0, self.debounce_seconds),
            )
            self._schedule_redis_timer()
            return
        with self._lock:
            if job_id not in self._queued_job_ids:
                self._queue.append((job_id, payload))
                self._queued_job_ids.add(job_id)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.flush)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> None:
        if self._redis.enabled:
            self._flush_redis(force=True)
            return
        with self._lock:
            timer = self._timer
            if timer is not None and timer is not threading.current_thread():
                timer.cancel()
            jobs = list(self._queue)
            self._queue.clear()
            self._queued_job_ids.clear()
            self._timer = None
        retry_jobs: list[tuple[str, MemoryWritePayload]] = []
        for job_id, payload in jobs:
            attempts = self.writer.service.store.mark_memory_job_running(job_id)
            if attempts == 0:
                continue
            try:
                self.writer.write(payload)
            except Exception as exc:
                self.writer.service.store.finish_memory_job(
                    job_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.exception("Memory write failed for thread %s", payload.thread_id)
                if attempts < 3:
                    retry_jobs.append((job_id, payload))
            else:
                self.writer.service.store.finish_memory_job(job_id)
        if retry_jobs:
            with self._lock:
                for job in retry_jobs:
                    if job[0] not in self._queued_job_ids:
                        self._queue.append(job)
                        self._queued_job_ids.add(job[0])
                if self._timer is None:
                    self._timer = threading.Timer(max(1.0, self.debounce_seconds), self.flush)
                    self._timer.daemon = True
                    self._timer.start()

    def _recover_jobs(self) -> None:
        self.writer.service.store.requeue_stale_memory_jobs(
            (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        )
        for row in self.writer.service.store.list_memory_jobs():
            if int(row.get("attempts") or 0) >= 3:
                continue
            try:
                raw = json.loads(str(row.get("payload_json") or "{}"))
                payload = MemoryWritePayload(**raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Skipping invalid persisted memory job id=%s", row.get("id"))
                continue
            job_id = str(row["id"])
            if self._redis.enabled:
                self._redis.schedule_memory_job(
                    job_id,
                    asdict(payload),
                    due_at=time.time() + max(1.0, self.debounce_seconds),
                )
                continue
            self._queue.append((job_id, payload))
            self._queued_job_ids.add(job_id)
        if self._redis.enabled:
            self._schedule_redis_timer()
            return
        if self._queue:
            self._timer = threading.Timer(max(1.0, self.debounce_seconds), self.flush)
            self._timer.daemon = True
            self._timer.start()

    def _schedule_redis_timer(self) -> None:
        due_at = self._redis.next_memory_job_due_at()
        if due_at is None:
            return
        delay = max(0.05, due_at - time.time())
        with self._lock:
            if self._timer is not None and self._timer.is_alive():
                return
            self._timer = threading.Timer(delay, self._flush_redis_due)
            self._timer.daemon = True
            self._timer.start()

    def _flush_redis_due(self) -> None:
        self._flush_redis(force=False)

    def _flush_redis(self, *, force: bool) -> None:
        with self._lock:
            timer = self._timer
            if timer is not None and timer is not threading.current_thread():
                timer.cancel()
            self._timer = None
        while True:
            job_ids = self._redis.claim_memory_jobs(force=force, limit=1000 if force else 50)
            if not job_ids:
                break
            for job_id in job_ids:
                raw = self._redis.get_memory_job(job_id)
                if raw is None:
                    self._redis.finish_memory_job(job_id)
                    continue
                try:
                    payload = MemoryWritePayload(**raw)
                except (TypeError, ValueError):
                    self.writer.service.store.finish_memory_job(job_id, error="Invalid Redis memory job payload")
                    self._redis.finish_memory_job(job_id)
                    continue
                attempts = self.writer.service.store.mark_memory_job_running(job_id)
                if attempts == 0:
                    self._redis.finish_memory_job(job_id)
                    continue
                try:
                    self.writer.write(payload)
                except Exception as exc:
                    self.writer.service.store.finish_memory_job(
                        job_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    logger.exception("Memory write failed for thread %s", payload.thread_id)
                    if attempts < 3:
                        retry_delay = self.debounce_seconds * (4 ** max(0, attempts - 1))
                        self._redis.reschedule_memory_job(
                            job_id,
                            due_at=time.time() + max(1.0, retry_delay),
                        )
                    else:
                        self._redis.finish_memory_job(job_id)
                else:
                    self.writer.service.store.finish_memory_job(job_id)
                    self._redis.finish_memory_job(job_id)
            break
        self._schedule_redis_timer()


def _memory_job_id(payload: MemoryWritePayload) -> str:
    source_context = payload.source_context or {}
    source_id = str(source_context.get("batch_id") or payload.event_id)
    digest = hashlib.sha256(f"{payload.user_id}\0{source_id}".encode()).hexdigest()
    return f"memory_job_{digest[:32]}"
