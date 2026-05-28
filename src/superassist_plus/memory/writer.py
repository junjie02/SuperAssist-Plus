from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any

from langchain_core.language_models import BaseChatModel

from superassist_plus.memory.service import MemoryService, MemoryWritePayload

logger = logging.getLogger(__name__)


MEMORY_WRITER_PROMPT = """You are a cognitive graph update agent for SuperAssist-Plus.
Given one completed conversation event and its context, produce an UpdatePlan with operations.

Return ONLY valid JSON with this shape:
{
  "reasoning": "brief analysis of durable memory changes",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "concept|intent|time",
      "data": {
        "ref": "short_ref_for_edges",
        "title": "short durable title",
        "description": "durable reusable memory, no raw transcript",
        "importance": 0.5
      },
      "reasoning": "why this node should exist",
      "grounded_in": ["current_event"]
    },
    {
      "op": "ADD_EDGE",
      "source_id": "current_event",
      "target_id": "short_ref_for_edges",
      "edge_type": "GROUNDS",
      "weight": 0.9
    },
    {
      "op": "UPDATE_NODE",
      "node_id": "existing_node_id",
      "data": {"importance": 0.7},
      "update_reasoning": "why the existing memory changed"
    },
    {
      "op": "MERGE_NODES",
      "node_ids": ["existing_node_a", "existing_node_b"],
      "merged_data": {"title": "merged title", "description": "merged durable memory"},
      "reasoning": "why these nodes are near-duplicates"
    }
  ],
  "symbolic_actions": []
}

NODE TYPES:
- event: raw conversational events. The current event already exists as "current_event"; do not add it again.
- concept: recurring patterns, stable user facts, preferences, reusable project context.
- intent: user goals, pending outcomes, unmet needs, or follow-up objectives.
- time: deadlines, schedules, or recurring temporal anchors.

EDGE TYPES with default weights:
- GROUNDS (0.9): event -> concept/intent, direct evidence.
- CAUSES (0.9): event -> event, causal relationship.
- TRIGGERS (0.8): event/concept -> intent, activates a goal.
- USER_FEEDBACK (0.8): event/concept -> intent, explicit correction or preference signal.
- REINFORCES (0.7): event -> existing concept, supporting evidence.
- PART_OF (0.7): concept -> concept, hierarchy or containment.
- DERIVED_FROM (0.6): concept -> concept, abstraction or derivation.
- DEADLINE_FOR (0.6): time -> event/concept/intent, temporal constraint.
- RELATED_TO (0.5): concept -> concept, only when no more specific edge fits.

RULES:
1. Store only durable preferences, goals, facts, recurring work context, useful concepts, and deadlines.
2. Do not store secrets, transient tool output, or one-off chat filler.
3. Create concepts for recurring patterns or stable facts; if similar context already exists, UPDATE or MERGE instead.
4. Link every new concept, intent, or time node to grounding evidence using grounded_in and explicit ADD_EDGE operations.
5. Create intents only when the event/context suggests unmet goals or follow-up work with supporting evidence.
6. Self-review before output: check for missing edges between concepts that share grounding events.
7. Prefer concise Chinese titles/descriptions when the source event is Chinese; otherwise match the source language.
8. Use "current_event" as the source id for this turn's existing event node.
"""


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

    def _build_plan(self, payload: MemoryWritePayload) -> dict[str, Any]:
        if self.llm_enabled and self.model is not None and getattr(self.model, "_llm_type", "") != "superassist-plus-fallback":
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
                return self._parse_json(str(response.content))
            except Exception as exc:
                logger.warning("LLM memory writer failed; using fallback plan: %s", exc)
        return self._fallback_plan(payload)

    @staticmethod
    def _fallback_plan(payload: MemoryWritePayload) -> dict[str, Any]:
        text = payload.user_message.strip()
        if not text or len(text) < 12:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [
                {
                    "ref": "turn_concept",
                    "type": "concept",
                    "title": text[:80],
                    "description": f"User discussed: {text[:500]}",
                    "reasoning": "Fallback durable concept from the user turn.",
                }
            ],
            "edges": [],
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
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
        return value if isinstance(value, dict) else {"nodes": [], "edges": []}


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
