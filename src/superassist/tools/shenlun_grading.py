from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from superassist.channels.store import FeishuThreadStore
from superassist.config import Settings
from superassist.models import AgentRunEvent
from superassist.shenlun_grading import ShenlunGradingError, ShenlunGradingService
from superassist.teams.context import current_team_thread_id

_SERVICES: dict[str, ShenlunGradingService] = {}
_SERVICES_LOCK = threading.Lock()


def get_shenlun_grading_service(settings: Settings) -> ShenlunGradingService:
    key = str((settings.resolved_tool_workspace_dir / "shenlun-grading").resolve())
    with _SERVICES_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = ShenlunGradingService(settings)
            _SERVICES[key] = service
        return service


def make_shenlun_grading_tool(
    settings: Settings,
    *,
    run_event_reporter: Callable[[AgentRunEvent], None] | None = None,
) -> BaseTool:
    service = get_shenlun_grading_service(settings)

    @tool("shenlun_grading")
    def shenlun_grading(
        action: Literal["reserve", "current", "check", "scope", "start", "next", "status"],
        question_count: Annotated[int, Field(ge=1, le=20)] = 5,
        manifest_path: str = "",
        job_id: str = "",
        wait_seconds: Annotated[float, Field(ge=0, le=300)] = 120,
    ) -> str:
        """Create and run a durable Shenlun grading batch.

        Use reserve once for a new question set. Save reliable content incrementally
        under questions_dir, one original-question-named file per question. Use current
        on later image batches to recover the latest unfinished set before listing,
        reading, and extending those files. Use check before start. Use scope only
        after the user explicitly asks to grade a smaller prefix. Use start only when manifest.status is ready
        and every question file explicitly marks both question information and candidate
        answer complete. Then call next repeatedly; it returns completed questions.
        Threshold checks, third review, and arbitration are controlled by this tool.
        """

        thread_id = current_team_thread_id()
        if not thread_id:
            return _response(error="shenlun grading requires an active conversation thread")
        try:
            if action == "reserve":
                return _response(**service.reserve(
                    user_scope=_resolve_user_scope(settings, thread_id),
                    thread_id=thread_id,
                    question_count=question_count,
                ))
            if action == "current":
                return _response(**service.current(
                    user_scope=_resolve_user_scope(settings, thread_id),
                    thread_id=thread_id,
                ))
            if action == "check":
                if not manifest_path.strip():
                    raise ShenlunGradingError("manifest_path is required for check")
                return _response(**service.check(
                    manifest_path,
                    parent_thread_id=thread_id,
                ))
            if action == "scope":
                if not manifest_path.strip():
                    raise ShenlunGradingError("manifest_path is required for scope")
                return _response(**service.scope(
                    manifest_path,
                    parent_thread_id=thread_id,
                    question_count=question_count,
                ))
            if action == "start":
                if not manifest_path.strip():
                    raise ShenlunGradingError("manifest_path is required for start")
                preflight = service.check(manifest_path, parent_thread_id=thread_id)
                if not preflight.get("ready"):
                    return _response(**preflight)
                return _response(**service.start(
                    manifest_path,
                    parent_thread_id=thread_id,
                    reporter=run_event_reporter,
                ))
            if not job_id.strip():
                raise ShenlunGradingError(f"job_id is required for {action}")
            if action == "next":
                return _response(**service.next_result(job_id, wait_seconds=wait_seconds))
            return _response(**service.status(job_id))
        except ShenlunGradingError as exc:
            return _response(error=str(exc))

    return shenlun_grading


def _resolve_user_scope(settings: Settings, thread_id: str) -> str:
    entry = FeishuThreadStore(settings.feishu_thread_store_path).get_entry_by_thread_id(thread_id)
    if entry is not None:
        chat_id = str(entry.get("chat_id") or "").strip()
        if chat_id:
            return chat_id
    return "local"


def _response(**value: Any) -> str:
    if value.get("error") and "status" not in value:
        value = {"status": "error", **value}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["get_shenlun_grading_service", "make_shenlun_grading_tool"]
