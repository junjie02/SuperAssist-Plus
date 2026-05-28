from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover - LangSmith is normally present via LangChain.
    _langsmith_traceable = None


def traceable(*args: Any, **kwargs: Any) -> Callable:
    if _langsmith_traceable is None:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            return func

        return decorator
    return _langsmith_traceable(*args, **kwargs)


def runnable_trace_config(
    *,
    run_name: str,
    user_id: str | None = None,
    thread_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_metadata = _compact_metadata(metadata or {})
    if user_id:
        trace_metadata["user_id"] = user_id
    if thread_id:
        trace_metadata["thread_id"] = thread_id
    return {
        "run_name": run_name,
        "tags": ["superassist-plus", *(tags or [])],
        "metadata": trace_metadata,
    }


def trace_extra(*, metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if metadata:
        extra["metadata"] = _compact_metadata(metadata)
    if tags:
        extra["tags"] = ["superassist-plus", *tags]
    return {"langsmith_extra": extra}


def without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            compact[key] = _preview(value, limit=500)
        elif isinstance(value, (int, float, bool)):
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = [_preview(str(item), limit=200) for item in value[:20]]
        elif isinstance(value, dict):
            compact[key] = {
                str(nested_key): _preview(str(nested_value), limit=200)
                for nested_key, nested_value in list(value.items())[:20]
            }
        else:
            compact[key] = _preview(str(value), limit=200)
    return compact


def _preview(value: str, *, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
