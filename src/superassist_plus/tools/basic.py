from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.tools import tool


@tool
def current_time() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(UTC).isoformat()


@tool
def echo(text: str) -> str:
    """Echo text back for smoke tests and simple tool-use validation."""

    return text

