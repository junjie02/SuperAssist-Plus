from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any


class AIEngineError(RuntimeError):
    """Raised when the local Python AI Engine cannot complete a chat request."""


async def iter_sse_events(lines: AsyncIterable[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Parse the one-line JSON SSE events emitted by ``/internal/chat``."""

    async for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AIEngineError("AI Engine returned invalid SSE JSON") from exc
        if isinstance(event, dict):
            yield event


class AIEngineClient:
    """Small async client for the existing local SuperAssist SSE endpoint."""

    def __init__(self, base_url: str, *, request_timeout_seconds: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self._session: Any = None

    async def start(self) -> None:
        if self._session is not None:
            return
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("aiohttp is required by the WeCom channel") from exc
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_connect=10,
            sock_read=self.request_timeout_seconds,
        )
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def stream_chat(
        self,
        *,
        user_id: str,
        message: str,
        thread_id: str,
        rag_mode: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        if self._session is None:
            await self.start()
        payload = {
            "user_id": user_id,
            "message": message,
            "thread_id": thread_id,
            "rag_mode": rag_mode,
        }
        try:
            async with self._session.post(
                f"{self.base_url}/internal/chat",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:500]
                    raise AIEngineError(f"AI Engine returned HTTP {response.status}: {detail}")
                async for event in iter_sse_events(response.content):
                    yield event
        except AIEngineError:
            raise
        except Exception as exc:
            raise AIEngineError(f"Cannot reach AI Engine at {self.base_url}: {exc}") from exc
