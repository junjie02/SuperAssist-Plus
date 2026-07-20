"""Run an asyncio event loop on a dedicated background thread.

LangChain runs synchronously and we cannot block its main thread on async
ACP calls, so each ACP team member owns one of these loop-threads. ``submit``
schedules a coroutine onto the loop and returns a concurrent.futures.Future
the caller can ``.result()`` on.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future


class AsyncLoopThread:
    """Owns an event loop running on a daemon thread."""

    def __init__(self, name: str) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


__all__ = ["AsyncLoopThread"]
