from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_current_team_thread_id: ContextVar[str | None] = ContextVar(
    "superassist_plus_current_team_thread_id",
    default=None,
)


def current_team_thread_id() -> str | None:
    return _current_team_thread_id.get()


@contextmanager
def team_thread_context(thread_id: str | None) -> Iterator[None]:
    token = _current_team_thread_id.set(thread_id)
    try:
        yield
    finally:
        _current_team_thread_id.reset(token)
