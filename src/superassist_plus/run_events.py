from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from superassist_plus.models import AgentRunEvent


RunEventReporter = Callable[[AgentRunEvent], None]

_current_run_event_reporter: ContextVar[RunEventReporter | None] = ContextVar(
    "superassist_plus_current_run_event_reporter",
    default=None,
)


def current_run_event_reporter() -> RunEventReporter | None:
    return _current_run_event_reporter.get()


@contextmanager
def run_event_reporter_context(reporter: RunEventReporter | None) -> Iterator[None]:
    token = _current_run_event_reporter.set(reporter)
    try:
        yield
    finally:
        _current_run_event_reporter.reset(token)
