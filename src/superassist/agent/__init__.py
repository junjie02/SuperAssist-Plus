"""Agent package entry points.

Exports are resolved lazily so that importing ``superassist.agent.state`` from
a middleware does not trigger ``agent.factory`` (which imports middlewares back),
i.e. it breaks the ``middlewares -> agent.state -> agent.__init__ -> factory``
circular import.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AgentBundle", "AgentRuntime", "build_agent"]


def __getattr__(name: str) -> Any:
    if name == "AgentRuntime":
        from superassist.agent.runtime import AgentRuntime

        return AgentRuntime
    if name in {"AgentBundle", "build_agent"}:
        from superassist.agent.factory import AgentBundle, build_agent

        return {"AgentBundle": AgentBundle, "build_agent": build_agent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
