from __future__ import annotations

from superassist_plus.config import Settings, get_settings

from .config import SubagentConfig, build_builtin_subagents


class SubagentRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._configs = build_builtin_subagents(
            timeout_seconds=self.settings.subagent_timeout_seconds,
            max_turns=self.settings.subagent_max_turns,
        )

    def get(self, name: str) -> SubagentConfig | None:
        return self._configs.get(name)

    def names(self) -> list[str]:
        return sorted(self._configs)

    def list(self) -> list[SubagentConfig]:
        return [self._configs[name] for name in self.names()]


def get_subagent_config(name: str, settings: Settings | None = None) -> SubagentConfig | None:
    return SubagentRegistry(settings).get(name)


def get_available_subagent_names(settings: Settings | None = None) -> list[str]:
    return SubagentRegistry(settings).names()
