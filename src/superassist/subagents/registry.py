from __future__ import annotations

from superassist.config import Settings, get_settings

from .config import SubagentConfig, load_subagent_configs


class SubagentRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._configs = load_subagent_configs(
            self.settings.resolved_agents_dir,
            default_timeout_seconds=self.settings.subagent_timeout_seconds,
            default_max_turns=self.settings.subagent_max_turns,
        )

    def get(self, name: str) -> SubagentConfig | None:
        return self._configs.get(name)

    def names(self) -> list[str]:
        return sorted(self._configs)

    def list(self) -> list[SubagentConfig]:
        return [self._configs[name] for name in self.names()]

    def default(self) -> SubagentConfig:
        return next(config for config in self._configs.values() if config.is_default)

    def available_agents_text(self) -> str:
        return "\n".join(f"- {config.name}: {config.description}" for config in self.list())
