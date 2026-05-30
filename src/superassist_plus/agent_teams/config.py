from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from superassist_plus.config import PROJECT_ROOT


class TeamAgentConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    description: str
    env: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    auto_approve_permissions: bool = False

    @field_validator("name", "command", "description")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class AgentTeamConfig(BaseModel):
    enabled: bool = False
    idle_ttl_seconds: int = 3600
    agents: list[TeamAgentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_agent_names(self) -> "AgentTeamConfig":
        names = [agent.name for agent in self.agents]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate agent name(s): {', '.join(duplicates)}")
        return self

    @property
    def agents_by_name(self) -> dict[str, TeamAgentConfig]:
        return {agent.name: agent for agent in self.agents}

    @classmethod
    def disabled(cls) -> "AgentTeamConfig":
        return cls(enabled=False, agents=[])

    @classmethod
    def from_file(cls, path: Path | None = None) -> "AgentTeamConfig":
        config_path = path or PROJECT_ROOT / "agent_team.toml"
        if not config_path.exists():
            return cls.disabled()
        try:
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
            raise AgentTeamConfigError(f"Invalid agent team config at {config_path}: {exc}") from exc


class AgentTeamConfigError(ValueError):
    pass
