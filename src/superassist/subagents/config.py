from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)
_AGENT_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
ModelProfile = Literal["main", "memory"]


class SubagentConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SubagentConfig:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None
    timeout_seconds: int
    max_turns: int
    model_profile: ModelProfile = "main"
    is_default: bool = False
    source_dir: Path | None = None


def load_subagent_configs(
    agents_dir: Path,
    *,
    default_timeout_seconds: int,
    default_max_turns: int,
) -> dict[str, SubagentConfig]:
    root = agents_dir.resolve()
    if not root.is_dir():
        raise SubagentConfigError(f"Agent definitions directory does not exist: {root}")

    configs: dict[str, SubagentConfig] = {}
    for directory in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name):
        manifest_path = directory / "agent.toml"
        if not manifest_path.is_file():
            continue
        try:
            config = _load_subagent_config(
                manifest_path,
                default_timeout_seconds=default_timeout_seconds,
                default_max_turns=default_max_turns,
            )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError, SubagentConfigError) as exc:
            logger.warning("Skipping invalid agent definition path=%s error=%s", manifest_path, exc)
            continue
        if config.name in configs:
            logger.warning("Skipping duplicate agent definition name=%s path=%s", config.name, manifest_path)
            continue
        configs[config.name] = config

    if not configs:
        raise SubagentConfigError(f"No valid enabled agents found in {root}")
    defaults = [config.name for config in configs.values() if config.is_default]
    if len(defaults) != 1:
        raise SubagentConfigError(
            f"Exactly one enabled agent must set default=true; found {len(defaults)} in {root}"
        )
    return configs


def _load_subagent_config(
    manifest_path: Path,
    *,
    default_timeout_seconds: int,
    default_max_turns: int,
) -> SubagentConfig:
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(data.get("enabled", True)):
        raise SubagentConfigError("agent is disabled")

    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    if not _AGENT_NAME_RE.fullmatch(name):
        raise SubagentConfigError("name must match [a-z0-9][a-z0-9-]{0,63}")
    if not description or len(description) > 500:
        raise SubagentConfigError("description must contain 1 to 500 characters")

    directory = manifest_path.parent.resolve()
    prompt_name = str(data.get("prompt_file") or "system.md").strip()
    prompt_path = (directory / prompt_name).resolve()
    if not prompt_path.is_relative_to(directory):
        raise SubagentConfigError("prompt_file must stay inside its agent directory")
    if not prompt_path.is_file():
        raise SubagentConfigError(f"prompt file does not exist: {prompt_name}")
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise SubagentConfigError("prompt file is empty")

    raw_tools = data.get("allowed_tools", "*")
    if raw_tools == "*":
        allowed_tools = None
    elif isinstance(raw_tools, list) and all(isinstance(item, str) and item.strip() for item in raw_tools):
        allowed_tools = list(dict.fromkeys(item.strip() for item in raw_tools))
    else:
        raise SubagentConfigError("allowed_tools must be '*' or a list of tool names")

    model_profile = str(data.get("model_profile") or "main").strip().lower()
    if model_profile not in {"main", "memory"}:
        raise SubagentConfigError("model_profile must be 'main' or 'memory'")
    timeout_seconds = _positive_int(data.get("timeout_seconds"), default_timeout_seconds, "timeout_seconds")
    max_turns = _positive_int(data.get("max_turns"), default_max_turns, "max_turns")
    return SubagentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        model_profile=model_profile,  # type: ignore[arg-type]
        is_default=bool(data.get("default", False)),
        source_dir=directory,
    )


def _positive_int(value: object, default: int, field: str) -> int:
    resolved = default if value is None else value
    if isinstance(resolved, bool):
        raise SubagentConfigError(f"{field} must be a positive integer")
    try:
        number = int(resolved)
    except (TypeError, ValueError) as exc:
        raise SubagentConfigError(f"{field} must be a positive integer") from exc
    if number <= 0:
        raise SubagentConfigError(f"{field} must be a positive integer")
    return number
