"""Error types and human-readable formatters for the ACP client."""

from __future__ import annotations

import shutil


class ACPClientError(RuntimeError):
    """Raised when the ACP client fails to start, prompt, or close a session."""


def missing_command_message(name: str, command: str) -> str:
    """Explain why an agent could not be started when its command is not on PATH."""

    msg = f"Team agent '{name}' command '{command}' was not found on PATH."
    if command == "codex-acp" and shutil.which("codex"):
        return (
            f"{msg} The installed `codex` CLI does not speak ACP directly. "
            "Use an ACP adapter such as `npx -y @zed-industries/codex-acp` in agent_team.toml."
        )
    return f"{msg} Install the agent binary or update agent_team.toml."


def format_start_error(name: str, exc: Exception) -> str:
    """Compose a useful error message from an arbitrary spawn-time exception."""

    text = str(exc)
    code = getattr(exc, "code", None)
    data = getattr(exc, "data", None)
    extras = ""
    if code is not None:
        extras += f" code={code}"
    if data is not None:
        extras += f" data={data!r}"
    if not text or text == "Internal error":
        return (
            f"Failed to start team agent '{name}': {text or type(exc).__name__}.{extras} "
            "Run the configured adapter command manually to inspect stderr, or install the ACP adapter locally."
        )
    if "EPERM" in text and "npm-cache" in text:
        return (
            f"Failed to start team agent '{name}': npm cache permission error. "
            "SuperAssist sets npm_config_cache to a project-local cache; restart the runtime and try again."
        )
    return f"Failed to start team agent '{name}': {text}.{extras}"


__all__ = ["ACPClientError", "format_start_error", "missing_command_message"]
