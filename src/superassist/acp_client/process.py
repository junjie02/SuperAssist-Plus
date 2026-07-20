"""Spawn an ACP agent process and connect a session to it.

This is the only file in superassist that imports the ``acp`` package
directly; downstream callers (``teams/supervisor``) work with the
``ACPSession`` returned by :func:`open_session`.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superassist.acp_client.errors import ACPClientError, format_start_error, missing_command_message
from superassist.acp_client.permissions import PermissionPolicy, build_permission_response

logger = logging.getLogger(__name__)


@dataclass
class ACPSpawnRequest:
    name: str
    command: str
    args: list[str]
    cwd: Path
    env: dict[str, str]
    model: str | None = None
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_APPROVE


@dataclass
class ACPSession:
    """An open ACP session bound to a spawned agent process."""

    name: str
    context: Any
    conn: Any
    session_id: str
    client: Any

    async def prompt(self, text: str) -> str:
        """Send ``text`` and return the agent's accumulated response."""

        from acp import text_block

        before = len(self.client.chunks)
        await self.conn.prompt(session_id=self.session_id, prompt=[text_block(text)])
        return "".join(self.client.chunks[before:]).strip() or "(no response)"

    async def close(self) -> None:
        try:
            await self.context.__aexit__(None, None, None)
        except Exception:
            logger.exception("Failed to close ACP session for '%s'", self.name)


async def open_session(request: ACPSpawnRequest) -> ACPSession:
    """Spawn the configured ACP agent and initialize a session."""

    try:
        from acp import PROTOCOL_VERSION, Client, spawn_agent_process
        from acp.schema import ClientCapabilities, Implementation, TextContentBlock
    except ImportError as exc:
        raise ACPClientError(
            "agent-client-protocol package is not installed. Install project dependencies before using agent teams."
        ) from exc

    cwd = request.cwd.resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    policy = request.permission_policy

    class _CollectingClient(Client):
        def __init__(self) -> None:
            self.chunks: list[str] = []

        async def session_update(self, session_id: str, update, **kwargs) -> None:  # type: ignore[override]
            try:
                if hasattr(update, "content") and isinstance(update.content, TextContentBlock):
                    self.chunks.append(update.content.text)
            except Exception:
                return

        async def request_permission(self, options, session_id: str, tool_call, **kwargs):  # type: ignore[override]
            response = build_permission_response(options, policy=policy)
            outcome = response.outcome.outcome
            tool_call_id = getattr(tool_call, "tool_call_id", "<unknown>")
            if outcome == "selected":
                logger.info("ACP permission auto-approved for '%s' tool call %s", request.name, tool_call_id)
            else:
                logger.warning("ACP permission denied for '%s' tool call %s", request.name, tool_call_id)
            return response

    try:
        client = _CollectingClient()
        command = shutil.which(request.command) or request.command
        context = spawn_agent_process(client, command, *request.args, env=request.env or None, cwd=str(cwd))
        conn, _proc = await context.__aenter__()
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="superassist", title="SuperAssist", version="0.1.0"),
        )
        session_kwargs: dict[str, Any] = {"cwd": str(cwd), "mcp_servers": []}
        if request.model:
            session_kwargs["model"] = request.model
        acp_session = await conn.new_session(**session_kwargs)
    except FileNotFoundError as exc:
        raise ACPClientError(missing_command_message(request.name, request.command)) from exc
    except ACPClientError:
        raise
    except Exception as exc:
        raise ACPClientError(format_start_error(request.name, exc)) from exc

    return ACPSession(
        name=request.name,
        context=context,
        conn=conn,
        session_id=acp_session.session_id,
        client=client,
    )


def resolve_env(env: dict[str, str], *, cache_dir: Path | None = None) -> dict[str, str]:
    """Expand ``$NAME`` placeholders against ``os.environ`` and add npm-friendly defaults."""

    resolved: dict[str, str] = {
        key: (os.environ.get(value[1:], "") if value.startswith("$") else value) for key, value in env.items()
    }
    if cache_dir is not None:
        cache_dir = cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved.setdefault("npm_config_cache", str(cache_dir))
        resolved.setdefault("npm_config_prefer_offline", "true")
        resolved.setdefault("npm_config_audit", "false")
        resolved.setdefault("npm_config_fund", "false")
    return resolved


__all__ = ["ACPSession", "ACPSpawnRequest", "open_session", "resolve_env"]
