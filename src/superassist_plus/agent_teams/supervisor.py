from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superassist_plus.config import Settings, get_settings

from .bus import JsonlBus, LedgerTamperError
from .config import AgentTeamConfig, TeamAgentConfig

logger = logging.getLogger(__name__)

_team_supervisor: "TeamSupervisor | None" = None


def get_team_supervisor() -> "TeamSupervisor | None":
    return _team_supervisor


def set_team_supervisor(supervisor: "TeamSupervisor | None") -> None:
    global _team_supervisor
    _team_supervisor = supervisor


@dataclass
class TeamTaskResult:
    agent: str
    task_id: str
    result: str
    ledger_id: str


class TeamSupervisor:
    def __init__(
        self,
        config: AgentTeamConfig,
        *,
        settings: Settings | None = None,
        bus: JsonlBus | None = None,
        member_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        self.bus = bus or JsonlBus(self.settings.data_dir / "teams" / "default")
        self._member_factory = member_factory or TeamMemberProcess
        self._members: dict[str, TeamMemberProcess] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and bool(self.config.agents)

    @property
    def agents_by_name(self) -> dict[str, TeamAgentConfig]:
        return self.config.agents_by_name

    def available_agents_text(self) -> str:
        return "\n".join(f"- {agent.name}: {agent.description}" for agent in self.config.agents)

    def invoke(
        self,
        agent: str,
        *,
        thread_id: str,
        description: str,
        prompt: str,
        wait: bool = True,
    ) -> TeamTaskResult:
        if not self.enabled:
            raise TeamSupervisorError("Agent teams are disabled or no team agents are configured.")
        if not wait:
            raise TeamSupervisorError("team_task wait=false is not supported in v1; use wait=true.")
        config = self.agents_by_name.get(agent)
        if config is None:
            available = ", ".join(sorted(self.agents_by_name))
            raise TeamSupervisorError(f"Unknown team agent '{agent}'. Available: {available}")
        try:
            self.bus.validate_thread(thread_id)
        except LedgerTamperError:
            raise
        except Exception as exc:
            raise TeamSupervisorError(f"Unable to validate team ledger for thread '{thread_id}': {exc}") from exc

        task_record = self.bus.append_message(
            thread_id,
            sender="superassist",
            recipient=agent,
            kind="task",
            body=prompt,
            extra={"description": description},
        )
        self.bus.append_inbox(
            thread_id,
            agent,
            {
                "task_id": task_record["id"],
                "description": description,
                "prompt": prompt,
            },
        )
        member = self._member(config)
        workspace = self.bus.workspace_dir(thread_id, agent)
        response = member.invoke(thread_id=thread_id, prompt=prompt, workspace=workspace)
        self.bus.append_raw(
            thread_id,
            agent,
            {
                "task_id": task_record["id"],
                "response": response,
            },
        )
        result_record = self.bus.append_message(
            thread_id,
            sender=agent,
            recipient="superassist",
            kind="result",
            body=response,
            parent_ids=[task_record["id"]],
            extra={"description": description},
        )
        return TeamTaskResult(
            agent=agent,
            task_id=str(task_record["id"]),
            result=response,
            ledger_id=str(result_record["id"]),
        )

    def sweep_idle(self) -> None:
        if self.config.idle_ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self.config.idle_ttl_seconds
        with self._lock:
            idle = [name for name, member in self._members.items() if member.last_used < cutoff]
            for name in idle:
                self._members.pop(name).close()

    def close(self) -> None:
        with self._lock:
            members = list(self._members.values())
            self._members.clear()
        for member in members:
            member.close()

    def _member(self, config: TeamAgentConfig) -> "TeamMemberProcess":
        with self._lock:
            member = self._members.get(config.name)
            if member is None:
                member = self._member_factory(config)
                self._members[config.name] = member
            return member


class TeamSupervisorError(RuntimeError):
    pass


class TeamMemberProcess:
    """Long-lived ACP adapter process wrapper for one configured team member."""

    def __init__(self, config: TeamAgentConfig) -> None:
        self.config = config
        self.last_used = time.monotonic()
        self._loop_thread = _AsyncLoopThread(f"team-agent-{config.name}")
        self._sessions: dict[tuple[str, str], _ACPSession] = {}
        self._closed = False

    def invoke(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        if self._closed:
            raise TeamSupervisorError(f"Team agent '{self.config.name}' is closed.")
        self.last_used = time.monotonic()
        future = self._loop_thread.submit(self._ainvoke(thread_id=thread_id, prompt=prompt, workspace=workspace))
        return future.result()

    async def _ainvoke(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        session = await self._ensure_session(thread_id=thread_id, workspace=workspace)
        before = len(session.client.chunks)
        from acp import text_block

        await session.conn.prompt(session_id=session.session_id, prompt=[text_block(prompt)])
        result = "".join(session.client.chunks[before:]).strip()
        return result or "(no response)"

    async def _ensure_session(self, *, thread_id: str, workspace: Path) -> "_ACPSession":
        key = (thread_id, str(workspace))
        session = self._sessions.get(key)
        if session is not None:
            return session
        try:
            from acp import PROTOCOL_VERSION, Client, spawn_agent_process
            from acp.schema import ClientCapabilities, Implementation
        except ImportError as exc:
            raise TeamSupervisorError(
                "agent-client-protocol package is not installed. Install project dependencies before using agent teams."
            ) from exc

        agent_config = self.config

        class _CollectingClient(Client):
            def __init__(self) -> None:
                self.chunks: list[str] = []

            async def session_update(self, session_id: str, update, **kwargs) -> None:  # type: ignore[override]
                try:
                    from acp.schema import TextContentBlock

                    if hasattr(update, "content") and isinstance(update.content, TextContentBlock):
                        self.chunks.append(update.content.text)
                except Exception:
                    return

            async def request_permission(self, options, session_id: str, tool_call, **kwargs):  # type: ignore[override]
                response = _build_permission_response(
                    options,
                    auto_approve=agent_config.auto_approve_permissions,
                )
                outcome = response.outcome.outcome
                if outcome == "selected":
                    logger.info(
                        "ACP permission auto-approved for team agent '%s' tool call %s in session %s",
                        agent_config.name,
                        getattr(tool_call, "tool_call_id", "<unknown>"),
                        session_id,
                    )
                else:
                    logger.warning(
                        "ACP permission denied for team agent '%s' tool call %s in session %s",
                        agent_config.name,
                        getattr(tool_call, "tool_call_id", "<unknown>"),
                        session_id,
                    )
                return response

        workspace = workspace.resolve()
        env = _resolved_env(self.config.env, cache_dir=workspace.parents[3] / "npm-cache")
        args = list(self.config.args)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            client = _CollectingClient()
            command = _resolve_command(self.config.command)
            context = spawn_agent_process(client, command, *args, env=env, cwd=str(workspace))
            conn, _proc = await context.__aenter__()
            await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="superassist-plus", title="SuperAssist-Plus", version="0.1.0"),
            )
            session_kwargs: dict[str, Any] = {"cwd": str(workspace), "mcp_servers": []}
            if self.config.model:
                session_kwargs["model"] = self.config.model
            acp_session = await conn.new_session(**session_kwargs)
        except FileNotFoundError as exc:
            raise TeamSupervisorError(_missing_command_message(self.config)) from exc
        except Exception as exc:
            raise TeamSupervisorError(_format_start_error(self.config, exc)) from exc

        session = _ACPSession(
            context=context,
            conn=conn,
            session_id=acp_session.session_id,
            client=client,
        )
        self._sessions[key] = session
        return session

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future = self._loop_thread.submit(self._aclose())
        try:
            future.result(timeout=10)
        except Exception:
            logger.exception("Failed to close team agent '%s' cleanly", self.config.name)
        self._loop_thread.close()

    async def _aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            try:
                await session.context.__aexit__(None, None, None)
            except Exception:
                logger.exception("Failed to close ACP session for '%s'", self.config.name)


@dataclass
class _ACPSession:
    context: Any
    conn: Any
    session_id: str
    client: Any


class _AsyncLoopThread:
    def __init__(self, name: str) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


def _resolved_env(env: dict[str, str], *, cache_dir: Path | None = None) -> dict[str, str] | None:
    resolved = {key: (os.environ.get(value[1:], "") if value.startswith("$") else value) for key, value in env.items()}
    if cache_dir is not None:
        cache_dir = cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved.setdefault("npm_config_cache", str(cache_dir))
        resolved.setdefault("npm_config_prefer_offline", "true")
        resolved.setdefault("npm_config_audit", "false")
        resolved.setdefault("npm_config_fund", "false")
    return resolved or None


def _resolve_command(command: str) -> str:
    return shutil.which(command) or command


def _build_permission_response(options: list[Any], *, auto_approve: bool) -> Any:
    from acp import RequestPermissionResponse
    from acp.schema import AllowedOutcome, DeniedOutcome

    if auto_approve:
        for preferred_kind in ("allow_once", "allow_always"):
            for option in options:
                if getattr(option, "kind", None) != preferred_kind:
                    continue
                option_id = getattr(option, "option_id", None) or getattr(option, "optionId", None)
                if option_id is not None:
                    return RequestPermissionResponse(
                        outcome=AllowedOutcome(outcome="selected", optionId=option_id),
                    )

    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _missing_command_message(config: TeamAgentConfig) -> str:
    message = f"Team agent '{config.name}' command '{config.command}' was not found on PATH."
    if config.command == "codex-acp" and shutil.which("codex"):
        return (
            f"{message} The installed `codex` CLI does not speak ACP directly. "
            "Use an ACP adapter such as `npx -y @zed-industries/codex-acp` in agent_team.toml."
        )
    return f"{message} Install the agent binary or update agent_team.toml."


def _format_start_error(config: TeamAgentConfig, exc: Exception) -> str:
    text = str(exc)
    code = getattr(exc, "code", None)
    data = getattr(exc, "data", None)
    details = ""
    if code is not None:
        details += f" code={code}"
    if data is not None:
        details += f" data={data!r}"
    if not text or text == "Internal error":
        return (
            f"Failed to start team agent '{config.name}': {text or type(exc).__name__}.{details} "
            "Run the configured adapter command manually to inspect stderr, or install the ACP adapter locally."
        )
    if "EPERM" in text and "npm-cache" in text:
        return (
            f"Failed to start team agent '{config.name}': npm cache permission error. "
            "SuperAssist-Plus now sets npm_config_cache to a project-local cache; restart the runtime and try again."
        )
    return f"Failed to start team agent '{config.name}': {text}.{details}"
