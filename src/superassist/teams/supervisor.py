"""High-level team supervisor.

The supervisor owns:

* the parsed ``agent_team.toml`` membership
* the per-thread JSONL ledger (hash-chained, HMAC-signed, file-locked)
* a pool of long-lived ACP team-member processes (one per configured agent)

It does **not** speak ACP directly; that work lives in
:mod:`superassist.acp_client`. Each ``TeamMember`` here is a thin holder of
an :class:`AsyncLoopThread` plus a per-(thread, workspace) cache of
:class:`ACPSession`.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from superassist.acp_client import (
    ACPClientError,
    ACPSession,
    ACPSpawnRequest,
    AsyncLoopThread,
    PermissionPolicy,
    open_session,
    resolve_env,
)
from superassist.config import Settings, get_settings

from .config import AgentTeamConfig, TeamAgentConfig
from .ledger import LedgerTamperError, TeamLedger

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


class TeamSupervisorError(RuntimeError):
    """Raised when the team supervisor cannot service a request."""


class TeamSupervisor:
    """Route ``team_task`` calls to a configured ACP agent."""

    def __init__(
        self,
        config: AgentTeamConfig,
        *,
        settings: Settings | None = None,
        bus: TeamLedger | None = None,
        member_factory: type[TeamMember] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        self.bus = bus or TeamLedger(self.settings.data_dir / "teams" / "default")
        self._member_factory = member_factory or TeamMember
        self._members: dict[str, TeamMember] = {}
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
            {"task_id": task_record["id"], "description": description, "prompt": prompt},
        )

        workspace = self.bus.workspace_dir(thread_id, agent)
        try:
            response = self._member(config).invoke(thread_id=thread_id, prompt=prompt, workspace=workspace)
        except ACPClientError as exc:
            raise TeamSupervisorError(str(exc)) from exc

        self.bus.append_raw(thread_id, agent, {"task_id": task_record["id"], "response": response})
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

    def _member(self, config: TeamAgentConfig) -> "TeamMember":
        with self._lock:
            member = self._members.get(config.name)
            if member is None:
                member = self._member_factory(config)
                self._members[config.name] = member
            return member


class TeamMember:
    """Long-lived holder of one configured team agent's ACP sessions."""

    def __init__(self, config: TeamAgentConfig) -> None:
        self.config = config
        self.last_used = time.monotonic()
        self._loop = AsyncLoopThread(f"team-agent-{config.name}")
        self._sessions: dict[tuple[str, str], ACPSession] = {}
        self._closed = False

    def invoke(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        if self._closed:
            raise ACPClientError(f"Team agent '{self.config.name}' is closed.")
        self.last_used = time.monotonic()
        return self._loop.submit(self._aprompt(thread_id=thread_id, prompt=prompt, workspace=workspace)).result()

    async def _aprompt(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        session = await self._ensure_session(thread_id=thread_id, workspace=workspace)
        return await session.prompt(prompt)

    async def _ensure_session(self, *, thread_id: str, workspace: Path) -> ACPSession:
        key = (thread_id, str(workspace))
        existing = self._sessions.get(key)
        if existing is not None:
            return existing
        request = ACPSpawnRequest(
            name=self.config.name,
            command=self.config.command,
            args=list(self.config.args),
            cwd=workspace,
            env=resolve_env(self.config.env, cache_dir=workspace.parents[3] / "npm-cache"),
            model=self.config.model,
            permission_policy=PermissionPolicy.AUTO_APPROVE if self.config.auto_approve_permissions else PermissionPolicy.DENY,
        )
        session = await open_session(request)
        self._sessions[key] = session
        return session

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.submit(self._aclose()).result(timeout=10)
        except Exception:
            logger.exception("Failed to close team agent '%s' cleanly", self.config.name)
        self._loop.close()

    async def _aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            await session.close()


# -- Backwards-compatible aliases for tests that import the old names ----

TeamMemberProcess = TeamMember


def _resolve_command(command: str) -> str:
    """Compatibility shim: previously exposed by supervisor; now backed by shutil.which."""

    return shutil.which(command) or command


__all__ = [
    "TeamMember",
    "TeamMemberProcess",
    "TeamSupervisor",
    "TeamSupervisorError",
    "TeamTaskResult",
    "_resolve_command",
    "get_team_supervisor",
    "set_team_supervisor",
]
