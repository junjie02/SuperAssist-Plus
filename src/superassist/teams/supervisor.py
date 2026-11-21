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
from concurrent.futures import Future, ThreadPoolExecutor
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
    status: str = "completed"


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
        self._futures: dict[str, Future[None]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(config.agents)),
            thread_name_prefix="team-task",
        )
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

        task_id = str(task_record["id"])
        if wait:
            return self._execute_task(task_record, config, description, prompt)

        future = self._executor.submit(self._execute_async, task_record, config, description, prompt)
        with self._lock:
            self._futures[task_id] = future
        future.add_done_callback(lambda _: self._forget_future(task_id))
        return TeamTaskResult(
            agent=agent,
            task_id=task_id,
            result="",
            ledger_id=task_id,
            status="queued",
        )

    def poll(self, task_id: str, *, thread_id: str) -> TeamTaskResult:
        """Read a task's durable status from the ledger."""
        records = self.bus.read_ledger(thread_id)
        task = next(
            (record for record in records if record.get("id") == task_id and record.get("kind") == "task"),
            None,
        )
        if task is None:
            raise TeamSupervisorError(f"Unknown team task '{task_id}'")
        child = next(
            (
                record
                for record in records
                if task_id in (record.get("parent_ids") or []) and record.get("kind") in {"result", "error"}
            ),
            None,
        )
        if child is None:
            return TeamTaskResult(
                agent=str(task["recipient"]),
                task_id=task_id,
                result="",
                ledger_id=task_id,
                status=(
                    "running"
                    if any(
                        record.get("kind") == "task_started" and task_id in (record.get("parent_ids") or [])
                        for record in records
                    )
                    else "queued"
                ),
            )
        status = "failed" if child.get("kind") == "error" else "completed"
        return TeamTaskResult(
            agent=str(task["recipient"]),
            task_id=task_id,
            result=str(child.get("body") or ""),
            ledger_id=str(child["id"]),
            status=status,
        )

    def _execute_async(
        self,
        task_record: dict[str, object],
        config: TeamAgentConfig,
        description: str,
        prompt: str,
    ) -> None:
        self._execute_task(task_record, config, description, prompt, mark_started=True)

    def _execute_task(
        self,
        task_record: dict[str, object],
        config: TeamAgentConfig,
        description: str,
        prompt: str,
        *,
        mark_started: bool = False,
    ) -> TeamTaskResult:
        task_id = str(task_record["id"])
        thread_id = str(task_record["thread_id"])
        if mark_started:
            self.bus.append_message(
                thread_id,
                sender="superassist",
                recipient=config.name,
                kind="task_started",
                body="",
                parent_ids=[task_id],
            )
        workspace = self.bus.workspace_dir(thread_id, config.name)
        try:
            response = self._member(config).invoke(thread_id=thread_id, prompt=prompt, workspace=workspace)
        except Exception as exc:
            self.bus.append_message(
                thread_id,
                sender=config.name,
                recipient="superassist",
                kind="error",
                body=str(exc),
                parent_ids=[task_id],
                extra={"description": description, "error_type": type(exc).__name__},
            )
            if not mark_started:
                if isinstance(exc, ACPClientError):
                    raise TeamSupervisorError(str(exc)) from exc
                raise
            return TeamTaskResult(config.name, task_id, str(exc), task_id, status="failed")

        self.bus.append_raw(thread_id, config.name, {"task_id": task_id, "response": response})
        result_record = self.bus.append_message(
            thread_id,
            sender=config.name,
            recipient="superassist",
            kind="result",
            body=response,
            parent_ids=[task_id],
            extra={"description": description},
        )
        return TeamTaskResult(config.name, task_id, response, str(result_record["id"]))

    def _forget_future(self, task_id: str) -> None:
        with self._lock:
            self._futures.pop(task_id, None)

    def sweep_idle(self) -> None:
        if self.config.idle_ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self.config.idle_ttl_seconds
        with self._lock:
            idle = [name for name, member in self._members.items() if member.last_used < cutoff]
            for name in idle:
                self._members.pop(name).close()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
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
        self._invoke_lock = threading.Lock()

    def invoke(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        with self._invoke_lock:
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
        with self._invoke_lock:
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
__all__ = [
    "TeamMember",
    "TeamSupervisor",
    "TeamSupervisorError",
    "TeamTaskResult",
    "get_team_supervisor",
    "set_team_supervisor",
]
