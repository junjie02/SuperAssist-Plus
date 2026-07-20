"""ACP (Agent Client Protocol) client utilities.

This subpackage owns everything that talks to an external agent through the
``agent-client-protocol`` library: spawning the process, holding sessions,
and applying the permission policy. Higher-level orchestration (which
agents exist, ledger persistence, ``team_task`` tool routing) lives in
``superassist.teams``.
"""

from superassist.acp_client.errors import ACPClientError, format_start_error, missing_command_message
from superassist.acp_client.loop import AsyncLoopThread
from superassist.acp_client.permissions import PermissionPolicy, build_permission_response
from superassist.acp_client.process import ACPSession, ACPSpawnRequest, open_session, resolve_env

__all__ = [
    "ACPClientError",
    "ACPSession",
    "ACPSpawnRequest",
    "AsyncLoopThread",
    "PermissionPolicy",
    "build_permission_response",
    "format_start_error",
    "missing_command_message",
    "open_session",
    "resolve_env",
]
