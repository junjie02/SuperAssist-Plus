from __future__ import annotations

import logging

from langchain_core.tools import tool

from superassist_plus.agent_teams import get_team_supervisor
from superassist_plus.agent_teams.bus import LedgerTamperError
from superassist_plus.agent_teams.context import current_team_thread_id
from superassist_plus.agent_teams.supervisor import TeamSupervisorError
from superassist_plus.observability import trace_extra, traceable

logger = logging.getLogger(__name__)


@tool("team_task")
def team_task(agent: str, description: str, prompt: str, wait: bool = True) -> str:
    """Delegate work to a persistent external team agent and wait for its result.

    Args:
        agent: Team agent name from agent_team.toml.
        description: Short 3-8 word description for tracking.
        prompt: Full task instructions for the team agent.
        wait: Must be true in v1.
    """

    return run_team_task(agent=agent, description=description, prompt=prompt, wait=wait)


def run_team_task(agent: str, description: str, prompt: str, wait: bool = True) -> str:
    return _run_team_task_traced(
        agent=agent,
        description=description,
        prompt=prompt,
        wait=wait,
        **trace_extra(
            metadata={
                "agent": agent,
                "description": description,
                "prompt_preview": prompt,
                "wait": wait,
            },
            tags=["tool", "team_task", agent],
        ),
    )


@traceable(name="team_task.dispatch", run_type="tool")
def _run_team_task_traced(agent: str, description: str, prompt: str, wait: bool = True) -> str:
    supervisor = get_team_supervisor()
    if supervisor is None or not supervisor.enabled:
        return "Error: Agent teams are disabled. Enable them in agent_team.toml."
    thread_id = current_team_thread_id()
    if not thread_id:
        return "Error: team_task requires an active SuperAssist thread context."
    try:
        result = supervisor.invoke(
            agent,
            thread_id=thread_id,
            description=description,
            prompt=prompt,
            wait=wait,
        )
    except LedgerTamperError as exc:
        logger.warning("Team ledger tamper check failed: %s", exc)
        return f"Error: Team ledger integrity check failed for this thread: {exc}"
    except TeamSupervisorError as exc:
        logger.warning("Team task failed: %s", exc)
        return f"Error: {exc}"
    except Exception as exc:
        logger.exception("Unexpected team task failure")
        return f"Error: team_task failed with {type(exc).__name__}: {exc}"
    return f"Team task succeeded via {result.agent}. Result: {result.result}"
