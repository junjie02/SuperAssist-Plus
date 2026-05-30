from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from superassist_plus.agent.runtime import AgentRuntime, team_prompt_section
from superassist_plus.agent_teams import AgentTeamConfig, TeamSupervisor, get_team_supervisor, set_team_supervisor
from superassist_plus.agent_teams.bus import JsonlBus, LedgerTamperError
from superassist_plus.agent_teams.config import AgentTeamConfigError, TeamAgentConfig
from superassist_plus.agent_teams.context import team_thread_context
from superassist_plus.agent_teams.supervisor import _resolve_command
from superassist_plus.config import Settings
from superassist_plus.tools import default_tools
from superassist_plus.tools.team import team_task


def test_agent_team_config_loads_toml(tmp_path: Path) -> None:
    path = tmp_path / "agent_team.toml"
    path.write_text(
        """
enabled = true
idle_ttl_seconds = 123

[[agents]]
name = "codex"
command = "npx"
args = ["-y", "@zed-industries/codex-acp"]
description = "Codex"
""".strip(),
        encoding="utf-8",
    )

    config = AgentTeamConfig.from_file(path)

    assert config.enabled is True
    assert config.idle_ttl_seconds == 123
    assert config.agents_by_name["codex"].args == ["-y", "@zed-industries/codex-acp"]


def test_agent_team_config_missing_file_is_disabled(tmp_path: Path) -> None:
    config = AgentTeamConfig.from_file(tmp_path / "missing.toml")

    assert config.enabled is False
    assert config.agents == []


def test_agent_team_config_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "agent_team.toml"
    path.write_text(
        """
enabled = true

[[agents]]
name = "codex"
command = "npx"
description = "Codex"

[[agents]]
name = "codex"
command = "npx"
description = "Duplicate"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(AgentTeamConfigError, match="duplicate agent"):
        AgentTeamConfig.from_file(path)


def test_agent_team_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "agent_team.toml"
    path.write_text(
        """
enabled = true

[[agents]]
name = "codex"
description = "Codex"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(AgentTeamConfigError, match="command"):
        AgentTeamConfig.from_file(path)


def test_jsonl_bus_appends_and_validates_hash_chain(tmp_path: Path) -> None:
    bus = JsonlBus(tmp_path)

    first = bus.append_message("t", sender="superassist", recipient="codex", kind="task", body="do it")
    second = bus.append_message("t", sender="codex", recipient="superassist", kind="result", body="done", parent_ids=[first["id"]])

    records = bus.read_ledger("t")
    assert [record["seq"] for record in records] == [1, 2]
    assert second["prev_hash"] == first["hash"]
    assert records[0]["body"] == "do it"
    assert records[1]["parent_ids"] == [first["id"]]


def test_jsonl_bus_detects_modified_record(tmp_path: Path) -> None:
    bus = JsonlBus(tmp_path)
    bus.append_message("t", sender="superassist", recipient="codex", kind="task", body="do it")
    path = bus.ledger_path("t")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    record["body"] = "tampered"
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(LedgerTamperError, match="hash mismatch"):
        bus.read_ledger("t")


def test_jsonl_bus_detects_deleted_middle_record(tmp_path: Path) -> None:
    bus = JsonlBus(tmp_path)
    bus.append_message("t", sender="superassist", recipient="codex", kind="task", body="one")
    bus.append_message("t", sender="codex", recipient="superassist", kind="result", body="two")
    bus.append_message("t", sender="superassist", recipient="codex", kind="task", body="three")
    path = bus.ledger_path("t")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    with pytest.raises(LedgerTamperError):
        bus.read_ledger("t")


def test_jsonl_bus_concurrent_appends_are_complete_and_ordered(tmp_path: Path) -> None:
    bus = JsonlBus(tmp_path)

    def append(index: int) -> None:
        bus.append_message("t", sender="superassist", recipient="codex", kind="task", body=f"task {index}")

    threads = [threading.Thread(target=append, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = bus.read_ledger("t")
    assert len(records) == 20
    assert [record["seq"] for record in records] == list(range(1, 21))


def test_jsonl_bus_writes_inbox_and_raw_outbox_separately(tmp_path: Path) -> None:
    bus = JsonlBus(tmp_path)
    bus.append_inbox("t", "codex", {"prompt": "do it"})
    bus.append_raw("t", "codex", {"response": "done"})

    assert "do it" in bus.inbox_path("t", "codex").read_text(encoding="utf-8")
    assert "done" in bus.outbox_path("t", "codex").read_text(encoding="utf-8")
    assert not bus.ledger_path("t").exists()


class FakeMember:
    instances: list["FakeMember"] = []

    def __init__(self, config: TeamAgentConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, str, Path]] = []
        self.last_used = 0.0
        self.closed = False
        FakeMember.instances.append(self)

    def invoke(self, *, thread_id: str, prompt: str, workspace: Path) -> str:
        self.calls.append((thread_id, prompt, workspace))
        return f"{self.config.name} saw {len(self.calls)}: {prompt}"

    def close(self) -> None:
        self.closed = True


def _team_config(enabled: bool = True) -> AgentTeamConfig:
    return AgentTeamConfig(
        enabled=enabled,
        agents=[
            TeamAgentConfig(
                name="codex",
                command="fake",
                description="Codex fake",
            )
        ],
    )


def test_team_supervisor_invokes_and_reuses_member(tmp_path: Path) -> None:
    FakeMember.instances.clear()
    supervisor = TeamSupervisor(
        _team_config(),
        settings=Settings(SUPERASSIST_PLUS_DATA_DIR=tmp_path, SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash"),
        member_factory=FakeMember,
    )

    first = supervisor.invoke("codex", thread_id="t", description="one", prompt="first")
    second = supervisor.invoke("codex", thread_id="t", description="two", prompt="second")

    assert first.result == "codex saw 1: first"
    assert second.result == "codex saw 2: second"
    assert len(FakeMember.instances) == 1
    assert len(FakeMember.instances[0].calls) == 2
    ledger = supervisor.bus.read_ledger("t")
    assert [record["kind"] for record in ledger] == ["task", "result", "task", "result"]


def test_team_supervisor_unknown_agent_returns_clear_error(tmp_path: Path) -> None:
    supervisor = TeamSupervisor(
        _team_config(),
        settings=Settings(SUPERASSIST_PLUS_DATA_DIR=tmp_path, SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash"),
        member_factory=FakeMember,
    )

    with pytest.raises(Exception, match="Unknown team agent"):
        supervisor.invoke("missing", thread_id="t", description="bad", prompt="no")


def test_team_task_uses_active_thread_context(tmp_path: Path) -> None:
    FakeMember.instances.clear()
    supervisor = TeamSupervisor(
        _team_config(),
        settings=Settings(SUPERASSIST_PLUS_DATA_DIR=tmp_path, SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash"),
        member_factory=FakeMember,
    )
    set_team_supervisor(supervisor)
    try:
        with team_thread_context("thread_1"):
            result = team_task.invoke({"agent": "codex", "description": "demo", "prompt": "do it", "wait": True})
    finally:
        set_team_supervisor(None)
        supervisor.close()

    assert "Team task succeeded via codex" in result
    assert supervisor.bus.read_ledger("thread_1")


def test_team_task_disabled_when_no_supervisor() -> None:
    set_team_supervisor(None)

    result = team_task.invoke({"agent": "codex", "description": "demo", "prompt": "do it", "wait": True})

    assert "Agent teams are disabled" in result


def test_default_tools_can_include_team_task() -> None:
    names = {tool.name for tool in default_tools(include_team_task=True)}

    assert "team_task" in names


def test_runtime_registers_team_supervisor_from_config(tmp_path: Path, monkeypatch) -> None:
    config = _team_config()
    monkeypatch.setattr("superassist_plus.agent.runtime.AgentTeamConfig.from_file", lambda: config)
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_ENABLE_TOOLS=True,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    try:
        assert runtime.team_supervisor is not None
        assert runtime.team_supervisor.enabled is True
        assert get_team_supervisor() is runtime.team_supervisor
    finally:
        runtime.close()


def test_runtime_skips_team_supervisor_when_config_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("superassist_plus.agent.runtime.AgentTeamConfig.from_file", lambda: _team_config(enabled=False))
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_ENABLE_TOOLS=True,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    runtime = AgentRuntime(settings)
    try:
        assert runtime.team_supervisor is None
    finally:
        runtime.close()


def test_team_prompt_section_lists_agents() -> None:
    section = team_prompt_section("- codex: Codex fake")

    assert "team_task" in section
    assert "codex" in section


def test_resolve_command_uses_path_executable(monkeypatch) -> None:
    monkeypatch.setattr("superassist_plus.agent_teams.supervisor.shutil.which", lambda command: "E:/nodejs/npx.CMD")

    assert _resolve_command("npx") == "E:/nodejs/npx.CMD"
