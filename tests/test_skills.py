from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist.middlewares import DynamicContextMiddleware, ToolEventMiddleware
from superassist.config import Settings
from superassist.skills import (
    active_skill_names,
    build_available_skills_section,
    build_loaded_skills_section,
    list_public_skills,
)
from superassist.tools.files import delete_path, read_file, write_file


def test_public_deep_research_skill_is_discovered() -> None:
    skills = list_public_skills()
    deep_research = next(skill for skill in skills if skill.name == "deep-research")

    assert deep_research.description.startswith("Use this skill instead of WebSearch")
    assert deep_research.virtual_file_path == "/mnt/skills/public/deep-research/SKILL.md"


def test_available_skill_prompt_contains_metadata_not_full_content() -> None:
    section = build_available_skills_section()

    assert "<name>deep-research</name>" in section
    assert "<location>/mnt/skills/public/deep-research/SKILL.md</location>" in section
    assert "# Deep Research Skill" not in section


def test_skill_discovery_is_one_level_and_does_not_register_references() -> None:
    names = {skill.name for skill in list_public_skills()}

    assert "gongkao-huasheng13" in names
    assert "ziliao-fenxi" not in names


def test_read_file_can_read_skill_virtual_path() -> None:
    content = read_file.invoke({"path": "/mnt/skills/public/deep-research/SKILL.md"})

    assert "# Deep Research Skill" in content
    assert "Research Methodology" in content


def test_write_and_delete_do_not_mutate_skill_virtual_path(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.files.get_settings", lambda: settings)

    for tool in (write_file, delete_path):
        try:
            if tool.name == "write_file":
                tool.invoke({"path": "/mnt/skills/public/deep-research/SKILL.md", "content": "nope"})
            else:
                tool.invoke({"path": "/mnt/skills/public/deep-research/SKILL.md"})
        except PermissionError as exc:
            assert "outside the tool workspace" in str(exc)
        else:
            raise AssertionError(f"{tool.name} should reject /mnt/skills mutation")


def test_reading_skill_records_timed_activation_and_full_content_can_be_injected() -> None:
    middleware = ToolEventMiddleware(clock=lambda: 1000.0)

    class DummyTool:
        name = "read_file"

    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "id": "call_1",
            "args": {"path": "/mnt/skills/public/deep-research/SKILL.md"},
        },
        tool=DummyTool(),
        state={"tool_events": [], "loaded_skills": [], "skill_activations": {}},
        runtime=None,
    )

    middleware.wrap_tool_call(
        request,
        lambda _request: ToolMessage(content="# Deep Research Skill", tool_call_id="call_1", name="read_file"),
    )

    assert request.state["loaded_skills"] == ["deep-research"]
    assert request.state["skill_activations"] == {"deep-research": 1000.0}
    assert "# Deep Research Skill" in build_loaded_skills_section(request.state["loaded_skills"])


def test_reading_skill_reference_refreshes_activation() -> None:
    middleware = ToolEventMiddleware(clock=lambda: 1200.0)

    class DummyTool:
        name = "read_file"

    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "id": "call_2",
            "args": {"path": "/mnt/skills/public/huasheng13/references/ziliao-fenxi.md"},
        },
        tool=DummyTool(),
        state={
            "tool_events": [],
            "loaded_skills": ["gongkao-huasheng13"],
            "skill_activations": {"gongkao-huasheng13": 1000.0},
        },
        runtime=None,
    )

    middleware.wrap_tool_call(
        request,
        lambda _request: ToolMessage(content="reference", tool_call_id="call_2", name="read_file"),
    )

    assert request.state["skill_activations"] == {"gongkao-huasheng13": 1200.0}


def test_dynamic_context_injects_available_and_loaded_skills() -> None:
    middleware = DynamicContextMiddleware(300, clock=lambda: 1100.0)

    class Request:
        state = {
            "user_id": "u",
            "thread_id": "t",
            "memory_recall": {},
            "loaded_skills": ["deep-research"],
            "skill_activations": {"deep-research": 1000.0},
            "active_skills_at_turn_start": ["deep-research"],
        }
        messages = [HumanMessage(content="Research AI")]

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)
    content = str(merged[0].content)

    assert "<available_skills>" in content
    assert '<skill name="deep-research">' in content
    assert "# Deep Research Skill" in content


def test_expired_skill_keeps_index_but_drops_full_content() -> None:
    middleware = DynamicContextMiddleware(300, clock=lambda: 1300.0)

    class Request:
        state = {
            "user_id": "u",
            "thread_id": "t",
            "memory_recall": {},
            "loaded_skills": ["deep-research"],
            "skill_activations": {"deep-research": 1000.0},
            "active_skills_at_turn_start": ["deep-research"],
        }
        messages = [HumanMessage(content="hello")]

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)
    content = str(merged[0].content)

    assert "<available_skills>" in content
    assert "<name>deep-research</name>" in content
    assert '<skill name="deep-research">' not in content
    assert "# Deep Research Skill" not in content
    assert active_skill_names({"deep-research": 1000.0}, 300, now=1300.0) == []


def test_newly_loaded_skill_is_not_duplicated_in_same_turn_context() -> None:
    middleware = DynamicContextMiddleware(300, clock=lambda: 1001.0)

    class Request:
        state = {
            "user_id": "u",
            "thread_id": "t",
            "memory_recall": {},
            "loaded_skills": ["deep-research"],
            "skill_activations": {"deep-research": 1000.0},
            "active_skills_at_turn_start": [],
        }
        messages = [HumanMessage(content="skill tool result is already in this turn")]

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)
    content = str(merged[0].content)

    assert "<name>deep-research</name>" in content
    assert '<skill name="deep-research">' not in content
