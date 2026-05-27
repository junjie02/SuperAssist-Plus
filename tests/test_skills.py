from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from superassist_plus.agent.middleware import DynamicContextMiddleware, ToolEventMiddleware
from superassist_plus.config import Settings
from superassist_plus.skills import (
    build_available_skills_section,
    build_loaded_skills_section,
    list_public_skills,
)
from superassist_plus.tools.files import delete_path, read_file, write_file


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


def test_read_file_can_read_skill_virtual_path() -> None:
    content = read_file.invoke({"path": "/mnt/skills/public/deep-research/SKILL.md"})

    assert "# Deep Research Skill" in content
    assert "Research Methodology" in content


def test_write_and_delete_do_not_mutate_skill_virtual_path(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path / "data",
        SUPERASSIST_PLUS_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist_plus.tools.files.get_settings", lambda: settings)

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


def test_reading_skill_records_loaded_skill_and_full_content_can_be_injected() -> None:
    middleware = ToolEventMiddleware()

    class DummyTool:
        name = "read_file"

    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "id": "call_1",
            "args": {"path": "/mnt/skills/public/deep-research/SKILL.md"},
        },
        tool=DummyTool(),
        state={"tool_events": [], "loaded_skills": []},
        runtime=None,
    )

    middleware.wrap_tool_call(
        request,
        lambda _request: ToolMessage(content="# Deep Research Skill", tool_call_id="call_1", name="read_file"),
    )

    assert request.state["loaded_skills"] == ["deep-research"]
    assert "# Deep Research Skill" in build_loaded_skills_section(request.state["loaded_skills"])


def test_dynamic_context_injects_available_and_loaded_skills() -> None:
    middleware = DynamicContextMiddleware("Base")

    class Request:
        state = {
            "user_id": "u",
            "thread_id": "t",
            "memory_recall": {},
            "loaded_skills": ["deep-research"],
        }
        messages = [HumanMessage(content="Research AI")]

        def override(self, **kwargs):
            return kwargs["messages"]

    merged = middleware.wrap_model_call(Request(), lambda value: value)
    content = str(merged[0].content)

    assert "<available_skills>" in content
    assert '<skill name="deep-research">' in content
    assert "# Deep Research Skill" in content
