from __future__ import annotations

from pathlib import Path

from superassist_plus.config import Settings
from superassist_plus.tools import default_tools
from superassist_plus.tools.files import delete_path, list_files, read_file, write_file
from superassist_plus.tools.web import web_fetch, web_search


def _tool(name: str):
    return next(tool for tool in default_tools() if tool.name == name)


def test_default_tools_include_file_and_web_tools() -> None:
    names = {tool.name for tool in default_tools()}

    assert {
        "list_files",
        "read_file",
        "write_file",
        "delete_path",
        "web_search",
        "web_fetch",
        "task",
    }.issubset(names)
    assert "current_time" not in names


def test_file_tools_are_workspace_scoped(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path / "data",
        SUPERASSIST_PLUS_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist_plus.tools.files.get_settings", lambda: settings)

    assert write_file.invoke({"path": "notes/todo.txt", "content": "hello"}) == "OK"
    assert read_file.invoke({"path": "notes/todo.txt"}) == "hello"
    assert "notes/todo.txt" in list_files.invoke({"path": "."})
    assert delete_path.invoke({"path": "notes/todo.txt"}) == "OK"
    assert "File not found" in read_file.invoke({"path": "notes/todo.txt"})


def test_file_tools_reject_path_escape(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path / "data",
        SUPERASSIST_PLUS_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist_plus.tools.files.get_settings", lambda: settings)

    try:
        write_file.invoke({"path": "../outside.txt", "content": "nope"})
    except PermissionError as exc:
        assert "outside the tool workspace" in str(exc)
    else:
        raise AssertionError("Path escape should raise PermissionError")


def test_network_tools_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path / "data",
        SUPERASSIST_PLUS_TOOL_NETWORK_ENABLED=False,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist_plus.tools.web.get_settings", lambda: settings)

    assert "Network tools are disabled" in web_search.invoke({"query": "test"})
    assert "Network tools are disabled" in web_fetch.invoke({"url": "https://example.com"})


def test_tool_lookup_by_name() -> None:
    assert _tool("read_file").name == "read_file"


def test_default_tools_can_exclude_task_for_subagents() -> None:
    assert "task" not in {tool.name for tool in default_tools(include_task=False)}
