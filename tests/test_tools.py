from __future__ import annotations

from pathlib import Path

from superassist.config import Settings
from superassist.tools import default_tools
from superassist.tools import web as web_module
from superassist.tools.files import delete_path, list_files, read_file, write_file
from superassist.tools.ocr import ocr_image
from superassist.tools.shell import shell
from superassist.tools.web import official_media_web_scope, web_fetch, web_search


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
        "image_search",
        "inspect_image",
        "present_images",
        "generate_image",
        "shell",
        "task",
    }.issubset(names)


def test_file_tools_are_workspace_scoped(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.files.get_settings", lambda: settings)

    assert write_file.invoke({"path": "notes/todo.txt", "content": "hello"}) == "OK"
    assert read_file.invoke({"path": "notes/todo.txt"}) == "hello"
    assert "notes/todo.txt" in list_files.invoke({"path": "."})
    assert delete_path.invoke({"path": "notes/todo.txt"}) == "OK"
    assert "File not found" in read_file.invoke({"path": "notes/todo.txt"})


def test_file_tools_reject_path_escape(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_WORKSPACE_DIR=tmp_path / "workspace",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.files.get_settings", lambda: settings)

    try:
        write_file.invoke({"path": "../outside.txt", "content": "nope"})
    except PermissionError as exc:
        assert "outside the tool workspace" in str(exc)
    else:
        raise AssertionError("Path escape should raise PermissionError")


def test_network_tools_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_NETWORK_ENABLED=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.web.get_settings", lambda: settings)

    assert "Network tools are disabled" in web_search.invoke({"query": "test"})
    assert "Network tools are disabled" in web_fetch.invoke({"url": "https://example.com"})


def test_web_search_parses_duckduckgo_lite_results(monkeypatch) -> None:
    body = """
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=x" class='result-link'>
      Example Result
    </a>
    <td class='result-snippet'>A <b>useful</b> snippet.</td>
    """

    monkeypatch.setattr(web_module, "_fetch_url", lambda url: (body, "text/html"))

    result = web_search.invoke({"query": "example", "max_results": 1})

    assert "Example Result" in result
    assert "https://example.com/page" in result
    assert "A useful snippet." in result


def test_web_search_falls_back_to_html_results(monkeypatch) -> None:
    empty_body = "<title>DuckDuckGo</title>"
    html_body = """
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org">
      HTML Result
    </a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org">
      HTML snippet.
    </a>
    """
    bodies = iter([(empty_body, "text/html"), (html_body, "text/html")])

    monkeypatch.setattr(web_module, "_fetch_url", lambda url: next(bodies))

    result = web_search.invoke({"query": "example", "max_results": 1})

    assert "HTML Result" in result
    assert "https://example.org" in result


def test_official_media_scope_filters_search_and_blocks_fetch(monkeypatch) -> None:
    body = """
    <a rel="nofollow" href="https://www.gov.cn/policy" class='result-link'>Official Result</a>
    <td class='result-snippet'>Official.</td>
    <a rel="nofollow" href="https://example.com/repost" class='result-link'>Repost</a>
    <td class='result-snippet'>Unofficial.</td>
    """
    requested: list[str] = []

    def fetch(url):
        requested.append(url)
        return body, "text/html"

    monkeypatch.setattr(web_module, "_fetch_url", fetch)

    with official_media_web_scope(["gov.cn"]):
        result = web_search.invoke({"query": "2026-08-01", "max_results": 5})
        blocked = web_fetch.invoke({"url": "https://example.com/repost"})

    assert "Official Result" in result
    assert "example.com" not in result
    assert "site%3Agov.cn" in requested[0]
    assert "outside the configured official-media" in blocked


def test_official_media_scope_enforces_shared_request_limit(monkeypatch) -> None:
    requested: list[str] = []

    def fetch(url):
        requested.append(url)
        return "plain text", "text/plain"

    monkeypatch.setattr(web_module, "_fetch_url", fetch)

    with official_media_web_scope(["gov.cn"], max_requests=2):
        first = web_fetch.invoke({"url": "https://www.gov.cn/a"})
        second = web_fetch.invoke({"url": "https://www.gov.cn/b"})
        blocked = web_fetch.invoke({"url": "https://www.gov.cn/c"})

    assert first == "plain text"
    assert second == "plain text"
    assert "request limit reached (2)" in blocked
    assert requested == ["https://www.gov.cn/a", "https://www.gov.cn/b"]


def test_tool_lookup_by_name() -> None:
    assert _tool("read_file").name == "read_file"


def test_default_tools_can_exclude_task_for_subagents() -> None:
    assert "task" not in {tool.name for tool in default_tools(include_task=False)}


def test_subagent_tools_can_exclude_lead_only_image_delivery_tools() -> None:
    names = {tool.name for tool in default_tools(include_task=False, include_images=False)}

    assert {"image_search", "inspect_image", "present_images", "generate_image"}.isdisjoint(names)


def test_shell_tool_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_SHELL_ENABLED=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.shell.get_settings", lambda: settings)

    assert "shell tool is disabled" in shell.invoke({"command": "echo hello"})


def test_shell_tool_runs_command_when_enabled(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_SHELL_ENABLED=True,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.shell.get_settings", lambda: settings)

    result = shell.invoke({"command": "Write-Output hello"})

    assert "hello" in result


def test_shell_tool_rejects_cwd_escape(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_SHELL_ENABLED=True,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.shell.get_settings", lambda: settings)

    result = shell.invoke({"command": "echo hello", "cwd": ".."})

    assert "outside the project root" in result


def test_shell_tool_blocks_destructive_commands(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_SHELL_ENABLED=True,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.shell.get_settings", lambda: settings)

    result = shell.invoke({"command": "Remove-Item . -Recurse -Force"})

    assert "blocked" in result


def test_ocr_tool_supports_sync_invocation(tmp_path: Path, monkeypatch) -> None:
    """The agent graph invokes tools synchronously; async tools silently fail."""
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    monkeypatch.setattr("superassist.tools.files.get_settings", lambda: settings)
    monkeypatch.setattr("superassist.tools.ocr.get_settings", lambda: settings)

    result = ocr_image.invoke({"path": "图片/20260101/missing.jpg"})

    assert "not found" in result


def test_shell_cwd_falls_back_to_tool_workspace(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_TOOL_SHELL_ENABLED=True,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    workspace_dir = settings.resolved_tool_workspace_dir / "probe_workspace_dir"
    workspace_dir.mkdir(parents=True)
    monkeypatch.setattr("superassist.tools.shell.get_settings", lambda: settings)

    result = shell.invoke({"command": "Write-Output hello", "cwd": "probe_workspace_dir"})

    assert "hello" in result
    assert "does not exist" not in result


def test_orientation_candidates_run_all_angles_in_worker_threads(monkeypatch) -> None:
    import threading

    from PIL import Image

    from superassist.tools.ocr import OCRCandidate, orientation_candidates

    seen: list[tuple[int, str]] = []

    def fake_candidate(_engine, _image, angle):
        seen.append((angle, threading.current_thread().name))
        return OCRCandidate(angle=angle, lines=(f"angle-{angle}",), character_count=8, average_confidence=0.9)

    monkeypatch.setattr("superassist.tools.ocr._candidate", fake_candidate)
    monkeypatch.setattr("superassist.tools.ocr._get_engine", lambda: object())

    candidates = orientation_candidates(Image.new("RGB", (8, 8)))

    assert [candidate.angle for candidate in candidates] == [0, 90, 180, 270]
    assert all(name.startswith("ocr-angle") for _, name in seen)
