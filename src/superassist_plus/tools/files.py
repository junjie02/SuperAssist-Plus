from __future__ import annotations

import shutil
from pathlib import Path

from langchain_core.tools import tool

from superassist_plus.config import get_settings
from superassist_plus.skills import resolve_skill_virtual_path


def _workspace_root() -> Path:
    root = get_settings().resolved_tool_workspace_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_workspace_path(path: str) -> Path:
    root = _workspace_root()
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(f"Path is outside the tool workspace: {path}") from None
    return resolved


def _resolve_read_path(path: str) -> Path:
    skill_path = resolve_skill_virtual_path(path)
    if skill_path is not None:
        return skill_path
    return _resolve_workspace_path(path)


@tool("list_files")
def list_files(path: str = ".", max_depth: int = 2) -> str:
    """List files and directories under the tool workspace.

    Args:
        path: Relative path inside the tool workspace. Defaults to ".".
        max_depth: Maximum directory depth to show. Defaults to 2.
    """

    target = _resolve_workspace_path(path)
    if not target.exists():
        return f"Error: Path not found: {path}"
    if target.is_file():
        return target.name

    root = _workspace_root()
    max_depth = max(0, min(max_depth, 8))
    lines: list[str] = []
    for item in sorted(target.rglob("*")):
        relative = item.relative_to(target)
        if len(relative.parts) > max_depth:
            continue
        suffix = "/" if item.is_dir() else ""
        lines.append(str(item.relative_to(root)).replace("\\", "/") + suffix)
        if len(lines) >= 300:
            lines.append("... [truncated]")
            break
    return "\n".join(lines) if lines else "(empty)"


@tool("read_file")
def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a UTF-8 text file from the tool workspace.

    Args:
        path: Relative path inside the tool workspace.
        start_line: Optional 1-based starting line.
        end_line: Optional 1-based ending line, inclusive.
    """

    target = _resolve_read_path(path)
    if not target.exists():
        return f"Error: File not found: {path}"
    if not target.is_file():
        return f"Error: Path is not a file: {path}"
    content = target.read_text(encoding="utf-8", errors="replace")
    if start_line is not None and end_line is not None:
        lines = content.splitlines()
        content = "\n".join(lines[max(0, start_line - 1) : max(0, end_line)])
    return content[:50000]


@tool("write_file")
def write_file(path: str, content: str, append: bool = False) -> str:
    """Create or write a UTF-8 text file inside the tool workspace.

    Args:
        path: Relative path inside the tool workspace.
        content: Text content to write.
        append: Append instead of overwrite. Defaults to false.
    """

    target = _resolve_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return "OK"


@tool("delete_path")
def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or an empty directory from the tool workspace.

    Args:
        path: Relative path inside the tool workspace.
        recursive: Delete a directory tree. Defaults to false.
    """

    target = _resolve_workspace_path(path)
    if not target.exists():
        return f"Error: Path not found: {path}"
    if target.is_dir():
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
    else:
        target.unlink()
    return "OK"
