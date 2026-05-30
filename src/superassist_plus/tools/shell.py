from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from superassist_plus.config import PROJECT_ROOT, get_settings


_DANGEROUS_PATTERNS = [
    re.compile(r"\bRemove-Item\b.*\s-Recurse\b.*\s-Force\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+--\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sq]\b", re.IGNORECASE),
    re.compile(r"\brmdir\s+/s\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
]


@tool("shell")
def shell(command: str, cwd: str = ".") -> str:
    """Run a non-interactive shell command in the project workspace.

    Args:
        command: Command to execute. Destructive commands are blocked.
        cwd: Relative directory under the project root. Defaults to ".".
    """

    settings = get_settings()
    if not settings.tool_shell_enabled:
        return "Error: shell tool is disabled. Set SUPERASSIST_PLUS_TOOL_SHELL_ENABLED=true to enable it."
    blocked = _blocked_reason(command)
    if blocked:
        return f"Error: shell command blocked: {blocked}"
    try:
        workdir = _resolve_cwd(cwd)
    except PermissionError as exc:
        return f"Error: {exc}"
    shell_path = _get_shell()
    timeout = max(1, min(settings.tool_shell_timeout_seconds, 600))
    try:
        result = subprocess.run(
            _shell_args(shell_path, command),
            cwd=str(workdir),
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except FileNotFoundError as exc:
        return f"Error: shell executable not found: {exc}"
    output = result.stdout or ""
    if result.stderr:
        output = f"{output}\nStd Error:\n{result.stderr}" if output else result.stderr
    if result.returncode != 0:
        output = f"{output}\nExit Code: {result.returncode}" if output else f"Exit Code: {result.returncode}"
    return _truncate(output or "(no output)", max_chars=settings.tool_shell_output_max_chars)


def _resolve_cwd(cwd: str) -> Path:
    raw = Path(cwd)
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(f"cwd is outside the project root: {cwd}") from None
    if not resolved.exists():
        raise PermissionError(f"cwd does not exist: {cwd}")
    if not resolved.is_dir():
        raise PermissionError(f"cwd is not a directory: {cwd}")
    return resolved


def _get_shell() -> str:
    if os.name == "nt":
        for candidate in (
            "pwsh",
            "pwsh.exe",
            "powershell",
            "powershell.exe",
            str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
            "cmd.exe",
        ):
            found = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
            if found and Path(found).exists():
                return found
        raise FileNotFoundError("No PowerShell or cmd.exe found")
    for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh", "sh"):
        found = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
        if found and Path(found).exists():
            return found
    raise FileNotFoundError("No POSIX shell found")


def _shell_args(shell_path: str, command: str) -> list[str]:
    name = Path(shell_path).name.lower()
    if name in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return [shell_path, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
    if name in {"cmd", "cmd.exe"}:
        return [shell_path, "/c", command]
    return [shell_path, "-c", command]


def _blocked_reason(command: str) -> str | None:
    normalized = command.strip()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(normalized):
            return "destructive command requires manual execution"
    return None


def _truncate(output: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(output) <= max_chars:
        return output
    half = max_chars // 2
    return f"{output[:half]}\n... [truncated {len(output) - max_chars} chars] ...\n{output[-half:]}"
