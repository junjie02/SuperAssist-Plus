# Tools Module Technical Documentation

IMPORTANT: Any change to available tools, tool schemas, tool side effects, or
tool registration must update this document.

## Purpose

The `tools` module owns LangChain `BaseTool` adapters exposed to the inner
LangChain agent. Tools should be simple, typed, testable units.

## Files

- `basic.py`: currently provides the `echo` smoke-test tool.
- `files.py`: workspace-scoped file listing, reading, writing, and deletion,
  plus read-only `/mnt/skills` access for built-in skill files.
- `web.py`: DuckDuckGo HTML search and HTTP/HTTPS fetch helpers.
- `shell.py`: gated local shell command execution scoped to the project root.
- `team.py`: persistent external agent-team task delegation.
- `__init__.py`: `default_tools()` returns the tools registered with the agent.

## Current Tools

- `echo(text)`
  - Returns input text.
  - Useful for basic tool-call validation.
- `list_files(path=".", max_depth=2)`
  - Lists files under `SUPERASSIST_PLUS_TOOL_WORKSPACE_DIR`, or
    `{SUPERASSIST_PLUS_DATA_DIR}/workspace` when unset.
- `read_file(path, start_line=None, end_line=None)`
  - Reads UTF-8 text files from the tool workspace, or read-only built-in skill
    files under `/mnt/skills`.
- `write_file(path, content, append=False)`
  - Creates or overwrites UTF-8 text files inside the tool workspace.
- `delete_path(path, recursive=False)`
  - Deletes files or directories inside the tool workspace.
- `web_search(query, max_results=5)`
  - Searches DuckDuckGo HTML results when network tools are enabled.
- `web_fetch(url, max_chars=12000)`
  - Fetches HTTP/HTTPS pages and returns readable text.
- `shell(command, cwd=".")`
  - Runs a non-interactive command under the project root when
    `SUPERASSIST_PLUS_TOOL_SHELL_ENABLED=true`.
  - Uses PowerShell/cmd fallback on Windows and POSIX shells elsewhere.
  - Blocks obvious destructive commands and truncates long output.
- `team_task(agent, description, prompt, wait=true)`
  - Delegates work to a persistent ACP-backed external team agent configured in
    `agent_team.toml`.
  - Requires an active SuperAssist thread context and agent teams to be enabled.
  - Writes auditable task/result records through the agent-team JSONL ledger.

## Registration

`default_tools()` is used by `AgentRuntime` when building the LangChain agent.
New default tools should be added there deliberately. `team_task` is included
only when the runtime has an enabled agent-team supervisor.
`AgentRuntime` also passes `SUPERASSIST_PLUS_MAX_TOOL_CALLS` into the tool-call
limit middleware so one turn cannot execute unbounded tool loops.
Current time is not exposed as a tool; it is injected by
`DynamicContextMiddleware` as `current_time_utc`.

## Maintenance Notes

- Prefer `@tool` from `langchain_core.tools` unless a custom `BaseTool` subclass
  is necessary.
- Document side effects and safety constraints for any tool that reads or writes
  files, shells out, uses the network, or touches credentials.
- File tools must stay scoped to the configured tool workspace and reject path
  traversal outside it. The only exception is read-only `/mnt/skills` access in
  `read_file`; mutation tools must continue rejecting skill paths.
- Network tools must respect `SUPERASSIST_PLUS_TOOL_NETWORK_ENABLED`.
- Shell execution must respect `SUPERASSIST_PLUS_TOOL_SHELL_ENABLED`, keep cwd
  inside the project root, and block obviously destructive commands.
- Add tests when introducing tools with nontrivial behavior.
