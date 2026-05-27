# Tools Module Technical Documentation

IMPORTANT: Any change to available tools, tool schemas, tool side effects, or
tool registration must update this document.

## Purpose

The `tools` module owns LangChain `BaseTool` adapters exposed to the inner
LangChain agent. Tools should be simple, typed, testable units.

## Files

- `basic.py`: currently provides `current_time` and `echo` tools.
- `__init__.py`: `default_tools()` returns the tools registered with the agent.

## Current Tools

- `current_time()`
  - Returns current UTC time as ISO-8601.
  - Useful for smoke testing tool availability.
- `echo(text)`
  - Returns input text.
  - Useful for basic tool-call validation.

## Registration

`default_tools()` is used by `AgentRuntime` when building the LangChain agent.
New default tools should be added there deliberately.

## Maintenance Notes

- Prefer `@tool` from `langchain_core.tools` unless a custom `BaseTool` subclass
  is necessary.
- Document side effects and safety constraints for any tool that reads or writes
  files, shells out, uses the network, or touches credentials.
- Add tests when introducing tools with nontrivial behavior.

