from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

from superassist_plus.agent import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SuperAssist-Plus locally.")
    parser.add_argument("message", nargs="*", help="User message to send to the agent.")
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--interactive", "-i", action="store_true", help="Start a continuous conversation session.")
    parser.add_argument("--flush-memory", action="store_true", help="Flush memory writes before exiting.")
    args = parser.parse_args()

    runtime = AgentRuntime(tool_event_reporter=_print_tool_event)
    if args.interactive:
        _run_interactive(runtime, user_id=args.user_id, thread_id=args.thread_id, flush_memory=args.flush_memory)
        return
    if not args.message:
        parser.error("message is required unless --interactive is used")
    result = runtime.run(" ".join(args.message), user_id=args.user_id, thread_id=args.thread_id)
    if args.flush_memory:
        runtime.memory_queue.flush()
    print(result.answer)


def _run_interactive(
    runtime: AgentRuntime,
    *,
    user_id: str,
    thread_id: str | None,
    flush_memory: bool,
) -> None:
    resolved_thread_id = thread_id or f"thread_{uuid4().hex[:12]}"
    print(f"SuperAssist-Plus interactive session. thread_id={resolved_thread_id}")
    print("Type 'exit' or 'quit' to leave.")
    try:
        while True:
            try:
                message = input("> ").strip()
            except EOFError:
                break
            if message.lower() in {"exit", "quit"}:
                break
            if not message:
                continue
            result = runtime.run(message, user_id=user_id, thread_id=resolved_thread_id)
            print(result.answer)
    except KeyboardInterrupt:
        print()
    finally:
        if flush_memory:
            runtime.memory_queue.flush()


def _print_tool_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    tool_name = str(event.get("tool") or "tool")
    if event_type == "agent_tool_call":
        content = str(event.get("content") or "").strip()
        if content:
            print(content)
        return
    if event_type == "tool_start":
        args = json.dumps(event.get("args") or {}, ensure_ascii=False, default=str)
        print(f"[tool:start] {tool_name} {args}")
        return
    if event_type == "tool_result":
        status = str(event.get("status") or "success")
        error = str(event.get("error") or "").strip()
        suffix = f" error={error}" if error else ""
        print(f"[tool:{status}] {tool_name}{suffix}")


if __name__ == "__main__":
    main()
