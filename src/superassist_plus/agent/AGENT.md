# Agent Module Technical Documentation

IMPORTANT: Any change to the LangGraph runtime, LangChain agent factory,
middleware chain, state schema, persistence handoff, or continuous conversation
behavior must update this document.

## Purpose

The `agent` module owns the SuperAssist-Plus execution graph. It combines:

- an outer LangGraph state graph for application-level turn orchestration;
- an inner LangChain `create_agent` graph for model/tool interaction;
- a required middleware chain for cross-cutting behavior.

## Files

- `state.py`: `SuperAssistState`, the outer LangGraph state schema.
- `middleware.py`: LangChain middleware classes and `build_middlewares()`.
- `runtime.py`: `AgentRuntime`, graph construction, run API, history loading,
  channel-facing run-event reporting, persistence, short-memory compression,
  and memory queue handoff.
- `short_memory.py`: token-budgeted thread history loading, compact tool-event
  persistence, and LLM summary compression helpers.
- `__init__.py`: exports `AgentRuntime`.

## Outer LangGraph Flow

The compiled graph is:

```text
START
  -> prepare_context
  -> agent
  -> persist_turn
  -> enqueue_memory_write
  -> END
```

`prepare_context`:

- loads recent thread history from `messages.jsonl`;
- creates an Event memory node for the incoming turn after assembling context;
- recalls read-path long-term memory tiers for the lead agent;
- records write-path memory tiers for the memory writer handoff.

`agent`:

- invokes the inner LangChain agent with `messages`, `user_id`, `thread_id`,
  `memory_recall`, `tool_events`, and `metadata`.
- registers default tools only when `SUPERASSIST_PLUS_ENABLE_TOOLS=true`.
- when project-root `agent_team.toml` enables agent teams, registers
  `team_task`, starts a `TeamSupervisor`, and exposes persistent external team
  agents in the lead prompt.
- injects DeerFlow-style skill metadata for built-in skills. The lead agent sees
  skill names, descriptions, and `/mnt/skills/.../SKILL.md` locations first; if
  it reads a skill file, that skill is remembered for the current thread and
  its full instructions are injected on later model calls.
- for MiniMax-compatible endpoints, keeps the LangGraph/middleware path active
  and relies on the LLM adapter to add MiniMax `reasoning_split` while retaining
  OpenAI-compatible tool schemas.
- when tools are disabled, uses a direct model call inside the same outer
  LangGraph node to avoid provider-specific tool/agent parameters rejected by
  MiniMax-style OpenAI-compatible endpoints.
- uses a structured lead prompt with clarification-first behavior,
  human-readable progress notes between tool rounds, citation requirements for
  web-sourced claims, and concise response style rules.

`persist_turn`:

- appends the current user input, compact tool events, and final assistant
  answer to JSONL.
- compresses older short-memory records into `thread_meta.json` summary when
  the thread exceeds `SUPERASSIST_PLUS_SHORT_MEMORY_TOKEN_LIMIT`.

`enqueue_memory_write`:

- creates a `MemoryWritePayload` with write-path memory context and queues it
  for debounced background writing.

`AgentRuntime` accepts an optional `run_event_reporter` callback. It reports
`preparing_context` at turn start, `thinking` before the model step, and forwards
model-authored `agent_text` from streamed messages and tool-call messages for
external surfaces such as Feishu cards. During a run, the callback is also
available to the `task` tool through a context variable so subagents can report
their own model-authored `subagent_text`. It intentionally does not expose tool
names, arguments, results, or finalization boilerplate. If a model emits a tool
call without assistant text, middleware does not invent a fallback progress
sentence; Feishu should stay on the previous human-readable card text instead
of showing raw tool names.

During model/tool execution, the active `thread_id` is also exposed through an
agent-team context variable so `team_task` can route work to the correct
per-thread persistent external session without adding internal arguments to the
tool schema.

## Inner LangChain Middleware Chain

`build_middlewares()` returns, in order:

1. `DynamicContextMiddleware`
   - Injects the lead system prompt, runtime context, and memory recall into
     model requests as a single system message. The runtime does not pass
     `system_prompt` directly to `create_agent`, because that would create a
     second system message after middleware injection.
   - Adds available skill metadata and already loaded thread skills.
2. `ToolErrorMiddleware`
   - Converts tool exceptions into readable `ToolMessage` objects.
3. `ToolEventMiddleware`
   - Captures normalized tool execution events into state.
   - Captures assistant text attached to tool-call messages as `agent_tool_call`
     events, without generating tool-name fallback prose.
   - Marks a skill as loaded when `read_file` reads its `/mnt/skills` `SKILL.md`.
4. `MemoryAfterAgentMiddleware`
   - Marks state as memory-ready and records final assistant text.

The middleware chain is mandatory. Future cross-cutting concerns should be added
here rather than scattered through runtime nodes.

## Continuous Conversation

`AgentRuntime.run()` accepts a `thread_id`. If the thread exists, it loads a
token-budgeted short-memory window from `messages.jsonl`, prepending any
conversation summary from `thread_meta.json`. The default budget is 80K
estimated tokens. When compression runs, the latest
`SUPERASSIST_PLUS_SHORT_MEMORY_KEEP_RECENT_TURNS` turns remain as raw JSONL
records and older records are folded into the summary.

Loaded skill names are stored in `thread_meta.json` beside `messages.jsonl`, so
skill instructions that were explicitly read remain available in later turns of
the same thread.

## Maintenance Notes

- Do not reintroduce a manual tool loop; LangChain owns tool routing.
- Keep `SUPERASSIST_PLUS_ENABLE_TOOLS=false` as the compatibility default for
  OpenAI-compatible providers that reject tool schemas.
- If `SUPERASSIST_PLUS_ENABLE_TOOLS=true` with MiniMax, runtime metadata should
  include `tool_schema_binding=openai_compatible_minimax`.
- Keep the MiniMax path to one system message. MiniMax accepts OpenAI-compatible
  tools, but rejected the previous two-system-message payload with
  `invalid chat setting (2013)`.
- The direct-model compatibility path must preserve the same metadata flags as
  the middleware path: `dynamic_context_injected`, `memory_ready`, and
  `final_assistant_text`.
- Keep persistence outside middleware. Middleware may annotate state, but durable
  writes remain in outer graph nodes.
- Update tests when changing graph node names, middleware order, or metadata
  keys.
