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
  persistence, and memory queue handoff.
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
- for MiniMax-compatible endpoints, keeps the LangGraph/middleware path active
  and relies on the LLM adapter to add MiniMax `reasoning_split` while retaining
  OpenAI-compatible tool schemas.
- when tools are disabled, uses a direct model call inside the same outer
  LangGraph node to avoid provider-specific tool/agent parameters rejected by
  MiniMax-style OpenAI-compatible endpoints.

`persist_turn`:

- appends only the current user input and final assistant answer to JSONL.

`enqueue_memory_write`:

- creates a `MemoryWritePayload` with write-path memory context and queues it
  for debounced background writing.

## Inner LangChain Middleware Chain

`build_middlewares()` returns, in order:

1. `DynamicContextMiddleware`
   - Injects the lead system prompt, runtime context, and memory recall into
     model requests as a single system message. The runtime does not pass
     `system_prompt` directly to `create_agent`, because that would create a
     second system message after middleware injection.
2. `ToolErrorMiddleware`
   - Converts tool exceptions into readable `ToolMessage` objects.
3. `ToolEventMiddleware`
   - Captures normalized tool execution events into state.
4. `MemoryAfterAgentMiddleware`
   - Marks state as memory-ready and records final assistant text.

The middleware chain is mandatory. Future cross-cutting concerns should be added
here rather than scattered through runtime nodes.

## Continuous Conversation

`AgentRuntime.run()` accepts a `thread_id`. If the thread exists, it loads up to
20 recent persisted messages and appends the new `HumanMessage`. This gives
interactive mode continuity across turns.

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
