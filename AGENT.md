# SuperAssist-Plus Technical Documentation

IMPORTANT: Any change to this project must update this technical document when
the change affects architecture, behavior, runtime flow, dependencies, tests, or
module responsibilities. Each subdirectory also has its own `AGENT.md`; update
the nearest module document after modifying that module.

## Project Purpose

SuperAssist-Plus is a re-architecture of `SuperAssist` based on LangChain and
LangGraph. It is not a direct copy migration. The implementation keeps useful
SuperAssist behavior while replacing the old hand-written agent loop with a
maintainable LangGraph runtime and LangChain agent/middleware chain.

The long-term memory mechanism must follow the latest and most correct
CogniFold-style graph memory semantics rather than the inaccurate parts of the
old `SuperAssist.agent_runtime.long_term_memory` implementation.

## Non-Negotiable Requirements

- Use LangChain and LangGraph for the agent runtime.
- Use `ProAssist` as the reference for LangChain/LangGraph agent structure and
  middleware concepts.
- Use `CogniFold` as the reference for typed graph memory, consolidation, and
  structural debt mitigation.
- Prefer mainstream, maintainable frameworks for all other pieces.
- Keep code readable, elegant, small, and easy to test.
- Install dependencies and run tests in the `CF` conda environment.
- Add automated tests or useful test scripts when implementing important
  functionality.

## Current Architecture

The Python package is `superassist_plus` under `src/`.

- `config.py`: Pydantic settings sourced from environment variables and `.env`.
- `README.md`: user-facing startup, memory scoring, embedding, FAISS, and test
  instructions.
- `models.py`: shared Pydantic models and fixed memory ontology.
- `llm.py`: LangChain chat model factory, MiniMax compatibility adapter, plus
  deterministic fallback model.
- `cli.py`: single-turn and interactive command-line entrypoint.
- `agent/`: outer LangGraph runtime, inner LangChain agent, middleware chain,
  and graph state schema.
- `channels/`: lightweight IM channel integrations. The Feishu channel uses a
  WebSocket long connection and calls `AgentRuntime` directly.
- `memory/`: CogniFold-style typed graph memory, SQLite persistence, FAISS dense
  vector recall over BGE embeddings, consolidation, deterministic writer, optional
  LLM-assisted UpdatePlan writer, and debounced write queue.
- `agent_teams/`: persistent external ACP-backed coding agents with project-root
  `agent_team.toml` membership, per-thread sessions, and append-only JSONL audit
  ledgers protected by cross-platform file locks.
- `tools/`: LangChain tool adapters.
- `ui/`: FastAPI local server for the memory graph viewer.
- `frontend/`: static memory graph visualization UI.
- `tests/`: pytest coverage for ontology, memory behavior, middleware order,
  runtime fallback, and continued thread history.

## Runtime Flow

Outer LangGraph flow:

1. `prepare_context`
   - Loads recent persisted thread history from `messages.jsonl`.
   - Creates the current user turn event after read/write context assembly.
   - Recalls relevant long-term memory into immediate/working/background tiers.
2. `agent`
   - Invokes a LangChain `create_agent` graph.
   - Passes state fields including `user_id`, `thread_id`, `memory_recall`,
     `tool_events`, and `metadata`.
   - Registers tools only when `SUPERASSIST_PLUS_ENABLE_TOOLS=true`.
   - Uses direct model invocation when tools are disabled to avoid
     OpenAI-compatible provider errors around unsupported agent/tool settings.
3. `persist_turn`
   - Appends the current user/assistant pair to the thread JSONL file.
4. `enqueue_memory_write`
   - Adds the completed turn to the debounced memory write queue.

The inner LangChain agent owns tool routing. SuperAssist-Plus must not recreate
the old manual tool-call loop.

`AgentRuntime` can optionally report channel-facing run events. It emits
`preparing_context` when a turn starts and forwards model-authored
`agent_text` from tool-call messages. It does not expose tool names, arguments,
tool results, or token streaming to the Feishu channel.

## LangChain Middleware Chain

The internal LangChain agent uses an explicit middleware chain:

- `DynamicContextMiddleware`
  - Injects current `user_id`, `thread_id`, and recalled memory into model calls.
- `ToolErrorMiddleware`
  - Converts tool exceptions into readable `ToolMessage` responses.
- `ToolEventMiddleware`
  - Records normalized tool start/result events for later memory writing.
- `MemoryAfterAgentMiddleware`
  - Marks the final state as memory-ready and stores final assistant text in
    metadata.

Middleware is a core architectural requirement, not an optional enhancement.
Future cross-cutting behavior should be added as middleware where possible.

## MiniMax Compatibility

MiniMax is configured through the OpenAI-compatible `ChatOpenAI` path, matching
the ProAssist configuration style and MiniMax's OpenAI-compatible documentation.
`MiniMaxCompatibleChatModel` keeps OpenAI tool schemas enabled and adds
`extra_body.reasoning_split=true` so MiniMax reasoning output can be preserved
as `reasoning_content`.

When `SUPERASSIST_PLUS_ENABLE_TOOLS=true` and the selected provider is MiniMax,
the outer LangGraph runtime, LangChain agent, middleware chain, and OpenAI-style
tool schema binding all remain active. Runtime metadata records:

```text
tool_schema_binding=openai_compatible_minimax
```

The lead system prompt is injected by `DynamicContextMiddleware` together with
runtime context and memory as a single system message. Do not pass the same
prompt through `create_agent(system_prompt=...)`; MiniMax rejected the previous
two-system-message payload with `invalid chat setting (2013)`.

## Long-Term Memory Design

Memory is a typed directed multigraph persisted in SQLite.

Node types:

- `event`: episodic trace.
- `concept`: semantic pattern.
- `intent`: crystallized goal.
- `time`: temporal anchor.

Edge types:

- `GROUNDS`: Event evidences concept/intent.
- `CAUSES`: Event causes event.
- `TRIGGERS`: Event or concept triggers intent.
- `REINFORCES`: Event supports concept.
- `PART_OF`: Structural hierarchy.
- `DERIVED_FROM`: Concept abstraction.
- `DEADLINE_FOR`: Temporal constraint.
- `RELATED_TO`: Associative link.
- `USER_FEEDBACK`: Feedback to intent.

Structural debt mitigation:

- Accumulation: similar incoming events reinforce existing concepts instead of
  duplicating them.
- Compression: similar concepts merge at the configured threshold.
- Decay: edge weights decay over time and weak edges are pruned.
- Completion: orphan concepts can be reconnected by embedding k-NN inference.

Current production memory embeddings use the configurable BGE provider backed by
`sentence-transformers`, defaulting to `BAAI/bge-base-zh-v1.5` on CPU. Dense
vectors are stored in persistent FAISS indexes under
`{SUPERASSIST_PLUS_DATA_DIR}/faiss/`, with per-user mapping files that resolve
FAISS integer ids back to memory node ids. SQLite `memory_nodes.embedding_json`
keeps an authoritative vector copy for rebuilding and audit. A deterministic
local hash provider remains available as
`SUPERASSIST_PLUS_EMBEDDING_PROVIDER=hash` for offline tests and development.

The BGE embedder is preloaded when `AgentRuntime` starts and is kept in memory
for the process lifetime. Memory writing is deterministic by default; LLM-based
memory plan generation is opt-in via `SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED=true`.
When enabled, the memory writer prompt follows CogniFold's `UpdatePlan`
operation format: `ADD_NODE`, `UPDATE_NODE`, `REMOVE_NODE`, `ADD_EDGE`,
`REMOVE_EDGE`, and `MERGE_NODES`, with explicit `grounded_in` evidence and
typed edge weights.

## Persistence and Conversation Modes

The project `.env` is resolved from the SuperAssist-Plus project root so CLI
commands keep the same configuration even if launched from another directory.

Thread history is stored under:

```text
{SUPERASSIST_PLUS_DATA_DIR}/threads/{thread_id}/messages.jsonl
```

Single-turn mode:

```powershell
superassist-plus "message" --flush-memory
```

Interactive continuous mode:

```powershell
superassist-plus --interactive --flush-memory
superassist-plus -i --thread-id my-thread --flush-memory
```

`--flush-memory` forces the debounced memory queue to write before exit. It is
recommended during development and testing.

Memory graph viewer:

```powershell
superassist-plus-memory-ui --user-id local-user --port 8765
```

Open the printed local URL to inspect nodes, typed edges, graph statistics, and
the right-side update ledger.

Feishu control entry:

```powershell
superassist-plus-feishu
```

This starts a Feishu/Lark WebSocket long-connection bot. Configure
`SUPERASSIST_PLUS_FEISHU_APP_ID`, `SUPERASSIST_PLUS_FEISHU_APP_SECRET`, and
optionally `SUPERASSIST_PLUS_FEISHU_DOMAIN`,
`SUPERASSIST_PLUS_FEISHU_ALLOWED_OPEN_IDS`, and
`SUPERASSIST_PLUS_FEISHU_MENTION_ONLY`. Private chats trigger directly; group
chats trigger on @ when mention-only mode is enabled. Feishu users map to
`feishu:<open_id>`, and conversation mappings are stored under
`{SUPERASSIST_PLUS_DATA_DIR}/channels/feishu_threads.json`. The Feishu service
creates and caches `AgentRuntime` at startup so the configured embedder, such as
BGE, is loaded before the first message arrives.

If PowerShell reports that `superassist-plus-memory-ui` is not recognized,
activate the `CF` conda environment or call the script by absolute path:

```powershell
conda activate CF
C:\Users\15746\.conda\envs\CF\Scripts\superassist-plus-memory-ui.exe --user-id local-user --port 8765
```

## Testing

Run all tests in the `CF` conda environment:

```powershell
conda run -n CF python -B -m pytest
```

Current expected result:

```text
125 passed
```

## Maintenance Rules

- Update this file when project-level architecture or behavior changes.
- Update the nearest directory-level `AGENT.md` when changing a module.
- Keep generated files out of the project tree; `.gitignore` covers cache,
  bytecode, runtime data, build output, and editable-install metadata.
- Do not modify `SuperAssist`, `ProAssist`, or `CogniFold` while implementing
  SuperAssist-Plus unless explicitly asked.
