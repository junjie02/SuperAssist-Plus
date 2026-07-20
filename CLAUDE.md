# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project assumes the conda env `CF` and is developed on Windows + PowerShell.

```powershell
conda activate CF
cd f:\CODE\SuperAssist\superAssist\SuperAssist
python -m pip install -e .

# single turn (CLI)
superassist "你好" --flush-memory

# continuous conversation
superassist -i --thread-id my-thread --flush-memory

# memory graph viewer (FastAPI on http://localhost:8765)
superassist-memory-ui --user-id local-user --port 8765

# Feishu bot (requires SUPERASSIST_FEISHU_APP_ID / _APP_SECRET in .env)
superassist-feishu

# tests — run `python -B` so .pyc files don't pollute the tree
python -B -m pytest

# single test file / single test
python -B -m pytest tests/test_memory.py
python -B -m pytest tests/test_memory.py::test_name -k name -x
```

Lint is `ruff` (configured in `pyproject.toml`, line-length 120, target py3.11). There is no separate build step — `pip install -e .` registers the three console scripts via `[project.scripts]`.

## Environment

All settings use the `SUPERASSIST_` prefix and are defined in [src/superassist/config.py](src/superassist/config.py). The `.env` file at the project root is auto-loaded. A blank `SUPERASSIST_API_KEY` falls back to a deterministic local responder so tests run without credentials. `SUPERASSIST_EMBEDDING_PROVIDER=hash` switches to an offline embedder for the same reason. Tools, subagents, and `team_task` are gated behind `SUPERASSIST_ENABLE_TOOLS=true`.

## Architecture

This is a LangChain 1.x + LangGraph assistant with **CogniFold-style typed-graph long-term memory**, **ACP-backed team agents** (Claude Code via `agent-client-protocol`), and a **Feishu** chat channel. It is a clean rewrite of `SuperAssist-Plus` — when in doubt about migration intent, the old repo is at `f:\CODE\SuperAssist\superAssist\SuperAssist-Plus`.

### Middleware-first agent (no outer LangGraph wrapper)

[src/superassist/agent/factory.py](src/superassist/agent/factory.py) composes a single `langchain.agents.create_agent` call with a deterministic middleware chain. There is **no outer LangGraph state graph** — cross-cutting concerns are middleware in a fixed order. Reading the docstring at the top of `factory.py` is the fastest way to understand control flow.

The chain (top→bottom; LangChain dispatches `before_*` in order, `after_*` in reverse):

1. `ToolErrorMiddleware` — wrap_tool_call: convert exceptions into ToolMessages
2. `ToolCallLimitMiddleware` — refuse new tool calls past per-turn budget
3. `MemoryRecallMiddleware` — before_agent: read graph memory, create user-turn event
4. `DynamicContextMiddleware` — wrap_model_call: prepend recall+skills+time to system message
5. `ShortMemoryMiddleware` — after_agent: persist `messages.jsonl`, compress when over budget
6. `ToolEventMiddleware` — collect tool start/result events
7. `SubagentLimitMiddleware` (only if `subagents_enabled`) — trim parallel `task` calls
8. `MemoryWriterMiddleware` — enqueue durable memory write payload
9. `FinalTextMiddleware` — surface `final_assistant_text` in state metadata

`SuperAssistState` ([agent/state.py](src/superassist/agent/state.py)) extends LangChain's `AgentState` with `user_id`, `thread_id`, `memory_event_id`, `memory_recall`, `tool_events`, `loaded_skills`, and `metadata`. Every middleware reads/writes this shape.

`AgentRuntime` ([agent/runtime.py](src/superassist/agent/runtime.py)) is the public turn driver — it loads thread history (`messages.jsonl` + `thread_meta.json`) into the initial state, calls `agent.invoke` (sync) or `agent.stream` (streaming with `accumulate_stream_text` from [agent/streaming.py](src/superassist/agent/streaming.py)), then extracts `AgentRunResult`. The streaming path is shared with the subagent executor.

### Memory: typed graph with FAISS

[src/superassist/memory/](src/superassist/memory/) implements a CogniFold-style graph:

- **storage.py** — SQLite-backed `MemoryGraphStore` (nodes + typed edges).
- **embedding.py** — pluggable `Embedder` (default `bge`, `hash` for offline tests). Embedder is preloaded by `MemoryService.preload_embedder()`.
- **vector_index.py** — per-user `PersistentFaissIndex` keyed by `user_id`.
- **scoring.py** — `MemoryContextRanker` does the read path: vector entry-point selection → bidirectional BFS / Personalized PageRank blend → `Score(v)` for highlighted recall nodes.
- **plans.py** — `UpdatePlan` is a Pydantic **discriminated union** of operations (`ADD_NODE`, `ADD_EDGE`, `REINFORCE`, etc.).
- **operations.py** — pure dispatch: `apply_plan(plan, ApplyContext)` is the only place per-op logic lives. Don't hand-roll dispatch elsewhere.
- **service.py** — `MemoryService` is the single public surface used by middleware. It owns the store, FAISS, embedder, ranker, and consolidation (concept merge, edge decay, orphan completion).
- **writer.py** — `MemoryWriteQueue` debounces durable writes (`SUPERASSIST_MEMORY_DEBOUNCE_SECONDS`). When `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED=true`, an LLM produces an `UpdatePlan`; otherwise a deterministic writer runs.

The ontology is fixed and asserted by `tests/test_smoke.py` — node types: `event|concept|intent|time`; edge types: `GROUNDS|CAUSES|TRIGGERS|REINFORCES|PART_OF|DERIVED_FROM|DEADLINE_FOR|RELATED_TO|USER_FEEDBACK`. Changing these requires updating the test.

### ACP team agents

[src/superassist/acp_client/](src/superassist/acp_client/) is the low-level Agent Client Protocol implementation: process spawning (`process.py`), the `AsyncLoopThread` event loop, `ACPSession`, `PermissionPolicy`, and error types. **It speaks ACP and nothing else.**

[src/superassist/teams/](src/superassist/teams/) sits on top: `TeamSupervisor` parses `agent_team.toml`, holds a pool of long-lived ACP team-member processes (one per configured agent, e.g. `claude_code`), and routes `team_task` tool invocations through them. It maintains a hash-chained, HMAC-signed, file-locked **JSONL ledger per thread** ([teams/ledger.py](src/superassist/teams/ledger.py)) — `LedgerTamperError` indicates the chain was broken. The `team_thread_context` (used by `AgentRuntime`) scopes ledger writes to the current thread.

`agent_team.toml` controls membership; setting `enabled=false` or removing all `[[agents]]` entries cleanly disables team mode (the supervisor returns `None` and the `team_task` tool isn't bound).

### Subagents

[src/superassist/subagents/](src/superassist/subagents/) is a separate concept from teams: it's the **in-process** general-purpose / research subagent invoked via the `task` tool. `executor.py` shares the streaming pipeline with the lead runtime. Concurrency is bounded by `SubagentLimitMiddleware` (parallel `task` calls trimmed) and `SUPERASSIST_SUBAGENT_MAX_CONCURRENT`.

### Channels

[src/superassist/channels/feishu.py](src/superassist/channels/feishu.py) is a Feishu WebSocket bot. It uses `lark-oapi`, maps Feishu chats to thread IDs through `FeishuThreadStore`, gates on `SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS` and `SUPERASSIST_FEISHU_MENTION_ONLY`, and only handles **text** in v1 (files/images return `UNSUPPORTED_FILE_MESSAGE`).

### Frontend (memory graph viewer)

[frontend/](frontend/) is a static, no-build SPA served by [src/superassist/ui/server.py](src/superassist/ui/server.py) at `/api/graph?user_id=<id>`. **Read [frontend/AGENT.md](frontend/AGENT.md) before changing the viewer** — its data contract (`nodes`, `edges`, `updates`, `stats`, plus `active_recall`/`recall_score`/`recall_components` on highlighted nodes), force-directed layout invariants, and "no build tooling, no network-loaded assets" rule are documented there.

### Skills

`SKILL.md`-based skill loading lives in [src/superassist/skills/registry.py](src/superassist/skills/registry.py). The single bundled skill is [skills/public/deep-research/SKILL.md](skills/public/deep-research/SKILL.md). `loaded_skills` flows through state and `thread_meta.json` so skill activation persists across turns within a thread.

## Data layout

`SUPERASSIST_DATA_DIR` (default `.superassist/`) holds:

- `superassist.sqlite3` — memory graph store
- `faiss/<user_id>/` — per-user FAISS vector index
- `threads/<thread_id>/messages.jsonl` + `thread_meta.json` — short memory history (loaded by `load_short_memory`)
- `teams/<thread_id>/ledger.jsonl` — hash-chained team ledger
- `channels/feishu_threads.json` — Feishu chat ↔ thread_id map
- `huggingface/` — embedder model cache
- `workspace/` — default tool sandbox (overridable via `SUPERASSIST_TOOL_WORKSPACE_DIR`)

The SQLite schema and FAISS layout are unchanged from `SuperAssist-Plus`, so old data can be copied directly (see [README.md](README.md) for the exact `Copy-Item` recipe).
