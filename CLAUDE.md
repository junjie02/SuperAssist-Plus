# CLAUDE.md

This file is the repository-level development guide. Keep it synchronized with the code whenever runtime boundaries, middleware ordering, storage layout, routes, or build commands change.

## Commands

The primary development environment is Windows, PowerShell, and the Conda environment `CF`.

```powershell
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
python -m pip install -e .

# Python tests and lint
python -B -m pytest
python -B -m pytest tests\test_rag.py -q -p no:cacheprovider
python -m ruff check src tests

# Go tests
Set-Location go-server
go test ./...
Set-Location ..

# Frontend build
Set-Location frontend
npm install
npm run build
Set-Location ..
```

Run the product in separate terminals:

```powershell
# terminal 1, repository root
superassist-ai-engine --port 8765

# terminal 2
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist\go-server
go run .

# optional terminal 3 for Vite HMR
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist\frontend
npm run dev
```

Other entry points:

```powershell
superassist "hello" --flush-memory
superassist -i --thread-id my-thread --flush-memory
superassist-feishu
superassist-wecom
superassist-memory-ui --user-id local-user --port 8765
```

Use `python -B` for tests so bytecode files do not pollute the tree. `ruff` uses line length 120 and Python 3.11. `frontend/dist` is a generated build artifact served by Go and should only change through `npm run build`.

## Configuration

Python settings live in [src/superassist/config.py](src/superassist/config.py), use the `SUPERASSIST_` prefix, and load the root `.env`. Go loads the same `.env` through [go-server/config/config.go](go-server/config/config.go). The two runtimes do not accept identical database URL syntax, so verify both sides before enabling MySQL.

Important test behavior:

- Empty `SUPERASSIST_API_KEY` selects the deterministic fallback chat model.
- `SUPERASSIST_EMBEDDING_PROVIDER=hash` selects the deterministic offline embedder.
- Normal tools require `SUPERASSIST_ENABLE_TOOLS=true`.
- RAG mode always binds `rag_search`, `web_search`, and `web_fetch`; the network tools still enforce `SUPERASSIST_TOOL_NETWORK_ENABLED`.
- Settings updates are written to the root `.env`. Memory values apply to newly constructed runtimes; Feishu and WeCom connection changes require the affected channel process to restart.

## Runtime Architecture

```text
React/Vite browser
  -> Go/Gin :8080 (JWT, users, threads, WebSocket, static assets, proxy)
      -> Python/FastAPI :8765 (SSE chat, memory graph, settings, LightRAG)
          -> LangChain create_agent
          -> SQLite/MySQL + FAISS + local LightRAG stores
```

The browser must never call Python `/internal/*` routes directly. Go derives `user_id` from JWT and adds it to proxied graph, settings, document, and chat calls. Python should remain bound to `127.0.0.1` unless a separate trusted network boundary is added.

### Go Gateway

[go-server/main.go](go-server/main.go) owns the public HTTP surface:

- unauthenticated `/api/auth/register` and `/api/auth/login`;
- authenticated threads, memory graph, settings, LightRAG documents, and LightRAG graph APIs;
- `/ws/chat?token=...`, which validates JWT before upgrading;
- React assets from `frontend/dist`, including SPA fallback routes.

The WebSocket handler forwards `{message, thread_id, rag_mode}` to Python `/internal/chat`, consumes SSE incrementally, and sends JSON events to the browser. A Python connection refusal therefore appears as a Go 502/stream error, not as an LLM response.

### Middleware-First Agent

[src/superassist/agent/factory.py](src/superassist/agent/factory.py) creates one LangChain `create_agent`; there is no outer lead-agent `StateGraph`. Hook ordering is behavioral API because `before_*` runs in registration order and `after_*` runs in reverse.

Base chain:

1. `ToolErrorMiddleware`
2. `ToolCallLimitMiddleware`
3. `MemoryRecallMiddleware`
4. `DynamicContextMiddleware`
5. `ShortMemoryMiddleware`
6. `ToolEventMiddleware`
7. optional `SubagentLimitMiddleware`
8. `MemoryWriterMiddleware`
9. `FinalTextMiddleware`

RAG mode inserts `RagRetrievalMiddleware` and `RagRetryMiddleware` after memory recall, and registers `RagAttributionMiddleware` after `FinalTextMiddleware`; reverse `after_agent` dispatch makes attribution run first so final-text extraction captures the attributed answer. Keep [src/CLAUDE.md](src/CLAUDE.md) synchronized when changing this order.

### Long-Term Memory

[src/superassist/memory/](src/superassist/memory/) implements a CogniFold-style typed graph:

- SQLite/MySQL `MemoryGraphStore` is authoritative for nodes, edges, jobs, and recall snapshots.
- BGE is the normal embedder; hash embeddings exist for deterministic tests.
- `PersistentFaissIndex` provides per-user semantic entry points.
- `MemoryContextRanker` combines vector entry points, bidirectional BFS, optional Personalized PageRank, recency, access count, urgency, and semantic affinity.
- `MemoryWriter` produces an `UpdatePlan` through an LLM or deterministic fallback; `operations.apply_plan` is the only operation dispatcher.
- consolidation merges similar concepts, decays edges, and grounds orphan concepts.

The ontology is fixed: `event|concept|intent|time` and `GROUNDS|CAUSES|TRIGGERS|REINFORCES|PART_OF|DERIVED_FROM|DEADLINE_FOR|RELATED_TO|USER_FEEDBACK`. Update `tests/test_smoke.py` if this contract intentionally changes.

### LightRAG Knowledge Base

[src/superassist/rag/](src/superassist/rag/) is separate from long-term memory. It stores uploaded-document content under a SHA-256-derived per-user directory and uses `lightrag-hku` local JSON KV, NanoVectorDB, GraphML, and document-status stores.

- Upload returns 202 and indexes in the service's dedicated asyncio loop.
- Supported formats are declared in `rag/documents.py`.
- `LightRAGService.retrieve` uses structured `aquery_data` and does not ask LightRAG to generate the final answer.
- `mix` is the automatic first attempt. The agent may call `rag_search`; retry middleware guarantees up to the configured attempt limit when no usable evidence is found.
- Attribution middleware deterministically appends uploaded/web/model provenance.
- RAG graph and document manifests are isolated from CogniFold memory data.

The complete storage, extraction, chunking, graph, update, retrieval, and deletion contract is in [src/superassist/rag/README.md](src/superassist/rag/README.md).

### Subagents And ACP Teams

`subagents/` contains in-process task agents invoked by the `task` tool. `teams/` is a distinct ACP integration configured by `agent_team.toml`; it manages long-lived external processes and a hash-chained, HMAC-signed, file-locked ledger. Do not conflate the two execution models.

### Frontend

[frontend/](frontend/) is a React 18 + Vite SPA, not the legacy no-build viewer. Development requests proxy to Go; production assets are built into `frontend/dist` and served by Go.

The authenticated shell keeps Chat mounted while switching pages so WebSocket and thread state survive navigation. Main views are Chat, Memory Graph, Knowledge, Users (admin only), and Settings. `SUPERASSIST_ADMIN_USERNAMES` is synchronized to persisted Go user roles at startup; all `/api/admin/*` routes must retain database-backed authorization. Read [frontend/CLAUDE.md](frontend/CLAUDE.md) before changing data contracts, responsive layout, graph behavior, or refresh events.

### Feishu

[src/superassist/channels/feishu.py](src/superassist/channels/feishu.py) receives Feishu events over WebSocket and sends incremental interactive-card updates. Private chats use `feishu:<open_id>` identities; every member of a group shares `feishu-group:<chat_id>`, a stable group thread, Memory graph, and RAG scope. Every visible group message is first inserted idempotently into `channels/feishu_messages.sqlite3`; only an explicit mention activates the Agent. Activation waits for a configurable 1.5-second quiet window (6-second hard limit), then projects every unconsumed message through the high-water mark into one speaker-attributed `FeishuConversationBatch`. Per-scope locks queue concurrent work instead of dropping it. Image bytes are cached in the inbox at ingress, sent in chronological speaker order with optional untrusted OCR, and retained in the existing sliding in-memory context for private follow-ups. GPT-5.6 uses Responses with `use_previous_response_id=False`; configured OpenAI-compatible Claude and DeepSeek routes provide ordered failover. Streaming card updates use one coalescing worker per message and must drain before the final patch. Non-image files remain unsupported.

### WeCom

[src/superassist/channels/wecom.py](src/superassist/channels/wecom.py) uses the official WeCom intelligent-robot WebSocket SDK, but delegates Agent execution to the existing Python `/internal/chat` SSE endpoint through [ai_engine_client.py](src/superassist/channels/ai_engine_client.py). Keep this process boundary: constructing another runtime in the channel would create unsafe concurrent writers for local LightRAG files. Private chats are isolated by sender; every member of a group chat shares the group thread, Memory identity, and RAG state. Explicit sender/group-to-browser mappings can attach either scope to uploaded knowledge. See [WECOM.md](src/superassist/channels/WECOM.md) for setup and operations.

[src/superassist/channels/wecom_rpa.py](src/superassist/channels/wecom_rpa.py) is a separate Windows-only visual adapter for ordinary-WeChat external groups opened in WeCom 5.x. It must keep the hard gates in this order: recognized `外部群` header, exact configured group allowlist, configured leading wake prefix, replay claim, then AI Engine call. It only monitors the active visible group and rechecks the group before every send. Never relax these gates to support private chats or background coordinate clicking.

## Data Layout

`SUPERASSIST_DATA_DIR` defaults to `.superassist`:

```text
superassist.sqlite3              relational data and typed memory graph
logs/model-input.jsonl          final provider payloads when input logging is enabled
faiss/<safe-user-id>.index      memory vector index
faiss/<safe-user-id>.mapping.json
rag/<user-hash>/documents.json  uploaded-document manifest
rag/<user-hash>/files/          original uploaded files
rag/<user-hash>/index/default/  LightRAG KV/vector/GraphML stores
threads/<thread-id>/             messages.jsonl + thread_meta.json
teams/<thread-id>/ledger.jsonl  audited ACP team communication
channels/feishu_threads.json    Feishu-to-thread mapping
channels/feishu_messages.sqlite3 durable Feishu inbox, image payloads, and consumption cursors
channels/wecom_threads.json     WeCom chat/sender-to-thread and RAG mapping
channels/wecom_rpa_state.json   desktop RPA visible-message replay guard
huggingface/                    embedding model cache
workspace/                      file/shell tool sandbox
```

Do not manually merge the CogniFold graph with the LightRAG graph. They have different ownership, lifecycle, deletion, and recall semantics.

## Documentation Ownership

- [README.md](README.md): user-facing setup, architecture, configuration, and operations.
- [src/CLAUDE.md](src/CLAUDE.md): detailed Python contracts down to fields and call ordering.
- [src/superassist/rag/README.md](src/superassist/rag/README.md): complete LightRAG technical design.
- [frontend/CLAUDE.md](frontend/CLAUDE.md): frontend contracts and interaction invariants.
- [.rag-eval-stage/README.md](.rag-eval-stage/README.md): reproducible Vector RAG versus LightRAG evaluation.

Generated dataset examples and `skills/**/SKILL.md` are content/protocol assets, not general project documentation. Change them only when their own behavior changes.
