# Tests Technical Documentation

IMPORTANT: Any change to test strategy, fixtures, test coverage ownership, or
expected test commands must update this document.

## Purpose

The `tests` directory contains pytest coverage for SuperAssist-Plus. Tests must
run in the `CF` conda environment.

## Test Command

```powershell
conda run -n CF python -B -m pytest
```

Expected current result:

```text
66 passed
```

## Files

- `test_smoke.py`
  - Verifies the fixed memory ontology of 4 node types and 9 edge types.
- `test_memory.py`
  - Verifies edge type constraints.
  - Verifies `event -> intent` `TRIGGERS` edges are valid.
  - Verifies reinforcement of an existing concept.
  - Verifies memory recall returns relevant persisted nodes.
  - Verifies FAISS index and node-id mapping files persist and reload.
  - Verifies similar concept merging and edge transfer behavior.
- `test_memory_scoring.py`
  - Verifies CogniFold write-path PageRank uses the real directed graph without
    temporary reverse projection.
  - Verifies read-path recall uses semantic entry points with bidirectional BFS
    and optional PPR blending.
  - Verifies node scoring combines PageRank, recency, access, and urgency.
  - Verifies `DEADLINE_FOR` urgency only boosts `time -> intent` targets.
  - Verifies `prepare_turn()` assembles and touches context before writing the
    current event.
- `test_embedding.py`
  - Verifies deterministic hash fallback behavior.
  - Verifies BGE provider selection.
  - Verifies BGE calls through `SentenceTransformer` without downloading a real
    model by monkeypatching the dependency.
  - Verifies BGE preload reuses one in-memory model instance.
- `test_memory_writer.py`
  - Verifies deterministic memory writing is the default and does not call the
    chat model.
  - Verifies LLM memory planning remains available when explicitly enabled.
  - Verifies the memory writer prompt uses CogniFold-style `UpdatePlan`
    operations, including `MERGE_NODES` and `grounded_in`.
  - Verifies operation plans can add nodes/edges, update nodes, and merge
    near-duplicate concepts.
- `test_runtime.py`
  - Verifies project-root `.env` loading configuration.
  - Verifies default tool-calling compatibility mode.
  - Verifies runtime initializes the embedding provider during startup.
  - Verifies middleware chain order.
  - Verifies fallback runtime mode and middleware metadata.
  - Verifies follow-up turns load persisted thread history.
  - Verifies MiniMax with `SUPERASSIST_PLUS_ENABLE_TOOLS=true` uses the
    OpenAI-compatible binding path.
  - Verifies runtime context is merged with the lead prompt into one system
    message.
  - Verifies runtime events expose context preparation and model-authored
    progress text for channel surfaces.
- `test_subagents.py`
  - Verifies built-in `general-purpose` and `research` subagent prompts.
  - Verifies subagent tool filtering removes recursive `task` access.
  - Verifies FastAPI subagent task status endpoints.
- `test_task_tool.py`
  - Verifies the `task` tool rejects unknown subagent types.
  - Verifies task success/failure/timeout result formatting.
  - Verifies subagent concurrency middleware keeps only the first 3 task calls.
- `test_feishu_channel.py`
  - Verifies Feishu text/rich-text parsing and mention cleanup.
  - Verifies Feishu private/group trigger rules and allowed-open-id filtering.
  - Verifies the Feishu channel caches one runtime so startup can preload BGE.
  - Verifies chat/topic to thread mapping persistence.
  - Verifies running card creation, stage patching, and final answer patching.
- `test_llm.py`
  - Verifies MiniMax model names default to `temperature=1.0`.
  - Verifies explicit temperature settings override the MiniMax default.
  - Verifies MiniMax detection, OpenAI tool-binding payload preservation,
    `reasoning_split`, and reasoning content preservation.
- `test_ui.py`
  - Verifies the memory graph API payload contains nodes, edges, update ledger
    entries, and aggregate stats.
  - Verifies the FastAPI `/api/graph` endpoint returns the graph payload.

## Maintenance Notes

- Add tests alongside important implementation work.
- Prefer deterministic tests that do not require network access or model API
  keys.
- Memory service tests explicitly configure `SUPERASSIST_PLUS_EMBEDDING_PROVIDER=hash`
  and should call `service.embed()` rather than global embedding helpers.
- Use temporary directories for runtime data.
- Run tests with `-B` to avoid bytecode artifacts in the project tree.
