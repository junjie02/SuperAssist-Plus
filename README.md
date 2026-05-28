# SuperAssist-Plus

SuperAssist-Plus is a LangChain/LangGraph rewrite of SuperAssist with a
CogniFold-style typed graph long-term memory system.

The project is intentionally small and explicit at this stage:

- LangGraph owns the runtime flow.
- LangChain owns model and tool orchestration.
- SQLite persists thread data and typed memory graph data.
- FAISS persists dense vector indexes for memory retrieval.
- FastAPI serves the local memory graph viewer.

See `AGENT.md` for the living migration requirements and implementation log.

## Technical Docs

`AGENT.md` files are technical documentation, not disposable notes. Update the
nearest `AGENT.md` whenever changing a module:

- `AGENT.md`: project architecture and operating rules.
- `src/AGENT.md`: source tree packaging boundary.
- `src/superassist_plus/AGENT.md`: package-level behavior.
- `src/superassist_plus/agent/AGENT.md`: LangGraph and middleware runtime.
- `src/superassist_plus/memory/AGENT.md`: CogniFold-style memory system.
- `src/superassist_plus/tools/AGENT.md`: LangChain tools.
- `tests/AGENT.md`: test strategy.

## Memory Embeddings

Memory similarity uses BGE by default through `sentence-transformers`:

```env
SUPERASSIST_PLUS_EMBEDDING_PROVIDER=bge
SUPERASSIST_PLUS_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
SUPERASSIST_PLUS_EMBEDDING_DEVICE=cpu
```

The first real run may download the BGE model if it is not already cached. For
offline tests, set `SUPERASSIST_PLUS_EMBEDDING_PROVIDER=hash`.

Dense vectors are written to persistent FAISS index files under:

```text
{SUPERASSIST_PLUS_DATA_DIR}/faiss/
```

SQLite still keeps `memory_nodes.embedding_json` as the authoritative vector
copy so FAISS can be rebuilt if needed.

## Memory Scoring

Current write-path context scoring is deterministic code, not LLM-generated.
Before a user turn is persisted as an `event`, SuperAssist-Plus scores the
existing graph globally, matching CogniFold's write-path context selection.

Base node score:

```text
Score(v) = [0.4 * PR(v) + 0.4 * Recency(v) + 0.2 * Acc(v)] * U(v)
Recency(v) = exp(-0.01 * hours_since_last_accessed)
Acc(v) = node.access_count / max_access_count
```

`PR(v)` is weighted PageRank on the real directed graph. The write path does
not seed PageRank from the new event and does not add temporary reverse edges.
Support evidence can still be surfaced later by collecting incoming/outgoing
relationships around selected context nodes.

After scoring all existing nodes, the ranker builds a candidate pool of up to
`SUPERASSIST_PLUS_MEMORY_CANDIDATE_POOL_SIZE` nodes, then reranks and selects up
to `SUPERASSIST_PLUS_MEMORY_TOP_K` final context nodes:

```text
Immediate:  10%, 0.7 * Recency + 0.3 * (U - 1)
Working:    30%, 0.5 * PR + 0.3 * Recency + 0.2 * TypeBonus
Background: 50%, 0.8 * PR + 0.2 * Diversity
Buffer:     remaining nodes by base Score(v)
```

`U(v)` only boosts `intent` nodes targeted by `time -> intent DEADLINE_FOR`
edges whose deadline is in the next 24 hours. It ramps from `1.0` to `2.0`.

Read-path recall for the lead agent is separate from write-path scoring. It
uses vector entry points, bidirectional BFS traversal, and optional PPR blending
by default:

```env
SUPERASSIST_PLUS_MEMORY_READ_USE_PPR=true
SUPERASSIST_PLUS_MEMORY_READ_ENTRY_POINTS=10
SUPERASSIST_PLUS_MEMORY_READ_MAX_DEPTH=3
SUPERASSIST_PLUS_MEMORY_READ_BFS_WEIGHT=0.6
SUPERASSIST_PLUS_MEMORY_READ_PPR_WEIGHT=0.4
SUPERASSIST_PLUS_MEMORY_READ_BFS_DECAY=0.7
```

The lead agent receives read-path context. The memory writer receives the
write-path context used for graph updates.

Common memory controls:

```env
SUPERASSIST_PLUS_MEMORY_TOP_K=12
SUPERASSIST_PLUS_MEMORY_CANDIDATE_POOL_SIZE=150
SUPERASSIST_PLUS_MEMORY_COMPLETION_TOP_K=5
SUPERASSIST_PLUS_MEMORY_DECAY_LAMBDA=0.005
SUPERASSIST_PLUS_MEMORY_EDGE_DELETE_THRESHOLD=0.15
```

Edge weights use fixed defaults by edge type:

```text
GROUNDS=0.9, CAUSES=0.9, TRIGGERS=0.8, USER_FEEDBACK=0.8,
REINFORCES=0.7, PART_OF=0.7, DERIVED_FROM=0.6,
DEADLINE_FOR=0.6, RELATED_TO=0.5
```

If an existing edge is activated again, its weight is boosted by `0.05` up to
`1.0`. Edge decay is exponential:

```text
decayed_weight = edge.weight * exp(-SUPERASSIST_PLUS_MEMORY_DECAY_LAMBDA * age_days)
```

Edges below `SUPERASSIST_PLUS_MEMORY_EDGE_DELETE_THRESHOLD` are pruned.

## Running

Use the `CF` conda environment:

```powershell
conda activate CF
cd C:\Users\15746\Desktop\CODE\SuperAssist-Plus
```

If PowerShell blocks conda activation scripts, use `conda run` instead:

```powershell
conda run -n CF superassist-plus "hello" --flush-memory
```

If the command-line scripts are missing after editing `pyproject.toml`, refresh
the editable install:

```powershell
python -m pip install -e .
```

Single turn:

```powershell
superassist-plus "你好，记住我喜欢简洁直接的回答" --flush-memory
```

Continuous conversation:

```powershell
superassist-plus -i --flush-memory
superassist-plus -i --thread-id my-thread --flush-memory
```

Memory graph UI:

```powershell
superassist-plus-memory-ui --user-id local-user --port 8765
```

If PowerShell says `superassist-plus-memory-ui` is not recognized, either
activate `CF` first or call the script directly:

```powershell
C:\Users\15746\.conda\envs\CF\Scripts\superassist-plus-memory-ui.exe --user-id local-user --port 8765
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765/?user_id=local-user
```

Run tests:

```powershell
python -B -m pytest
```

## LangSmith Tracing

Set these in `.env` to send traces to LangSmith:

```env
LANGSMITH_TRACING=true
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=superassist-plus-dev
```

Then run through the `CF` environment:

```powershell
conda run -n CF superassist-plus "用 task 分发两个检查并汇总" --flush-memory
```

The trace includes `superassist.turn`, the lead LangChain agent, tool calls,
`task.dispatch`, and `subagent.run` spans with task description, subagent type,
task id, allowed tools, status, and prompt previews. Model token usage is shown
on LangSmith model spans when the provider returns usage metadata.
