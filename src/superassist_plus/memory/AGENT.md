# Memory Module Technical Documentation

IMPORTANT: Any change to memory ontology, storage schema, recall scoring,
write plans, consolidation, embeddings, queue behavior, or memory tests must
update this document.

## Purpose

The `memory` module implements the CogniFold-style long-term memory mechanism
for SuperAssist-Plus. It replaces the inaccurate parts of SuperAssist's old
`long_term_memory` implementation with a typed graph model and explicit
structural debt mitigation.

## Files

- `embedding.py`: configurable embedding providers, including BGE via
  `sentence-transformers`, deterministic hash fallback, and cosine similarity.
- `scoring.py`: CogniFold write-path weighted PageRank, read-path BFS/PPR,
  node scoring, urgency, and tiered context assembly.
- `vector_index.py`: persistent FAISS dense vector index plus node-id mapping
  for semantic retrieval.
- `storage.py`: SQLite-backed typed graph persistence and strict edge
  validation.
- `service.py`: high-level memory operations: turn preparation, recall,
  reinforcement, CogniFold-style UpdatePlan application, consolidation.
- `writer.py`: LLM-assisted memory writer, fallback writer, and debounced
  in-process write queue.
- `__init__.py`: exports `MemoryService`.

## Ontology

Node types:

- `event`
- `concept`
- `intent`
- `time`

Edge types:

- `GROUNDS`
- `CAUSES`
- `TRIGGERS`
- `REINFORCES`
- `PART_OF`
- `DERIVED_FROM`
- `DEADLINE_FOR`
- `RELATED_TO`
- `USER_FEEDBACK`

`storage.py` validates edge source/target compatibility before writing.
`TRIGGERS` supports both `event -> intent` and `concept -> intent`, matching
CogniFold's event/context-triggered intent semantics.

## Write Path

1. `MemoryService.prepare_turn()` creates a transient reference probe from the
   incoming message timestamp and embedding. The probe is not persisted.
2. `MemoryContextRanker` computes weighted PageRank on the existing real
   directed graph. The write path does not seed PageRank from the current event
   and does not add temporary reverse edges.
3. Nodes are scored with:

   ```text
   Score(v) = [0.4 * PR(v) + 0.4 * Recency(v) + 0.2 * Acc(v)] * U(v)
   ```

   `Recency(v)` uses `exp(-0.01 * age_hours)`, `Acc(v)` is normalized
   `access_count`, and `U(v)` only boosts `intent` targets of
   `time -> intent DEADLINE_FOR` edges within the next 24 hours.
4. The ranker takes a candidate pool of up to
   `SUPERASSIST_PLUS_MEMORY_CANDIDATE_POOL_SIZE` nodes by base score and reranks
   it into Immediate, Working, Background, and Buffer tiers. Immediate
   emphasizes recency/urgency, Working emphasizes PR/recency/type, Background
   emphasizes PR/diversity, and Buffer fills by base score. The final injected
   context size is capped by `SUPERASSIST_PLUS_MEMORY_TOP_K`.
5. `prepare_turn()` touches selected context nodes so `access_count` becomes a
   real Hebbian activation signal.
6. Only after proactive context assembly does `prepare_turn()` create the
   current user-turn `event` node.
7. If semantic concept matching exceeds `memory_reinforce_similarity`, it
   writes or boosts an `event -> concept REINFORCES` edge. This matching is for
   reinforcement only; it does not seed write-path PageRank.
8. After the assistant answers, `MemoryWriteQueue` receives a
   `MemoryWritePayload` with the write-path memory context.
9. `MemoryWriter` creates a deterministic concept plan by default. Optional LLM
   memory planning is gated by `SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED=true`.
10. When LLM planning is enabled, the memory writer prompt asks for a
   CogniFold-style `UpdatePlan` with `operations`, including `ADD_NODE`,
   `ADD_EDGE`, `UPDATE_NODE`, `REMOVE_NODE`, `REMOVE_EDGE`, and `MERGE_NODES`.
11. `MemoryService.apply_structured_memory()` validates and executes operations.
   It creates nodes first, resolves short refs such as `current_event`, applies
   updates/edges/merges, and uses `grounded_in` to add missing grounding edges
   where the edge ontology allows it.
12. The old `nodes`/`edges` plan shape remains supported for deterministic
   fallback writes and backwards-compatible tests.
13. Consolidation runs after write.

## Embedding Providers

Default production configuration uses BGE:

```text
SUPERASSIST_PLUS_EMBEDDING_PROVIDER=bge
SUPERASSIST_PLUS_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
SUPERASSIST_PLUS_EMBEDDING_DEVICE=cpu
```

`BGEEmbedder` lazily loads `SentenceTransformer`, so model initialization occurs
only when memory embedding is first needed. The deterministic `HashEmbedder`
remains available with `SUPERASSIST_PLUS_EMBEDDING_PROVIDER=hash` for unit tests
and offline smoke runs.

Dense vectors are written to a persistent FAISS index under
`{SUPERASSIST_PLUS_DATA_DIR}/faiss/`. SQLite `memory_nodes.embedding_json`
remains as the authoritative vector copy for rebuilding, migration, and audit.

For each user, the index files are:

- `<safe_user_id>.index`: FAISS `IndexIDMap2(IndexFlatIP)` storing normalized
  dense vectors.
- `<safe_user_id>.mapping.json`: ordered mapping from FAISS integer ids to
  memory node ids.

Recall flow:

1. Embed the query with BGE or the configured test embedder.
2. Use the vector index to select semantic entry points.
3. Traverse outgoing and incoming graph edges from those entry points using
   BFS decay. If `SUPERASSIST_PLUS_MEMORY_READ_USE_PPR=true`, blend the BFS
   scores with Personalized PageRank seeded by the same entry points.
4. Rerank into Immediate, Working, Background, and Buffer tiers for the lead
   agent.
5. Touch selected nodes after recall so repeated explicit recall also
   contributes to access activation.

The FAISS index remains available for concept matching, rebuilds, persistence
tests, and read-path entry points, but the write-path ranking formula is no
longer `0.70 * semantic + 0.30 * graph_score`.

## Consolidation

- `merge_similar_concepts()`: merges concepts above threshold and transfers
  edges.
- `decay_edges()`: applies exponential decay and prunes weak edges.
- `complete_orphans()`: uses embedding similarity to reconnect orphan concepts
  to likely grounding events.
- `consolidate()`: runs all consolidation passes and returns counts.

## Memory Writer Prompt

The LLM memory writer prompt is defined in `writer.py` as
`MEMORY_WRITER_PROMPT`. It adapts CogniFold's cognitive graph update style:

- role: cognitive graph update agent;
- output: valid JSON `UpdatePlan`;
- node types: `event`, `concept`, `intent`, `time`;
- typed edge rules with default weights;
- required `grounded_in` evidence for non-event nodes;
- explicit operations including `MERGE_NODES`;
- self-review for missing edges and near-duplicate concepts.

LLM writing remains disabled by default because provider JSON compliance can be
fragile. The prompt and operation executor are ready for opt-in use once the
configured chat provider reliably returns valid JSON.

## Persistence

SQLite tables:

- `memory_nodes`
- `memory_edges`
- `memory_jobs`

The current queue is in-process and debounced. `memory_jobs` exists for durable
job evolution but is not yet the primary queue mechanism.

## Startup Loading

`AgentRuntime` calls `MemoryService.preload_embedder()` during initialization.
For BGE this constructs the `SentenceTransformer` model before the interactive
prompt accepts the first user message. The embedder object is cached at process
scope and remains resident for the lifetime of the process.

## Maintenance Notes

- Keep ontology changes synchronized with `models.py`, tests, and root docs.
- Treat BGE as the default semantic similarity mechanism. Keep the hash provider
  as a test fallback and do not use it as the production-quality ranking path.
- Keep memory code split across small files; do not rebuild a monolithic
  `long_term_memory.py`.
- Keep FAISS persistence isolated in `vector_index.py`; do not spread
  provider-specific vector search code through `service.py`.
- Keep SQLite as the graph/source-of-truth layer and FAISS as the vector
  retrieval layer. If they diverge, rebuild FAISS from SQLite.
