# RAG-EVAL

This project compares a conventional dense-vector RAG pipeline with the existing
SuperAssist LightRAG index over the same uploaded papers.

It also contains a BEIR retrieval-only ablation runner for the current production
`HybridRAGService`. The runner imports SuperAssist rather than copying the engine and
compares SQLite FTS5 BM25, FAISS Dense, and the production RRF Hybrid path.

The checked-in `.rag-eval-stage` directory is the source snapshot used to populate the
sibling `../RAG-EVAL` workspace. Run evaluation commands from `RAG-EVAL`, where generated
questions, indexes, per-question JSONL, and summaries are stored under `artifacts/`.

## Fair comparison

- Both systems reuse the exact LightRAG chunks: 1200 tokens with 100-token overlap.
- Both use SuperAssist's configured BGE embedding model and chat LLM.
- Vector RAG independently embeds and ranks chunks by cosine similarity.
- LightRAG opens the existing graph/vector stores and queries in `mix` mode.
- Both receive an approximately 6000-token retrieval budget and use the same answer prompt.
- LightRAG explicitly reserves 1500 tokens for entities, 1500 for relationships, and the remainder for original chunks.
- Answer generation has no tools. Web search is not registered or callable.
- The prompt forbids model-knowledge fallback and requires an explicit insufficient-evidence answer.
- Judge tokens are excluded from system token cost.

The generated 100-question bank targets all three uploaded papers and normally contains:

- about 90 single-paper questions, balanced across the three documents;
- about 10 within-paper cross-section questions;
- cross-paper questions may be used as fallback when grounded evidence is available.

Every item includes a reference answer, gold chunk IDs, source paper names, and verbatim evidence.

## Environment

By default, the scripts expect the sibling project at `../SuperAssist` and load its `.env`, model,
embedding implementation, uploaded documents, and LightRAG database. Override it with
`SUPERASSIST_ROOT` when needed.

Use the same Python environment as SuperAssist:

```powershell
E:\Conda\envs\CF\python.exe -m pip install -e .
```

## Run

Run the BEIR SciFact ablation with the English BGE model:

```powershell
E:\Conda\envs\CF\python.exe -m rag_eval.beir_ablation --dataset scifact --model BAAI/bge-base-en-v1.5
```

Add `--modes bm25,dense,hybrid,hybrid_rerank` to evaluate cross-encoder reranking.
The English default reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`; override it with
`--reranker-model`. Production can use the multilingual `BAAI/bge-reranker-v2-m3`.

Use `--limit 20 --corpus-limit 500` for a fast smoke run. Data, production-compatible indexes, per-query JSONL,
and summaries are written under `artifacts/beir/`. BEIR is an English retrieval benchmark;
its result must be paired with a separately labelled Chinese domain set before changing
the production model from `BAAI/bge-base-zh-v1.5`.

The BEIR run treats each corpus document as one gold-addressable evidence unit. This
isolates retrieval ranking from SuperAssist's document chunking quality; chunk-size and
Agentic query-rewrite ablations belong in a separate domain evaluation.

## Current BEIR SciFact Baseline

The full run uses all 5,183 SciFact documents and 300 test queries. It evaluates the
production-compatible retrieval implementation with candidate Top-100 and returned Top-20.
Metrics were independently checked against BEIR `EvaluateRetrieval`.

| Metric | BM25 | Dense | Hybrid RRF |
| --- | ---: | ---: | ---: |
| NDCG@10 | 0.6741 | **0.7376** | 0.7294 |
| MAP@10 | 0.6305 | **0.6918** | 0.6852 |
| MRR@10 | 0.6411 | **0.7004** | 0.6988 |
| Recall@10 | 0.7956 | **0.8659** | 0.8494 |
| Recall@20 | 0.8326 | **0.9132** | 0.8926 |
| Latency mean | **0.1108s** | 0.1443s | 0.1317s |
| Latency P50 | **0.0523s** | 0.1280s | 0.1247s |
| Latency P95 | 0.4299s | 0.2350s | **0.1904s** |
| Latency P99 | 0.8360s | 0.2983s | **0.2193s** |

On this dataset, `BAAI/bge-base-en-v1.5` Dense retrieval beats the current fixed-weight
RRF Hybrid path on every reported quality metric. Parallel branch execution makes Hybrid
latency comparable to Dense in this run, but does not fix the ranking loss. This is evidence
to tune or learn fusion on a development split, not evidence to remove BM25 from the Chinese
product: SciFact is English and its documents are used as whole evidence units.

The latency rows above are from the 2026-08-22 full rerun after Dense and BM25 were made
concurrent in Hybrid mode. Retrieval quality is unchanged. Compared with the preceding
sequential artifact, Hybrid P95 fell from `0.5580s` to `0.1904s` and P99 from `0.9547s`
to `0.2193s`. Absolute latency varies with host load, so the raw per-query JSONL files are
retained with each run.

Model policy:

- use `BAAI/bge-base-en-v1.5` for this English BEIR baseline;
- keep `BAAI/bge-base-zh-v1.5` as the production default until a labelled Chinese domain
  evaluation compares it with candidates such as `BAAI/bge-m3`;
- do not compare models unless corpus, queries, qrels, Top-K, and metric code are fixed;
- rebuild the FAISS index whenever the embedding model changes.

### English embedding model ablation

Both models use the same full SciFact corpus, 300 queries, production embedding wrapper,
FAISS search, and Top-K settings. Index time includes corpus embedding and FAISS construction
on the current Windows CPU while the normal SuperAssist service remained running.

| Metric | bge-small-en-v1.5 | bge-base-en-v1.5 |
| --- | ---: | ---: |
| NDCG@10 | 0.7200 | **0.7376** |
| MAP@10 | 0.6764 | **0.6918** |
| MRR@10 | 0.6845 | **0.7004** |
| Recall@10 | 0.8452 | **0.8659** |
| Recall@20 | 0.8753 | **0.9132** |
| Query latency P50 | **0.0598s** | 0.1037s |
| Query latency P95 | **0.0715s** | 0.1356s |
| First index build | **420.0s** | 1267.6s |

`bge-base-en-v1.5` remains the primary English benchmark model because it wins every
reported quality metric. `bge-small-en-v1.5` is a viable CPU-constrained option: it built
the index about 3x faster and reduced P50 query latency by about 42%, at the cost of 3.79
percentage points of Recall@20. This comparison says nothing about the Chinese production
model; that decision requires Chinese qrels and domain documents.

### Cross-encoder reranker ablation

The reranker receives the actual document text only: evaluation corpus IDs are not included,
and a title already present at the beginning of `text` is not duplicated. The runner also
stores each reranker/model/sample combination in a separate experiment directory and fails
instead of reporting results if retrieval falls back to the original RRF order.

| Metric | Hybrid RRF | Hybrid + Reranker |
| --- | ---: | ---: |
| NDCG@10 | **0.7294** | 0.6981 |
| MAP@10 | **0.6852** | 0.6475 |
| MRR@10 | **0.6988** | 0.6639 |
| Recall@10 | **0.8494** | 0.8378 |
| Recall@20 | 0.8926 | **0.9026** |
| Latency P50 | **0.1344s** | 1.5137s |
| Latency P95 | **0.1642s** | 1.7792s |

After correcting the pair text, the same MiniLM full run changes to NDCG@10 `0.6981`,
MAP@10 `0.6475`, Recall@10 `0.8378`, and Recall@20 `0.9026`; it still trails the Hybrid
baseline on top-rank quality. This confirms the earlier degradation was partly caused by
the malformed pair text, but this small MS MARCO model is still not a production choice.

A fixed random three-query smoke test with the production candidate
`BAAI/bge-reranker-v2-m3` and Top-10 reranker candidates gives NDCG@10 `0.8770` versus
`0.6667` for Hybrid, and Recall@10 `1.0000` versus `0.6667`. It takes a CPU median of
`18.55s/query`, so this is a quality smoke test, not a deployment benchmark. A statistically
meaningful full run requires GPU or a dedicated reranking service. Keep production reranking
disabled until a larger labelled Chinese evaluation and an infrastructure latency budget are
available.

### Legacy LightRAG comparison

Generate or resume the grounded question bank:

```powershell
E:\Conda\envs\CF\python.exe -m rag_eval.generate_questions --count 100
```

Run a two-question smoke evaluation first:

```powershell
E:\Conda\envs\CF\python.exe -m rag_eval.evaluate --limit 2 --results artifacts/smoke-results.jsonl
```

Run the complete evaluation:

```powershell
E:\Conda\envs\CF\python.exe -m rag_eval.evaluate
```

The evaluator checkpoints every completed question. Re-running resumes from `artifacts/results.jsonl`.
Final metrics are written to `artifacts/summary.json` and `artifacts/summary.md`.

## Metrics

- answer accuracy and average 0-2 judge score;
- retrieval hit rate, all-evidence hit rate, evidence recall, and evidence precision;
- retrieval and end-to-end P50/P95 latency;
- average input, output, and total tokens per question;
- token metadata coverage, so unavailable provider usage is visible instead of silently reported as zero.

Metric definitions:

- **Answer accuracy**: percentage of answers receiving score 2 from the evidence-grounded LLM judge.
- **Retrieval hit rate**: percentage of questions retrieving at least one gold evidence chunk.
- **All-evidence hit rate**: percentage retrieving every gold chunk required by a question; this is binary per question.
- **Mean evidence recall**: average `retrieved gold / all gold`; partial retrieval receives partial credit.
- **Mean evidence precision**: average `retrieved gold / all returned provenance chunks`.

For LightRAG, returned provenance includes source chunk IDs attached to selected entities and relationships,
not only raw chunks included verbatim. This improves measured evidence coverage but lowers precision, so precision
should be read as source concentration rather than final-answer correctness.

## Current 100-Question Baseline

The completed baseline uses 91 shared chunks from `deepseekv4.pdf`, `GLM5.2.pdf`, and `lightRAG.pdf`.
The question bank contains 89 single-paper questions and 11 within-paper cross-section questions.

| Metric | Vector RAG | LightRAG |
| --- | ---: | ---: |
| Answer accuracy | 55.00% | 60.00% |
| Average answer score (0-2) | 1.160 | 1.270 |
| Retrieval hit rate | 50.00% | 81.00% |
| All-evidence hit rate | 43.00% | 80.00% |
| Mean evidence recall | 46.50% | 80.50% |
| Mean evidence precision | 10.40% | 4.85% |
| Retrieval latency P50 | 0.067s | 6.531s |
| Retrieval latency P95 | 0.301s | 19.534s |
| End-to-end latency P50 | 6.395s | 14.545s |
| End-to-end latency P95 | 24.426s | 26.855s |
| Average total tokens/question | 6694.5 | 4918.8 |

Both systems have 100% provider token-metadata coverage. Judge usage was 91,548 tokens and is excluded from
system cost. The final run hit the cached Vector RAG index, so the latency table measures query-time behavior,
not first-time vector indexing or LightRAG graph construction.

This is a project baseline, not a publication-grade claim: questions and scores use LLM generation/judging and
cover only three papers. Human review and additional external datasets are required before generalizing the result.
