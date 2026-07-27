# RAG-EVAL

This project compares a conventional dense-vector RAG pipeline with the existing
SuperAssist LightRAG index over the same uploaded papers.

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
