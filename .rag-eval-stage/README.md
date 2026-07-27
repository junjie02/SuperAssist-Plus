# RAG-EVAL

This project compares a conventional dense-vector RAG pipeline with the existing
SuperAssist LightRAG index over the same uploaded papers.

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
