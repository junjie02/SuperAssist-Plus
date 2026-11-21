from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from rag_eval.beir_metrics import evaluate_rankings, latency_summary, markdown_report


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "beir"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


def _load_superassist():
    configured = os.environ.get("SUPERASSIST_ROOT", "").strip()
    candidates = [Path(configured).resolve()] if configured else [ROOT.parent / "SuperAssist", ROOT.parent]
    project = next((path.resolve() for path in candidates if (path / "src" / "superassist").exists()), None)
    if project is None:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"SuperAssist project not found in: {rendered}")
    source = project / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from superassist.config import Settings
    from superassist.rag.chunking import lexical_text
    from superassist.rag.reranking import CrossEncoderReranker
    from superassist.rag.service import HybridRAGService

    return Settings, HybridRAGService, CrossEncoderReranker, lexical_text, project


def download_dataset(dataset: str, data_root: Path) -> Path:
    expected = data_root / dataset
    if (expected / "corpus.jsonl").exists():
        return expected
    data_root.mkdir(parents=True, exist_ok=True)
    return Path(util.download_and_unzip(BEIR_URL.format(dataset=dataset), str(data_root)))


def _index_signature(dataset: str, model: str, corpus: dict[str, dict[str, str]]) -> dict[str, Any]:
    corpus_digest = hashlib.sha256()
    for document_id in sorted(corpus):
        item = corpus[document_id]
        corpus_digest.update(document_id.encode("utf-8"))
        corpus_digest.update(str(item.get("title") or "").encode("utf-8"))
        corpus_digest.update(str(item.get("text") or "").encode("utf-8"))
    return {
        "dataset": dataset,
        "model": model,
        "corpus_documents": len(corpus),
        "corpus_sha256": corpus_digest.hexdigest(),
        "format_version": 1,
    }


def build_production_index(
    service: Any,
    user_id: str,
    corpus: dict[str, dict[str, str]],
    signature: dict[str, Any],
    *,
    batch_size: int,
    force: bool,
    lexical_text: Any,
) -> float:
    user_dir = service._user_dir(user_id)
    metadata_path = user_dir / "beir-index.json"
    database_path = service._database_path(user_id)
    dense_path, mapping_path = service._dense_paths(user_id)
    if not force and metadata_path.exists() and database_path.exists() and dense_path.exists() and mapping_path.exists():
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
                service._write_manifest(user_id, [_ready_manifest(signature)])
                return 0.0
        except (OSError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    service._ensure_store(user_id)
    with service._connect(user_id) as connection:
        connection.execute("DELETE FROM chunk_fts")
        connection.execute("DELETE FROM chunks")
        connection.commit()

    document_ids = sorted(corpus)
    total_batches = (len(document_ids) + batch_size - 1) // batch_size
    for offset in range(0, len(document_ids), batch_size):
        batch_number = offset // batch_size + 1
        batch_ids = document_ids[offset : offset + batch_size]
        texts = [_document_text(corpus[document_id]) for document_id in batch_ids]
        vectors = service._embedder.embed_many(texts)
        with service._connect(user_id) as connection:
            for document_id, text, vector in zip(batch_ids, texts, vectors, strict=True):
                item = corpus[document_id]
                title = str(item.get("title") or "").strip()
                array = np.asarray(vector, dtype="float32")
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, document_name, ordinal, parent_id, heading,
                        text, token_count, content_hash, embedding, embedding_dim
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        document_id,
                        document_id,
                        0,
                        "",
                        title,
                        text,
                        max(1, len(text.split())),
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        array.tobytes(),
                        len(array),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_fts (chunk_id, document_name, heading, body) VALUES (?, ?, ?, ?)",
                    (document_id, lexical_text(document_id), lexical_text(title), lexical_text(text)),
                )
            connection.commit()
        elapsed = time.perf_counter() - started
        print(
            f"[index] batch {batch_number}/{total_batches}; "
            f"documents {min(offset + len(batch_ids), len(document_ids))}/{len(document_ids)}; "
            f"elapsed {elapsed:.1f}s",
            flush=True,
        )

    with service._connect(user_id) as connection:
        service._rebuild_dense_index(user_id, connection)
    service._write_manifest(user_id, [_ready_manifest(signature)])
    metadata_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return time.perf_counter() - started


def run_ablation(
    *,
    dataset: str,
    split: str,
    model: str,
    modes: tuple[str, ...],
    top_k: int,
    candidate_top_k: int,
    batch_size: int,
    limit: int | None,
    sample_seed: int,
    corpus_limit: int | None,
    reranker_model: str,
    reranker_candidate_top_k: int,
    force_reindex: bool,
) -> dict[str, Any]:
    Settings, HybridRAGService, CrossEncoderReranker, lexical_text, project = _load_superassist()
    dataset_path = download_dataset(dataset, ARTIFACTS / "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)
    query_ids = sorted(qrels)
    if limit is not None:
        query_ids = sorted(random.Random(sample_seed).sample(query_ids, min(limit, len(query_ids))))
        qrels = {query_id: qrels[query_id] for query_id in query_ids}
    if corpus_limit is not None and corpus_limit > 0 and corpus_limit < len(corpus):
        gold_ids = {document_id for query_id in query_ids for document_id in qrels[query_id]}
        selected_ids = sorted(gold_ids)
        selected_ids.extend(document_id for document_id in sorted(corpus) if document_id not in gold_ids)
        corpus = {document_id: corpus[document_id] for document_id in selected_ids[:corpus_limit]}
        qrels = {
            query_id: {document_id: score for document_id, score in relevance.items() if document_id in corpus}
            for query_id, relevance in qrels.items()
        }

    run_name = f"{dataset}-{_safe_name(model)}"
    index_run_dir = ARTIFACTS / "runs" / run_name
    data_dir = index_run_dir / "superassist-data"
    if "hybrid_rerank" in modes:
        sample_name = "full" if limit is None else f"sample-{len(query_ids)}-seed-{sample_seed}"
        experiment_name = (
            f"rerank-{_safe_name(reranker_model)}-top-{reranker_candidate_top_k}-{sample_name}"
        )
        run_dir = index_run_dir / "experiments" / experiment_name
    else:
        run_dir = index_run_dir
    settings = Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=data_dir,
        SUPERASSIST_EMBEDDING_PROVIDER="bge",
        SUPERASSIST_EMBEDDING_MODEL=model,
        SUPERASSIST_EMBEDDING_DEVICE=os.environ.get("SUPERASSIST_EMBEDDING_DEVICE", "cpu"),
        SUPERASSIST_RAG_CANDIDATE_TOP_K=candidate_top_k,
        SUPERASSIST_RAG_CHUNK_TOP_K=top_k,
        SUPERASSIST_RAG_CONTEXT_MAX_TOKENS=50000,
        SUPERASSIST_RAG_RERANKER_ENABLED=False,
        SUPERASSIST_RAG_RERANKER_MODEL=reranker_model,
        SUPERASSIST_RAG_RERANKER_CANDIDATE_TOP_K=reranker_candidate_top_k,
    )
    reranker = None
    if "hybrid_rerank" in modes:
        # Keep the Hugging Face cache path short enough for legacy Windows path limits.
        reranker = CrossEncoderReranker(
            reranker_model,
            device=settings.embedding_device,
            cache_dir=str(ARTIFACTS / "huggingface"),
            batch_size=settings.rag_reranker_batch_size,
            max_length=settings.rag_reranker_max_length,
        )
    service = HybridRAGService(settings, reranker=reranker)
    user_id = f"beir:{dataset}:{model}"
    signature = _index_signature(dataset, model, corpus)
    try:
        index_seconds = build_production_index(
            service,
            user_id,
            corpus,
            signature,
            batch_size=batch_size,
            force=force_reindex,
            lexical_text=lexical_text,
        )
        summary: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "model": model,
            "modes": {},
            "corpus_documents": len(corpus),
            "queries": len(query_ids),
            "sample_seed": sample_seed if limit is not None else None,
            "candidate_top_k": candidate_top_k,
            "top_k": top_k,
            "corpus_limit": corpus_limit,
            "index_seconds": index_seconds,
            "reranker_model": reranker_model if "hybrid_rerank" in modes else None,
            "reranker_candidate_top_k": reranker_candidate_top_k,
            "superassist_root": str(project),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            rankings: dict[str, list[str]] = {}
            latencies: list[float] = []
            reranked_queries = 0
            result_path = run_dir / f"{mode}.jsonl"
            with result_path.open("w", encoding="utf-8") as handle:
                for index, query_id in enumerate(query_ids, start=1):
                    started = time.perf_counter()
                    retrieval_mode = "hybrid" if mode == "hybrid_rerank" else mode
                    result = service.retrieve(
                        user_id,
                        queries[query_id],
                        retrieval_mode,
                        rerank=mode == "hybrid_rerank",
                    )
                    elapsed = time.perf_counter() - started
                    if mode == "hybrid_rerank" and result.hits:
                        if not all(hit.get("reranker_score") is not None for hit in result.hits):
                            raise RuntimeError(
                                "Cross-encoder reranking fell back to RRF; benchmark results would be invalid"
                            )
                        reranked_queries += 1
                    ranked = [str(hit["chunk_id"]) for hit in result.hits]
                    rankings[query_id] = ranked
                    latencies.append(elapsed)
                    handle.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "query": queries[query_id],
                                "gold_document_ids": sorted(qrels[query_id]),
                                "retrieved_document_ids": ranked,
                                "latency_seconds": elapsed,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if index % 25 == 0 or index == len(query_ids):
                        print(f"[{mode}] {index}/{len(query_ids)} queries", flush=True)
            summary["modes"][mode] = {
                "metrics": evaluate_rankings(qrels, rankings),
                "latency": latency_summary(latencies),
                "reranked_queries": reranked_queries if mode == "hybrid_rerank" else None,
            }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "summary.md").write_text(markdown_report(summary), encoding="utf-8")
        return summary
    finally:
        service.close()


def _document_text(item: dict[str, str]) -> str:
    return "\n".join(part for part in (str(item.get("title") or "").strip(), str(item.get("text") or "").strip()) if part)


def _ready_manifest(signature: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"beir-{signature['dataset']}",
        "name": f"BEIR {signature['dataset']}",
        "storage_name": "",
        "content_hash": signature["corpus_sha256"],
        "size": 0,
        "status": "ready",
        "error": None,
        "characters": None,
        "chunks": signature["corpus_documents"],
        "created_at": now,
        "updated_at": now,
    }


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEIR retrieval ablations through SuperAssist production RAG")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", default="bm25,dense,hybrid")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-seed", type=int, default=42, help="Fixed random seed used with --limit")
    parser.add_argument("--corpus-limit", type=int, help="Limit corpus for a fast smoke run; omit for the official full benchmark")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=30)
    parser.add_argument("--force-reindex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    invalid = sorted(set(modes) - {"bm25", "dense", "hybrid", "hybrid_rerank"})
    if invalid:
        raise ValueError(f"Unsupported modes: {', '.join(invalid)}")
    summary = run_ablation(
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        modes=modes,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        batch_size=args.batch_size,
        limit=args.limit,
        sample_seed=args.sample_seed,
        corpus_limit=args.corpus_limit,
        reranker_model=args.reranker_model,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        force_reindex=args.force_reindex,
    )
    print(markdown_report(summary))


if __name__ == "__main__":
    main()
