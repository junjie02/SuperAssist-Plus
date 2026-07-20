#!/usr/bin/env python3
"""Generate MuSiQue trajectories using the configured LLM.

Captures every LLM call (input prompts + output UpdatePlan operations)
as JSONL trajectories for dataset construction.

Usage:
    # Quick test: 10 trajectories
    python dataset/generate_musique.py --max-trajectories 10

    # Full run: 10000 trajectories
    python dataset/generate_musique.py --max-trajectories 10000 \\
        --model deepseek:deepseek-v4-pro

    # Resume interrupted run
    python dataset/generate_musique.py --max-trajectories 10000 --resume

    # Concurrent: 4 workers
    python dataset/generate_musique.py --max-trajectories 1000 --workers 4

    # Custom output
    python dataset/generate_musique.py --max-trajectories 100 \\
        --output-dir dataset/my_output --output-file test_trajectories.jsonl

Environment:
    DEEPSEEK_API_KEY   DeepSeek API key
    DEEPSEEK_BASE_URL  DeepSeek base URL (default: https://api.deepseek.com/v1)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows consoles (avoids GBK errors with non-ASCII text)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure project root and src/ are on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_dir = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Auto-load .env file if present
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Fallback: parse .env manually
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _key = _key.strip()
                    _val = _val.strip().strip('"').strip("'")
                    if _key and _key not in os.environ:
                        os.environ[_key] = _val

from dataset.collector import TrajectoryCollector
from dataset.config import TrajectoryDatasetConfig


# ---------------------------------------------------------------------------
# Per-example worker (used by both sequential and concurrent paths)
# ---------------------------------------------------------------------------

def _process_one_example(
    example: dict[str, Any],
    example_index: int,
    agent: Any,  # CognifoldAgent
    runner: Any,  # MuSiQueRunner
    collector: TrajectoryCollector,
    max_trajectories: int,
    context_max_nodes: int,
    done_event: threading.Event,
) -> int:
    """Process one MuSiQue example, generating trajectories for each event.

    Thread-safe. Each call creates its own ConceptGraph, PlanExecutor, and
    ContextRanker to avoid shared mutable state between threads.

    Args:
        example: MuSiQue example dict.
        example_index: Global example index (for logging).
        agent: Shared CognifoldAgent (thread-safe).
        runner: Shared MuSiQueRunner (read-only).
        collector: Shared TrajectoryCollector (thread-safe via lock).
        max_trajectories: Target trajectory count.
        context_max_nodes: Max context nodes for the LLM prompt.
        done_event: Set when target reached; workers check this periodically.

    Returns:
        Number of events successfully processed.
    """
    from cognifold.executor.runner import PlanExecutor
    from cognifold.graph.store import ConceptGraph
    from cognifold.scoring.ranker import ContextRanker, ScoringConfig

    example_id = example.get("id", f"ex_{example_index}")
    thread_tag = f"[W{threading.current_thread().name.split('-')[-1][:3]}]"
    print(f"{thread_tag} Example {example_index + 1}: {example_id}", flush=True)

    # Per-thread / per-example objects (no shared mutable state)
    graph = ConceptGraph()
    executor = PlanExecutor(graph, skip_embedding=True)
    scoring = ScoringConfig(context_window_size=context_max_nodes)
    ranker = ContextRanker(scoring)

    events = runner.build_events(example, example_index)
    print(f"{thread_tag}   Events: {len(events)}", flush=True)

    events_processed = 0

    for j, event in enumerate(events):
        # Early termination: target reached
        if done_event.is_set() or collector.count >= max_trajectories:
            break

        # Context window: PageRank-based node ranking
        try:
            node_scores_dict: dict[str, float] = {}
            scored = ranker.score_nodes(graph)
            for sn in scored[:context_max_nodes]:
                node_scores_dict[sn.node_id] = sn.composite_score
            context_ids = list(node_scores_dict.keys())
        except Exception:
            context_ids = []
            node_scores_dict = {}

        # Process event → LLM call → trajectory recorded
        try:
            plan = agent.process_event(
                event=event,
                graph=graph,
                context_node_ids=context_ids,
                node_scores=node_scores_dict,
                trajectory_collector=collector,
            )
            executor.execute(plan)

            events_processed += 1
            n_ops = len(plan.operations) if plan else 0
            ops_summary = ", ".join(
                f"{op.op.value}" for op in (plan.operations[:3] if plan else [])
            )
            safe_title = event.title.encode("ascii", errors="replace").decode("ascii")[:50]
            print(
                f"{thread_tag}   [{j + 1}/{len(events)}] {safe_title:<50s}  "
                f"{n_ops} ops [{ops_summary}]  trajectories={collector.count}",
                flush=True,
            )

        except Exception as e:
            msg = str(e)[:100]
            print(f"{thread_tag}   [{j + 1}/{len(events)}] ERROR: {msg}", flush=True)
            if "429" in str(e) or "rate" in str(e).lower() or "Rate" in str(e):
                print(f"{thread_tag}   Rate limit hit, sleeping 30s...", flush=True)
                time.sleep(30)
                # Retry once
                try:
                    plan = agent.process_event(
                        event=event,
                        graph=graph,
                        context_node_ids=context_ids,
                        node_scores=node_scores_dict,
                        trajectory_collector=collector,
                    )
                    executor.execute(plan)
                    events_processed += 1
                except Exception as e2:
                    print(f"{thread_tag}   Retry also failed: {e2}", flush=True)

    return events_processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MuSiQue trajectories for dataset construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 10 trajectories
  python dataset/generate_musique.py --max-trajectories 10

  # Production run with DeepSeek V4 Pro
  python dataset/generate_musique.py --max-trajectories 10000 --model deepseek:deepseek-v4-pro

  # Concurrent run with 4 workers
  python dataset/generate_musique.py --max-trajectories 1000 --workers 4

  # Resume a previous run
  python dataset/generate_musique.py --max-trajectories 10000 --resume
""",
    )

    # Model
    parser.add_argument("--model", type=str, default="deepseek:deepseek-v4-pro",
                        help="LLM model (default: deepseek:deepseek-v4-pro)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max output tokens (default: 8192)")

    # Agent exploration
    parser.add_argument("--max-exploration-steps", type=int, default=0,
                        help="Max tool-calling iterations per event. "
                             "0 = no tools, single LLM call (default, recommended). "
                             "Set to 1+ to enable graph exploration tools.")

    # Sampling
    parser.add_argument("--max-trajectories", type=int, default=10000,
                        help="Target number of trajectories (default: 10000)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max examples to process, 0=unlimited (default: 0)")
    parser.add_argument("--start-example", type=int, default=0,
                        help="Skip first N examples (for resume)")
    parser.add_argument("--limit", type=str, default="answerable",
                        help="Dataset limit: answerable|all|N (default: answerable)")

    # Concurrency
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads. 1 = sequential (default). "
                             "Set to 4-20 for concurrent processing. "
                             "Higher values increase throughput but also rate-limit risk.")

    # Output
    parser.add_argument("--output-dir", type=str, default="dataset/output",
                        help="Output directory (default: dataset/output)")
    parser.add_argument("--output-file", type=str, default="musique_trajectories.jsonl",
                        help="Output file name (default: musique_trajectories.jsonl)")
    parser.add_argument("--flush-every", type=int, default=10,
                        help="Flush to disk every N trajectories (default: 10)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-written IDs)")

    # Data
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Override MuSiQue data file path")
    parser.add_argument("--include-unanswerable", action="store_true",
                        help="Include unanswerable examples (default: answerable-only)")

    # Profile
    parser.add_argument("--profile", type=str, default="configs/prompt_profiles.yaml",
                        help="Prompt profiles YAML (default: configs/prompt_profiles.yaml)")
    parser.add_argument("--profile-name", type=str, default=None,
                        help="Prompt profile name (default: auto-detect)")

    # Graph context
    parser.add_argument("--context-max-nodes", type=int, default=60,
                        help="Max context nodes for retrieval (default: 60)")

    # Quality
    parser.add_argument("--min-operations", type=int, default=0,
                        help="Minimum ops in UpdatePlan to record (default: 0)")

    args = parser.parse_args()

    # --- Initialize ---
    print(f"Model: {args.model}")
    print(f"Target trajectories: {args.max_trajectories}")
    print(f"Workers: {args.workers}")
    print(f"Max exploration steps: {args.max_exploration_steps}")
    print(f"Output: {Path(args.output_dir) / args.output_file}")

    # --- Collector ---
    output_path = Path(args.output_dir) / args.output_file
    collector = TrajectoryCollector(output_path, flush_every=args.flush_every)
    # Attach min_operations filter (read by graph.py trajectory hook)
    collector._min_operations = args.min_operations  # type: ignore[attr-defined]

    if args.resume:
        existing = collector.total_count
        print(f"Resume mode: {existing} trajectories already in output file")
        # Compute start_example based on existing trajectories
        if existing > 0 and args.start_example == 0:
            # Rough estimate: ~20 events per example
            estimated_examples_done = existing // 20
            args.start_example = max(0, estimated_examples_done - 1)
            print(f"Estimated start example: {args.start_example}")

    # --- Load MuSiQue data ---
    from benchmarks.musique.run_benchmark import MuSiQueRunner
    from benchmarks.musique.download_data import download_musique

    runner = MuSiQueRunner()
    runner._include_unanswerable = args.include_unanswerable

    data_path = args.data_path or Path("benchmarks/musique/data/musique_validation.json")
    if not data_path.exists():
        print("Downloading MuSiQue data...")
        data_path = download_musique(split="validation")
    if not data_path.exists():
        data_path = Path("benchmarks/musique/data/musique_dev.json")
    if not data_path.exists():
        print("Error: MuSiQue data not found. Run benchmarks/musique/download_data.py first.")
        return 1

    examples = runner.load_dataset(data_path)

    # Apply limit
    if args.limit and args.limit != "answerable" and args.limit != "all":
        try:
            n = int(args.limit)
            examples = examples[:n]
        except ValueError:
            pass

    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    # Slice for resume
    examples = examples[args.start_example:]
    print(f"Processing {len(examples)} examples (starting from index {args.start_example})")

    # --- Agent setup ---
    from cognifold.agent.agent import CognifoldAgent
    from cognifold.agent.config import AgentConfig

    agent_config = AgentConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_exploration_steps=args.max_exploration_steps,
        domain="wiki",
    )

    agent = CognifoldAgent(config=agent_config)
    print(f"Agent initialized with model: {args.model}")

    # --- Dispatch ---
    done_event = threading.Event()
    total_events_processed = 0

    if args.workers <= 1:
        # ===== Sequential path (default) =====
        for i, example in enumerate(examples):
            if collector.count >= args.max_trajectories:
                print(f"\nReached target of {args.max_trajectories} trajectories. Stopping.")
                break

            n = _process_one_example(
                example=example,
                example_index=args.start_example + i,
                agent=agent,
                runner=runner,
                collector=collector,
                max_trajectories=args.max_trajectories,
                context_max_nodes=args.context_max_nodes,
                done_event=done_event,
            )
            total_events_processed += n

            if i % 5 == 0:
                collector.flush()

    else:
        # ===== Concurrent path =====
        print(f"\nProcessing with {args.workers} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures: list[concurrent.futures.Future] = []

            for i, example in enumerate(examples):
                # Stop submitting if target reached
                if done_event.is_set() or collector.count >= args.max_trajectories:
                    print(
                        f"\nReached target of {args.max_trajectories} trajectories. "
                        f"Stopping submission ({len(futures)} already in flight)."
                    )
                    break

                fut = pool.submit(
                    _process_one_example,
                    example=example,
                    example_index=args.start_example + i,
                    agent=agent,
                    runner=runner,
                    collector=collector,
                    max_trajectories=args.max_trajectories,
                    context_max_nodes=args.context_max_nodes,
                    done_event=done_event,
                )
                futures.append(fut)

            # Wait for in-flight examples to complete
            print(f"Waiting for {len(futures)} submitted examples to complete...")
            for fut in concurrent.futures.as_completed(futures):
                try:
                    n = fut.result()
                    total_events_processed += n
                except Exception as e:
                    print(f"Worker failed: {e}", file=sys.stderr)

            # Signal completion and flush
            done_event.set()
            collector.flush()

    # --- Done ---
    collector.close()
    print(f"\n{'=' * 60}")
    print(f"Done. Collected {collector.count} new trajectories.")
    # Total in file includes pre-existing (if resumed) plus new
    pre_existing = collector.total_count - collector.count
    print(f"Total in file: {pre_existing + collector.count}")
    print(f"Total events processed: {total_events_processed}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
