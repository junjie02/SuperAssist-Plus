#!/usr/bin/env python3
"""Clean CogniFold JSONL trajectories by removing low-quality entries.

Filters (each trajectory is removed if ANY filter matches):
  1. Parse error      — parse_error is not null
  2. Empty response   — raw_response is empty or whitespace
  3. Zero operations  — len(operations) == 0
  4. Empty reasoning  — plan.reasoning is empty (model violated prompt)
  5. Orphan node      — non-event ADD_NODE without corresponding ADD_EDGE in
                        the same plan (violates connectivity rules)

Usage:
    python dataset/clean_trajectories.py
    python dataset/clean_trajectories.py -i in.jsonl -o out.jsonl
    python dataset/clean_trajectories.py --dry-run   # stats only, no output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _check_orphan(operations: list[dict]) -> bool:
    """Return True if any non-event ADD_NODE lacks a connecting ADD_EDGE."""
    created_ids: set[str] = set()
    edge_refs: set[str] = set()

    for op in operations:
        if op.get("op") == "ADD_NODE" and op.get("node_type") in ("concept", "intent"):
            data = op.get("data") or {}
            nid = data.get("concept_id") or data.get("intent_id") or ""
            if nid:
                created_ids.add(nid)
        if op.get("op") == "ADD_EDGE":
            edge_refs.add(op.get("source_id") or "")
            edge_refs.add(op.get("target_id") or "")

    for nid in created_ids:
        if nid not in edge_refs:
            return True  # orphan found
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean CogniFold JSONL trajectories"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("dataset/output/musique_trajectories.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: <input_stem>_clean.jsonl)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, don't write output",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_clean.jsonl"
    )

    # ---- Read all trajectories ----
    trajectories: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(json.loads(line))

    total = len(trajectories)
    print(f"Loaded {total} trajectories from {input_path}")

    # ---- Apply filters ----
    filter_counts: Counter = Counter()
    kept: list[dict[str, Any]] = []

    for traj in trajectories:
        output = traj.get("output") or {}
        raw = (output.get("raw_response") or "").strip()
        err = output.get("parse_error")
        plan = output.get("parsed_plan") or {}
        ops: list[dict] = plan.get("operations") or []
        reasoning = (plan.get("reasoning") or "").strip()

        removed = False

        # Filter 1: Parse error
        if err is not None:
            filter_counts["parse_error"] += 1
            removed = True

        # Filter 2: Empty response
        if not raw:
            filter_counts["empty_response"] += 1
            removed = True

        # Filter 3: Zero operations
        if len(ops) == 0:
            filter_counts["zero_operations"] += 1
            removed = True

        # Filter 4: Empty reasoning (only for trajectories that have operations)
        if len(ops) > 0 and not reasoning:
            filter_counts["empty_reasoning"] += 1
            removed = True

        # Filter 5: Orphan concept/intent
        if _check_orphan(ops):
            filter_counts["orphan_node"] += 1
            removed = True

        if not removed:
            kept.append(traj)

    # ---- Report ----
    removed_total = total - len(kept)
    print(f"\n{'=' * 50}")
    print(f"Total:    {total:>6}")
    print(f"Kept:     {len(kept):>6}  ({len(kept) / total * 100:.1f}%)")
    print(f"Removed:  {removed_total:>6}  ({removed_total / total * 100:.1f}%)")
    print(f"\nBreakdown (may overlap — a trajectory can match multiple filters):")
    for name in ["parse_error", "empty_response", "zero_operations", "empty_reasoning", "orphan_node"]:
        c = filter_counts.get(name, 0)
        bar = "#" * (c * 40 // total) if total else ""
        print(f"  {name:<20s} {c:>5}  {bar}")

    # Overlap: how many match 2+ filters
    overlap = sum(1 for v in filter_counts.values() if v > 0)
    unique_removed = removed_total  # simplified — actual unique would need per-trajectory tracking
    print(f"\n  Unique trajectories removed: {removed_total}")

    # ---- Write ----
    if args.dry_run:
        print("\n[Dry run — no output written]")
        return 0

    with open(output_path, "w", encoding="utf-8") as f:
        for traj in kept:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"\nCleaned data written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
