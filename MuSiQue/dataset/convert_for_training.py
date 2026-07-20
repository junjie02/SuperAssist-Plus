#!/usr/bin/env python3
"""Convert raw trajectory JSONL to LLaMA-Factory training format.

Usage:
    # Convert to ShareGPT format (recommended for LLaMA-Factory)
    python dataset/convert_for_training.py \
        --input dataset/output/musique_trajectories.jsonl \
        --output dataset/output/train_sharegpt.jsonl \
        --format sharegpt

    # Only include successful parses with >=2 operations
    python dataset/convert_for_training.py \
        --input dataset/output/musique_trajectories.jsonl \
        --output dataset/output/train_filtered.jsonl \
        --min-operations 2 --skip-errors

    # Split into train/val (90/10)
    python dataset/convert_for_training.py \
        --input dataset/output/musique_trajectories.jsonl \
        --output dataset/output/train.jsonl \
        --val-output dataset/output/val.jsonl \
        --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def convert_sharegpt(trajectory: dict) -> dict | None:
    """Convert one trajectory to ShareGPT format.

    Returns None if the trajectory should be skipped.
    """
    inp = trajectory["input"]
    out = trajectory["output"]

    system_prompt = inp.get("system_prompt", "")
    user_prompt = inp.get("user_prompt", "")

    # Extract the assistant response — use RAW LLM output (what we want to train on),
    # fall back to serialized parsed plan only if raw is empty.
    raw = out.get("raw_response", "")
    parsed = out.get("parsed_plan")
    if raw.strip():
        assistant_msg = raw.strip()
    elif parsed is not None:
        assistant_msg = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        return None  # Skip empty responses

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_msg},
        ]
    }


def convert_alpaca(trajectory: dict) -> dict | None:
    """Convert one trajectory to Alpaca format."""
    inp = trajectory["input"]
    out = trajectory["output"]

    system_prompt = inp.get("system_prompt", "")
    user_prompt = inp.get("user_prompt", "")

    parsed = out.get("parsed_plan")
    raw = out.get("raw_response", "")

    if raw.strip():
        output = raw.strip()
    elif parsed is not None:
        output = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        return None

    return {
        "instruction": user_prompt,
        "input": "",
        "output": output,
        "system": system_prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert trajectory JSONL to LLaMA-Factory training format"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trajectory JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output training JSONL")
    parser.add_argument("--val-output", type=Path, default=None, help="Validation set output")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Fraction for validation (default: 0.0 = no split)")
    parser.add_argument("--format", choices=["sharegpt", "alpaca"], default="sharegpt",
                        help="Conversation format (default: sharegpt)")
    parser.add_argument("--fmt", choices=["jsonl", "json"], default="json",
                        help="Output file format: json (single array) or jsonl (one per line)")
    parser.add_argument("--min-operations", type=int, default=0,
                        help="Skip trajectories with fewer operations (default: 0)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip trajectories with parse errors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--max", type=int, default=0, help="Max trajectories to output (0=all)")

    args = parser.parse_args()

    random.seed(args.seed)

    converter = convert_sharegpt if args.format == "sharegpt" else convert_alpaca

    # Read and convert
    converted = []
    skipped_empty = 0
    skipped_ops = 0
    skipped_error = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)

            # Filters
            if args.skip_errors and traj["output"].get("parse_error"):
                skipped_error += 1
                continue

            ops = traj["output"].get("parsed_plan", {}).get("operations", []) or []
            if len(ops) < args.min_operations:
                skipped_ops += 1
                continue

            result = converter(traj)
            if result is None:
                skipped_empty += 1
                continue

            converted.append(result)

            if args.max > 0 and len(converted) >= args.max:
                break

    print(f"Converted: {len(converted)} trajectories")
    print(f"  Skipped: {skipped_empty} empty, {skipped_ops} low-ops, {skipped_error} errors")

    if not converted:
        print("ERROR: No valid trajectories to output!")
        return 1

    def _write(path: Path, data: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if args.fmt == "json":
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Split train/val
    if args.val_split > 0 and args.val_output:
        random.shuffle(converted)
        split_idx = int(len(converted) * (1 - args.val_split))
        train_data = converted[:split_idx]
        val_data = converted[split_idx:]

        _write(args.output, train_data)
        _write(args.val_output, val_data)

        print(f"Train: {len(train_data)} → {args.output}")
        print(f"Val:   {len(val_data)} → {args.val_output}")
    else:
        _write(args.output, converted)
        print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
