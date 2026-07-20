#!/usr/bin/env python3
"""Convert CogniFold JSONL trajectories to ShareGPT format.

Usage:
    # Default: convert musique_trajectories.jsonl
    python dataset/convert_to_sharegpt.py

    # Custom input/output
    python dataset/convert_to_sharegpt.py -i dataset/output/musique_trajectories.jsonl -o dataset/output/musique_sharegpt.json

    # Keep only successful trajectories (default: skip errors)
    python dataset/convert_to_sharegpt.py --keep-errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CogniFold JSONL trajectories to ShareGPT format"
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
        help="Output JSON file (default: <input_stem>_sharegpt.json)",
    )
    parser.add_argument(
        "--keep-errors",
        action="store_true",
        help="Keep trajectories with parse errors (raw_response empty)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_sharegpt.jsonl"
    )

    conversations: list[dict] = []
    skipped = 0
    kept_errors = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            traj = json.loads(line)
            raw = (traj.get("output") or {}).get("raw_response", "")

            if not raw or not raw.strip():
                if args.keep_errors:
                    kept_errors += 1
                else:
                    skipped += 1
                    continue

            inp = traj.get("input") or {}
            system_prompt = inp.get("system_prompt", "")
            user_prompt = inp.get("user_prompt", "")

            # Strip tool-related content (tools are not used in this dataset)
            import re as _re
            system_prompt = _re.sub(
                r"\n*## Available Tools\n\n.*?(?=\n## )",
                "",
                system_prompt,
                flags=_re.DOTALL,
            )
            user_prompt = _re.sub(
                r"\n*Explore the graph with tools if needed, then provide your update plan as JSON\.\n*",
                "\n",
                user_prompt,
            )

            conv_turns = []

            # System prompt as system message
            if system_prompt:
                conv_turns.append({"from": "system", "value": system_prompt})

            # User prompt
            conv_turns.append({"from": "human", "value": user_prompt})

            # Model response
            conv_turns.append({"from": "gpt", "value": raw})

            conversations.append({
                "conversations": conv_turns,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Converted {len(conversations)} trajectories → {output_path}")
    if skipped:
        print(f"Skipped {skipped} trajectories with empty/invalid output "
              f"(use --keep-errors to include)")
    if kept_errors:
        print(f"Kept {kept_errors} trajectories with empty output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
