#!/usr/bin/env python3
"""Generate a distribution report from judge scores.

Usage:
    python dataset/report_scores.py
    python dataset/report_scores.py -i dataset/output/judge_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DIMENSIONS = ["format", "reasoning", "operations", "connectivity",
              "grounding", "dedup", "symbolic", "rules"]

DIM_LABELS = {
    "format": "格式正确性",
    "reasoning": "推理质量",
    "operations": "操作合理性",
    "connectivity": "连通性",
    "grounding": "证据引用",
    "dedup": "去重意识",
    "symbolic": "符号化动作",
    "rules": "规则遵循",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge score distribution report")
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("dataset/output/judge_scores.jsonl"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run judge_score.py first.")
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        trajs = [json.loads(l) for l in f if l.strip()]

    total = len(trajs)

    # Count scored vs unscored
    scored = [t for t in trajs if "_judge" in t and "error" not in t["_judge"]]
    errors = [t for t in trajs if "_judge" in t and "error" in t["_judge"]]
    print(f"Total: {total}  Scored: {len(scored)}  Errors: {len(errors)}")
    print()

    if not scored:
        print("No scores to report.")
        return 0

    # ── Overall distribution ──
    overalls = [t["_judge"]["overall"] for t in scored]
    bins = [(1.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5),
            (3.5, 4.0), (4.0, 4.5), (4.5, 5.0)]

    print("=== Overall 分数分布 ===")
    max_count = 0
    bin_counts = []
    for lo, hi in bins:
        c = sum(1 for o in overalls if lo <= o < hi)
        bin_counts.append((lo, hi, c))
        max_count = max(max_count, c)
    # Handle 5.0 exactly
    c_5 = sum(1 for o in overalls if o == 5.0)
    if c_5 > 0:
        bin_counts.append((5.0, 5.0, c_5))
        max_count = max(max_count, c_5)

    bar_max = 40
    for lo, hi, c in bin_counts:
        bar = "█" * int(c / max(max_count, 1) * bar_max)
        pct = c / len(scored) * 100
        lo_str = f"{lo:.1f}" if lo != 5.0 else "5.0"
        hi_str = f"{hi:.1f}" if hi != 5.0 else "5.0"
        print(f"  [{lo_str}-{hi_str})  {bar:<{bar_max+2}} {c:>4}  ({pct:.0f}%)")

    avg_overall = sum(overalls) / len(overalls)
    median_overall = sorted(overalls)[len(overalls) // 2]
    print(f"\n  Mean: {avg_overall:.2f}  Median: {median_overall:.2f}  "
          f"Min: {min(overalls):.1f}  Max: {max(overalls):.1f}")
    print()

    # ── Per-dimension averages ──
    print("=== 各维度均分 ===")
    dim_avgs = {}
    for dim in DIMENSIONS:
        scores = [t["_judge"]["scores"].get(dim, 0) for t in scored]
        avg = sum(scores) / len(scores)
        dim_avgs[dim] = avg
        bar = "▓" * int(avg * 8)
        print(f"  {DIM_LABELS[dim]:<8s} ({dim:<12s}): {avg:.2f}  {bar}")

    print()

    # ── Low-score analysis ──
    low = [t for t in scored if t["_judge"]["overall"] < 3.0]
    if low:
        print(f"=== 低分段 (overall < 3.0): {len(low)} 条 ===")
        # Find which dimensions are pulling scores down
        for dim in DIMENSIONS:
            low_dim = sum(1 for t in low
                          if t["_judge"]["scores"].get(dim, 0) <= 2)
            if low_dim > 0:
                print(f"  {DIM_LABELS[dim]}: {low_dim}/{len(low)} 条评分 <= 2 "
                      f"({low_dim/len(low)*100:.0f}%)")

        # Show a few critiques
        print("\n  样例 critique:")
        for t in low[:3]:
            c = t["_judge"].get("critique", "")[:150]
            title = t.get("conversations", [{}])[1].get("value", "")
            title_line = [l for l in title.split("\n") if l.startswith("Title:")]
            event_title = title_line[0][7:] if title_line else "?"
            print(f"  [{event_title[:40]}] overall={t['_judge']['overall']}")
            print(f"    {c}")

    print()

    # ── High-score analysis ──
    high = [t for t in scored if t["_judge"]["overall"] >= 4.0]
    print(f"=== 高分 (overall >= 4.0): {len(high)} 条 ({len(high)/len(scored)*100:.0f}%) ===")

    mid = [t for t in scored if 2.5 <= t["_judge"]["overall"] < 4.0]
    print(f"=== 中分 (2.5 <= overall < 4.0): {len(mid)} 条 ({len(mid)/len(scored)*100:.0f}%) ===")

    very_low = [t for t in scored if t["_judge"]["overall"] < 2.5]
    print(f"=== 极低 (overall < 2.5): {len(very_low)} 条 ({len(very_low)/len(scored)*100:.0f}%) ===")
    if very_low:
        print("  建议丢弃或人工审核")

    print(f"\n建议阈值参考:")
    print(f"  SFT  (高质量):  overall >= 4.0  →  {len(high)} 条")
    print(f"  DPO  (需改进):  2.5 <= overall < 4.0  →  {len(mid)} 条")
    print(f"  丢弃 (质量差):  overall < 2.5  →  {len(very_low)} 条")

    return 0


if __name__ == "__main__":
    sys.exit(main())
