#!/usr/bin/env python3
"""LLM-as-Judge: score all trajectories on 8 quality dimensions.

Uses DeepSeek V4 Flash for cost-efficient batch scoring.
Outputs judge_scores.jsonl with original data + scores + critique.

Usage:
    python dataset/judge_score.py
    python dataset/judge_score.py --workers 50 --limit 10  # dry run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# UTF-8 for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load .env
_env = _PROJECT_ROOT / ".env"
if _env.exists():
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v


JUDGE_PROMPT = """You are a strict quality evaluator for a cognitive graph update dataset. You will receive:

1. A SYSTEM PROMPT that defines the rules the model must follow
2. A USER PROMPT containing an event and context window
3. A MODEL RESPONSE (JSON) with reasoning, operations, and symbolic_actions

Score the model response on these 8 dimensions (each 1-5, where 5 = perfect):

### Scoring Dimensions

1. **format** (格式正确性): Is the JSON valid? Does it follow the expected schema? Are operation types correct?

2. **reasoning** (推理质量): Is the reasoning specific to this event? Does it reference actual content from the event text? Or is it vague/generic?

3. **operations** (操作合理性): Are the operations appropriate for this event + context? Check for:
   - Is the event properly added as ADD_NODE?
   - Are concepts/intents created ONLY when there's clear evidence?
   - Are edges created between related nodes?
   - Are there missing operations that SHOULD have been created?
   - Are there unnecessary operations that shouldn't exist?

4. **connectivity** (连通性): Are non-event nodes (concept/intent/time) properly connected with ADD_EDGE operations? No orphan nodes?

5. **grounding** (证据引用): Are grounded_in references non-empty and valid for concept/intent nodes? Do they reference actual evidence?

6. **dedup** (去重意识): If the context window already has similar concepts, did the model UPDATE instead of creating duplicates? Deduct points if the model created a duplicate concept when UPDATE should have been used.

7. **symbolic** (符号化动作): Are extracted symbolic_actions accurate? Are facts correctly extracted from the event text? No hallucinated facts?

8. **rules** (规则遵循): Does the response follow ALL important rules in the system prompt? Check for:
   - Operations in correct order (ADD_NODE before ADD_EDGE referencing it)
   - No orphan nodes without edges
   - grounded_in present for non-event nodes
   - reasoning present for non-event nodes
   - Return ONLY valid JSON (no extra text)

### Output Format

Return ONLY valid JSON (no markdown, no extra text):

{
  "scores": {
    "format": 5,
    "reasoning": 4,
    "operations": 3,
    "connectivity": 4,
    "grounding": 5,
    "dedup": 5,
    "symbolic": 4,
    "rules": 3
  },
  "overall": 4.0,
  "critique": "Specific issues found: ...",
  "improvement_hint": "To improve, the model should: ..."
}
"""


def _score_one(traj: dict, model: str, api_key: str, base_url: str) -> dict | None:
    """Score a single trajectory. Returns traj with scores merged in, or None on failure."""
    from openai import OpenAI

    conv = traj["conversations"]
    system_prompt = conv[0]["value"]
    user_prompt = conv[1]["value"]
    gpt_response = conv[2]["value"]

    judge_input = f"""## SYSTEM PROMPT

{system_prompt}

## USER PROMPT

{user_prompt}

## MODEL RESPONSE

{gpt_response}"""

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": judge_input},
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            # Clean code fences
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)
            # Merge scores into traj
            traj["_judge"] = result
            return traj
        except Exception as e:
            wait = 2 ** attempt + random.random()  # exponential backoff: 1, 2, 4, 8, 16s
            if attempt < 4:
                time.sleep(wait)
            else:
                traj["_judge"] = {"error": str(e)[:200]}
                return traj


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-Judge dataset scoring")
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("dataset/output/musique_sharegpt.jsonl"))
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("dataset/output/judge_scores.jsonl"))
    parser.add_argument("--model", type=str, default="deepseek:deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit trajectories (0=all)")
    args = parser.parse_args()

    # Resolve model
    model_name = args.model
    if model_name.startswith("deepseek:"):
        model_name = model_name.replace("deepseek:", "")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"

    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        return 1

    # Load trajectories
    with open(args.input, "r", encoding="utf-8") as f:
        trajs = [json.loads(l) for l in f if l.strip()]

    if args.limit > 0:
        trajs = trajs[:args.limit]

    print(f"Scoring {len(trajs)} trajectories with {args.workers} workers...")
    print(f"Model: {model_name}")

    # Score in parallel with progress bar
    scored: list[dict] = []
    failed = 0
    t0 = time.monotonic()
    total = len(trajs)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Scoring", unit="traj")
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score_one, t, model_name, api_key, base_url): i
                   for i, t in enumerate(trajs)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                if result and result.get("_judge", {}).get("error"):
                    failed += 1
                scored.append(result)
            except Exception as e:
                failed += 1
                trajs[idx]["_judge"] = {"error": str(e)[:200]}
                scored.append(trajs[idx])

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(failed=failed, rate=f"{len(scored)/(time.monotonic()-t0)*60:.0f}/min")
            elif len(scored) % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = len(scored) / elapsed * 60
                pct = len(scored) / total * 100
                print(f"  [{len(scored)}/{total} {pct:.0f}%] failed={failed} {rate:.0f}/min",
                      flush=True)

    if use_tqdm:
        pbar.close()

    # Sort back to original order and write
    scored.sort(key=lambda t: trajs.index(t) if t in trajs else 99999)  # best effort

    with open(args.output, "w", encoding="utf-8") as f:
        for t in scored:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - t0
    print(f"\nDone. {len(scored)} scored in {elapsed:.0f}s ({len(scored)/elapsed*60:.0f}/min)")
    print(f"Failed: {failed}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
