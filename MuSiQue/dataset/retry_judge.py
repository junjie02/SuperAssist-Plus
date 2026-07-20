#!/usr/bin/env python3
"""Retry failed judge scoring entries with 50 workers and better extraction."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
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

from dataset.judge_score import JUDGE_PROMPT


def _extract_json_safe(text: str) -> dict:
    """Robust JSON extraction handling code fences and truncation."""
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    # Try to repair truncated JSON
    if text:
        # Count braces
        depth = 0
        in_string = False
        escape = False
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        # Close unclosed braces
        if in_string:
            text += '"'
        text += "}" * max(depth, 0)
    return json.loads(text)


def _score_one(traj: dict, model: str, api_key: str, base_url: str) -> dict | None:
    from openai import OpenAI

    conv = traj["conversations"]
    judge_input = f"## SYSTEM PROMPT\n\n{conv[0]['value']}\n\n## USER PROMPT\n\n{conv[1]['value']}\n\n## MODEL RESPONSE\n\n{conv[2]['value']}"

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
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            result = _extract_json_safe(content)
            traj["_judge"] = result
            return traj
        except Exception as e:
            wait = 2 ** attempt + random.random()
            if attempt < 4:
                time.sleep(wait)
            else:
                traj["_judge"] = {"error": str(e)[:200]}
                return traj


def main():
    input_path = Path("dataset/output/judge_scores.jsonl")
    output_path = Path("dataset/output/judge_scores_fixed.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        all_trajs = [json.loads(l) for l in f if l.strip()]

    ok = [t for t in all_trajs if "error" not in t.get("_judge", {})]
    failed = [t for t in all_trajs if "error" in t.get("_judge", {})]

    print(f"Total: {len(all_trajs)}  OK: {len(ok)}  Failed: {len(failed)}")

    if not failed:
        print("Nothing to retry.")
        return

    model_name = "deepseek-v4-flash"
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"

    print(f"Retrying {len(failed)} with 50 workers...")
    t0 = time.monotonic()
    retried = 0
    retry_ok = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(failed), desc="Retrying", unit="traj")
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(_score_one, t, model_name, api_key, base_url): i
                   for i, t in enumerate(failed)}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result and "error" not in result.get("_judge", {}):
                    retry_ok += 1
                retried += 1
                # Update the failed list entry
                idx = futures[fut]
                failed[idx] = result
            except Exception:
                retried += 1

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(ok=retry_ok)
            elif retried % 50 == 0:
                print(f"  {retried}/{len(failed)} ok={retry_ok}", flush=True)

    if use_tqdm:
        pbar.close()

    elapsed = time.monotonic() - t0
    print(f"\nRetry done in {elapsed:.0f}s. Recovered: {retry_ok}/{len(failed)}")

    # Merge: OK + retried failed
    all_fixed = ok + failed
    with open(output_path, "w", encoding="utf-8") as f:
        for t in all_fixed:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    still_failed = sum(1 for t in all_fixed if "error" in t.get("_judge", {}))
    print(f"Output: {output_path}  ({len(all_fixed)} total, {still_failed} still failed)")


if __name__ == "__main__":
    main()
