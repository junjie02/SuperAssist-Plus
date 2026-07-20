#!/usr/bin/env python3
"""Split judge scores into SFT (perfect) and regenerate non-perfect for DPO.

Usage:
    python dataset/split_and_improve.py
    python dataset/split_and_improve.py --workers 100
"""

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


def _extract_json(text: str) -> str:
    """Clean LLM output to extract valid JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return text


def _improve_one(traj: dict, model: str, api_key: str, base_url: str) -> dict | None:
    """Regenerate one response using the judge's improvement hint."""
    from openai import OpenAI

    conv = traj["conversations"]
    system_prompt = conv[0]["value"]
    user_prompt = conv[1]["value"]
    original_response = conv[2]["value"]
    hint = traj["_judge"].get("improvement_hint", "")
    critique = traj["_judge"].get("critique", "")

    # Build improved user prompt with the critique
    improved_user = (
        f"{user_prompt}\n\n"
        f"[IMPORTANT IMPROVEMENT GUIDANCE]\n"
        f"Your previous response had these issues: {critique}\n"
        f"To improve: {hint}\n"
        f"Please produce a better response that addresses all the issues above."
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": improved_user},
                ],
                temperature=0.3,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            improved = _extract_json(content)
            # Verify it's valid JSON
            json.loads(improved)
            return {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": user_prompt},
                ],
                "chosen": improved,
                "rejected": original_response,
                "_meta": {
                    "overall": traj["_judge"]["overall"],
                    "critique": critique,
                },
            }
        except Exception as e:
            wait = 2 ** attempt + random.random()
            if attempt < 4:
                time.sleep(wait)
            else:
                return None


def main():
    parser = __import__("argparse").ArgumentParser(
        description="Split SFT + regenerate DPO")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--model", type=str, default="deepseek:deepseek-v4-pro")
    args = parser.parse_args()

    input_path = Path("dataset/output/judge_scores_fixed.jsonl")
    sft_path = Path("dataset/output/train_sft.jsonl")
    dpo_path = Path("dataset/output/train_dpo.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        trajs = [json.loads(l) for l in f if l.strip()]

    # ── Step 1: Split ──
    perfect = [t for t in trajs if t["_judge"]["overall"] == 5.0]
    imperfect = [t for t in trajs if t["_judge"]["overall"] < 5.0]

    print(f"Total: {len(trajs)}")
    print(f"Perfect (overall=5.0): {len(perfect)} → train_sft.jsonl")
    print(f"Imperfect (overall<5.0): {len(imperfect)} → regenerate for DPO")

    # Write SFT
    with open(sft_path, "w", encoding="utf-8") as f:
        for t in perfect:
            entry = {"conversations": t["conversations"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"SFT written: {sft_path} ({len(perfect)} entries)")

    if not imperfect:
        print("No imperfect entries to regenerate.")
        return

    # ── Step 2: Regenerate for DPO ──
    model_name = args.model
    if model_name.startswith("deepseek:"):
        model_name = model_name.replace("deepseek:", "")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"

    print(f"\nRegenerating {len(imperfect)} with {args.workers} workers...")
    print(f"Model: {args.model}")

    t0 = time.monotonic()
    dpo_entries = []
    failed = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(imperfect), desc="Improving", unit="traj")
        use_tqdm = True
    except ImportError:
        pbar = None
        use_tqdm = False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_improve_one, t, model_name, api_key, base_url): i
            for i, t in enumerate(imperfect)
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    dpo_entries.append(result)
                else:
                    failed += 1
            except Exception:
                failed += 1

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(ok=len(dpo_entries), failed=failed)
            elif (len(dpo_entries) + failed) % 100 == 0:
                print(f"  {len(dpo_entries)+failed}/{len(imperfect)} "
                      f"ok={len(dpo_entries)} failed={failed}", flush=True)

    if use_tqdm:
        pbar.close()

    # Write DPO
    with open(dpo_path, "w", encoding="utf-8") as f:
        for entry in dpo_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - t0
    print(f"\nDPO written: {dpo_path} ({len(dpo_entries)} entries, {failed} failed)")
    print(f"Done in {elapsed:.0f}s")
    print(f"\nFinal:")
    print(f"  train_sft.jsonl: {len(perfect)} entries")
    print(f"  train_dpo.jsonl: {len(dpo_entries)} entries (chosen=improved, rejected=original)")


if __name__ == "__main__":
    main()
