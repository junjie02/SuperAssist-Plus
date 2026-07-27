from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SUPERASSIST_ROOT = Path(
    os.getenv("SUPERASSIST_ROOT", str(EVAL_ROOT.parent / "SuperAssist"))
).resolve()
SUPERASSIST_SRC = SUPERASSIST_ROOT / "src"
if str(SUPERASSIST_SRC) not in sys.path:
    sys.path.insert(0, str(SUPERASSIST_SRC))

ARTIFACTS_DIR = EVAL_ROOT / "artifacts"
QUESTION_PATH = ARTIFACTS_DIR / "questions.json"
RESULTS_PATH = ARTIFACTS_DIR / "results.jsonl"
SUMMARY_JSON_PATH = ARTIFACTS_DIR / "summary.json"
SUMMARY_MD_PATH = ARTIFACTS_DIR / "summary.md"
VECTOR_INDEX_PATH = ARTIFACTS_DIR / "vector_index.npz"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    file_path: str
    order: int
    tokens: int
    content: str


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    measured_calls: int = 0

    def add_message(self, message: Any) -> None:
        self.calls += 1
        usage = getattr(message, "usage_metadata", None) or {}
        if usage:
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
            self._add(input_tokens, output_tokens, total_tokens)
            return

        metadata = getattr(message, "response_metadata", None) or {}
        raw = metadata.get("token_usage") or metadata.get("usage") or {}
        if raw:
            input_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
            output_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
            total_tokens = int(raw.get("total_tokens") or input_tokens + output_tokens)
            self._add(input_tokens, output_tokens, total_tokens)

    def _add(self, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.measured_calls += 1

    def snapshot(self) -> dict[str, int]:
        return asdict(self)

    def delta(self, before: dict[str, int]) -> dict[str, int]:
        now = self.snapshot()
        return {key: now[key] - int(before.get(key, 0)) for key in now}

    def add_usage(self, usage: dict[str, int]) -> None:
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.calls += int(usage.get("calls") or 0)
        self.measured_calls += int(usage.get("measured_calls") or 0)


def get_settings():
    from superassist.config import Settings

    return Settings(SUPERASSIST_DATA_DIR=str(SUPERASSIST_ROOT / ".superassist"))


def find_chunk_store() -> Path:
    candidates = sorted(
        (SUPERASSIST_ROOT / ".superassist" / "rag").glob(
            "*/index/default/kv_store_text_chunks.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No SuperAssist LightRAG chunk store was found")
    return candidates[0]


def load_chunks() -> list[Chunk]:
    path = find_chunk_store()
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(
            chunk_id=str(chunk_id),
            document_id=str(item.get("full_doc_id") or ""),
            file_path=str(item.get("file_path") or "unknown"),
            order=int(item.get("chunk_order_index") or 0),
            tokens=int(item.get("tokens") or 0),
            content=str(item.get("content") or ""),
        )
        for chunk_id, item in raw.items()
    ]
    return sorted(chunks, key=lambda item: (item.file_path.casefold(), item.order))


def document_names(chunks: list[Chunk]) -> list[str]:
    return sorted({item.file_path for item in chunks}, key=str.casefold)


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


def parse_json_response(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    starts = [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0]
    if not starts:
        raise ValueError("Model response does not contain JSON")
    start = min(starts)
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if end < start:
        raise ValueError("Model response contains incomplete JSON")
    return json.loads(cleaned[start : end + 1])


def normalize_text(value: str) -> str:
    return re.sub(r"\W+", "", value or "", flags=re.UNICODE).casefold()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
