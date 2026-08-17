"""Deterministic, structure-aware chunking for uploaded documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PLAIN_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百零0-9]+[章节篇部]|(?:\d+\.){0,3}\d+\s+|Slide\s+\d+|Sheet:\s*)\S.*$",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z0-9])")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LOCAL_TOKEN_RE = re.compile(r"\s*(?:[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]{1,4}|[^\s])")


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    document_name: str
    ordinal: int
    parent_id: str
    heading: str
    text: str
    token_count: int
    content_hash: str

    @property
    def searchable_text(self) -> str:
        prefix = f"{self.document_name}\n{self.heading}".strip()
        return f"{prefix}\n{self.text}" if prefix else self.text


def chunk_document(
    text: str,
    *,
    document_id: str,
    document_name: str,
    target_tokens: int = 384,
    max_tokens: int = 480,
    overlap_tokens: int = 64,
) -> list[DocumentChunk]:
    """Split normalized text along headings, paragraphs, and sentence boundaries."""

    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("Chunk sizes must satisfy 0 <= overlap < target <= max")
    normalized = normalize_document_text(text)
    if not normalized:
        return []

    raw_chunks: list[tuple[str, str]] = []
    for heading, section_text in _sections(normalized):
        raw_chunks.extend((heading, item) for item in _pack_section(section_text, target_tokens, max_tokens, overlap_tokens))

    chunks: list[DocumentChunk] = []
    for ordinal, (heading, chunk_text) in enumerate(raw_chunks):
        clean_text = chunk_text.strip()
        if not clean_text:
            continue
        digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        chunk_key = hashlib.sha256(f"{document_id}\0{ordinal}\0{digest}".encode()).hexdigest()[:24]
        parent_key = hashlib.sha256(f"{document_id}\0{heading}".encode()).hexdigest()[:20]
        chunks.append(
            DocumentChunk(
                id=f"chunk-{chunk_key}",
                document_id=document_id,
                document_name=document_name,
                ordinal=ordinal,
                parent_id=f"section-{parent_key}",
                heading=heading,
                text=clean_text,
                token_count=count_tokens(clean_text),
                content_hash=digest,
            )
        )
    return chunks


def normalize_document_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(character for character in value if character in "\n\t" or unicodedata.category(character) != "Cc")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    output: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if output and not blank:
                output.append("")
            blank = True
            continue
        output.append(line)
        blank = False
    return "\n".join(output).strip()


def lexical_terms(text: str) -> list[str]:
    """Return deterministic Latin terms and CJK bigrams for SQLite FTS5."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    terms = [token.casefold() for token in _LATIN_TOKEN_RE.findall(value)]
    for run in _CJK_RUN_RE.findall(value):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def lexical_text(text: str) -> str:
    return " ".join(lexical_terms(text))


def count_tokens(text: str) -> int:
    """Estimate tokens without downloading a provider-specific tokenizer."""

    return len(_local_tokens(text))


def truncate_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    tokens = _local_tokens(text)
    return text if len(tokens) <= limit else "".join(tokens[:limit]).rstrip()


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n\n".join(part for part in body if part.strip()).strip()
        if content:
            sections.append((" > ".join(heading_stack), content))
        body.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        markdown = _MARKDOWN_HEADING_RE.match(block)
        if markdown and "\n" not in block:
            flush()
            level = len(markdown.group(1))
            heading_stack[level - 1 :] = [markdown.group(2).strip()]
            continue
        if "\n" not in block and len(block) <= 120 and _PLAIN_HEADING_RE.match(block):
            flush()
            heading_stack = [block]
            continue
        body.append(block)
    flush()
    return sections or [("", text)]


def _pack_section(text: str, target: int, maximum: int, overlap: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(paragraph) if item.strip()]
        units.extend(sentences or [paragraph])

    output: list[str] = []
    current = ""
    for unit in units:
        if count_tokens(unit) > maximum:
            if current:
                output.append(current)
                current = _tail(current, overlap)
            pieces = _split_long_text(unit, maximum, overlap)
            output.extend(pieces[:-1])
            current = pieces[-1] if pieces else current
            continue

        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and (count_tokens(candidate) > maximum or count_tokens(current) >= target):
            output.append(current)
            prefix = _tail(current, overlap)
            candidate = f"{prefix}\n{unit}".strip() if prefix else unit
        current = candidate
    if current:
        output.append(current)
    return output


def _split_long_text(text: str, maximum: int, overlap: int) -> list[str]:
    tokens = _local_tokens(text)
    step = maximum - overlap
    return ["".join(tokens[start : start + maximum]).strip() for start in range(0, len(tokens), step)]


def _tail(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return "".join(_local_tokens(text)[-limit:]).strip()


def _local_tokens(text: str) -> list[str]:
    return _LOCAL_TOKEN_RE.findall(str(text or ""))


__all__ = [
    "DocumentChunk",
    "chunk_document",
    "count_tokens",
    "lexical_terms",
    "lexical_text",
    "normalize_document_text",
    "truncate_tokens",
]
