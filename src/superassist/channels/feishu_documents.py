from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from lark_oapi.api.docx.v1 import (
    Block,
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    CreateDocumentRequest,
    CreateDocumentRequestBody,
    Equation,
    Text,
    TextElement,
    TextRun,
)
from lark_oapi.api.drive.v1 import BaseMember, CreatePermissionMemberRequest

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_DELIMITED_MATH_RE = re.compile(
    r"\$\$(?P<double>.+?)\$\$"
    r"|\\\[(?P<bracket>.+?)\\\]"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|(?<!\\)(?<!\$)\$(?!\$)(?P<single>[^$\n]+?)(?<!\\)\$(?!\$)",
    re.DOTALL,
)
_LATEX_COMMAND_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|oint|lim|begin|end|left|right|"
    r"alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|omega|cdot|times|div|"
    r"leq|geq|neq|approx|infty|partial|nabla|mathbf|mathrm|text)\b"
)
_BLOCK_MARKER_RE = re.compile(r"^(?:#{1,9}\s+|[-*+]\s+|\d+[.)]\s+|>\s+|```|\$\$|\\\[)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


@dataclass(frozen=True)
class PublishedDocument:
    document_id: str
    title: str
    url: str


def contains_math_formula(text: str) -> bool:
    """Return true for LaTeX-like math while ignoring code and ordinary currency."""

    cleaned = _INLINE_CODE_RE.sub("", _FENCED_CODE_RE.sub("", str(text or "")))
    if _LATEX_COMMAND_RE.search(cleaned):
        return True
    for match in _DELIMITED_MATH_RE.finditer(cleaned):
        content = next((value for value in match.groupdict().values() if value is not None), "")
        if match.lastgroup != "single" or _looks_like_math(content):
            return True
    return False


def derive_document_title(markdown: str) -> str:
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("```", "$$", r"\[")):
            continue
        line = re.sub(r"^#{1,9}\s+", "", line)
        line = _DELIMITED_MATH_RE.sub("公式", line)
        line = _plain_markdown(line)
        if line:
            return line[:80]
    return "SuperAssist 公式回答"


def markdown_to_feishu_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1 if index < len(lines) else 0
            blocks.extend(_text_blocks("code", "\n".join(code_lines), parse_math=False))
            continue

        if stripped == "$$" or stripped == r"\[":
            closing = "$$" if stripped == "$$" else r"\]"
            formula_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != closing:
                formula_lines.append(lines[index].strip())
                index += 1
            index += 1 if index < len(lines) else 0
            formula = " ".join(item for item in formula_lines if item).strip()
            if formula:
                blocks.append(_formula_paragraph(formula))
            continue

        heading = re.match(r"^(#{1,9})\s+(.+)$", stripped)
        if heading:
            blocks.extend(_text_blocks(f"heading{len(heading.group(1))}", heading.group(2)))
            index += 1
            continue
        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)
        if bullet:
            blocks.extend(_text_blocks("bullet", bullet.group(1)))
            index += 1
            continue
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ordered:
            blocks.extend(_text_blocks("ordered", ordered.group(1)))
            index += 1
            continue
        quote = re.match(r"^>\s+(.+)$", stripped)
        if quote:
            blocks.extend(_text_blocks("quote", quote.group(1)))
            index += 1
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if not next_line or _BLOCK_MARKER_RE.match(next_line):
                break
            paragraph.append(next_line)
            index += 1
        blocks.extend(_text_blocks("text", "\n".join(paragraph)))
    return blocks or _text_blocks("text", "(empty response)")


class FeishuDocumentPublisher:
    def __init__(self, api_client: Any, *, doc_url_base: str) -> None:
        self.api_client = api_client
        self.doc_url_base = str(doc_url_base or "https://feishu.cn/docx").rstrip("/")

    def publish(
        self,
        markdown: str,
        *,
        chat_id: str,
        sender_open_id: str,
        is_private: bool,
        idempotency_key: str,
    ) -> PublishedDocument:
        title = derive_document_title(markdown)
        create_request = (
            CreateDocumentRequest.builder()
            .request_body(CreateDocumentRequestBody.builder().title(title).build())
            .build()
        )
        create_response = self.api_client.docx.v1.document.create(create_request)
        _require_success(create_response, "create document")
        document = getattr(getattr(create_response, "data", None), "document", None)
        document_id = str(getattr(document, "document_id", "") or "")
        if not document_id:
            raise RuntimeError("Feishu document creation returned no document_id")

        blocks = markdown_to_feishu_blocks(markdown)
        for chunk_index in range(0, len(blocks), 50):
            chunk = blocks[chunk_index : chunk_index + 50]
            client_token = hashlib.sha256(f"{idempotency_key}:{chunk_index // 50}".encode()).hexdigest()[:32]
            request = (
                CreateDocumentBlockChildrenRequest.builder()
                .document_id(document_id)
                .block_id(document_id)
                .client_token(client_token)
                .request_body(CreateDocumentBlockChildrenRequestBody.builder().children(chunk).build())
                .build()
            )
            response = self.api_client.docx.v1.document_block_children.create(request)
            _require_success(response, "write document blocks")

        self._grant_view_permission(
            document_id,
            member_type="openid" if is_private else "openchat",
            member_id=sender_open_id if is_private else chat_id,
        )
        return PublishedDocument(
            document_id=document_id,
            title=title,
            url=f"{self.doc_url_base}/{document_id}",
        )

    def _grant_view_permission(self, document_id: str, *, member_type: str, member_id: str) -> None:
        if not member_id:
            raise RuntimeError("Cannot grant Feishu document access without a member ID")
        member = BaseMember.builder().member_type(member_type).member_id(member_id).perm("view").build()
        request = (
            CreatePermissionMemberRequest.builder()
            .token(document_id)
            .type("docx")
            .need_notification(False)
            .request_body(member)
            .build()
        )
        response = self.api_client.drive.v1.permission_member.create(request)
        _require_success(response, "grant document permission")


def _text_blocks(kind: str, content: str, *, parse_math: bool = True) -> list[Block]:
    chunks = _split_text(content, 1800)
    return [_text_block(kind, chunk, parse_math=parse_math) for chunk in chunks if chunk]


def _text_block(kind: str, content: str, *, parse_math: bool) -> Block:
    elements = _inline_elements(content) if parse_math else [_text_run(content)]
    text = Text.builder().elements(elements).build()
    block_types = {
        "text": 2,
        "heading1": 3,
        "heading2": 4,
        "heading3": 5,
        "heading4": 6,
        "heading5": 7,
        "heading6": 8,
        "heading7": 9,
        "heading8": 10,
        "heading9": 11,
        "bullet": 12,
        "ordered": 13,
        "code": 14,
        "quote": 15,
    }
    builder = Block.builder().block_type(block_types[kind])
    return getattr(builder, kind)(text).build()


def _formula_paragraph(formula: str) -> Block:
    text = Text.builder().elements([_equation(formula)]).build()
    return Block.builder().block_type(2).text(text).build()


def _inline_elements(content: str) -> list[TextElement]:
    elements: list[TextElement] = []
    cursor = 0
    matches = list(_DELIMITED_MATH_RE.finditer(content))
    if not matches:
        command = _LATEX_COMMAND_RE.search(content)
        if command:
            end_match = re.search(r"[\u3400-\u9fff，。；：！？]", content[command.end() :])
            end = command.end() + end_match.start() if end_match else len(content)
            if command.start() > 0:
                elements.append(_text_run(_plain_markdown(content[: command.start()])))
            elements.append(_equation(content[command.start() : end].strip()))
            if end < len(content):
                elements.append(_text_run(_plain_markdown(content[end:])))
            return [element for element in elements if _element_has_content(element)]
    for match in matches:
        formula = next((value for value in match.groupdict().values() if value is not None), "").strip()
        if match.lastgroup == "single" and not _looks_like_math(formula):
            continue
        if match.start() > cursor:
            elements.append(_text_run(_plain_markdown(content[cursor : match.start()])))
        elements.append(_equation(formula))
        cursor = match.end()
    if cursor < len(content):
        elements.append(_text_run(_plain_markdown(content[cursor:])))
    return [element for element in elements if _element_has_content(element)] or [_text_run(_plain_markdown(content))]


def _text_run(content: str) -> TextElement:
    return TextElement.builder().text_run(TextRun.builder().content(content).build()).build()


def _equation(content: str) -> TextElement:
    return TextElement.builder().equation(Equation.builder().content(content.strip()).build()).build()


def _element_has_content(element: TextElement) -> bool:
    text_run = getattr(element, "text_run", None)
    equation = getattr(element, "equation", None)
    return bool(getattr(text_run, "content", "") or getattr(equation, "content", ""))


def _looks_like_math(content: str) -> bool:
    value = str(content or "").strip()
    if not value or re.fullmatch(r"[\d\s,.]+", value):
        return False
    return bool(re.search(r"[A-Za-z\\^_=+*/<>]|\d\s*[-+]\s*\d", value))


def _plain_markdown(text: str) -> str:
    value = _MARKDOWN_LINK_RE.sub(r"\1 (\2)", str(text or ""))
    value = re.sub(r"(?<!\\)(?:\*\*|__|~~)", "", value)
    value = re.sub(r"(?<!\\)`([^`]*)`", r"\1", value)
    return value.replace(r"\*", "*")


def _split_text(text: str, limit: int) -> list[str]:
    value = str(text or "")
    if len(value) <= limit:
        return [value]
    chunks: list[str] = []
    while value:
        cut = min(limit, len(value))
        if cut < len(value):
            newline = value.rfind("\n", 0, cut)
            space = value.rfind(" ", 0, cut)
            cut = max(newline, space, limit // 2)
        chunks.append(value[:cut].strip())
        value = value[cut:].lstrip()
    return chunks


def _require_success(response: Any, operation: str) -> None:
    success = getattr(response, "success", None)
    if callable(success) and success():
        return
    code = getattr(response, "code", None)
    message = getattr(response, "msg", None) or getattr(response, "message", None) or "unknown error"
    raise RuntimeError(f"Feishu failed to {operation}: code={code} message={message}")


__all__ = [
    "FeishuDocumentPublisher",
    "PublishedDocument",
    "contains_math_formula",
    "derive_document_title",
    "markdown_to_feishu_blocks",
]
