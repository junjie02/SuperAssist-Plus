"""Text extraction for documents that reach the agent as real files.

Feishu delivers PDFs, Word files and spreadsheets as ``file`` messages. The
agent cannot read those with ``read_file`` (which only returns UTF-8 text), so
the channel extracts plain text here and hands it to the model while keeping the
original file on disk for later tool use.
"""

from __future__ import annotations

from pathlib import Path

DOCUMENT_MAX_CHARS = 24_000

TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".log",
        ".html",
        ".htm",
        ".xml",
        ".py",
        ".ini",
        ".toml",
        ".cfg",
    }
)
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})
DOC_SUFFIXES = frozenset({".doc"})
SHEET_SUFFIXES = frozenset({".xlsx", ".xlsm"})
LEGACY_SHEET_SUFFIXES = frozenset({".xls"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})

SUPPORTED_DOCUMENT_SUFFIXES = frozenset(
    TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | SHEET_SUFFIXES
)


def document_kind(path: str | Path) -> str:
    """Return a coarse kind used for prompts and routing."""

    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in DOCX_SUFFIXES:
        return "docx"
    if suffix in DOC_SUFFIXES:
        return "doc"
    if suffix in SHEET_SUFFIXES or suffix in LEGACY_SHEET_SUFFIXES:
        return "spreadsheet"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "binary"


def extract_document_text(path: str | Path, *, max_chars: int = DOCUMENT_MAX_CHARS) -> str:
    """Extract readable text from a document, truncated to ``max_chars``.

    Returns an empty string when the format is unsupported or extraction fails;
    callers are expected to surface the file path instead of an exception.
    """

    target = Path(path)
    if not target.is_file():
        return ""
    kind = document_kind(target)
    try:
        if kind == "pdf":
            text = _extract_pdf(target)
        elif kind == "docx":
            text = _extract_docx(target)
        elif kind == "spreadsheet":
            text = _extract_spreadsheet(target)
        elif kind == "text":
            text = _read_text_file(target)
        else:
            return ""
    except Exception:  # noqa: BLE001 - a broken document must not break the turn
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[内容已截断]"
    return text


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one damaged page should not kill the PDF
            page_text = ""
        page_text = page_text.strip()
        if page_text:
            pages.append(f"[第{index}页]\n{page_text}")
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    blocks: list[str] = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def _extract_spreadsheet(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    blocks: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if cell is None else str(cell) for cell in row]
                if any(cell.strip() for cell in cells):
                    rows.append(" | ".join(cells))
                if len(rows) >= 400:
                    rows.append("... [工作表已截断]")
                    break
            if rows:
                blocks.append(f"[工作表 {sheet.title}]\n" + "\n".join(rows))
    finally:
        workbook.close()
    return "\n\n".join(blocks)


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


__all__ = [
    "DOCUMENT_MAX_CHARS",
    "SUPPORTED_DOCUMENT_SUFFIXES",
    "document_kind",
    "extract_document_text",
]
