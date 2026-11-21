"""On-demand OCR for images that were saved into the tool workspace.

Design note
-----------
OCR is deliberately **not** part of the default image input path. Feishu images
are handed to the vision model as original bytes; calling OCR is an explicit
decision the model makes via the ``ocr_image`` tool.

This module owns the (slow) orientation detection: it is only executed when the
tool is actually called, never on every inbound image.
"""

from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import tool
from PIL import Image, ImageOps

from superassist.config import get_settings
from superassist.tools.files import _resolve_workspace_path

OCR_ORIENTATION_ANGLES = (0, 90, 180, 270)
OCR_ORIENTATION_MIN_CHARS = 12
OCR_ORIENTATION_MIN_CONFIDENCE = 0.65
OCR_ORIENTATION_MIN_IMPROVEMENT = 1.18
OCR_ORIENTATION_MIN_SECOND_BEST_IMPROVEMENT = 1.05


@dataclass
class OCRCandidate:
    angle: int
    lines: tuple[str, ...]
    character_count: int
    average_confidence: float
    horizontal_layout: float = 1.0

    @property
    def score(self) -> float:
        return self.character_count * self.average_confidence * self.horizontal_layout


_thread_local = threading.local()
_initialization_lock = threading.Lock()
_orientation_pool: ThreadPoolExecutor | None = None


def _get_engine() -> Any:
    engine = getattr(_thread_local, "engine", None)
    if engine is not None:
        return engine
    if getattr(_thread_local, "initialization_attempted", False):
        raise RuntimeError("OCR engine initialization previously failed")
    _thread_local.initialization_attempted = True
    with _initialization_lock:
        from rapidocr_onnxruntime import RapidOCR

        # One session per rotation runs at the same time; cap each session so
        # four concurrent sessions do not oversubscribe the CPU.
        engine = RapidOCR(intra_op_num_threads=_session_threads())
    _thread_local.engine = engine
    return engine


def _session_threads() -> int:
    cpus = os.cpu_count() or 4
    return max(1, cpus // len(OCR_ORIENTATION_ANGLES))


def _safe_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _horizontal_layout(value: Any) -> float | None:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.shape != (4, 2):
        return None
    width = (
        float(np.linalg.norm(points[1] - points[0]))
        + float(np.linalg.norm(points[2] - points[3]))
    ) / 2
    height = (
        float(np.linalg.norm(points[3] - points[0]))
        + float(np.linalg.norm(points[2] - points[1]))
    ) / 2
    if width <= 0 or height <= 0:
        return None
    return width / (width + height)


def _candidate(engine: Any, image: Image.Image, angle: int) -> OCRCandidate:
    candidate_image = image.rotate(angle, expand=True) if angle else image
    pixels = np.asarray(candidate_image)
    try:
        output = engine(pixels, use_cls=False)
    except TypeError:
        output = engine(pixels)
    results = output[0] if isinstance(output, tuple) else output
    lines: list[str] = []
    weighted_confidence = 0.0
    weighted_layout = 0.0
    layout_characters = 0
    character_count = 0
    for item in results or []:
        if not isinstance(item, (list, tuple)) or len(item) <= 1:
            continue
        text = str(item[1]).strip()
        valid_characters = sum(character.isalnum() for character in text)
        if valid_characters <= 0:
            continue
        lines.append(text)
        character_count += valid_characters
        weighted_confidence += _safe_confidence(item[2] if len(item) > 2 else 0.0) * valid_characters
        layout = _horizontal_layout(item[0] if item else None)
        if layout is not None:
            weighted_layout += layout * valid_characters
            layout_characters += valid_characters
    average_confidence = weighted_confidence / character_count if character_count else 0.0
    return OCRCandidate(
        angle=angle,
        lines=tuple(lines),
        character_count=character_count,
        average_confidence=average_confidence,
        horizontal_layout=weighted_layout / layout_characters if layout_characters else 1.0,
    )


def select_orientation(candidates: list[OCRCandidate]) -> OCRCandidate | None:
    """Pick the reading angle with the strongest, most confident text signal."""
    if not candidates:
        return None
    original = next((candidate for candidate in candidates if candidate.angle == 0), candidates[0])
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    best = ranked[0]
    if best.angle == 0:
        return original
    second_best_score = ranked[1].score if len(ranked) > 1 else 0.0
    if (
        best.character_count < OCR_ORIENTATION_MIN_CHARS
        or best.average_confidence < OCR_ORIENTATION_MIN_CONFIDENCE
        or best.score < max(original.score, 1.0) * OCR_ORIENTATION_MIN_IMPROVEMENT
        or best.score < max(second_best_score, 1.0) * OCR_ORIENTATION_MIN_SECOND_BEST_IMPROVEMENT
    ):
        return original
    return best


def _candidate_with_local_engine(image: Image.Image, angle: int) -> OCRCandidate:
    # The RapidOCR engine is thread-local, so each worker builds its own once.
    return _candidate(_get_engine(), image, angle)


def _get_orientation_pool() -> ThreadPoolExecutor:
    """Long-lived pool so worker threads keep their warm OCR engines."""
    global _orientation_pool
    if _orientation_pool is None:
        with _initialization_lock:
            if _orientation_pool is None:
                _orientation_pool = ThreadPoolExecutor(
                    max_workers=len(OCR_ORIENTATION_ANGLES),
                    thread_name_prefix="ocr-angle",
                )
    return _orientation_pool


def orientation_candidates(image: Image.Image) -> list[OCRCandidate]:
    """OCR all four rotations concurrently; order stays 0/90/180/270."""
    pool = _get_orientation_pool()
    return list(
        pool.map(lambda angle: _candidate_with_local_engine(image, angle), OCR_ORIENTATION_ANGLES)
    )


def _ocr_sync(path: Path, detect_orientation: bool) -> str:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if detect_orientation:
        selected = select_orientation(orientation_candidates(image))
    else:
        selected = _candidate(_get_engine(), image, 0)
    text = "\n".join(selected.lines).strip() if selected is not None else ""
    max_chars = get_settings().feishu_image_ocr_max_chars
    if max_chars > 0:
        text = text[:max_chars]
    if not text:
        return "OCR returned no legible text. Rely on the original image instead."
    rotation_note = f" (chosen rotation {selected.angle} degrees)" if selected.angle else ""
    return f"OCR text{rotation_note}:\n{text}"


@tool("ocr_image")
def ocr_image(path: str, detect_orientation: bool = True) -> str:
    """Extract text from an image stored in the tool workspace.

    Only call this when a transcription is genuinely useful — for example tiny
    print, a rotated page, or when you need to quote exact wording. The inbound
    image is already visible to you, so always verify OCR output against it.

    Args:
        path: Workspace-relative image path, e.g. ``图片/20261121/abc123.jpg``.
              The path is printed next to each attached image; do not OCR
              unrelated files that merely happen to sit in the same folder.
        detect_orientation: OCR 0/90/180/270 degrees in parallel and keep the
                            best reading. Disable it for a single fast pass on
                            an image you know is upright.
    """
    try:
        resolved = _resolve_workspace_path(path)
    except PermissionError as exc:
        return f"OCR failed: {exc}"
    if not resolved.is_file():
        return f"OCR failed: image not found in tool workspace: {path}"
    try:
        return _ocr_sync(resolved, bool(detect_orientation))
    except Exception as exc:  # noqa: BLE001 - tool must degrade, never crash the turn
        return f"OCR failed: {type(exc).__name__}: {exc}"
