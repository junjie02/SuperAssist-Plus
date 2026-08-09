from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from openai import OpenAI
from PIL import Image

from superassist.config import get_settings

MAX_SEARCH_RESULTS = 8
MAX_PRESENTED_IMAGES = 3
MAX_INSPECTED_IMAGES = 4
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
MAX_ORIGINAL_BYTES = 10 * 1024 * 1024
MAX_GENERATED_IMAGE_BYTES = MAX_ORIGINAL_BYTES
MAX_IMAGE_PROMPT_CHARS = 4000
GENERATED_IMAGE_TTL_SECONDS = 24 * 60 * 60
GPT_IMAGE_MODEL = "gpt-image-2"
GPT_IMAGE_SIZES = {"auto", "1024x1024", "1536x1024", "1024x1536"}
GPT_IMAGE_QUALITIES = {"auto", "low", "medium", "high"}
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


@tool("image_search")
def image_search(
    query: str,
    max_results: int = 4,
    *,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """Search for images and show temporary low-detail candidates to the agent.

    Search results are not sent to the user. Review the returned candidate images,
    optionally call inspect_image or search again, and call present_images only for
    candidates that should appear in the final Feishu response.

    Args:
        query: A focused image-search query.
        max_results: Number of candidates to inspect, from 1 to 8. Defaults to 4.
    """

    if not get_settings().tool_network_enabled:
        return _command_message(
            tool_call_id,
            "Image search is disabled by SUPERASSIST_TOOL_NETWORK_ENABLED=false.",
            status="error",
        )
    query = str(query or "").strip()
    if not query:
        return _command_message(tool_call_id, "Image search requires a non-empty query.", status="error")

    from ddgs import DDGS

    limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    raw_results = DDGS().images(query, max_results=limit, safesearch="moderate")
    candidates: dict[str, dict[str, Any]] = {}
    content: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw in raw_results or []:
        candidate = _normalize_candidate(raw, query=query)
        if candidate is None:
            continue
        candidate_id = f"img_{uuid4().hex[:10]}"
        candidate["candidate_id"] = candidate_id
        try:
            data, mime_type = download_image_url(
                candidate["thumbnail_url"] or candidate["image_url"],
                max_bytes=MAX_THUMBNAIL_BYTES,
                timeout=12,
            )
        except Exception as exc:  # noqa: BLE001 - one bad result must not fail the search
            errors.append(f"{candidate_id}: {type(exc).__name__}")
            continue
        candidates[candidate_id] = candidate
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": json.dumps(_model_candidate(candidate), ensure_ascii=False),
                },
                {
                    "type": "input_image",
                    "image_url": _image_data_url(data, mime_type),
                    "detail": "low",
                },
            ]
        )

    if not candidates:
        detail = f" Download errors: {', '.join(errors)}" if errors else ""
        return _command_message(tool_call_id, f"No usable image candidates found for {query!r}.{detail}")

    content.insert(
        0,
        {
            "type": "input_text",
            "text": (
                f"Temporary image-search candidates for query {query!r}. "
                "Visually assess them. They will not be sent unless you call present_images."
            ),
        },
    )
    return Command(
        update={
            "image_search_results": candidates,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )


@tool("inspect_image")
def inspect_image(
    candidate_ids: list[str],
    *,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """Inspect selected image-search candidates at higher detail.

    Args:
        candidate_ids: Candidate IDs returned by image_search. At most 4.
    """

    registry = dict(state.get("image_search_results") or {})
    selected, invalid = _resolve_candidates(candidate_ids, registry, MAX_INSPECTED_IMAGES)
    if invalid or not selected:
        message = _invalid_candidates_message(invalid, bool(selected))
        return _command_message(tool_call_id, message, status="error")

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "Higher-detail temporary image candidates. These are still not visible to the user.",
        }
    ]
    failures: list[str] = []
    for candidate in selected:
        try:
            data, mime_type = _download_candidate(candidate, max_bytes=MAX_ORIGINAL_BYTES)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{candidate['candidate_id']}: {type(exc).__name__}")
            continue
        content.extend(
            [
                {"type": "input_text", "text": json.dumps(_model_candidate(candidate), ensure_ascii=False)},
                {
                    "type": "input_image",
                    "image_url": _image_data_url(data, mime_type),
                    "detail": "high",
                },
            ]
        )
    if len(content) == 1:
        return _command_message(
            tool_call_id,
            f"Unable to download the selected candidates: {', '.join(failures)}",
            status="error",
        )
    if failures:
        content.append({"type": "input_text", "text": f"Some candidates failed: {', '.join(failures)}"})
    return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})


@tool("present_images")
def present_images(
    candidate_ids: list[str],
    *,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """Select searched images for the final Feishu response.

    This is the only image-search tool that authorizes user-visible delivery.
    Call it only after visually checking relevance. At most 3 images may be selected.

    Args:
        candidate_ids: Candidate IDs returned by image_search. At most 3.
    """

    registry = dict(state.get("image_search_results") or {})
    selected, invalid = _resolve_candidates(candidate_ids, registry, MAX_PRESENTED_IMAGES)
    if invalid or not selected:
        message = _invalid_candidates_message(invalid, bool(selected))
        return _command_message(tool_call_id, message, status="error")
    outbound = [_outbound_candidate(item) for item in selected]
    ids = ", ".join(item["candidate_id"] for item in outbound)
    return Command(
        update={
            "outbound_images": outbound,
            "messages": [
                ToolMessage(
                    content=(
                        f"Selected {len(outbound)} image(s) for the final Feishu response: {ids}. "
                        "Now write the final answer and explain why these images are relevant."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool("generate_image")
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "medium",
    *,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """Generate one user-visible image with GPT Image 2.

    The generated image is returned for visual inspection and is automatically
    attached to the final Feishu response. Do not call present_images for it.

    Args:
        prompt: A concrete visual description of the image to create.
        size: Output size: 1024x1024, 1536x1024, 1024x1536, or auto.
        quality: Output quality: low, medium, high, or auto.
    """

    del state  # Injected for a stable tool interface; generated images do not need a candidate registry.
    settings = get_settings()
    if not settings.tool_network_enabled:
        return _command_message(
            tool_call_id,
            "Image generation is disabled by SUPERASSIST_TOOL_NETWORK_ENABLED=false.",
            status="error",
        )
    prompt = str(prompt or "").strip()
    if not prompt:
        return _command_message(tool_call_id, "Image generation requires a non-empty prompt.", status="error")
    if len(prompt) > MAX_IMAGE_PROMPT_CHARS:
        return _command_message(
            tool_call_id,
            f"Image prompt is too long (maximum {MAX_IMAGE_PROMPT_CHARS} characters).",
            status="error",
        )
    size = str(size or "").strip().lower()
    quality = str(quality or "").strip().lower()
    if size not in GPT_IMAGE_SIZES:
        return _command_message(tool_call_id, f"Unsupported image size: {size}.", status="error")
    if quality not in GPT_IMAGE_QUALITIES:
        return _command_message(tool_call_id, f"Unsupported image quality: {quality}.", status="error")
    if not settings.resolved_image_generation_api_key:
        return _command_message(
            tool_call_id,
            "Image generation requires SUPERASSIST_IMAGE_GENERATION_API_KEY or SUPERASSIST_API_KEY.",
            status="error",
        )

    try:
        client = OpenAI(
            api_key=settings.resolved_image_generation_api_key,
            base_url=settings.resolved_image_generation_base_url,
        )
        response = client.images.generate(
            model=settings.image_generation_model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
            output_format="png",
        )
        data = _generated_image_bytes(response)
        if len(data) > MAX_GENERATED_IMAGE_BYTES:
            raise ValueError("Generated image exceeds the 10 MB Feishu upload limit")
        mime_type = detect_image_mime_type(data)
        if mime_type != "image/png":
            raise ValueError("Image generation API did not return a PNG image")
        validate_image_bytes(data)
        path = _save_generated_image(data, settings.generated_image_cache_dir)
    except Exception as exc:  # noqa: BLE001 - provider failures must remain tool-level errors
        return _command_message(
            tool_call_id,
            f"Image generation failed ({type(exc).__name__}): {_safe_error_text(exc)}",
            status="error",
        )

    candidate_id = f"gen_{path.stem}"
    title = _generated_image_title(prompt)
    candidate = {
        "candidate_id": candidate_id,
        "title": title,
        "local_path": str(path),
        "mime_type": mime_type,
        "source": settings.image_generation_model,
        "source_url": "",
    }
    return Command(
        update={
            "outbound_images": [candidate],
            "messages": [
                ToolMessage(
                    content=[
                        {
                            "type": "input_text",
                            "text": (
                                f"Generated image {candidate_id} with {settings.image_generation_model}. "
                                "It is already selected for the final Feishu response. Inspect it before replying."
                            ),
                        },
                        {"type": "input_image", "image_url": _image_data_url(data, mime_type), "detail": "high"},
                    ],
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def download_image_url(url: str, *, max_bytes: int, timeout: int = 15) -> tuple[bytes, str]:
    """Download a public HTTP image with SSRF, type, and size checks."""

    _validate_public_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; SuperAssist/0.1)",
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/gif,*/*;q=0.5",
        },
    )
    opener = build_opener(_PublicRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        _validate_public_url(response.geturl())
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("Image exceeds the download size limit")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Image exceeds the download size limit")
    mime_type = detect_image_mime_type(data)
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError("Downloaded resource is not a supported raster image")
    validate_image_bytes(data)
    return data, mime_type


def detect_image_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def validate_image_bytes(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("Downloaded image data is corrupt or incomplete") from exc


def load_generated_image(path_value: str, allowed_root: Path, *, max_bytes: int = MAX_GENERATED_IMAGE_BYTES) -> bytes:
    """Read a generated image only when it is inside SuperAssist's controlled cache."""

    root = allowed_root.resolve()
    path = Path(str(path_value or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Generated image path is outside the configured cache directory") from exc
    if not path.is_file():
        raise ValueError("Generated image file is missing")
    if path.stat().st_size > max_bytes:
        raise ValueError("Generated image exceeds the upload size limit")
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("Generated image exceeds the upload size limit")
    validate_image_bytes(data)
    return data


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(str(url or ""))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URL must be public HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("Image URL credentials are not allowed")
    try:
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or default_port)}
    except socket.gaierror as exc:
        raise ValueError("Image host could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(address.split("%", 1)[0]).is_global for address in addresses):
        raise ValueError("Image URL resolves to a non-public address")


def _normalize_candidate(raw: Any, *, query: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    image_url = str(raw.get("image") or "").strip()
    thumbnail_url = str(raw.get("thumbnail") or "").strip()
    source_url = str(raw.get("url") or "").strip()
    if not image_url or not source_url:
        return None
    return {
        "query": query[:500],
        "title": str(raw.get("title") or "Untitled image").strip()[:500],
        "image_url": image_url,
        "thumbnail_url": thumbnail_url,
        "source_url": source_url,
        "source": str(raw.get("source") or "").strip()[:200],
        "width": _positive_int(raw.get("width")),
        "height": _positive_int(raw.get("height")),
    }


def _model_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "title": candidate["title"],
        "source_page": candidate["source_url"],
        "source": candidate["source"],
        "width": candidate["width"],
        "height": candidate["height"],
    }


def _outbound_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "title": candidate["title"],
        "image_url": candidate["image_url"],
        "thumbnail_url": candidate["thumbnail_url"],
        "source_url": candidate["source_url"],
    }


def _resolve_candidates(
    candidate_ids: list[str],
    registry: dict[str, dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    requested = list(dict.fromkeys(str(value).strip() for value in (candidate_ids or []) if str(value).strip()))
    if len(requested) > limit:
        return [], [f"too many IDs (maximum {limit})"]
    invalid = [candidate_id for candidate_id in requested if candidate_id not in registry]
    return [registry[candidate_id] for candidate_id in requested if candidate_id in registry], invalid


def _invalid_candidates_message(invalid: list[str], has_selected: bool) -> str:
    if invalid:
        return f"Invalid or expired image candidate IDs: {', '.join(invalid)}. Use IDs from this turn's image_search."
    if not has_selected:
        return "No image candidate IDs were supplied."
    return "No valid image candidates were selected."


def _download_candidate(candidate: dict[str, Any], *, max_bytes: int) -> tuple[bytes, str]:
    errors: list[Exception] = []
    for url in (candidate.get("image_url"), candidate.get("thumbnail_url")):
        if not url:
            continue
        try:
            return download_image_url(str(url), max_bytes=max_bytes)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    raise errors[-1] if errors else ValueError("Candidate has no downloadable URL")


def _command_message(tool_call_id: str, content: str, *, status: str = "success") -> Command[Any]:
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id, status=status)],
        }
    )


def _image_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _generated_image_bytes(response: Any) -> bytes:
    items = getattr(response, "data", None) or []
    if not items:
        raise ValueError("Image generation API returned no image")
    item = items[0]
    encoded = str(getattr(item, "b64_json", None) or "").strip()
    if encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image generation API returned invalid base64 data") from exc
    url = str(getattr(item, "url", None) or "").strip()
    if url:
        data, _mime_type = download_image_url(url, max_bytes=MAX_GENERATED_IMAGE_BYTES)
        return data
    raise ValueError("Image generation API returned neither base64 data nor a URL")


def _save_generated_image(data: bytes, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _cleanup_generated_images(root)
    path = root / f"{uuid4().hex}.png"
    path.write_bytes(data)
    return path.resolve()


def _cleanup_generated_images(root: Path) -> None:
    cutoff = time.time() - GENERATED_IMAGE_TTL_SECONDS
    for path in root.glob("*.png"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _generated_image_title(prompt: str) -> str:
    compact = " ".join(prompt.split())
    return compact[:80] or "Generated image"


def _safe_error_text(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    return text[:300] or "provider returned no details"


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


__all__ = [
    "GPT_IMAGE_MODEL",
    "detect_image_mime_type",
    "download_image_url",
    "generate_image",
    "image_search",
    "inspect_image",
    "load_generated_image",
    "present_images",
    "validate_image_bytes",
]
