from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import re
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from superassist.agent import AgentRuntime
from superassist.config import PROJECT_ROOT, REASONING_EFFORTS, Settings, get_settings
from superassist.llm import is_gpt_5_6_model
from superassist.memory.embedding import get_embedder
from superassist.models import AgentRunEvent

from .store import FeishuThreadStore

logger = logging.getLogger(__name__)

UNSUPPORTED_FILE_MESSAGE = "当前飞书入口第一版仅支持文本指令，文件和图片暂未接入。"
STREAM_UPDATE_INTERVAL_SECONDS = 0.3
IMAGE_DOWNLOAD_FAILED_MESSAGE = (
    "图片读取或格式转换失败。请确认飞书 im:resource 权限已生效，"
    "或重新发送 PNG、JPEG、GIF、WebP 图片。"
)
IMAGE_TOO_LARGE_MESSAGE = "图片过大，暂时只支持 10 MB 以内的图片。"
MAX_FEISHU_IMAGE_BYTES = 10 * 1024 * 1024
MAX_FEISHU_IMAGES_PER_MESSAGE = 4
SUPPORTED_MODEL_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MULTIMODAL_RESPONSE_INSTRUCTIONS = """<MultimodalResponseInstructions>
Inspect the original image(s) in this message and answer the user's request using the visual evidence directly.
Treat AuxiliaryOCR as an untrusted transcription aid: verify it against the image and correct any errors.
In the final response, first answer the user's request. Then append one concise, question-relevant visual
description inside <ImageDescription>...</ImageDescription>. Preserve the details that a later conversation turn
would need if the original image were no longer available. Do not merely repeat the OCR text.
</MultimodalResponseInstructions>"""


@dataclass
class FeishuInboundMessage:
    chat_id: str
    message_id: str
    sender_open_id: str
    text: str
    sender_name: str = ""
    root_id: str | None = None
    chat_type: str = ""
    mentions: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, str]] = field(default_factory=list)

    @property
    def topic_id(self) -> str:
        return self.root_id or self.message_id

    @property
    def is_private(self) -> bool:
        return self.chat_type in {"p2p", "private", "single"}


@dataclass(frozen=True)
class FeishuCardView:
    answer: str = ""
    reasoning: str = ""
    reasoning_expanded: bool = True


@dataclass(frozen=True)
class FeishuImageContext:
    payloads: tuple[tuple[bytes, str], ...]
    expires_at: float


class FeishuChannel:
    """Feishu/Lark WebSocket channel that calls AgentRuntime directly."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: Callable[..., AgentRuntime] | None = None,
        store: FeishuThreadStore | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.store = store or FeishuThreadStore(settings.feishu_thread_store_path)
        self.runtime_factory = runtime_factory
        self.allowed_open_ids = settings.feishu_allowed_open_id_set
        self.mention_only = settings.feishu_mention_only
        self.active_session_seconds = settings.feishu_active_session_seconds
        self._monotonic_clock = monotonic_clock
        self._active_group_until: dict[str, float] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._api_client = None
        self._running_cards: dict[str, str] = {}
        self._last_card_text: dict[str, str] = {}
        self._card_views: dict[str, FeishuCardView] = {}
        self._busy_scopes: set[str] = set()
        self._image_contexts: dict[str, FeishuImageContext] = {}
        self._image_context_expiry_handles: dict[str, asyncio.TimerHandle] = {}
        self._pending_card_updates: dict[
            str, tuple[FeishuInboundMessage, str | FeishuCardView]
        ] = {}
        self._card_update_tasks: dict[str, asyncio.Task[None]] = {}
        self._lark = None
        self._CreateMessageRequest = None
        self._CreateMessageRequestBody = None
        self._ReplyMessageRequest = None
        self._ReplyMessageRequestBody = None
        self._PatchMessageRequest = None
        self._PatchMessageRequestBody = None
        self._GetMessageResourceRequest = None
        self._ocr_engine: Any | None = None
        self._ocr_initialization_attempted = False
        self._ocr_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        if not self.settings.feishu_app_id or not self.settings.feishu_app_secret:
            raise RuntimeError(
                "Feishu channel requires SUPERASSIST_FEISHU_APP_ID and "
                "SUPERASSIST_FEISHU_APP_SECRET."
            )
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (
                CreateMessageRequest,
                CreateMessageRequestBody,
                GetMessageResourceRequest,
                PatchMessageRequest,
                PatchMessageRequestBody,
                ReplyMessageRequest,
                ReplyMessageRequestBody,
            )
        except ImportError as exc:
            raise RuntimeError("lark-oapi is not installed. Install project dependencies first.") from exc

        self._lark = lark
        self._CreateMessageRequest = CreateMessageRequest
        self._CreateMessageRequestBody = CreateMessageRequestBody
        self._ReplyMessageRequest = ReplyMessageRequest
        self._ReplyMessageRequestBody = ReplyMessageRequestBody
        self._PatchMessageRequest = PatchMessageRequest
        self._PatchMessageRequestBody = PatchMessageRequestBody
        self._GetMessageResourceRequest = GetMessageResourceRequest
        self._api_client = (
            lark.Client.builder()
            .app_id(self.settings.feishu_app_id)
            .app_secret(self.settings.feishu_app_secret)
            .domain(self.settings.feishu_domain)
            .build()
        )
        self._main_loop = asyncio.get_running_loop()
        await asyncio.to_thread(get_embedder(self.settings).preload)
        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()
        logger.info("Feishu channel started with domain %s", self.settings.feishu_domain)

    async def stop(self) -> None:
        self._running = False
        for handle in self._image_context_expiry_handles.values():
            handle.cancel()
        self._image_context_expiry_handles.clear()
        self._image_contexts.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    async def handle_inbound(self, inbound: FeishuInboundMessage) -> None:
        if self.allowed_open_ids and inbound.sender_open_id not in self.allowed_open_ids:
            logger.info("Ignoring Feishu message from non-allowed open_id=%s", inbound.sender_open_id)
            return
        clean_text = clean_mention_text(inbound.text, inbound.mentions).strip()
        accepted = self._should_accept_message(inbound)
        logger.info(
            "Feishu inbound gate chat_type=%s chat_suffix=%s mentioned=%s accepted=%s",
            inbound.chat_type or "unknown",
            inbound.chat_id[-8:] if inbound.chat_id else "unknown",
            has_bot_mention(inbound),
            accepted,
        )
        if not accepted:
            return
        if not clean_text and not inbound.files:
            return

        scope_key = f"private:{inbound.chat_id}" if inbound.is_private else f"group:{inbound.chat_id}"
        if scope_key in self._busy_scopes:
            logger.info(
                "Ignoring Feishu message while scope is busy scope=%s message_suffix=%s",
                scope_key,
                inbound.message_id[-8:] if inbound.message_id else "unknown",
            )
            return
        self._busy_scopes.add(scope_key)
        try:
            active_image_context = self._get_image_context(scope_key)
            images = [item for item in inbound.files if item.get("image_key")]
            other_files = [item for item in inbound.files if not item.get("image_key")]
            if other_files:
                await self._send_or_patch(inbound, UNSUPPORTED_FILE_MESSAGE, final=True)
                return
            image_payloads: list[tuple[bytes, str]] = []
            has_new_images = bool(images)
            if images:
                await self._send_or_patch(inbound, "Reading image...", final=False)
                try:
                    clean_text = strip_file_placeholders(clean_text)
                    image_payloads = await self._download_image_payloads(images, inbound.message_id)
                except ImageTooLargeError:
                    await self._send_or_patch(inbound, IMAGE_TOO_LARGE_MESSAGE, final=True)
                    return
                except ImageDownloadError:
                    logger.exception("Failed to download Feishu image")
                    await self._send_or_patch(inbound, IMAGE_DOWNLOAD_FAILED_MESSAGE, final=True)
                    return
            elif active_image_context is not None:
                image_payloads = list(active_image_context.payloads)
            if not clean_text and not image_payloads:
                return
            try:
                await self._handle_scoped_message(
                    inbound,
                    clean_text,
                    image_payloads,
                    extract_image_ocr=has_new_images,
                )
            finally:
                if image_payloads:
                    self._set_image_context(scope_key, image_payloads)
        finally:
            self._busy_scopes.discard(scope_key)

    def _should_accept_message(self, inbound: FeishuInboundMessage) -> bool:
        if inbound.is_private or not self.mention_only:
            return True

        now = self._monotonic_clock()
        expires_at = self._active_group_until.get(inbound.chat_id)
        mentioned = has_bot_mention(inbound)
        if not mentioned and (expires_at is None or now >= expires_at):
            self._active_group_until.pop(inbound.chat_id, None)
            return False

        self._active_group_until[inbound.chat_id] = now + self.active_session_seconds
        return True

    async def _download_image_payloads(
        self,
        images: list[dict[str, str]],
        message_id: str,
    ) -> list[tuple[bytes, str]]:
        return list(
            await asyncio.gather(
                *(
                    self._download_image(message_id, item["image_key"])
                    for item in images[:MAX_FEISHU_IMAGES_PER_MESSAGE]
                )
            )
        )

    async def _download_image(self, message_id: str, image_key: str) -> tuple[bytes, str]:
        try:
            if not self._api_client or not self._GetMessageResourceRequest:
                raise RuntimeError("Feishu image client is not initialized")
            request = (
                self._GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = await asyncio.to_thread(self._api_client.im.v1.message_resource.get, request)
            if not response.success() or not response.file:
                raise RuntimeError(f"Feishu image download failed with code={getattr(response, 'code', None)}")
            data = response.file.read(MAX_FEISHU_IMAGE_BYTES + 1)
            if len(data) > MAX_FEISHU_IMAGE_BYTES:
                raise ImageTooLargeError
            original_data, mime_type = original_image_payload(data)
            logger.info(
                "Downloaded original Feishu image message_suffix=%s bytes=%d mime=%s",
                message_id[-8:] if message_id else "unknown",
                len(data),
                mime_type,
            )
            return original_data, mime_type
        except ImageTooLargeError:
            raise
        except Exception as exc:
            raise ImageDownloadError from exc

    async def _handle_scoped_message(
        self,
        inbound: FeishuInboundMessage,
        clean_text: str,
        image_payloads: list[tuple[bytes, str]] | None = None,
        *,
        extract_image_ocr: bool = True,
    ) -> None:
        user_id, conversation_scope = feishu_memory_scope(inbound)
        thread_id = self.store.get_or_create_thread_id(
            chat_id=inbound.chat_id,
            topic_id=conversation_scope,
            user_id=user_id,
        )
        reasoning_effort = self.store.get_reasoning_effort(
            chat_id=inbound.chat_id,
            topic_id=conversation_scope,
            default=self.settings.reasoning_effort,
        )
        if reasoning_effort not in REASONING_EFFORTS:
            reasoning_effort = self.settings.reasoning_effort

        is_effort_command, requested_effort = parse_effort_command(clean_text)
        if is_effort_command:
            if requested_effort is None:
                response = (
                    f"当前推理强度：`{reasoning_effort}`\n\n"
                    f"可选值：`{'`, `'.join(REASONING_EFFORTS)}`。"
                )
            elif requested_effort not in REASONING_EFFORTS:
                response = f"不支持 `{requested_effort}`。可选值：`{'`, `'.join(REASONING_EFFORTS)}`。"
            elif not is_gpt_5_6_model(self.settings.model):
                response = f"当前模型 `{self.settings.model}` 不支持此处的 GPT-5.6 推理强度设置。"
            else:
                self.store.set_reasoning_effort(
                    chat_id=inbound.chat_id,
                    topic_id=conversation_scope,
                    effort=requested_effort,
                )
                response = f"已将当前会话的推理强度设置为 `{requested_effort}`。"
            await self._send_or_patch(inbound, response, final=True)
            return

        await self._send_or_patch(inbound, "Preparing context...", final=False)
        memory_text = clean_text or default_image_only_request(len(image_payloads or []))
        runtime_text = attributed_feishu_text(inbound, memory_text)
        message_content: str | list[dict[str, Any]] = runtime_text
        if image_payloads:
            ocr_text = await self._extract_images_ocr(image_payloads) if extract_image_ocr else ""
            persisted_text = build_image_memory_text(runtime_text, ocr_text)
            message_content = build_multimodal_image_content(
                build_multimodal_request_text(runtime_text, ocr_text),
                image_payloads,
            )
            runtime_text = persisted_text

        def report(event: AgentRunEvent) -> None:
            if event.type == "thinking":
                text = event.message.strip() or "Thinking..."
                view: str | FeishuCardView = text
            elif event.type == "agent_reasoning":
                reasoning = event.message.strip()
                if not reasoning:
                    return
                previous = self._card_views.get(inbound.message_id, FeishuCardView())
                view = FeishuCardView(
                    answer=previous.answer,
                    reasoning=reasoning,
                    reasoning_expanded=not bool(previous.answer),
                )
                self._card_views[inbound.message_id] = view
            elif event.type == "agent_text":
                text = event.message.strip()
                previous = self._card_views.get(inbound.message_id)
                if previous and previous.reasoning:
                    view = FeishuCardView(answer=text, reasoning=previous.reasoning, reasoning_expanded=False)
                    self._card_views[inbound.message_id] = view
                else:
                    view = text
            elif event.type == "subagent_text":
                text = format_subagent_card_text(event)
                view = text
            else:
                return
            if isinstance(view, str) and not view:
                return
            if self._main_loop and self._main_loop.is_running():
                self._main_loop.call_soon_threadsafe(
                    self._queue_card_update,
                    inbound,
                    view,
                )

        runtime: AgentRuntime | None = None
        try:
            runtime = self._create_runtime(report, reasoning_effort=reasoning_effort)
            runtime_kwargs: dict[str, Any] = {"user_id": user_id, "thread_id": thread_id}
            if image_payloads:
                runtime_kwargs["message_content"] = message_content
            result = await asyncio.to_thread(
                runtime.run_streaming,
                runtime_text,
                **runtime_kwargs,
            )
            if result.metadata.get("model_error"):
                logger.warning(
                    "Feishu model request failed error_type=%s error=%s",
                    result.metadata.get("model_error"),
                    result.metadata.get("model_error_message"),
                )
            runtime.memory_queue.flush()
            await self._flush_card_updates(inbound.message_id)
            final_text = result.answer.strip() or self._last_card_text.get(inbound.message_id, "") or "(empty response)"
            if result.metadata.get("model_error"):
                final_text = format_model_error(result.metadata, has_images=bool(image_payloads))
            previous = self._card_views.get(inbound.message_id)
            final_view: str | FeishuCardView = final_text
            if previous and previous.reasoning:
                final_view = FeishuCardView(
                    answer=final_text,
                    reasoning=previous.reasoning,
                    reasoning_expanded=False,
                )
            await self._send_or_patch(inbound, final_view, final=True)
        except Exception:
            logger.exception("Feishu agent run failed")
            await self._flush_card_updates(inbound.message_id)
            await self._send_or_patch(inbound, "处理这条飞书消息时出错了，请稍后重试。", final=True)
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()

    def _create_runtime(
        self,
        reporter: Callable[[AgentRunEvent], None],
        *,
        reasoning_effort: str,
    ) -> AgentRuntime:
        if self.runtime_factory is None:
            runtime_settings = self.settings.model_copy(update={"reasoning_effort": reasoning_effort})
            return AgentRuntime(runtime_settings, run_event_reporter=reporter)
        return self.runtime_factory(reporter)

    async def _extract_images_ocr(self, image_payloads: list[tuple[bytes, str]]) -> str:
        if not self.settings.feishu_image_ocr_enabled or self.settings.feishu_image_ocr_max_chars <= 0:
            return ""
        try:
            text = await asyncio.to_thread(self._extract_images_ocr_sync, image_payloads)
        except Exception as exc:  # noqa: BLE001 - optional OCR must never block original-image input
            logger.warning("Local Feishu OCR unavailable error_type=%s", type(exc).__name__)
            return ""
        bounded = text[: self.settings.feishu_image_ocr_max_chars].strip()
        logger.info(
            "Local Feishu OCR completed images=%d chars=%d truncated=%s",
            len(image_payloads),
            len(bounded),
            len(text) > len(bounded),
        )
        return bounded

    def _extract_images_ocr_sync(self, image_payloads: list[tuple[bytes, str]]) -> str:
        with self._ocr_lock:
            engine = self._get_ocr_engine()
            image_texts: list[str] = []
            for index, (data, _mime_type) in enumerate(image_payloads, start=1):
                with Image.open(io.BytesIO(data)) as image:
                    pixels = np.asarray(image.convert("RGB"))
                output = engine(pixels)
                results = output[0] if isinstance(output, tuple) else output
                lines = [
                    str(item[1]).strip()
                    for item in (results or [])
                    if isinstance(item, (list, tuple)) and len(item) > 1 and str(item[1]).strip()
                ]
                if lines:
                    image_texts.append(f"[Image {index}]\n" + "\n".join(lines))
            return "\n\n".join(image_texts)

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine
        if self._ocr_initialization_attempted:
            raise RuntimeError("RapidOCR initialization previously failed")
        self._ocr_initialization_attempted = True
        from rapidocr_onnxruntime import RapidOCR

        self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _get_image_context(self, scope_key: str) -> FeishuImageContext | None:
        context = self._image_contexts.get(scope_key)
        if context is None:
            return None
        if self._monotonic_clock() >= context.expires_at:
            self._clear_image_context(scope_key)
            return None
        return context

    def _set_image_context(self, scope_key: str, payloads: list[tuple[bytes, str]]) -> None:
        if not payloads:
            self._clear_image_context(scope_key)
            return
        expires_at = self._monotonic_clock() + self.settings.feishu_image_context_ttl_seconds
        self._image_contexts[scope_key] = FeishuImageContext(tuple(payloads), expires_at)
        self._schedule_image_context_expiry(scope_key, expires_at)

    def _schedule_image_context_expiry(self, scope_key: str, expires_at: float) -> None:
        previous = self._image_context_expiry_handles.pop(scope_key, None)
        if previous is not None:
            previous.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._image_context_expiry_handles[scope_key] = loop.call_later(
            self.settings.feishu_image_context_ttl_seconds,
            self._expire_image_context,
            scope_key,
            expires_at,
        )

    def _expire_image_context(self, scope_key: str, expected_expires_at: float) -> None:
        context = self._image_contexts.get(scope_key)
        if context is None or context.expires_at != expected_expires_at:
            return
        remaining = context.expires_at - self._monotonic_clock()
        if remaining > 0:
            loop = asyncio.get_running_loop()
            self._image_context_expiry_handles[scope_key] = loop.call_later(
                remaining,
                self._expire_image_context,
                scope_key,
                expected_expires_at,
            )
            return
        self._clear_image_context(scope_key)
        logger.info("Expired full Feishu image context scope=%s", scope_key)

    def _clear_image_context(self, scope_key: str) -> None:
        self._image_contexts.pop(scope_key, None)
        handle = self._image_context_expiry_handles.pop(scope_key, None)
        if handle is not None:
            handle.cancel()

    def _queue_card_update(self, inbound: FeishuInboundMessage, text: str | FeishuCardView) -> None:
        message_id = inbound.message_id
        self._pending_card_updates[message_id] = (inbound, text)
        task = self._card_update_tasks.get(message_id)
        if task is None or task.done():
            self._card_update_tasks[message_id] = asyncio.create_task(
                self._drain_card_updates(message_id)
            )

    async def _drain_card_updates(self, message_id: str) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                pending = self._pending_card_updates.pop(message_id, None)
                if pending is None:
                    return
                inbound, text = pending
                try:
                    await self._send_or_patch(inbound, text, final=False)
                except Exception:
                    logger.exception("Failed to update streaming Feishu card")
                if message_id not in self._pending_card_updates:
                    return
                await asyncio.sleep(STREAM_UPDATE_INTERVAL_SECONDS)
        finally:
            if self._card_update_tasks.get(message_id) is current_task:
                self._card_update_tasks.pop(message_id, None)
            if message_id in self._pending_card_updates and message_id not in self._card_update_tasks:
                self._card_update_tasks[message_id] = asyncio.create_task(
                    self._drain_card_updates(message_id)
                )

    async def _flush_card_updates(self, message_id: str) -> None:
        # Let call_soon_threadsafe callbacks queued by the model thread run first.
        await asyncio.sleep(0)
        while True:
            task = self._card_update_tasks.get(message_id)
            if task is None:
                if message_id not in self._pending_card_updates:
                    return
                self._card_update_tasks[message_id] = asyncio.create_task(
                    self._drain_card_updates(message_id)
                )
                task = self._card_update_tasks[message_id]
            await task
            await asyncio.sleep(0)

    def _on_message(self, event: Any) -> None:
        try:
            inbound = parse_feishu_event(event)
        except Exception:
            logger.exception("Failed to parse Feishu event")
            return
        if self._main_loop and self._main_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.handle_inbound(inbound), self._main_loop)
        else:
            logger.warning("Feishu main loop is not running; message ignored")

    def _run_ws(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import lark_oapi.ws.client as ws_client_module

            ws_client_module.loop = loop
            event_handler = self._lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(
                self._on_message
            ).build()
            ws_client = self._lark.ws.Client(
                app_id=self.settings.feishu_app_id,
                app_secret=self.settings.feishu_app_secret,
                event_handler=event_handler,
                log_level=self._lark.LogLevel.INFO,
                domain=self.settings.feishu_domain,
            )
            ws_client.start()
        except Exception:
            if self._running:
                logger.exception("Feishu WebSocket error")

    async def _send_or_patch(
        self, inbound: FeishuInboundMessage, text: str | FeishuCardView, *, final: bool
    ) -> None:
        if isinstance(text, str):
            text = text.strip()
        if (isinstance(text, str) and not text) or (
            isinstance(text, FeishuCardView) and not text.reasoning and not text.answer
        ):
            return
        visible_text = text.answer if isinstance(text, FeishuCardView) else text
        if visible_text:
            self._last_card_text[inbound.message_id] = visible_text
        card_id = self._running_cards.get(inbound.message_id)
        if card_id:
            await self._update_card(card_id, text)
        elif inbound.is_private and inbound.message_id:
            card_id = await self._reply_card(inbound.message_id, text)
            if card_id:
                self._running_cards[inbound.message_id] = card_id
        else:
            card_id = await self._create_card(inbound.chat_id, text)
            if card_id:
                self._running_cards[inbound.message_id] = card_id
        if final:
            self._pending_card_updates.pop(inbound.message_id, None)
            self._running_cards.pop(inbound.message_id, None)
            self._last_card_text.pop(inbound.message_id, None)
            self._card_views.pop(inbound.message_id, None)

    async def _reply_card(self, message_id: str, text: str | FeishuCardView) -> str | None:
        if not self._api_client:
            return None
        request = self._ReplyMessageRequest.builder().message_id(message_id).request_body(
            self._ReplyMessageRequestBody.builder()
            .msg_type("interactive")
            .content(build_card_content(text))
            .reply_in_thread(True)
            .build()
        ).build()
        response = await asyncio.to_thread(self._api_client.im.v1.message.reply, request)
        response_data = getattr(response, "data", None)
        return getattr(response_data, "message_id", None)

    async def _create_card(self, chat_id: str, text: str | FeishuCardView) -> str | None:
        if not self._api_client:
            return None
        request = self._CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            self._CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(build_card_content(text))
            .build()
        ).build()
        response = await asyncio.to_thread(self._api_client.im.v1.message.create, request)
        response_data = getattr(response, "data", None)
        return getattr(response_data, "message_id", None)

    async def _update_card(self, message_id: str, text: str | FeishuCardView) -> None:
        if not self._api_client:
            return
        request = self._PatchMessageRequest.builder().message_id(message_id).request_body(
            self._PatchMessageRequestBody.builder().content(build_card_content(text)).build()
        ).build()
        await asyncio.to_thread(self._api_client.im.v1.message.patch, request)


class FeishuChannelService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.channel = FeishuChannel(self.settings)

    async def run_forever(self) -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        await self.channel.start()
        try:
            await stop_event.wait()
        finally:
            await self.channel.stop()


def parse_feishu_event(event: Any) -> FeishuInboundMessage:
    message = event.event.message
    sender_info = event.event.sender
    sender = sender_info.sender_id
    content = json.loads(message.content)
    text, files = parse_feishu_content(content)
    mentions = _coerce_mentions(getattr(message, "mentions", []) or content.get("mentions", []))
    return FeishuInboundMessage(
        chat_id=str(message.chat_id),
        message_id=str(message.message_id),
        root_id=getattr(message, "root_id", None) or None,
        sender_open_id=str(sender.open_id),
        sender_name=str(getattr(sender_info, "name", "") or getattr(sender, "name", "") or ""),
        text=text.strip(),
        chat_type=str(getattr(message, "chat_type", "") or ""),
        mentions=mentions,
        files=files,
    )


def feishu_memory_scope(inbound: FeishuInboundMessage) -> tuple[str, str]:
    """Return the shared-memory identity and stable short-memory scope."""
    if inbound.is_private:
        return f"feishu:{inbound.sender_open_id}", "__private__"
    return f"feishu-group:{inbound.chat_id}", "__group__"


def parse_effort_command(text: str) -> tuple[bool, str | None]:
    match = re.fullmatch(r"/effort(?:\s+|=|$)(.*)", text.strip(), flags=re.IGNORECASE)
    if match is None:
        return False, None
    value = match.group(1).strip().lower()
    return True, value or None


def attributed_feishu_text(inbound: FeishuInboundMessage, clean_text: str) -> str:
    """Prefix group messages so shared memory retains speaker provenance."""
    if inbound.is_private:
        return clean_text
    speaker = normalize_sender_name(inbound.sender_open_id) or "unknown"
    return f"[飞书群成员: {speaker}] {clean_text}"


def normalize_sender_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:128]


def parse_feishu_content(content: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    files: list[dict[str, str]] = []
    if isinstance(content.get("text"), str):
        return content["text"], files
    if isinstance(content.get("file_key"), str):
        files.append({"file_key": content["file_key"]})
        return "[file]", files
    if isinstance(content.get("image_key"), str):
        files.append({"image_key": content["image_key"]})
        return "[image]", files
    paragraphs = content.get("content")
    if not isinstance(paragraphs, list):
        return "", files
    text_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, list):
            continue
        parts: list[str] = []
        for element in paragraph:
            if not isinstance(element, dict):
                continue
            tag = element.get("tag")
            if tag in {"text", "at"} and isinstance(element.get("text"), str):
                parts.append(element["text"])
            elif tag == "img" and isinstance(element.get("image_key"), str):
                files.append({"image_key": element["image_key"]})
                parts.append("[image]")
            elif tag in {"file", "media"} and isinstance(element.get("file_key"), str):
                files.append({"file_key": element["file_key"]})
                parts.append("[file]")
        if parts:
            text_paragraphs.append(" ".join(parts))
    return "\n\n".join(text_paragraphs), files


def should_trigger_agent(inbound: FeishuInboundMessage, *, mention_only: bool) -> bool:
    if inbound.is_private:
        return True
    if not mention_only:
        return True
    return has_bot_mention(inbound)


def has_bot_mention(inbound: FeishuInboundMessage) -> bool:
    return bool(inbound.mentions) or bool(re.search(r"(^|\s)@[^\s]+", inbound.text))


def clean_mention_text(text: str, mentions: list[dict[str, Any]]) -> str:
    cleaned = text
    for mention in mentions:
        name = str(mention.get("name") or mention.get("text") or "").strip()
        if name:
            cleaned = cleaned.replace(name, " ")
    cleaned = re.sub(r"(^|\s)@[^\s]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def strip_file_placeholders(text: str) -> str:
    return re.sub(r"\[(?:image|file)\]", " ", text, flags=re.IGNORECASE).strip()


def default_image_only_request(image_count: int) -> str:
    image_label = "一张图片" if image_count == 1 else f"{max(1, image_count)} 张图片"
    return (
        f"用户只发送了{image_label}，没有附加文字。请仔细查看图片并推断用户最可能希望获得的帮助，"
        "然后直接回应。如果图片中包含题目、问题或待完成的任务，请识别并解答；如果意图仍不明确，"
        "请概括关键内容，并给出最有帮助的判断或提出一个简短的澄清问题。不要只做泛泛的图片描述。"
    )


def build_multimodal_image_content(
    text: str,
    payloads: list[tuple[bytes, str]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for data, mime_type in payloads:
        encoded = base64.b64encode(data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    content.append({"type": "text", "text": text})
    return content


def format_auxiliary_ocr(ocr_text: str) -> str:
    if not ocr_text.strip():
        return ""
    return (
        '<AuxiliaryOCR source="local" reliability="unverified">\n'
        "The following text was extracted automatically and may contain errors. Verify it against the image.\n"
        f"{ocr_text.strip()}\n"
        "</AuxiliaryOCR>"
    )


def build_image_memory_text(user_text: str, ocr_text: str) -> str:
    ocr_section = format_auxiliary_ocr(ocr_text)
    return f"{ocr_section}\n\n{user_text}" if ocr_section else user_text


def build_multimodal_request_text(user_text: str, ocr_text: str) -> str:
    return f"{MULTIMODAL_RESPONSE_INSTRUCTIONS}\n\n{build_image_memory_text(user_text, ocr_text)}"


def image_mime_type(data: bytes, file_name: str | None = None) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    guessed, _encoding = mimetypes.guess_type(file_name or "")
    if guessed and guessed.startswith("image/"):
        return guessed
    return "application/octet-stream"


def original_image_payload(data: bytes) -> tuple[bytes, str]:
    """Validate the downloaded resource while preserving its original bytes."""
    if not data:
        raise ImageDownloadError("Feishu returned an empty image")
    mime_type = image_mime_type(data)
    if mime_type not in SUPPORTED_MODEL_IMAGE_MIME_TYPES:
        raise ImageDownloadError("Feishu resource is not a supported model image")
    return data, mime_type


def format_model_error(metadata: dict[str, Any], *, has_images: bool) -> str:
    error_type = str(metadata.get("model_error") or "ModelError").strip()
    detail = re.sub(r"\s+", " ", str(metadata.get("model_error_message") or "")).strip()
    if len(detail) > 600:
        detail = detail[:597] + "..."
    prefix = "模型处理图片失败" if has_images else "模型服务请求失败"
    return f"{prefix}（`{error_type}`）：{detail or '服务端没有返回详细原因。'}"


class ImageTooLargeError(ValueError):
    pass


class ImageDownloadError(RuntimeError):
    pass


def build_card_content(text: str | FeishuCardView) -> str:
    if isinstance(text, FeishuCardView) and text.reasoning:
        elements: list[dict[str, Any]] = [
            {
                "tag": "collapsible_panel",
                "expanded": text.reasoning_expanded,
                "header": {
                    "title": {"tag": "plain_text", "content": "思考过程"},
                    "vertical_align": "center",
                },
                "elements": [{"tag": "markdown", "content": text.reasoning}],
            }
        ]
        if text.answer:
            elements.append({"tag": "markdown", "content": text.answer})
        return json.dumps(
            {
                "schema": "2.0",
                "config": {"wide_screen_mode": True, "update_multi": True},
                "body": {"elements": elements},
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [{"tag": "markdown", "content": text}],
        },
        ensure_ascii=False,
    )


def format_subagent_card_text(event: AgentRunEvent) -> str:
    text = event.message.strip()
    if not text:
        return ""
    metadata = event.metadata or {}
    description = str(metadata.get("description") or "").strip()
    subagent_type = str(metadata.get("subagent_type") or "subagent").strip()
    label = description or subagent_type
    return f"Subagent [{label}]: {text}"


def _coerce_mentions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    mentions: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            mentions.append(item)
        else:
            text = getattr(item, "name", None) or getattr(item, "text", None)
            open_id = getattr(getattr(item, "id", None), "open_id", None)
            mentions.append({"name": text, "open_id": open_id})
    return mentions


def load_feishu_settings(env_path: str | Path = PROJECT_ROOT / ".env") -> Settings:
    load_dotenv(env_path, override=True)
    get_settings.cache_clear()
    return get_settings()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_feishu_settings()
    logger.info(
        "Feishu channel configuration model=%s base_url=%s env_file=%s",
        settings.model,
        settings.base_url,
        PROJECT_ROOT / ".env",
    )
    asyncio.run(FeishuChannelService(settings).run_forever())
