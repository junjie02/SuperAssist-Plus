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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from superassist.agent import AgentRuntime
from superassist.agent.short_memory import append_jsonl
from superassist.config import PROJECT_ROOT, REASONING_EFFORTS, Settings, get_settings
from superassist.llm import is_gpt_5_6_model
from superassist.memory.embedding import get_embedder
from superassist.models import AgentRunEvent
from superassist.redis_store import get_redis_store
from superassist.tools.images import MAX_ORIGINAL_BYTES, download_image_url, load_generated_image

from .daily_brief import (
    DailyBriefProgress,
    DailyBriefProgressReporter,
    DailyBriefRunResult,
    DailyBriefScheduler,
)
from .daily_quiz import (
    DailyQuizScheduler,
    DailyQuizStore,
    get_daily_quiz_store,
)
from .feishu_documents import FeishuDocumentPublisher, PublishedDocument, contains_math_formula
from .store import FeishuMessageStore, FeishuStoredMessage, FeishuThreadStore

logger = logging.getLogger(__name__)

UNSUPPORTED_FILE_MESSAGE = "当前飞书入口第一版仅支持文本指令，文件和图片暂未接入。"
STREAM_UPDATE_INTERVAL_SECONDS = 0.3
IMAGE_DOWNLOAD_FAILED_MESSAGE = (
    "图片读取或格式转换失败。请确认飞书 im:resource 权限已生效，"
    "或重新发送 PNG、JPEG、GIF、WebP 图片。"
)
IMAGE_TOO_LARGE_MESSAGE = "图片过大，暂时只支持 10 MB 以内的图片。"
MAX_FEISHU_IMAGE_BYTES = 10 * 1024 * 1024
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
    created_at: str = ""

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
    images: tuple[FeishuCardImage, ...] = ()


@dataclass(frozen=True)
class FeishuCardImage:
    title: str
    source_url: str
    image_key: str = ""


@dataclass(frozen=True)
class FeishuImageContext:
    payloads: tuple[tuple[bytes, str], ...]
    expires_at: float


@dataclass
class FeishuPendingActivation:
    trigger: FeishuInboundMessage
    started_at: float
    last_message_at: float
    changed: asyncio.Event


@dataclass(frozen=True)
class FeishuBatchImage:
    message_id: str
    sender_open_id: str
    index: int
    data: bytes
    mime_type: str


class FeishuChannel:
    """Feishu/Lark WebSocket channel that calls AgentRuntime directly."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: Callable[..., AgentRuntime] | None = None,
        store: FeishuThreadStore | None = None,
        message_store: FeishuMessageStore | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        daily_brief_trigger: (
            Callable[[str, DailyBriefProgressReporter | None], Awaitable[DailyBriefRunResult]] | None
        ) = None,
        daily_quiz_store: DailyQuizStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or FeishuThreadStore(settings.feishu_thread_store_path)
        self.message_store = message_store or FeishuMessageStore(settings.feishu_message_store_path)
        self.runtime_factory = runtime_factory
        self.allowed_open_ids = settings.feishu_allowed_open_id_set
        self.mention_only = settings.feishu_mention_only
        self.active_session_seconds = settings.feishu_active_session_seconds
        self._monotonic_clock = monotonic_clock
        self.daily_brief_trigger = daily_brief_trigger
        self.daily_quiz_store = daily_quiz_store or get_daily_quiz_store(settings)
        self._redis = get_redis_store(settings)
        self._active_group_until: dict[str, float] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._api_client = None
        self._document_publisher: FeishuDocumentPublisher | None = None
        self._published_documents: dict[str, PublishedDocument] = {}
        self._running_cards: dict[str, str] = {}
        self._last_card_text: dict[str, str] = {}
        self._card_views: dict[str, FeishuCardView] = {}
        self._scope_locks: dict[str, asyncio.Lock] = {}
        self._pending_activations: dict[str, FeishuPendingActivation] = {}
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
        self._CreateImageRequest = None
        self._CreateImageRequestBody = None
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
                CreateImageRequest,
                CreateImageRequestBody,
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
        self._CreateImageRequest = CreateImageRequest
        self._CreateImageRequestBody = CreateImageRequestBody
        self._api_client = (
            lark.Client.builder()
            .app_id(self.settings.feishu_app_id)
            .app_secret(self.settings.feishu_app_secret)
            .domain(self.settings.feishu_domain)
            .build()
        )
        self._document_publisher = FeishuDocumentPublisher(
            self._api_client,
            doc_url_base=self.settings.feishu_doc_url_base,
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
        for pending in self._pending_activations.values():
            pending.changed.set()
        self._pending_activations.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    async def handle_inbound(self, inbound: FeishuInboundMessage) -> None:
        sender_allowed = not self.allowed_open_ids or inbound.sender_open_id in self.allowed_open_ids
        if inbound.is_private and not sender_allowed:
            logger.info("Ignoring Feishu message from non-allowed open_id=%s", inbound.sender_open_id)
            return
        inserted, message_seq = self.message_store.add_message(
            message_id=inbound.message_id,
            chat_id=inbound.chat_id,
            sender_open_id=inbound.sender_open_id,
            sender_name=inbound.sender_name,
            text=inbound.text,
            root_id=inbound.root_id,
            chat_type=inbound.chat_type,
            mentions=inbound.mentions,
            files=inbound.files,
            created_at=inbound.created_at,
        )
        if not inserted:
            logger.info("Ignoring duplicate Feishu message id=%s", inbound.message_id)
            return
        if not self._redis.claim_once("feishu-message", inbound.message_id, ttl_seconds=7 * 86400):
            logger.info("Ignoring Redis-deduplicated Feishu message id=%s", inbound.message_id)
            return
        await self._cache_inbound_images(inbound)

        scope_key = f"private:{inbound.chat_id}" if inbound.is_private else f"group:{inbound.chat_id}"
        pending = self._pending_activations.get(scope_key)
        if self._redis.activation_times(scope_key) is not None:
            self._redis.touch_activation(
                scope_key,
                ttl_seconds=max(30, int(self.settings.feishu_activation_max_wait_seconds) + 30),
            )
        if pending is not None:
            pending.last_message_at = asyncio.get_running_loop().time()
            pending.changed.set()

        clean_text = clean_mention_text(inbound.text, inbound.mentions).strip()
        accepted = self._should_accept_message(inbound)
        if not sender_allowed:
            accepted = False
        logger.info(
            "Feishu inbound gate chat_type=%s chat_suffix=%s mentioned=%s accepted=%s",
            inbound.chat_type or "unknown",
            inbound.chat_id[-8:] if inbound.chat_id else "unknown",
            has_bot_mention(inbound),
            accepted,
        )
        if not accepted:
            return
        if re.fullmatch(r"/brief(?:\s+now)?", clean_text, flags=re.IGNORECASE):
            if self.daily_brief_trigger is None:
                await self._send_or_patch(inbound, "申论官媒简报功能尚未启用。", final=True)
                return
            await self._send_or_patch(inbound, "正在扫描官媒最新内容并生成申论简报，请稍候。", final=False)
            user_id, conversation_scope = feishu_memory_scope(inbound)
            self.store.get_or_create_thread_id(
                chat_id=inbound.chat_id,
                topic_id=conversation_scope,
                user_id=user_id,
            )
            loop = asyncio.get_running_loop()

            def report_progress(progress: DailyBriefProgress) -> None:
                loop.call_soon_threadsafe(
                    self._queue_card_update,
                    inbound,
                    format_daily_brief_progress(progress),
                )

            result = await self.daily_brief_trigger(inbound.chat_id, report_progress)
            await self._flush_card_updates(inbound.message_id)
            status_text = {
                "sent": "申论官媒简报已生成并发送。",
                "skipped": f"本次简报未执行：{result.message}",
                "failed": f"本次简报生成失败：{result.message}",
            }.get(result.status, result.message)
            await self._send_or_patch(inbound, status_text, final=True)
            self.message_store.commit_consumed(inbound.chat_id, message_seq)
            return
        if not clean_text and not inbound.files:
            return

        if not inbound.is_private:
            if pending is not None:
                if has_bot_mention(inbound):
                    pending.trigger = inbound
                return
            loop_time = asyncio.get_running_loop().time()
            pending = FeishuPendingActivation(
                trigger=inbound,
                started_at=loop_time,
                last_message_at=loop_time,
                changed=asyncio.Event(),
            )
            self._pending_activations[scope_key] = pending
            self._redis.touch_activation(
                scope_key,
                started=True,
                ttl_seconds=max(30, int(self.settings.feishu_activation_max_wait_seconds) + 30),
            )
            await self._collect_and_process_group(scope_key, pending)
            return

        lock = self._scope_locks.setdefault(scope_key, asyncio.Lock())
        async with lock:
            success = await self._process_single_message(inbound, clean_text, scope_key)
        if success:
            self.message_store.commit_consumed(inbound.chat_id, message_seq)

    async def _collect_and_process_group(
        self,
        scope_key: str,
        pending: FeishuPendingActivation,
    ) -> None:
        debounce = self.settings.feishu_activation_debounce_seconds
        hard_deadline = pending.started_at + self.settings.feishu_activation_max_wait_seconds
        try:
            while debounce > 0 and not self._redis.enabled:
                now = asyncio.get_running_loop().time()
                deadline = min(pending.last_message_at + debounce, hard_deadline)
                if now >= deadline:
                    break
                try:
                    await asyncio.wait_for(pending.changed.wait(), timeout=deadline - now)
                except TimeoutError:
                    break
                pending.changed.clear()
            while debounce > 0 and self._redis.enabled:
                times = self._redis.activation_times(scope_key)
                if times is None:
                    break
                started_at, last_message_at = times
                now_wall = time.time()
                deadline = min(
                    last_message_at + debounce,
                    started_at + self.settings.feishu_activation_max_wait_seconds,
                )
                if now_wall >= deadline:
                    break
                await asyncio.sleep(min(0.1, max(0.01, deadline - now_wall)))
        finally:
            if self._pending_activations.get(scope_key) is pending:
                self._pending_activations.pop(scope_key, None)

        with self._redis.lock("feishu-scope", scope_key, ttl_seconds=1200) as acquired:
            if not acquired:
                return
            through_seq = self.message_store.latest_seq(pending.trigger.chat_id)
            lock = self._scope_locks.setdefault(scope_key, asyncio.Lock())
            async with lock:
                messages = self.message_store.list_unconsumed(
                    pending.trigger.chat_id,
                    through_seq=through_seq,
                )
                if not messages:
                    self._redis.clear_activation(scope_key)
                    return
                success = await self._process_group_batch(pending.trigger, messages, scope_key)
                if success:
                    self.message_store.commit_consumed(pending.trigger.chat_id, through_seq)
            self._redis.clear_activation(scope_key)

    async def _process_single_message(
        self,
        inbound: FeishuInboundMessage,
        clean_text: str,
        scope_key: str,
    ) -> bool:
        active_image_context = self._get_image_context(scope_key)
        images = [item for item in inbound.files if item.get("image_key")]
        other_files = [item for item in inbound.files if not item.get("image_key")]
        if other_files:
            await self._send_or_patch(inbound, UNSUPPORTED_FILE_MESSAGE, final=True)
            return True
        image_payloads: list[tuple[bytes, str]] = []
        has_new_images = bool(images)
        if images:
            await self._send_or_patch(inbound, "Reading image...", final=False)
            try:
                clean_text = strip_file_placeholders(clean_text)
                image_payloads = await self._download_image_payloads(images, inbound.message_id)
            except ImageTooLargeError:
                await self._send_or_patch(inbound, IMAGE_TOO_LARGE_MESSAGE, final=True)
                return False
            except ImageDownloadError:
                logger.exception("Failed to download Feishu image")
                await self._send_or_patch(inbound, IMAGE_DOWNLOAD_FAILED_MESSAGE, final=True)
                return False
        elif active_image_context is not None:
            image_payloads = list(active_image_context.payloads)
        if not clean_text and not image_payloads:
            return True
        try:
            return await self._handle_scoped_message(
                inbound,
                clean_text,
                image_payloads,
                extract_image_ocr=has_new_images,
            )
        finally:
            if image_payloads:
                self._set_image_context(scope_key, image_payloads)

    def _should_accept_message(self, inbound: FeishuInboundMessage) -> bool:
        if inbound.is_private or not self.mention_only:
            return True

        if self.daily_quiz_store.is_active_reply(inbound.chat_id, inbound.root_id):
            return True
        return has_bot_mention(inbound)

    async def _download_image_payloads(
        self,
        images: list[dict[str, str]],
        message_id: str,
    ) -> list[tuple[bytes, str]]:
        return list(
            await asyncio.gather(
                *(
                    self._get_or_download_image(message_id, item["image_key"])
                    for item in images
                )
            )
        )

    async def _cache_inbound_images(self, inbound: FeishuInboundMessage) -> None:
        images = [item for item in inbound.files if item.get("image_key")]
        if not images:
            return
        for item in images:
            image_key = item["image_key"]
            if self.message_store.get_image(message_id=inbound.message_id, image_key=image_key):
                continue
            try:
                data, mime_type = await self._download_image(inbound.message_id, image_key)
            except (ImageTooLargeError, ImageDownloadError):
                logger.warning(
                    "Could not cache Feishu image at ingress message_suffix=%s",
                    inbound.message_id[-8:] if inbound.message_id else "unknown",
                )
                continue
            self.message_store.save_image(
                message_id=inbound.message_id,
                image_key=image_key,
                data=data,
                mime_type=mime_type,
            )

    async def _get_or_download_image(self, message_id: str, image_key: str) -> tuple[bytes, str]:
        cached = self.message_store.get_image(message_id=message_id, image_key=image_key)
        if cached is not None:
            return cached
        data, mime_type = await self._download_image(message_id, image_key)
        self.message_store.save_image(
            message_id=message_id,
            image_key=image_key,
            data=data,
            mime_type=mime_type,
        )
        return data, mime_type

    async def _process_group_batch(
        self,
        trigger: FeishuInboundMessage,
        messages: list[FeishuStoredMessage],
        scope_key: str,
    ) -> bool:
        trigger_text = clean_mention_text(trigger.text, trigger.mentions).strip()
        is_effort_command, _requested_effort = parse_effort_command(trigger_text)
        if is_effort_command:
            return await self._handle_scoped_message(trigger, trigger_text)

        image_specs: list[tuple[FeishuStoredMessage, dict[str, str]]] = []
        for message in messages:
            image_specs.extend(
                (message, item)
                for item in message.files
                if item.get("image_key")
            )
        omitted_images = max(0, len(image_specs) - self.settings.feishu_max_images_per_activation)
        image_specs = image_specs[: self.settings.feishu_max_images_per_activation]
        batch_images: list[FeishuBatchImage] = []
        if image_specs:
            await self._send_or_patch(trigger, "Reading images...", final=False)
            try:
                payloads = await asyncio.gather(
                    *(
                        self._get_or_download_image(message.message_id, item["image_key"])
                        for message, item in image_specs
                    )
                )
            except ImageTooLargeError:
                await self._send_or_patch(trigger, IMAGE_TOO_LARGE_MESSAGE, final=True)
                return False
            except ImageDownloadError:
                logger.exception("Failed to download Feishu batch images")
                await self._send_or_patch(trigger, IMAGE_DOWNLOAD_FAILED_MESSAGE, final=True)
                return False
            for index, ((message, _item), (data, mime_type)) in enumerate(
                zip(image_specs, payloads, strict=True),
                start=1,
            ):
                batch_images.append(
                    FeishuBatchImage(
                        message_id=message.message_id,
                        sender_open_id=message.sender_open_id,
                        index=index,
                        data=data,
                        mime_type=mime_type,
                    )
                )

        ocr_text = await self._extract_images_ocr(
            [(item.data, item.mime_type) for item in batch_images]
        ) if batch_images else ""
        runtime_text = build_feishu_conversation_batch(
            messages,
            ocr_text=ocr_text,
            omitted_images=omitted_images,
            trigger_message_ids={trigger.message_id},
        )
        message_content: str | list[dict[str, Any]] = runtime_text
        if batch_images:
            message_content = build_feishu_batch_multimodal_content(runtime_text, batch_images)
        success = await self._handle_scoped_message(
            trigger,
            trigger_text or default_image_only_request(len(batch_images)),
            runtime_text_override=runtime_text,
            message_content_override=message_content,
            memory_query=trigger_text or runtime_text[-4000:],
            memory_source_context={
                "batch_id": f"feishu:{trigger.chat_id}:{messages[0].seq}:{messages[-1].seq}",
                "message_ids": [message.message_id for message in messages],
                "sender_ids": list(dict.fromkeys(message.sender_open_id for message in messages)),
                "channel": "feishu",
            },
        )
        if success and batch_images:
            self._set_image_context(
                scope_key,
                [(item.data, item.mime_type) for item in batch_images],
            )
        return success

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
        runtime_text_override: str | None = None,
        message_content_override: str | list[dict[str, Any]] | None = None,
        memory_query: str | None = None,
        memory_source_context: dict[str, Any] | None = None,
    ) -> bool:
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
            return True

        await self._send_or_patch(inbound, "Preparing context...", final=False)
        memory_text = clean_text or default_image_only_request(len(image_payloads or []))
        runtime_text = runtime_text_override or attributed_feishu_text(inbound, memory_text)
        message_content: str | list[dict[str, Any]] = message_content_override or runtime_text
        if image_payloads and runtime_text_override is None:
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
            runtime = self._create_runtime(
                report,
                reasoning_effort=reasoning_effort,
            )
            runtime_kwargs: dict[str, Any] = {"user_id": user_id, "thread_id": thread_id}
            if image_payloads or isinstance(message_content, list):
                runtime_kwargs["message_content"] = message_content
            if memory_query and isinstance(runtime, AgentRuntime):
                runtime_kwargs["memory_query"] = memory_query
            if memory_source_context and isinstance(runtime, AgentRuntime):
                runtime_kwargs["memory_source_context"] = memory_source_context
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
                final_text = format_model_error(
                    result.metadata,
                    has_images=bool(image_payloads) or isinstance(message_content, list),
                )
            previous = self._card_views.get(inbound.message_id)
            outbound_images = await self._prepare_outbound_images(result.metadata.get("outbound_images"))
            final_view: str | FeishuCardView = final_text
            if (previous and previous.reasoning) or outbound_images:
                final_view = FeishuCardView(
                    answer=final_text,
                    reasoning=previous.reasoning if previous else "",
                    reasoning_expanded=False,
                    images=tuple(outbound_images),
                )
            card_id = await self._send_or_patch(inbound, final_view, final=True)
            active_quiz = self.daily_quiz_store.active_session(thread_id)
            if (
                card_id
                and active_quiz
                and not active_quiz.get("question_message_id")
                and "政治理论·单项选择" in final_text
            ):
                self.daily_quiz_store.set_question_message_id(thread_id, card_id)
            return not bool(result.metadata.get("model_error"))
        except Exception:
            logger.exception("Feishu agent run failed")
            await self._flush_card_updates(inbound.message_id)
            await self._send_or_patch(inbound, "处理这条飞书消息时出错了，请稍后重试。", final=True)
            return False
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

    async def _prepare_outbound_images(self, raw_images: Any) -> list[FeishuCardImage]:
        if not isinstance(raw_images, list):
            return []
        prepared: list[FeishuCardImage] = []
        for raw in raw_images[:3]:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "Image").strip()[:200]
            source_url = _public_link(str(raw.get("source_url") or ""))
            image_key = ""
            try:
                local_path = str(raw.get("local_path") or "").strip()
                if local_path:
                    data = await asyncio.to_thread(
                        load_generated_image,
                        local_path,
                        self.settings.generated_image_cache_dir,
                    )
                else:
                    data, _mime_type = await asyncio.to_thread(_download_outbound_candidate, raw)
                image_key = await self._upload_image(data)
            except Exception as exc:  # noqa: BLE001 - media failure must not fail the text response
                logger.warning(
                    "Failed to prepare selected outbound image candidate_id=%s error_type=%s",
                    raw.get("candidate_id"),
                    type(exc).__name__,
                )
            if image_key or source_url:
                prepared.append(FeishuCardImage(title=title, source_url=source_url, image_key=image_key))
        return prepared

    async def _upload_image(self, data: bytes) -> str:
        if not self._api_client or not self._CreateImageRequest or not self._CreateImageRequestBody:
            raise RuntimeError("Feishu image upload client is not initialized")
        request = self._CreateImageRequest.builder().request_body(
            self._CreateImageRequestBody.builder()
            .image_type("message")
            .image(io.BytesIO(data))
            .build()
        ).build()
        response = await asyncio.to_thread(self._api_client.im.v1.image.create, request)
        if not response.success():
            raise RuntimeError(f"Feishu image upload failed with code={getattr(response, 'code', None)}")
        image_key = str(getattr(getattr(response, "data", None), "image_key", "") or "")
        if not image_key:
            raise RuntimeError("Feishu image upload returned no image_key")
        return image_key

    def _get_image_context(self, scope_key: str) -> FeishuImageContext | None:
        context = self._image_contexts.get(scope_key)
        if context is None:
            payloads = self._redis.load_image_context(scope_key)
            if not payloads:
                return None
            self._redis.save_image_context(
                scope_key,
                payloads,
                ttl_seconds=self.settings.feishu_image_context_ttl_seconds,
            )
            expires_at = self._monotonic_clock() + self.settings.feishu_image_context_ttl_seconds
            context = FeishuImageContext(tuple(payloads), expires_at)
            self._image_contexts[scope_key] = context
            self._schedule_image_context_expiry(scope_key, expires_at)
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
        self._redis.save_image_context(
            scope_key,
            payloads,
            ttl_seconds=self.settings.feishu_image_context_ttl_seconds,
        )
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
        self._redis.delete("feishu:image-context", scope_key)
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
    ) -> str | None:
        if isinstance(text, str):
            text = text.strip()
        if (isinstance(text, str) and not text) or (
            isinstance(text, FeishuCardView) and not text.reasoning and not text.answer and not text.images
        ):
            return None
        visible_text = text.answer if isinstance(text, FeishuCardView) else text
        if visible_text:
            self._last_card_text[inbound.message_id] = visible_text
        remote_card = self._redis.get_card_state(inbound.message_id)
        card_id = self._running_cards.get(inbound.message_id) or (
            str(remote_card.get("card_id") or "") if remote_card else ""
        )
        if final and visible_text and contains_math_formula(visible_text):
            try:
                document_text = visible_text
                if isinstance(text, FeishuCardView):
                    source_lines = [
                        f"- [{image.title or '图片来源'}]({image.source_url})"
                        for image in text.images
                        if image.source_url
                    ]
                    if source_lines:
                        document_text += "\n\n## 图片来源\n" + "\n".join(source_lines)
                document = await self._publish_math_document(inbound, document_text)
                link_message_id = await self._send_document_link(inbound, document)
                if link_message_id:
                    if card_id:
                        await self._delete_message_best_effort(card_id)
                    if isinstance(text, FeishuCardView):
                        for image in text.images:
                            if image.image_key:
                                try:
                                    await self._send_image_message(inbound, image.image_key)
                                except Exception as exc:  # noqa: BLE001 - document delivery already succeeded
                                    logger.warning(
                                        "Failed to send image beside Feishu document error_type=%s",
                                        type(exc).__name__,
                                    )
                    self._finish_card_state(inbound.message_id)
                    return link_message_id
                logger.warning("Feishu document was created but its link message could not be sent")
            except Exception as exc:  # noqa: BLE001 - card fallback must preserve the answer
                logger.warning(
                    "Failed to publish math answer as Feishu document message_id=%s error_type=%s",
                    inbound.message_id,
                    type(exc).__name__,
                )
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
        if card_id:
            self._redis.set_card_state(
                inbound.message_id,
                card_id=card_id,
                last_text=str(visible_text or ""),
            )
        if final:
            self._finish_card_state(inbound.message_id)
        return card_id

    async def _publish_math_document(
        self,
        inbound: FeishuInboundMessage,
        markdown: str,
    ) -> PublishedDocument:
        cached = self._published_documents.get(inbound.message_id)
        if cached is not None:
            return cached
        remote = self._redis.get_json("feishu:math-document", inbound.message_id)
        if isinstance(remote, dict) and remote.get("document_id") and remote.get("url"):
            cached = PublishedDocument(
                document_id=str(remote["document_id"]),
                title=str(remote.get("title") or "SuperAssist 公式回答"),
                url=str(remote["url"]),
            )
            self._published_documents[inbound.message_id] = cached
            return cached
        if self._document_publisher is None:
            if self._api_client is None:
                raise RuntimeError("Feishu document client is not initialized")
            self._document_publisher = FeishuDocumentPublisher(
                self._api_client,
                doc_url_base=self.settings.feishu_doc_url_base,
            )
        published = await asyncio.to_thread(
            self._document_publisher.publish,
            markdown,
            chat_id=inbound.chat_id,
            sender_open_id=inbound.sender_open_id,
            is_private=inbound.is_private,
            idempotency_key=inbound.message_id,
        )
        self._published_documents[inbound.message_id] = published
        self._redis.set_json(
            "feishu:math-document",
            inbound.message_id,
            {
                "document_id": published.document_id,
                "title": published.title,
                "url": published.url,
            },
            ttl_seconds=30 * 86400,
        )
        return published

    async def _send_document_link(
        self,
        inbound: FeishuInboundMessage,
        document: PublishedDocument,
    ) -> str | None:
        text = f"包含公式的完整回答已整理为飞书云文档：\n{document.title}\n{document.url}"
        return await self._send_plain_message(inbound, "text", json.dumps({"text": text}, ensure_ascii=False))

    async def _send_image_message(self, inbound: FeishuInboundMessage, image_key: str) -> str | None:
        return await self._send_plain_message(
            inbound,
            "image",
            json.dumps({"image_key": image_key}, ensure_ascii=False),
        )

    async def _send_plain_message(
        self,
        inbound: FeishuInboundMessage,
        msg_type: str,
        content: str,
    ) -> str | None:
        if not self._api_client:
            return None
        if inbound.is_private and inbound.message_id:
            request = self._ReplyMessageRequest.builder().message_id(inbound.message_id).request_body(
                self._ReplyMessageRequestBody.builder()
                .msg_type(msg_type)
                .content(content)
                .reply_in_thread(True)
                .build()
            ).build()
            response = await asyncio.to_thread(self._api_client.im.v1.message.reply, request)
        else:
            request = self._CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
                self._CreateMessageRequestBody.builder()
                .receive_id(inbound.chat_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            ).build()
            response = await asyncio.to_thread(self._api_client.im.v1.message.create, request)
        response_data = getattr(response, "data", None)
        return getattr(response_data, "message_id", None)

    async def _delete_message_best_effort(self, message_id: str) -> None:
        if not self._api_client or not message_id:
            return
        try:
            from lark_oapi.api.im.v1 import DeleteMessageRequest

            request = DeleteMessageRequest.builder().message_id(message_id).build()
            response = await asyncio.to_thread(self._api_client.im.v1.message.delete, request)
            success = getattr(response, "success", None)
            if callable(success) and not success():
                logger.warning("Failed to remove temporary Feishu card message_id=%s", message_id)
        except Exception as exc:  # noqa: BLE001 - the document link was already delivered
            logger.warning(
                "Failed to remove temporary Feishu card message_id=%s error_type=%s",
                message_id,
                type(exc).__name__,
            )

    def _finish_card_state(self, message_id: str) -> None:
        self._pending_card_updates.pop(message_id, None)
        self._running_cards.pop(message_id, None)
        self._last_card_text.pop(message_id, None)
        self._card_views.pop(message_id, None)
        self._redis.delete("feishu:card", message_id)

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

    async def send_proactive_card(self, chat_id: str, text: str) -> str | None:
        """Send a standalone card for a scheduled or manually triggered task."""

        message_id = await self._create_card(chat_id, text)
        if message_id:
            self._remember_proactive_assistant_message(chat_id, text)
        return message_id

    async def start_daily_quiz(self, chat_id: str, scheduled_for: datetime) -> str:
        """Dispatch the scheduled question set to the dedicated quiz subagent."""

        entry = self.store.get_latest_chat_entry(chat_id)
        if entry is None:
            text = "政治理论练习尚未生成：请先在当前会话与主 Agent 对话一次，以建立会话。"
            await self._create_card(chat_id, text)
            return text
        if not self.daily_quiz_store.has_notebook(chat_id):
            text = "近三日日报笔记本目前还是空的；至少完成一次定时日报推送后再开始测验。"
            await self._create_card(chat_id, text)
            return text

        thread_id = str(entry["thread_id"])
        self.daily_quiz_store.start_session(chat_id, thread_id, scheduled_for, replace_existing=True)
        try:
            from superassist.tools.task import run_task

            result = await asyncio.to_thread(
                run_task,
                "生成今日政治理论练习",
                "根据工具提供的近几日日报和错题材料，生成、核验并保存今天的完整政治理论题组。",
                subagent_type="shenlun-quiz",
                parent_thread_id=thread_id,
                settings=self.settings,
            )
            if not result.startswith("Task Succeeded."):
                raise RuntimeError(result)
            current = self.daily_quiz_store.current_quiz_text(thread_id)
            if not current:
                raise RuntimeError("quiz subagent did not finalize a complete quiz set")
            message_id = await self._create_card(chat_id, current)
            self.daily_quiz_store.set_question_message_id(thread_id, message_id)
            if message_id:
                self._remember_quiz_visible_question(thread_id, current)
            return "政治理论测验已生成并完成检查，请在新卡片中一次提交全部答案。"
        except Exception as exc:
            logger.exception("Quiz subagent failed to start daily quiz chat_suffix=%s", chat_id[-8:])
            text = f"政治理论测验启动失败：{type(exc).__name__}: {exc}"
            await self._create_card(chat_id, text)
            return text

    def _remember_proactive_assistant_message(self, chat_id: str, text: str) -> None:
        entry = self.store.get_latest_chat_entry(chat_id)
        if entry is None:
            logger.warning(
                "Cannot add proactive Feishu message to main-agent history: no thread mapping chat_suffix=%s",
                chat_id[-8:] if chat_id else "unknown",
            )
            return
        thread_id = str(entry["thread_id"])
        append_jsonl(
            self.settings.data_dir / "threads" / thread_id / "messages.jsonl",
            [
                {
                    "role": "assistant",
                    "content": text,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": "daily_brief",
                }
            ],
        )
        logger.info(
            "Added proactive Feishu message to main-agent short memory thread_id=%s chars=%d",
            thread_id,
            len(text),
        )

    def _remember_quiz_visible_question(self, thread_id: str, question: str) -> None:
        append_jsonl(
            self.settings.data_dir / "threads" / thread_id / "messages.jsonl",
            [
                {
                    "role": "assistant",
                    "content": question,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": "daily_quiz",
                }
            ],
        )

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
        self.daily_quiz_store = get_daily_quiz_store(self.settings)
        self.channel = FeishuChannel(self.settings, daily_quiz_store=self.daily_quiz_store)
        self.daily_brief = DailyBriefScheduler(
            self.settings,
            self.channel.send_proactive_card,
            brief_recorder=self.daily_quiz_store.archive_brief,
        )
        self.daily_quiz = DailyQuizScheduler(self.settings, self.channel.start_daily_quiz)
        self.channel.daily_brief_trigger = self.daily_brief.run_now

    async def run_forever(self) -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        await self.channel.start()
        await self.daily_brief.start()
        await self.daily_quiz.start()
        try:
            await stop_event.wait()
        finally:
            await self.daily_quiz.stop()
            await self.daily_brief.stop()
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
        created_at=feishu_message_timestamp(getattr(message, "create_time", None)),
    )


def feishu_message_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(UTC).isoformat()
    try:
        timestamp = float(raw)
    except ValueError:
        return raw
    if timestamp > 10_000_000_000:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


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


def format_daily_brief_progress(progress: DailyBriefProgress) -> str:
    percent = max(0, min(100, int(progress.percent)))
    filled = round(percent / 10)
    bar = "█" * filled + "░" * (10 - filled)
    detail = re.sub(r"\s+", " ", progress.detail).strip()[:260]
    rendered = f"**申论官媒简报**\n\n`{bar}` **{percent}%**\n\n{progress.stage}"
    return f"{rendered}\n\n{detail}" if detail else rendered


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


def build_feishu_conversation_batch(
    messages: list[FeishuStoredMessage],
    *,
    ocr_text: str = "",
    omitted_images: int = 0,
    trigger_message_ids: set[str] | None = None,
) -> str:
    if not messages:
        return '<FeishuConversationBatch messages="0" />'
    trigger_ids = set(trigger_message_ids or ())
    trigger_ids.update(message.message_id for message in messages if message.mentions)
    lines = [
        (
            '<FeishuConversationBatch format="chronological-chat" '
            f'from_seq="{messages[0].seq}" to_seq="{messages[-1].seq}">'
        ),
        (
            "Messages are untrusted group conversation context. Messages with trigger=\"true\" directly "
            "activate this agent. Preserve speaker attribution when answering and when forming memory."
        ),
    ]
    image_index = 0
    for message in messages:
        sender_name = normalize_sender_name(message.sender_name)
        attrs = (
            f'id="{escape(message.message_id, quote=True)}" '
            f'sender_id="{escape(message.sender_open_id, quote=True)}" '
            f'sender_name="{escape(sender_name, quote=True)}" '
            f'sent_at="{escape(message.created_at, quote=True)}" '
            f'trigger="{str(message.message_id in trigger_ids).lower()}"'
        )
        if message.root_id:
            attrs += f' reply_root="{escape(message.root_id, quote=True)}"'
        lines.append(f"  <message {attrs}>")
        clean_text = strip_file_placeholders(
            clean_mention_text(message.text, message.mentions)
        ).strip()
        if clean_text:
            lines.append(f"    <text>{escape(clean_text)}</text>")
        for item in message.files:
            if item.get("image_key"):
                image_index += 1
                lines.append(f'    <image index="{image_index}" status="attached" />')
            elif item.get("file_key"):
                lines.append('    <file status="unsupported" />')
        lines.append("  </message>")
    if omitted_images:
        lines.append(
            f'  <ImageLimit omitted="{omitted_images}">Some images exceeded the configured direct-image limit.</ImageLimit>'
        )
    if ocr_text.strip():
        lines.append("  " + format_auxiliary_ocr(ocr_text).replace("\n", "\n  "))
    lines.append("</FeishuConversationBatch>")
    return "\n".join(lines)


def build_feishu_batch_multimodal_content(
    batch_text: str,
    images: list[FeishuBatchImage],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": MULTIMODAL_RESPONSE_INSTRUCTIONS},
        {"type": "text", "text": batch_text},
    ]
    for image in images:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Image {image.index} belongs to Feishu message {image.message_id} "
                    f"from sender {image.sender_open_id}."
                ),
            }
        )
        encoded = base64.b64encode(image.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
            }
        )
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


def _download_outbound_candidate(candidate: dict[str, Any]) -> tuple[bytes, str]:
    errors: list[Exception] = []
    for key in ("image_url", "thumbnail_url"):
        url = str(candidate.get(key) or "").strip()
        if not url:
            continue
        try:
            return download_image_url(url, max_bytes=MAX_ORIGINAL_BYTES, timeout=15)
        except Exception as exc:  # noqa: BLE001 - original-to-thumbnail fallback
            errors.append(exc)
    raise errors[-1] if errors else ImageDownloadError("Selected image has no downloadable URL")


def _public_link(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return url.strip()


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
    if isinstance(text, FeishuCardView):
        elements: list[dict[str, Any]] = []
        if text.reasoning:
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "expanded": text.reasoning_expanded,
                    "header": {
                        "title": {"tag": "plain_text", "content": "思考过程"},
                        "vertical_align": "center",
                    },
                    "elements": [{"tag": "markdown", "content": text.reasoning}],
                }
            )
        if text.answer:
            elements.append({"tag": "markdown", "content": text.answer})
        for item in text.images:
            if item.image_key:
                elements.append(
                    {
                        "tag": "img",
                        "img_key": item.image_key,
                        "alt": {"tag": "plain_text", "content": item.title or "Image"},
                        "preview": True,
                    }
                )
            if item.source_url:
                label = (item.title or "Image source").replace("[", "\\[").replace("]", "\\]")
                elements.append({"tag": "markdown", "content": f"[图片来源：{label}]({item.source_url})"})
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
