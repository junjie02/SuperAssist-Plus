from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import re
import signal
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageOps

from superassist.agent import AgentRuntime
from superassist.agent.short_memory import append_jsonl
from superassist.config import PROJECT_ROOT, REASONING_EFFORTS, Settings, get_settings
from superassist.conversation_state import (
    CLEAR_CONFIRMATION,
    CLEAR_GROUP_CONFIRMATION,
    CLEAR_WITH_ATTACHMENTS_MESSAGE,
    ConversationStateBusyError,
    clear_short_state,
    conversation_state_lock,
    parse_clear_command,
)
from superassist.llm import supports_reasoning_effort
from superassist.memory.embedding import get_embedder
from superassist.models import AgentRunEvent
from superassist.redis_store import get_redis_store
from superassist.tools.documents import document_kind, extract_document_text
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

SHENLUN_GRADING_EVENT_TYPES = {
    "grading_question_started",
    "grading_grader_completed",
    "grading_third_review_started",
    "grading_arbitration_started",
    "grading_question_completed",
    "grading_batch_completed",
}

UNSUPPORTED_FILE_MESSAGE = (
    "这个文件类型暂时读不了（已保存到工作区）。目前支持 PDF、Word、Excel、CSV 和纯文本；"
    "如果是旧版 .doc/.xls，请另存为 .docx/.xlsx 后再发一次。"
)
FILE_DOWNLOAD_FAILED_MESSAGE = (
    "文件下载失败。请确认飞书 im:resource 权限已生效，或把文件重新发送一次。"
)
FILE_TOO_LARGE_MESSAGE = "文件过大，暂时只支持 30 MB 以内的文件。"
MAX_FEISHU_FILE_BYTES = 30 * 1024 * 1024
# How much of an inbound document's text is replayed into every conversation
# batch; the full text stays readable with read_file.
FILE_CONTEXT_MAX_CHARS = 6000
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
The original pixels are supplied to you; do not claim the image was lost or that you only saw transcribed text.
If the text is too small to read reliably, or you must quote exact wording, call the `ocr_image` tool with the
workspace-relative path printed next to that image (inbound images live under 图片/YYYYMMDD/) and verify the returned
text against the image. Only OCR images attached to this message; never OCR other files found on disk, and skip the
tool entirely when you can already read the question, options and labels yourself.
Images may carry EXIF orientation metadata; trust what you actually see rather than assuming a rotation.
When the answer depends on an option letter, number, or ordering, verify the reasoning first and then separately
map it back to the exact visible label in the image.
In the final response, first answer the user's request. Then append one concise, question-relevant visual
description inside <ImageDescription>...</ImageDescription>. Preserve the details that a later conversation turn
would need if the original image were no longer available. Do not merely repeat the OCR text.
</MultimodalResponseInstructions>"""

OCR_ORIENTATION_ANGLES = (0, 90, 180, 270)
OCR_ORIENTATION_MIN_CHARS = 12
OCR_ORIENTATION_MIN_CONFIDENCE = 0.65
OCR_ORIENTATION_MIN_IMPROVEMENT = 1.18
OCR_ORIENTATION_MIN_SECOND_BEST_IMPROVEMENT = 1.05


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


@dataclass(frozen=True)
class FeishuOCRCandidate:
    angle: int
    lines: tuple[str, ...]
    character_count: int
    average_confidence: float
    horizontal_layout: float = 1.0

    @property
    def score(self) -> float:
        return self.character_count * self.average_confidence * self.horizontal_layout


@dataclass
class FeishuPendingActivation:
    trigger: FeishuInboundMessage
    started_at: float
    last_message_at: float
    changed: asyncio.Event
    cancelled: bool = False


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
        self.message_store = message_store or FeishuMessageStore(
            settings.feishu_message_store_path,
            image_dir=settings.resolved_tool_workspace_dir / "图片",
        )
        self.runtime_factory = runtime_factory
        self.allowed_open_ids = settings.feishu_allowed_open_id_set
        self.mention_only = settings.feishu_mention_only
        # Own open_id, used to tell "@this bot" apart from "@someone else" in groups.
        # Auto-resolved on start() via /open-apis/bot/v3/info unless configured explicitly.
        self._bot_open_id = str(settings.feishu_bot_open_id or "").strip()
        self._bot_name = str(settings.feishu_bot_name or "").strip()
        # Names that count as "@this bot" when Feishu omits the open_id on
        # image/mixed messages. Config values are comma-separated.
        self._bot_alias_names = tuple(
            dict.fromkeys(
                part.strip().lstrip("@")
                for part in str(settings.feishu_bot_name or "").split(",")
                if part.strip()
            )
        )
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
        self._grading_card_states: dict[str, dict[str, Any]] = {}
        self._grading_state_lock = threading.Lock()
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
        self._ocr_thread_local = threading.local()
        self._ocr_initialization_lock = threading.Lock()
        self._ocr_executor_lock = threading.Lock()
        self._ocr_executor: ThreadPoolExecutor | None = self._new_ocr_executor()
        self._image_io_semaphore = asyncio.Semaphore(
            self.settings.feishu_image_processing_concurrency
        )

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
        await self._resolve_bot_identity()
        self._document_publisher = FeishuDocumentPublisher(
            self._api_client,
            doc_url_base=self.settings.feishu_doc_url_base,
        )
        self._main_loop = asyncio.get_running_loop()
        await asyncio.to_thread(get_embedder(self.settings).preload)
        self._running = True
        self._enable_event_log_file()
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()
        logger.info("Feishu channel started with domain %s", self.settings.feishu_domain)

    def _enable_event_log_file(self) -> None:
        """Mirror Feishu ingress logs to a dedicated file for ops debugging.

        The service is often launched by a supervisor whose console output is
        not retained. A file handler makes it possible to confirm whether a
        pushed event actually reached this process.
        """
        try:
            log_path = self.settings.data_dir / "logs" / "feishu-events.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            for name in (__name__, "lark_oapi.ws", "lark_oapi.event"):
                target = logging.getLogger(name)
                if not any(
                    isinstance(h, logging.FileHandler)
                    and getattr(h, "baseFilename", "") == str(log_path.resolve())
                    for h in target.handlers
                ):
                    target.addHandler(handler)
        except Exception:  # noqa: BLE001 - logging must never block the channel
            logger.warning("Could not enable Feishu event log file", exc_info=True)

    async def _resolve_bot_identity(self) -> None:
        """Resolve our own open_id so "@someone else" is not mistaken for "@bot"."""
        if self._bot_open_id:
            logger.info("Feishu bot open_id configured explicitly: %s", self._bot_open_id)
            return
        try:
            from lark_oapi.channel import fetch_bot_identity
        except ImportError:
            logger.warning(
                "lark_oapi.channel.fetch_bot_identity is unavailable; set "
                "SUPERASSIST_FEISHU_BOT_OPEN_ID to enable accurate group @mention detection."
            )
            return
        config = getattr(self._api_client, "config", None)
        if config is None:
            logger.warning("Feishu API client has no config; cannot auto-resolve bot open_id.")
            return
        try:
            identity = await fetch_bot_identity(config)
        except Exception:
            logger.exception("Failed to resolve Feishu bot identity")
            identity = None
        if identity and identity.open_id:
            self._bot_open_id = identity.open_id
            identity_name = str(getattr(identity, "name", "") or "").strip()
            self._bot_name = self._bot_name or identity_name
            if identity_name:
                # Feishu resolves @ using the live display name, which may differ
                # from the configured alias (e.g. "小小娇助手版" vs "小小娇").
                self._bot_alias_names = tuple(
                    dict.fromkeys([*self._bot_alias_names, identity_name])
                )
            logger.info("Resolved Feishu bot open_id=%s", self._bot_open_id)
        else:
            logger.warning(
                "Could not resolve Feishu bot open_id; group @mentions will not trigger this bot. "
                "Set SUPERASSIST_FEISHU_BOT_OPEN_ID (or SUPERASSIST_FEISHU_BOT_NAME) to fix."
            )

    def _has_bot_mention(self, inbound: FeishuInboundMessage) -> bool:
        return has_bot_mention(
            inbound,
            bot_open_id=self._bot_open_id,
            bot_name=self._bot_name,
            bot_names=self._bot_alias_names,
        )

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
        with self._ocr_executor_lock:
            executor = self._ocr_executor
            self._ocr_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

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
        scope_key = f"private:{inbound.chat_id}" if inbound.is_private else f"group:{inbound.chat_id}"
        clean_text = clean_mention_text(inbound.text, inbound.mentions).strip()
        accepted = self._should_accept_message(inbound)
        if not sender_allowed:
            accepted = False
        logger.info(
            "Feishu inbound gate chat_type=%s chat_suffix=%s mentioned=%s "
            "mention_names=%s text=%r accepted=%s",
            inbound.chat_type or "unknown",
            inbound.chat_id[-8:] if inbound.chat_id else "unknown",
            self._has_bot_mention(inbound),
            [_mention_name(m) for m in inbound.mentions],
            inbound.text[:60],
            accepted,
        )
        if parse_clear_command(clean_text) and accepted:
            if not inbound.is_private and not self._has_bot_mention(inbound):
                logger.info("Ignoring unmentioned Feishu group /clear chat_suffix=%s", inbound.chat_id[-8:])
                return
            await self._handle_clear_command(inbound, message_seq=message_seq, scope_key=scope_key)
            return

        await self._cache_inbound_images(inbound)
        await self._cache_inbound_files(inbound)
        pending = self._pending_activations.get(scope_key)
        if self._redis.activation_times(scope_key) is not None:
            self._redis.touch_activation(
                scope_key,
                ttl_seconds=max(30, int(self.settings.feishu_activation_max_wait_seconds) + 30),
            )
        if pending is not None:
            pending.last_message_at = asyncio.get_running_loop().time()
            pending.changed.set()

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
        explicit_group_activation = not inbound.is_private and self._has_bot_mention(inbound)
        if not clean_text and not inbound.files and not explicit_group_activation:
            return

        if not inbound.is_private:
            if pending is not None:
                if self._has_bot_mention(inbound):
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

        if pending.cancelled:
            return

        with self._redis.lock("feishu-scope", scope_key, ttl_seconds=1200) as acquired:
            if not acquired:
                return
            if pending.cancelled:
                self._redis.clear_activation(scope_key)
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

    async def _handle_clear_command(
        self,
        inbound: FeishuInboundMessage,
        *,
        message_seq: int,
        scope_key: str,
    ) -> None:
        if inbound.files:
            await self._send_or_patch(inbound, CLEAR_WITH_ATTACHMENTS_MESSAGE, final=True)
            if inbound.is_private:
                self.message_store.commit_consumed(inbound.chat_id, message_seq)
            return

        with self._redis.lock("feishu-scope", scope_key, ttl_seconds=1200) as acquired:
            if not acquired:
                await self._send_or_patch(inbound, "当前会话正在处理中，请稍后重试 /clear。", final=True)
                return
            lock = self._scope_locks.setdefault(scope_key, asyncio.Lock())
            async with lock:
                pending = self._pending_activations.pop(scope_key, None)
                if pending is not None:
                    pending.cancelled = True
                    pending.changed.set()
                self._redis.clear_activation(scope_key)

                user_id, conversation_scope = feishu_memory_scope(inbound)
                thread_id = self.store.get_or_create_thread_id(
                    chat_id=inbound.chat_id,
                    topic_id=conversation_scope,
                    user_id=user_id,
                )
                try:
                    await asyncio.to_thread(
                        clear_short_state,
                        self.settings,
                        thread_id=thread_id,
                        user_id=user_id,
                        redis_store=self._redis,
                    )
                except ConversationStateBusyError:
                    await self._send_or_patch(inbound, "当前会话正在处理中，请稍后重试 /clear。", final=True)
                    return
                except (PermissionError, ValueError):
                    logger.exception("Rejected Feishu /clear thread_id=%s", thread_id)
                    await self._send_or_patch(inbound, "无法清除当前会话：会话不存在或无权访问。", final=True)
                    return

                self._clear_image_context(scope_key)
                self.message_store.commit_consumed(inbound.chat_id, message_seq)
                confirmation = CLEAR_CONFIRMATION if inbound.is_private else CLEAR_GROUP_CONFIRMATION
                await self._send_or_patch(inbound, confirmation, final=True)

    async def _process_single_message(
        self,
        inbound: FeishuInboundMessage,
        clean_text: str,
        scope_key: str,
    ) -> bool:
        active_image_context = self._get_image_context(scope_key)
        images = [item for item in inbound.files if item.get("image_key")]
        documents = [item for item in inbound.files if item.get("file_key")]
        if any(not item.get("path") for item in documents):
            await self._send_or_patch(inbound, "Reading file...", final=False)
            try:
                documents = await self._prepare_inbound_files(inbound, documents)
            except FileTooLargeError:
                await self._send_or_patch(inbound, FILE_TOO_LARGE_MESSAGE, final=True)
                return True
            except FileDownloadError:
                logger.exception(
                    "Failed to download Feishu file message_suffix=%s",
                    inbound.message_id[-8:] if inbound.message_id else "unknown",
                )
                await self._send_or_patch(inbound, FILE_DOWNLOAD_FAILED_MESSAGE, final=True)
                return True
            images = [item for item in inbound.files if item.get("image_key")]
            self.message_store.update_message_files(inbound.message_id, [*images, *documents])
            inbound.files[:] = [*images, *documents]
        if documents:
            if all(item.get("kind") == "binary" for item in documents):
                await self._send_or_patch(inbound, UNSUPPORTED_FILE_MESSAGE, final=True)
                return True
            clean_text = strip_file_placeholders(clean_text)
        image_payloads: list[tuple[bytes, str]] = []
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
        if not clean_text and not image_payloads and not documents:
            return True
        try:
            return await self._handle_scoped_message(
                inbound,
                clean_text,
                image_payloads,
            )
        finally:
            if image_payloads:
                self._set_image_context(scope_key, image_payloads)

    def _should_accept_message(self, inbound: FeishuInboundMessage) -> bool:
        if inbound.is_private or not self.mention_only:
            return True

        if self.daily_quiz_store.is_active_reply(inbound.chat_id, inbound.root_id):
            return True
        return self._has_bot_mention(inbound)

    async def _download_image_payloads(
        self,
        images: list[dict[str, str]],
        message_id: str,
    ) -> list[tuple[bytes, str]]:
        return list(
            await asyncio.gather(
                *(
                    self._run_bounded_image_io(
                        self._get_or_download_image,
                        message_id,
                        item["image_key"],
                    )
                    for item in images
                )
            )
        )

    async def _run_bounded_image_io(
        self,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
    ) -> Any:
        async with self._image_io_semaphore:
            return await operation(*args)

    async def _cache_inbound_images(self, inbound: FeishuInboundMessage) -> None:
        images = [item for item in inbound.files if item.get("image_key")]
        if not images:
            return

        async def cache_one(item: dict[str, str]) -> None:
            image_key = item["image_key"]
            if self.message_store.get_image(message_id=inbound.message_id, image_key=image_key):
                return
            try:
                data, mime_type = await self._run_bounded_image_io(
                    self._download_image,
                    inbound.message_id,
                    image_key,
                )
            except (ImageTooLargeError, ImageDownloadError):
                logger.warning(
                    "Could not cache Feishu image at ingress message_suffix=%s",
                    inbound.message_id[-8:] if inbound.message_id else "unknown",
                )
                return
            self.message_store.save_image(
                message_id=inbound.message_id,
                image_key=image_key,
                data=data,
                mime_type=mime_type,
            )

        await asyncio.gather(*(cache_one(item) for item in images))

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

        messages = [
            message
            for message in messages
            if not parse_clear_command(clean_mention_text(message.text, message.mentions))
        ]
        if not messages:
            return True

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
                        self._run_bounded_image_io(
                            self._get_or_download_image,
                            message.message_id,
                            item["image_key"],
                        )
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

        runtime_text = build_feishu_conversation_batch(
            messages,
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

    async def _cache_inbound_files(self, inbound: FeishuInboundMessage) -> None:
        """Download and extract files at ingress so "send file, then @" works."""

        documents = [
            item for item in inbound.files if item.get("file_key") and not item.get("path")
        ]
        if not documents:
            return
        try:
            prepared = await self._prepare_inbound_files(inbound, documents)
        except (FileTooLargeError, FileDownloadError):
            logger.warning(
                "Could not cache Feishu file at ingress message_suffix=%s",
                inbound.message_id[-8:] if inbound.message_id else "unknown",
            )
            return
        images = [item for item in inbound.files if item.get("image_key")]
        merged = [*images, *prepared]
        self.message_store.update_message_files(inbound.message_id, merged)
        inbound.files[:] = merged

    async def _prepare_inbound_files(
        self,
        inbound: FeishuInboundMessage,
        documents: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Download each inbound file, mirror it to the workspace and extract text."""

        prepared: list[dict[str, str]] = []
        for item in documents:
            if item.get("path"):
                prepared.append(dict(item))
                continue
            file_key = str(item.get("file_key") or "")
            raw_name = str(item.get("name") or "").strip() or f"{file_key[-10:] or 'file'}.bin"
            name = _safe_file_name(raw_name)
            data = await self._download_file(inbound.message_id, file_key)
            relative_path = await asyncio.to_thread(self._save_inbound_file, name, data)
            kind = document_kind(relative_path)
            text = ""
            if kind != "binary":
                absolute = self.settings.resolved_tool_workspace_dir / relative_path
                text = await asyncio.to_thread(extract_document_text, absolute)
            prepared.append(
                {
                    "file_key": file_key,
                    "name": name,
                    "path": relative_path,
                    "kind": kind,
                    "size": str(len(data)),
                    "text": text,
                }
            )
            logger.info(
                "Ingested Feishu file message_suffix=%s kind=%s bytes=%d chars=%d",
                inbound.message_id[-8:] if inbound.message_id else "unknown",
                kind,
                len(data),
                len(text),
            )
        return prepared

    async def _download_file(self, message_id: str, file_key: str) -> bytes:
        try:
            if not self._api_client or not self._GetMessageResourceRequest:
                raise RuntimeError("Feishu file client is not initialized")
            request = (
                self._GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("file")
                .build()
            )
            response = await asyncio.to_thread(self._api_client.im.v1.message_resource.get, request)
            if not response.success() or not response.file:
                raise RuntimeError(
                    f"Feishu file download failed with code={getattr(response, 'code', None)}"
                )
            data = response.file.read(MAX_FEISHU_FILE_BYTES + 1)
            if len(data) > MAX_FEISHU_FILE_BYTES:
                raise FileTooLargeError
            return data
        except FileTooLargeError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a typed download failure
            raise FileDownloadError from exc

    def _save_inbound_file(self, name: str, data: bytes) -> str:
        root = self.settings.resolved_tool_workspace_dir
        target_dir = root / "文件" / datetime.now().strftime("%Y%m%d")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        if target.exists():
            digest = hashlib.sha256(data).hexdigest()[:8]
            target = target_dir / f"{Path(name).stem}-{digest}{Path(name).suffix}"
        target.write_bytes(data)
        return str(target.relative_to(root)).replace("\\", "/")

    async def _handle_scoped_message(
        self,
        inbound: FeishuInboundMessage,
        clean_text: str,
        image_payloads: list[tuple[bytes, str]] | None = None,
        *,
        extract_image_ocr: bool = False,
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
            elif not supports_reasoning_effort(self.settings.model, self.settings.reasoning_model_prefixes):
                response = (
                    f"当前模型 `{self.settings.model}` 未列入可调思考强度的模型清单"
                    f"（`SUPERASSIST_REASONING_MODELS`）。"
                )
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
            image_paths = [
                self.message_store.find_image_path(data) or "" for data, _mime in image_payloads
            ]
            saved_paths = [path for path in image_paths if path]
            if saved_paths:
                persisted_text = (
                    f"{persisted_text}\n\n<ImageFiles>\n"
                    + "\n".join(saved_paths)
                    + "\n</ImageFiles>"
                )
            message_content = build_multimodal_image_content(
                build_multimodal_request_text(runtime_text, ocr_text),
                image_payloads,
                image_paths=image_paths,
            )
            runtime_text = persisted_text

        def report(event: AgentRunEvent) -> None:
            if event.type == "model_failover_reset":
                if self._main_loop and self._main_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self._reset_card_for_model_failover(inbound, event.message),
                        self._main_loop,
                    )
                    try:
                        future.result(timeout=10)
                    except Exception as exc:  # noqa: BLE001 - failover must continue if card reset fails
                        logger.warning(
                            "Failed to reset Feishu card before model failover error_type=%s",
                            type(exc).__name__,
                        )
                return
            if event.type in SHENLUN_GRADING_EVENT_TYPES:
                with self._grading_state_lock:
                    state = self._grading_card_states.get(inbound.message_id)
                    if state is None:
                        state = self._redis.get_grading_card_state(inbound.message_id)
                    state = apply_shenlun_grading_event(state, event)
                    self._grading_card_states[inbound.message_id] = state
                    self._redis.set_grading_card_state(inbound.message_id, state)
                    view = format_shenlun_grading_card(state)
            elif event.type == "thinking":
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
                grading_state = self._grading_card_states.get(inbound.message_id)
                if grading_state and grading_state.get("status") != "completed":
                    return
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
            await self._deliver_outbound_files(inbound, result.metadata.get("outbound_files"))
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
        ocr_enabled = (
            self.settings.feishu_image_ocr_enabled
            and self.settings.feishu_image_ocr_max_chars > 0
        )
        executor = self._get_ocr_executor()
        futures = [
            executor.submit(
                self._extract_single_image_ocr_sync,
                index,
                data,
                mime_type,
                ocr_enabled,
            )
            for index, (data, mime_type) in enumerate(image_payloads, start=1)
        ]
        image_texts: list[str] = []
        for index, future in enumerate(futures, start=1):
            transformed_payload, text = future.result()
            image_payloads[index - 1] = transformed_payload
            if text:
                image_texts.append(f"[Image {index}]\n{text}")
        return "\n\n".join(image_texts)

    def _new_ocr_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=self.settings.feishu_image_processing_concurrency,
            thread_name_prefix="feishu-ocr",
        )

    def _get_ocr_executor(self) -> ThreadPoolExecutor:
        with self._ocr_executor_lock:
            if self._ocr_executor is None:
                self._ocr_executor = self._new_ocr_executor()
            return self._ocr_executor

    def _extract_single_image_ocr_sync(
        self,
        index: int,
        data: bytes,
        mime_type: str,
        ocr_enabled: bool,
    ) -> tuple[tuple[bytes, str], str]:
        engine = self._get_ocr_engine() if ocr_enabled else None
        with Image.open(io.BytesIO(data)) as image:
            exif_orientation = int(image.getexif().get(274, 1) or 1)
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            candidates = (
                [self._ocr_candidate(engine, oriented, angle) for angle in OCR_ORIENTATION_ANGLES]
                if engine is not None
                else []
            )
            selected = select_ocr_orientation(candidates)
            selected_angle = selected.angle if selected is not None else 0
            final_image = oriented.rotate(selected_angle, expand=True) if selected_angle else oriented
            changed = exif_orientation != 1 or selected_angle != 0
            payload = (
                encode_recognition_image(final_image, source_mime_type=mime_type)
                if changed
                else (data, mime_type)
            )
            text = "\n".join(selected.lines) if selected is not None and selected.lines else ""
            logger.info(
                "Feishu image orientation analyzed index=%d size=%dx%d exif=%d selected=%d "
                "ocr_candidates=%s transformed=%s",
                index,
                oriented.width,
                oriented.height,
                exif_orientation,
                selected_angle,
                format_ocr_candidate_scores(candidates),
                changed,
            )
            return payload, text

    @staticmethod
    def _ocr_candidate(engine: Any, image: Image.Image, angle: int) -> FeishuOCRCandidate:
        candidate_image = image.rotate(angle, expand=True) if angle else image
        pixels = np.asarray(candidate_image)
        try:
            output = engine(pixels, use_cls=False)
        except TypeError:
            output = engine(pixels)
        results = output[0] if isinstance(output, tuple) else output
        lines: list[str] = []
        weighted_confidence = 0.0
        weighted_horizontal_layout = 0.0
        layout_character_count = 0
        character_count = 0
        for item in results or []:
            if not isinstance(item, (list, tuple)) or len(item) <= 1:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            confidence = _safe_ocr_confidence(item[2] if len(item) > 2 else 0.0)
            valid_characters = sum(character.isalnum() for character in text)
            if valid_characters <= 0:
                continue
            lines.append(text)
            character_count += valid_characters
            weighted_confidence += confidence * valid_characters
            horizontal_layout = _ocr_horizontal_layout(item[0] if item else None)
            if horizontal_layout is not None:
                weighted_horizontal_layout += horizontal_layout * valid_characters
                layout_character_count += valid_characters
        average_confidence = weighted_confidence / character_count if character_count else 0.0
        return FeishuOCRCandidate(
            angle=angle,
            lines=tuple(lines),
            character_count=character_count,
            average_confidence=average_confidence,
            horizontal_layout=(
                weighted_horizontal_layout / layout_character_count
                if layout_character_count
                else 1.0
            ),
        )

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine
        engine = getattr(self._ocr_thread_local, "engine", None)
        if engine is not None:
            return engine
        if getattr(self._ocr_thread_local, "initialization_attempted", False):
            raise RuntimeError("RapidOCR initialization previously failed")
        self._ocr_thread_local.initialization_attempted = True
        with self._ocr_initialization_lock:
            from rapidocr_onnxruntime import RapidOCR

            engine = RapidOCR()
        self._ocr_thread_local.engine = engine
        return engine

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

    async def _reset_card_for_model_failover(self, inbound: FeishuInboundMessage, message: str) -> None:
        """Replace partial provider output before the runtime replays on its fallback route."""

        await self._flush_card_updates(inbound.message_id)
        self._pending_card_updates.pop(inbound.message_id, None)
        self._last_card_text.pop(inbound.message_id, None)
        self._card_views.pop(inbound.message_id, None)
        await self._send_or_patch(
            inbound,
            message.strip() or "主模型响应中断，正在切换备用模型...",
            final=False,
        )

    def _on_message(self, event: Any) -> None:
        # Diagnostic: log the raw event shape at ingress so operators can tell
        # whether Feishu pushed the event at all, and whether it carried mentions.
        try:
            raw_message = getattr(getattr(event, "event", None), "message", None)
            logger.info(
                "Feishu event received msg_type=%s chat_type=%s has_mentions=%s content_prefix=%r",
                getattr(raw_message, "message_type", None) or getattr(raw_message, "msg_type", "unknown"),
                getattr(raw_message, "chat_type", "unknown"),
                bool(getattr(raw_message, "mentions", None)),
                str(getattr(raw_message, "content", "") or "")[:160],
            )
        except Exception:  # noqa: BLE001 - diagnostics must never block handling
            pass
        try:
            inbound = parse_feishu_event(event)
        except Exception:
            logger.exception("Failed to parse Feishu event")
            return
        if self._main_loop and self._main_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.handle_inbound(inbound), self._main_loop)
            # Without this callback any exception inside handle_inbound is swallowed
            # by the un-awaited Future, dropping the message with no trace in logs.
            future.add_done_callback(_log_inbound_failure)
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

    async def _deliver_outbound_files(self, inbound: FeishuInboundMessage, raw_files: Any) -> None:
        """Upload and send files the agent queued with the ``send_file`` tool."""

        items = [item for item in (raw_files or []) if isinstance(item, dict)][:5]
        workspace = self.settings.resolved_tool_workspace_dir.resolve()
        for item in items:
            relative = str(item.get("path") or "").strip()
            if not relative:
                continue
            try:
                target = (workspace / relative).resolve()
                target.relative_to(workspace)
                if not target.is_file():
                    raise FileNotFoundError(relative)
                file_key = await self._upload_file(target, str(item.get("name") or target.name))
                await self._send_file_message(inbound, file_key)
                logger.info(
                    "Sent Feishu file name=%s bytes=%d", target.name, target.stat().st_size
                )
            except Exception as exc:  # noqa: BLE001 - the text answer already went out
                logger.warning(
                    "Failed to send Feishu file path=%s error_type=%s",
                    relative,
                    type(exc).__name__,
                )
                await self._send_plain_message(
                    inbound,
                    "text",
                    json.dumps(
                        {"text": f"文件已保存在工作区：{relative}（上传到飞书失败，可直接取用该路径）"},
                        ensure_ascii=False,
                    ),
                )

    async def _upload_file(self, path: Path, name: str) -> str:
        if not self._api_client:
            raise RuntimeError("Feishu file client is not initialized")
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        request = CreateFileRequest.builder().request_body(
            CreateFileRequestBody.builder()
            .file_type(_feishu_file_type(name))
            .file_name(name)
            .file(io.BytesIO(path.read_bytes()))
            .build()
        ).build()
        response = await asyncio.to_thread(self._api_client.im.v1.file.create, request)
        if not response.success():
            raise RuntimeError(f"Feishu file upload failed with code={getattr(response, 'code', None)}")
        file_key = str(getattr(getattr(response, "data", None), "file_key", "") or "")
        if not file_key:
            raise RuntimeError("Feishu file upload returned no file_key")
        return file_key

    async def _send_file_message(self, inbound: FeishuInboundMessage, file_key: str) -> str | None:
        return await self._send_plain_message(
            inbound,
            "file",
            json.dumps({"file_key": file_key}, ensure_ascii=False),
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
        self._grading_card_states.pop(message_id, None)
        self._redis.delete("feishu:card", message_id)
        self._redis.delete("feishu:grading-card", message_id)

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
        self._append_proactive_history(thread_id, text, source="daily_brief")
        logger.info(
            "Added proactive Feishu message to main-agent short memory thread_id=%s chars=%d",
            thread_id,
            len(text),
        )

    def _remember_quiz_visible_question(self, thread_id: str, question: str) -> None:
        self._append_proactive_history(thread_id, question, source="daily_quiz")

    def _append_proactive_history(self, thread_id: str, content: str, *, source: str) -> None:
        try:
            with conversation_state_lock(self.settings, thread_id, redis_store=self._redis):
                append_jsonl(
                    self.settings.data_dir / "threads" / thread_id / "messages.jsonl",
                    [
                        {
                            "role": "assistant",
                            "content": content,
                            "created_at": datetime.now(UTC).isoformat(),
                            "source": source,
                        }
                    ],
                )
        except ConversationStateBusyError:
            logger.warning("Skipped proactive short-memory append for busy thread_id=%s source=%s", thread_id, source)

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


def _log_inbound_failure(future: Any) -> None:
    """Surface exceptions raised by ``handle_inbound`` instead of dropping them.

    ``asyncio.run_coroutine_threadsafe`` returns a Future nobody awaits, so an
    exception there is invisible: the message simply never appears in the store
    or the logs. Log it with the full traceback.
    """
    try:
        error = future.exception()
    except Exception:  # cancelled futures raise here
        return
    if error is not None:
        logger.error(
            "Feishu inbound handling failed: %s: %s",
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


def parse_feishu_event(event: Any) -> FeishuInboundMessage:
    # Feishu event shapes vary by subscription scope: @-mention deliveries may
    # omit/restructure `sender` or `message.mentions`. Never let a missing field
    # raise here — that silently drops the whole message in `_on_message`.
    event_body = getattr(event, "event", None)
    message = getattr(event_body, "message", None)
    if message is None:
        raise ValueError("Feishu event has no message body")
    sender_info = getattr(event_body, "sender", None)
    sender = getattr(sender_info, "sender_id", None)
    raw_content = getattr(message, "content", None)
    try:
        content = json.loads(raw_content) if isinstance(raw_content, str) else dict(raw_content or {})
    except (ValueError, TypeError):
        content = {}
    if not isinstance(content, dict):
        content = {}
    open_id = getattr(sender, "open_id", None) if sender is not None else None
    text, files = parse_feishu_content(content)
    raw_mentions = getattr(message, "mentions", []) or content.get("mentions", [])
    mentions = _coerce_mentions(raw_mentions)
    # Merge at-tags found in rich text so image/mixed messages keep their @ identity.
    known_ids = {_mention_open_id(m) for m in mentions if _mention_open_id(m)}
    known_names = {_mention_name(m).lstrip("@").casefold() for m in mentions if _mention_name(m)}
    for mention in _mentions_from_content(content):
        mid = _mention_open_id(mention)
        mname = _mention_name(mention).lstrip("@").casefold()
        if (mid and mid in known_ids) or (mname and mname in known_names):
            continue
        mentions.append(mention)
        if mid:
            known_ids.add(mid)
        if mname:
            known_names.add(mname)
    return FeishuInboundMessage(
        chat_id=str(getattr(message, "chat_id", "") or ""),
        message_id=str(getattr(message, "message_id", "") or ""),
        root_id=getattr(message, "root_id", None) or None,
        sender_open_id=str(open_id or ""),
        sender_name=str(
            (getattr(sender_info, "name", None) if sender_info is not None else None)
            or (getattr(sender, "name", None) if sender is not None else None)
            or ""
        ),
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


def _mentions_from_content(content: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract @-mentions embedded as ``at`` elements in rich-text (post) content.

    Feishu sometimes reports the mention only inside the rich-text body (with
    ``user_id`` / ``open_id`` and ``user_name``) and leaves ``message.mentions``
    empty — notably for image+mixed messages. Recover those identities here.
    """
    paragraphs = content.get("content")
    if not isinstance(paragraphs, list):
        return []
    mentions: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, list):
            continue
        for element in paragraph:
            if not isinstance(element, dict) or element.get("tag") != "at":
                continue
            open_id = element.get("open_id") or element.get("user_id") or ""
            name = element.get("user_name") or element.get("name") or element.get("text") or ""
            mentions.append(
                {"name": str(name), "open_id": str(open_id), "id": str(open_id), "id_type": "open_id"}
            )
    return mentions


def parse_feishu_content(content: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    files: list[dict[str, str]] = []
    if isinstance(content.get("text"), str):
        return content["text"], files
    if isinstance(content.get("file_key"), str):
        item = {"file_key": content["file_key"]}
        if isinstance(content.get("file_name"), str):
            item["name"] = content["file_name"]
        files.append(item)
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
                item = {"file_key": element["file_key"]}
                if isinstance(element.get("file_name"), str):
                    item["name"] = element["file_name"]
                files.append(item)
                parts.append("[file]")
        if parts:
            text_paragraphs.append(" ".join(parts))
    return "\n\n".join(text_paragraphs), files


def should_trigger_agent(
    inbound: FeishuInboundMessage,
    *,
    mention_only: bool,
    bot_open_id: str = "",
    bot_name: str = "",
    bot_names: tuple[str, ...] | list[str] = (),
) -> bool:
    if inbound.is_private:
        return True
    if not mention_only:
        return True
    return has_bot_mention(
        inbound,
        bot_open_id=bot_open_id,
        bot_name=bot_name,
        bot_names=bot_names,
    )


def _mention_open_id(mention: Any) -> str:
    """Extract the mentioned user's/bot's open_id from a Feishu event mention.

    Feishu sends ``id`` as a plain open_id string with ``id_type=open_id``,
    but some SDK versions / dict payloads wrap it as ``id.open_id``.
    """
    if isinstance(mention, dict):
        identifier = mention.get("id")
        if isinstance(identifier, dict):
            return str(identifier.get("open_id") or "")
        if identifier is not None and not isinstance(identifier, (str, int)):
            # ``MentionEvent.id`` is a UserId object when passed through raw.
            return str(getattr(identifier, "open_id", None) or "")
        explicit = str(mention.get("open_id") or "")
        if explicit:
            return explicit
        id_type = str(mention.get("id_type") or "").lower()
        if isinstance(identifier, str) and identifier and id_type in ("", "open_id", "bot"):
            return identifier
        return ""
    identifier = getattr(mention, "id", None)
    if isinstance(identifier, str):
        return identifier
    return str(getattr(identifier, "open_id", None) or "")


def _mention_name(mention: Any) -> str:
    if isinstance(mention, dict):
        return str(mention.get("name") or mention.get("text") or "")
    return str(getattr(mention, "name", None) or getattr(mention, "text", None) or "")


def has_bot_mention(
    inbound: FeishuInboundMessage,
    *,
    bot_open_id: str = "",
    bot_name: str = "",
    bot_names: tuple[str, ...] | list[str] = (),
) -> bool:
    """True when the message @-mentions this bot.

    Accept either an exact open_id match or an exact display-name match. Names
    are checked in addition to (not only when missing) open_id because Feishu
    sometimes omits ``mentions[].id`` for image/mixed messages while still
    sending ``mentions[].name``. A bare ``@`` with no structured mention never
    counts.
    """
    mentions = inbound.mentions or []
    if not mentions:
        return False
    if bot_open_id and any(_mention_open_id(m) == bot_open_id for m in mentions):
        return True
    names = [bot_name, *bot_names]
    expected = {n.lstrip("@").casefold() for n in names if n and n.strip()}
    if expected and any(_mention_name(m).lstrip("@").casefold() in expected for m in mentions):
        return True
    return False


def clean_mention_text(text: str, mentions: list[dict[str, Any]]) -> str:
    cleaned = text
    for mention in mentions:
        # ``_mention_name`` tolerates dicts and raw SDK Mention objects alike so a
        # malformed mention can never abort message handling.
        name = _mention_name(mention).strip()
        if name:
            cleaned = cleaned.replace(name, " ")
    cleaned = re.sub(r"(^|\s)@[^\s]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def strip_file_placeholders(text: str) -> str:
    return re.sub(r"\[(?:image|file)\]", " ", text, flags=re.IGNORECASE).strip()


def _feishu_file_type(name: str) -> str:
    """Map a file extension to the file_type Feishu's upload API expects."""

    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".doc", ".docx"}:
        return "doc"
    if suffix in {".xls", ".xlsx", ".xlsm", ".csv"}:
        return "xls"
    if suffix in {".ppt", ".pptx"}:
        return "ppt"
    if suffix == ".mp4":
        return "mp4"
    if suffix == ".opus":
        return "opus"
    return "stream"


def _safe_file_name(name: str) -> str:
    """Keep user-supplied file names usable on disk without allowing traversal."""

    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(name or "")).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return "file"
    if len(cleaned) > 120:
        stem, suffix = Path(cleaned).stem[:100], Path(cleaned).suffix[:12]
        cleaned = f"{stem}{suffix}"
    return cleaned


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
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, (data, mime_type) in enumerate(payloads):
        encoded = base64.b64encode(data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
        if image_paths and index < len(image_paths) and image_paths[index]:
            content.append(
                {
                    "type": "text",
                    "text": f"Workspace path for this image: {image_paths[index]}",
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
            "activate this agent. A trigger with no text asks the agent to handle the relevant preceding "
            "messages; if none are actionable, ask one brief clarifying question. Preserve speaker "
            "attribution when answering and when forming memory."
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
        elif message.message_id in trigger_ids and not message.files:
            lines.append('    <activation intent="handle-preceding-context" />')
        for item in message.files:
            if item.get("image_key"):
                image_index += 1
                lines.append(f'    <image index="{image_index}" status="attached" />')
            elif item.get("file_key"):
                raw_path = str(item.get("path") or "")
                name = escape(str(item.get("name") or Path(raw_path).name or "file"))
                path = escape(raw_path, quote=True)
                kind = escape(str(item.get("kind") or "binary"))
                size = escape(str(item.get("size") or "0"))
                text = str(item.get("text") or "").strip()
                if not raw_path:
                    lines.append(
                        f'    <file name="{name}" kind="{kind}" bytes="{size}" status="not-downloaded">'
                        "This file was not downloaded yet. Tell the user to resend it in the message "
                        "that mentions you, or ask them to paste the key text.</file>"
                    )
                elif text:
                    clipped = text[:FILE_CONTEXT_MAX_CHARS]
                    lines.append(
                        f'    <file name="{name}" path="{path}" kind="{kind}" bytes="{size}" status="extracted">'
                    )
                    for text_line in clipped.split("\n"):
                        lines.append(f"      {escape(text_line)}")
                    if len(text) > FILE_CONTEXT_MAX_CHARS:
                        lines.append(
                            f"      [以上为前 {FILE_CONTEXT_MAX_CHARS} 字，全文可用 read_file 读取 {raw_path}]"
                        )
                    lines.append("    </file>")
                else:
                    lines.append(
                        f'    <file name="{name}" path="{path}" kind="{kind}" bytes="{size}" '
                        'status="text-unavailable">Content could not be extracted (scanned PDF or legacy '
                        "Office format). Tell the user the file was received and saved, and ask them to "
                        "paste the key text or resend as PDF/DOCX/TXT.</file>"
                    )
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


def select_ocr_orientation(candidates: list[FeishuOCRCandidate]) -> FeishuOCRCandidate | None:
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


def format_ocr_candidate_scores(candidates: list[FeishuOCRCandidate]) -> str:
    if not candidates:
        return "disabled"
    return ",".join(
        f"{candidate.angle}:{candidate.character_count}/{candidate.average_confidence:.3f}"
        f"/{candidate.horizontal_layout:.3f}"
        for candidate in candidates
    )


def encode_recognition_image(image: Image.Image, *, source_mime_type: str) -> tuple[bytes, str]:
    output = io.BytesIO()
    if source_mime_type == "image/jpeg":
        image.save(output, format="JPEG", quality=92, optimize=True)
        return output.getvalue(), "image/jpeg"
    if source_mime_type == "image/webp":
        image.save(output, format="WEBP", quality=92, method=4)
        return output.getvalue(), "image/webp"
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), "image/png"


def _safe_ocr_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _ocr_horizontal_layout(value: Any) -> float | None:
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


class FileTooLargeError(ValueError):
    pass


class FileDownloadError(RuntimeError):
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


def apply_shenlun_grading_event(
    current: dict[str, Any] | None,
    event: AgentRunEvent,
) -> dict[str, Any]:
    """Merge one grading event into a complete, serializable card snapshot."""

    state = deepcopy(current) if isinstance(current, dict) else {
        "job_id": "",
        "status": "running",
        "completed": 0,
        "total": 5,
        "questions": {},
    }
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    state["job_id"] = str(metadata.get("job_id") or state.get("job_id") or "")
    try:
        state["total"] = max(1, int(metadata.get("total") or state.get("total") or 5))
    except (TypeError, ValueError):
        state["total"] = 5
    questions = state.setdefault("questions", {})
    if not isinstance(questions, dict):
        questions = {}
        state["questions"] = questions

    if event.type == "grading_batch_completed":
        state["status"] = "completed"
        summary = metadata.get("summary")
        if isinstance(summary, dict):
            state["summary"] = summary
            for result in summary.get("questions", []):
                if isinstance(result, dict):
                    _merge_completed_grading_question(questions, result)
        state["completed"] = int(metadata.get("completed") or state.get("total") or 0)
        return state

    try:
        number = int(metadata.get("question_number") or 0)
    except (TypeError, ValueError):
        return state
    if number <= 0:
        return state
    key = str(number)
    question = questions.setdefault(key, {"number": number, "graders": {}})
    question.update(
        {
            "number": number,
            "type": str(metadata.get("question_type") or question.get("type") or ""),
            "label": str(metadata.get("question_type_label") or question.get("label") or ""),
            "max_score": metadata.get("max_score", question.get("max_score")),
            "threshold": metadata.get("threshold", question.get("threshold")),
        }
    )
    report_question = metadata.get("question")
    if isinstance(report_question, dict):
        question["question"] = deepcopy(report_question)
    graders = question.setdefault("graders", {})
    if not isinstance(graders, dict):
        graders = {}
        question["graders"] = graders

    if event.type == "grading_question_started":
        question["status"] = "grading"
        state["current_question"] = number
    elif event.type == "grading_grader_completed":
        grader_id = str(metadata.get("grader_id") or "").lower()
        grader = metadata.get("grader")
        if grader_id and isinstance(grader, dict):
            graders[grader_id] = grader
        question["status"] = "grading"
    elif event.type == "grading_third_review_started":
        question["third_review_triggered"] = True
        question["status"] = "third_review"
    elif event.type == "grading_arbitration_started":
        question["third_review_triggered"] = True
        question["arbitration_triggered"] = True
        question["status"] = "arbitration"
    elif event.type == "grading_question_completed":
        _merge_completed_grading_question(questions, metadata)
        state["completed"] = sum(
            1 for item in questions.values()
            if isinstance(item, dict) and item.get("status") == "completed"
        )
        state["current_question"] = None
    return state


def _merge_completed_grading_question(questions: dict[str, Any], result: dict[str, Any]) -> None:
    try:
        number = int(result.get("question_number") or result.get("number") or 0)
    except (TypeError, ValueError):
        return
    if number <= 0:
        return
    question = questions.setdefault(str(number), {"number": number})
    report_question = result.get("question")
    question.update(
        {
            "number": number,
            "type": str(result.get("question_type") or question.get("type") or ""),
            "label": str(result.get("question_type_label") or question.get("label") or ""),
            "max_score": result.get("max_score", question.get("max_score")),
            "threshold": result.get("threshold", question.get("threshold")),
            "graders": result.get("graders") if isinstance(result.get("graders"), dict) else question.get("graders", {}),
            "score_difference": result.get("score_difference"),
            "third_review_triggered": bool(result.get("third_review_triggered")),
            "arbitration_triggered": bool(result.get("arbitration_triggered")),
            "arbitration": result.get("arbitration") if isinstance(result.get("arbitration"), dict) else None,
            "adopted_grader_ids": result.get("adopted_grader_ids") or [],
            "decision_method": str(result.get("decision_method") or ""),
            "calculation": str(result.get("calculation") or ""),
            "final_score": result.get("final_score"),
            "revision_advice": result.get("revision_advice") or [],
            "status": "completed",
        }
    )
    if isinstance(report_question, dict):
        question["question"] = deepcopy(report_question)


def format_shenlun_grading_card(state: dict[str, Any]) -> str:
    total = _safe_int(state.get("total"), 5)
    completed = _safe_int(state.get("completed"), 0)
    done = state.get("status") == "completed"
    title = "申论评分完成" if done else "申论评分中"
    lines = [f"**{title} · 已完成 {completed}/{total}**"]
    questions = state.get("questions") if isinstance(state.get("questions"), dict) else {}
    for number in range(1, total + 1):
        question = questions.get(str(number))
        if not isinstance(question, dict):
            lines.extend(["", f"**第{number}题｜尚未开始**"])
            continue
        label = str(question.get("label") or question.get("type") or "待识别")
        report_question = question.get("question") if isinstance(question.get("question"), dict) else {}
        display_number = str(report_question.get("display_number") or f"第{number}题")
        stem = str(report_question.get("stem") or "").strip()
        lines.extend(["", f"**{display_number}｜{label}**"])
        if stem:
            lines.extend(["**题目**", stem, "", "**解析**"])
        graders = question.get("graders") if isinstance(question.get("graders"), dict) else {}
        for grader_id in ("a", "b", "c"):
            grader = graders.get(grader_id)
            if not isinstance(grader, dict):
                continue
            score = _format_score(grader.get("score"))
            lines.append(f"评分员{_grader_number(grader_id)}：{score}分")
            scoring_points = _format_grading_scoring_points(grader.get("derived_scoring_points"))
            components = _format_grading_components(grader.get("score_components"))
            deductions = _format_grading_deductions(grader.get("deductions"))
            grader_calculation = _compact_grading_text(grader.get("calculation"), 180)
            if scoring_points:
                lines.append(f"采分依据：{scoring_points}")
            if components:
                lines.append(f"分项：{components}")
            if deductions:
                lines.append(f"扣分：{deductions}")
            if grader_calculation:
                lines.append(f"评分算式：{grader_calculation}")
            hit_points = _compact_grading_items(grader.get("hit_points"))
            missed_points = _compact_grading_items(grader.get("missed_points"))
            comments = _compact_grading_text(grader.get("comments"), 180)
            if hit_points:
                lines.append(f"命中：{hit_points}")
            if missed_points:
                lines.append(f"遗漏：{missed_points}")
            if comments:
                lines.append(f"意见：{comments}")

        if question.get("status") == "third_review" and "c" not in graders:
            lines.append("分差超过阈值，正在等待评分员3复核...")
        elif question.get("status") == "arbitration":
            lines.append("三评分歧仍未收敛，正在仲裁...")
        elif question.get("status") == "grading":
            lines.append("正在评分...")

        if question.get("status") == "completed":
            difference = _format_score(question.get("score_difference"))
            threshold = _format_score(question.get("threshold"))
            lines.append(f"初评分差：{difference}分，阈值：{threshold}分")
            if question.get("third_review_triggered"):
                lines.append("已触发三评")
            if question.get("arbitration_triggered"):
                lines.append("已触发仲裁")
                arbitration = question.get("arbitration")
                if isinstance(arbitration, dict) and arbitration.get("final_score") is not None:
                    lines.append(f"仲裁分：{_format_score(arbitration.get('final_score'))}分")
                    reason = _compact_grading_text(arbitration.get("reason"), 220)
                    if reason:
                        lines.append(f"仲裁理由：{reason}")
            adopted = question.get("adopted_grader_ids")
            if isinstance(adopted, list) and adopted:
                labels = "、".join(f"评分员{_grader_number(str(item))}" for item in adopted)
                lines.append(f"采用：{labels}")
            calculation = _compact_grading_text(question.get("calculation"), 120)
            if calculation:
                lines.append(f"计算：{calculation}")
            lines.append(f"最终得分：**{_format_score(question.get('final_score'))}分**")
            advice = _compact_grading_items(question.get("revision_advice"), max_items=3, max_chars=240)
            if advice:
                lines.append(f"修改建议：{advice}")
    return "\n".join(lines)


def _grader_number(grader_id: str) -> str:
    return {"a": "1", "b": "2", "c": "3"}.get(grader_id.lower(), grader_id.upper())


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{score:g}"


def _compact_grading_items(value: Any, *, max_items: int = 4, max_chars: int = 320) -> str:
    if not isinstance(value, list):
        return ""
    items = [_compact_grading_text(item, 100) for item in value]
    rendered = "；".join(item for item in items[:max_items] if item)
    return rendered[:max_chars]


def _format_grading_scoring_points(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    rendered: list[str] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        point = _compact_grading_text(item.get("point"), 70)
        awarded = item.get("awarded_score")
        maximum = item.get("max_score")
        score_text = ""
        if awarded is not None and maximum is not None:
            score_text = f"({_format_score(awarded)}/{_format_score(maximum)})"
        if point:
            rendered.append(f"{point}{score_text}")
    return "；".join(rendered)[:360]


def _format_grading_components(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return "；".join(
        f"{_compact_grading_text(key, 40)}={_format_score(score)}"
        for key, score in list(value.items())[:8]
    )[:300]


def _format_grading_deductions(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    rendered: list[str] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        points = _format_score(item.get("points"))
        if points in {"-", "0"}:
            continue
        kind = _compact_grading_text(item.get("type"), 30)
        basis = _compact_grading_text(item.get("basis"), 70)
        rendered.append(f"{kind} {points}分（{basis}）")
    return "；".join(rendered)[:320]


def _compact_grading_text(value: Any, max_chars: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_chars]


def _coerce_mentions(value: Any) -> list[dict[str, Any]]:
    """Normalize Feishu mentions into JSON-safe dicts.

    ``im.message.receive_v1`` delivers ``MentionEvent`` objects whose ``id`` is a
    ``UserId`` *object*, not a string. Keeping that object in the payload made
    ``json.dumps`` raise ``TypeError`` in the message store, which silently
    dropped every @-mention message before it could be logged. Only primitives
    may leave this function.
    """
    if not isinstance(value, list):
        return []
    mentions: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            raw = dict(item)
        else:
            raw = {
                "key": getattr(item, "key", None),
                "id": getattr(item, "id", None),
                "id_type": getattr(item, "id_type", None) or getattr(item, "mentioned_type", None),
                "name": getattr(item, "name", None) or getattr(item, "text", None),
                "tenant_key": getattr(item, "tenant_key", None),
            }
        mention: dict[str, Any] = {
            key: val for key, val in raw.items() if val is None or isinstance(val, (str, int, float, bool))
        }
        open_id = _mention_open_id(raw)
        if open_id:
            mention["open_id"] = open_id
        mentions.append(mention)
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
