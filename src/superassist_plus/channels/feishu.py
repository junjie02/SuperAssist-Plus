from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from superassist_plus.agent import AgentRuntime
from superassist_plus.config import Settings, get_settings
from superassist_plus.memory.embedding import get_embedder
from superassist_plus.models import AgentRunEvent

from .store import FeishuThreadStore

logger = logging.getLogger(__name__)

UNSUPPORTED_FILE_MESSAGE = "当前飞书入口第一版仅支持文本指令，文件和图片暂未接入。"


@dataclass
class FeishuInboundMessage:
    chat_id: str
    message_id: str
    sender_open_id: str
    text: str
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


class FeishuChannel:
    """Feishu/Lark WebSocket channel that calls AgentRuntime directly."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: Callable[[Callable[[AgentRunEvent], None]], AgentRuntime] | None = None,
        store: FeishuThreadStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or FeishuThreadStore(settings.feishu_thread_store_path)
        self.runtime_factory = runtime_factory or (
            lambda reporter: AgentRuntime(settings, run_event_reporter=reporter)
        )
        self.allowed_open_ids = settings.feishu_allowed_open_id_set
        self.mention_only = settings.feishu_mention_only
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._api_client = None
        self._running_cards: dict[str, str] = {}
        self._last_card_text: dict[str, str] = {}
        self._lark = None
        self._CreateMessageRequest = None
        self._CreateMessageRequestBody = None
        self._ReplyMessageRequest = None
        self._ReplyMessageRequestBody = None
        self._PatchMessageRequest = None
        self._PatchMessageRequestBody = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        if not self.settings.feishu_app_id or not self.settings.feishu_app_secret:
            raise RuntimeError(
                "Feishu channel requires SUPERASSIST_PLUS_FEISHU_APP_ID and "
                "SUPERASSIST_PLUS_FEISHU_APP_SECRET."
            )
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (
                CreateMessageRequest,
                CreateMessageRequestBody,
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
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    async def handle_inbound(self, inbound: FeishuInboundMessage) -> None:
        if self.allowed_open_ids and inbound.sender_open_id not in self.allowed_open_ids:
            logger.info("Ignoring Feishu message from non-allowed open_id=%s", inbound.sender_open_id)
            return
        if not should_trigger_agent(inbound, mention_only=self.mention_only):
            return
        clean_text = clean_mention_text(inbound.text, inbound.mentions).strip()
        if not clean_text and inbound.files:
            await self._send_or_patch(inbound, UNSUPPORTED_FILE_MESSAGE, final=True)
            return
        if not clean_text:
            return

        await self._send_or_patch(inbound, "Preparing context...", final=False)
        user_id = f"feishu:{inbound.sender_open_id}"
        thread_id = self.store.get_or_create_thread_id(
            chat_id=inbound.chat_id,
            topic_id=inbound.topic_id,
            user_id=user_id,
        )

        def report(event: AgentRunEvent) -> None:
            if event.type == "thinking":
                text = "Thinking..."
            elif event.type == "agent_text":
                text = event.message.strip()
            else:
                return
            if not text:
                return
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_or_patch(inbound, text, final=False),
                    self._main_loop,
                )

        try:
            runtime = self.runtime_factory(report)
            result = await asyncio.to_thread(runtime.run_streaming, clean_text, user_id=user_id, thread_id=thread_id)
            runtime.memory_queue.flush()
            final_text = result.answer.strip() or self._last_card_text.get(inbound.message_id, "") or "(empty response)"
            await self._send_or_patch(inbound, final_text, final=True)
        except Exception:
            logger.exception("Feishu agent run failed")
            await self._send_or_patch(inbound, "处理这条飞书消息时出错了，请稍后重试。", final=True)

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

    async def _send_or_patch(self, inbound: FeishuInboundMessage, text: str, *, final: bool) -> None:
        text = text.strip()
        if not text:
            return
        self._last_card_text[inbound.message_id] = text
        card_id = self._running_cards.get(inbound.message_id)
        if card_id:
            await self._update_card(card_id, text)
        elif inbound.message_id:
            card_id = await self._reply_card(inbound.message_id, text)
            if card_id:
                self._running_cards[inbound.message_id] = card_id
        else:
            await self._create_card(inbound.chat_id, text)
        if final:
            self._running_cards.pop(inbound.message_id, None)
            self._last_card_text.pop(inbound.message_id, None)

    async def _reply_card(self, message_id: str, text: str) -> str | None:
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

    async def _create_card(self, chat_id: str, text: str) -> None:
        if not self._api_client:
            return
        request = self._CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            self._CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(build_card_content(text))
            .build()
        ).build()
        await asyncio.to_thread(self._api_client.im.v1.message.create, request)

    async def _update_card(self, message_id: str, text: str) -> None:
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
    sender = event.event.sender.sender_id
    content = json.loads(message.content)
    text, files = parse_feishu_content(content)
    mentions = _coerce_mentions(getattr(message, "mentions", []) or content.get("mentions", []))
    return FeishuInboundMessage(
        chat_id=str(message.chat_id),
        message_id=str(message.message_id),
        root_id=getattr(message, "root_id", None) or None,
        sender_open_id=str(sender.open_id),
        text=text.strip(),
        chat_type=str(getattr(message, "chat_type", "") or ""),
        mentions=mentions,
        files=files,
    )


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
    return bool(inbound.mentions) or bool(re.search(r"(^|\s)@[^\s]+", inbound.text))


def clean_mention_text(text: str, mentions: list[dict[str, Any]]) -> str:
    cleaned = text
    for mention in mentions:
        name = str(mention.get("name") or mention.get("text") or "").strip()
        if name:
            cleaned = cleaned.replace(name, " ")
    cleaned = re.sub(r"(^|\s)@[^\s]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_card_content(text: str) -> str:
    return json.dumps(
        {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [{"tag": "markdown", "content": text}],
        },
        ensure_ascii=False,
    )


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(FeishuChannelService().run_forever())
