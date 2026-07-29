from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from superassist.config import Settings, get_settings

from .ai_engine_client import AIEngineClient
from .store import WeComThreadStore

logger = logging.getLogger(__name__)

UNSUPPORTED_MESSAGE = "当前企业微信入口支持文本、图文中的文本和语音转写；文件与图片请在网页 Knowledge 页面上传。"
ENGINE_ERROR_MESSAGE = "AI Engine 当前不可用。请确认 superassist-ai-engine 已在配置的地址启动，然后重试。"


@dataclass(frozen=True)
class WeComInboundMessage:
    frame: dict[str, Any]
    message_id: str
    sender_user_id: str
    chat_id: str
    chat_type: str
    message_type: str
    text: str

    @property
    def is_group(self) -> bool:
        return self.chat_type.lower() in {"group", "groupchat", "group_chat"}

    @property
    def memory_scope_id(self) -> str:
        return "__group__" if self.is_group else self.sender_user_id


class RecentMessageCache:
    """Bounded in-memory retry filter for duplicate WebSocket callbacks."""

    def __init__(self, *, ttl_seconds: float = 600, max_entries: int = 5000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._seen: OrderedDict[str, float] = OrderedDict()

    def add(self, message_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.ttl_seconds
        while self._seen and next(iter(self._seen.values())) < cutoff:
            self._seen.popitem(last=False)
        if message_id in self._seen:
            return False
        self._seen[message_id] = now
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return True


class WeComChannel:
    """Enterprise WeCom intelligent-robot channel backed by the local AI Engine."""

    def __init__(
        self,
        settings: Settings,
        *,
        engine_client: AIEngineClient | None = None,
        store: WeComThreadStore | None = None,
        sdk_client: Any = None,
    ) -> None:
        self.settings = settings
        self.engine_client = engine_client or AIEngineClient(settings.wecom_ai_engine_url)
        self.store = store or WeComThreadStore(settings.wecom_thread_store_path)
        self.allowed_user_ids = settings.wecom_allowed_user_id_set
        self.user_id_mapping = settings.wecom_user_id_mapping
        self._sdk_client = sdk_client
        self._semaphore = asyncio.Semaphore(settings.wecom_max_concurrent)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._dedupe = RecentMessageCache()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        if not self.settings.wecom_bot_id or not self.settings.wecom_bot_secret:
            raise RuntimeError(
                "WeCom channel requires SUPERASSIST_WECOM_BOT_ID and "
                "SUPERASSIST_WECOM_BOT_SECRET."
            )
        if self._sdk_client is None:
            try:
                from aibot import WSClient, WSClientOptions
            except ImportError as exc:
                raise RuntimeError(
                    "wecom-aibot-python-sdk is not installed. Install project dependencies first."
                ) from exc
            self._sdk_client = WSClient(
                WSClientOptions(
                    bot_id=self.settings.wecom_bot_id,
                    secret=self.settings.wecom_bot_secret,
                    max_reconnect_attempts=-1,
                )
            )
        self._register_handlers()
        await self.engine_client.start()
        try:
            await self._sdk_client.connect()
        except Exception:
            await self.engine_client.close()
            raise
        self._running = True
        logger.info("WeCom channel connected; AI Engine=%s", self.settings.wecom_ai_engine_url)

    async def stop(self) -> None:
        self._running = False
        if self._sdk_client is not None:
            self._sdk_client.disconnect()
        await self.engine_client.close()

    def _register_handlers(self) -> None:
        self._sdk_client.on("authenticated", lambda: logger.info("WeCom bot authenticated"))
        self._sdk_client.on("reconnecting", lambda attempt: logger.warning("WeCom reconnect attempt %s", attempt))
        self._sdk_client.on("error", lambda error: logger.error("WeCom SDK error: %s", error))
        for event_name in ("message.text", "message.voice", "message.mixed"):
            self._sdk_client.on(event_name, self.handle_frame)
        for event_name in ("message.image", "message.file"):
            self._sdk_client.on(event_name, self.handle_unsupported_frame)
        self._sdk_client.on("event.enter_chat", self.handle_enter_chat)

    async def handle_frame(self, frame: dict[str, Any]) -> None:
        inbound = parse_wecom_frame(frame)
        if not inbound.sender_user_id:
            logger.warning("Ignoring WeCom message without sender userid")
            return
        if self.allowed_user_ids and inbound.sender_user_id not in self.allowed_user_ids:
            await self._reply_once(inbound.frame, "当前企业微信账号未获授权使用此助手。")
            return
        if not inbound.text:
            await self._reply_once(inbound.frame, UNSUPPORTED_MESSAGE)
            return
        if not self._dedupe.add(inbound.message_id):
            logger.info("Ignoring duplicate WeCom message id=%s", inbound.message_id)
            return

        mapping_key = f"chat:{inbound.chat_id}" if inbound.is_group else inbound.sender_user_id
        default_identity = (
            f"wecom-group:{self.settings.wecom_bot_id}:{inbound.chat_id}"
            if inbound.is_group
            else f"wecom:{self.settings.wecom_bot_id}:{inbound.sender_user_id}"
        )
        identity = self.user_id_mapping.get(mapping_key, default_identity)
        thread_id, rag_mode = self.store.resolve(
            chat_id=inbound.chat_id,
            sender_user_id=inbound.sender_user_id,
            user_id=identity,
            rag_mode_default=self.settings.wecom_rag_mode_default,
            scope_id=inbound.memory_scope_id,
        )
        rag_command = parse_rag_command(inbound.text)
        if rag_command is not None:
            if isinstance(rag_command, bool):
                self.store.set_rag_mode(
                    chat_id=inbound.chat_id,
                    sender_user_id=inbound.sender_user_id,
                    enabled=rag_command,
                    scope_id=inbound.memory_scope_id,
                )
                state = "已开启" if rag_command else "已关闭"
                await self._reply_once(inbound.frame, f"知识库 RAG 模式{state}。")
            else:
                state = "开启" if rag_mode else "关闭"
                await self._reply_once(inbound.frame, f"当前知识库 RAG 模式：{state}。")
            return

        stream_id = f"superassist_{uuid4().hex}"
        await self._sdk_client.reply_stream(inbound.frame, stream_id, "正在准备上下文...", False)
        lock_key = f"{inbound.chat_id}:{inbound.memory_scope_id}"
        lock = self._conversation_locks.setdefault(lock_key, asyncio.Lock())
        async with self._semaphore:
            async with lock:
                await self._run_agent(inbound, identity, thread_id, rag_mode, stream_id)

    async def _run_agent(
        self,
        inbound: WeComInboundMessage,
        user_id: str,
        thread_id: str,
        rag_mode: bool,
        stream_id: str,
    ) -> None:
        visible_text = "正在思考..."
        last_sent = "正在准备上下文..."
        last_sent_at = time.monotonic()
        interval = self.settings.wecom_stream_interval_ms / 1000
        final_text = ""
        try:
            async for event in self.engine_client.stream_chat(
                user_id=user_id,
                message=inbound.text,
                thread_id=thread_id,
                rag_mode=rag_mode,
            ):
                event_type = str(event.get("type") or "")
                if event_type == "agent_text":
                    visible_text = str(event.get("content") or "").strip() or visible_text
                elif event_type == "subagent_text":
                    progress = str(event.get("content") or "").strip()
                    if progress and not final_text:
                        visible_text = f"子任务进展：{progress}"
                elif event_type == "thinking" and last_sent == "正在准备上下文...":
                    visible_text = "正在思考..."
                elif event_type == "done":
                    final_text = str(event.get("answer") or "").strip() or visible_text
                    break
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message") or "AI Engine chat failed"))
                else:
                    continue

                now = time.monotonic()
                if visible_text != last_sent and now - last_sent_at >= interval:
                    await self._sdk_client.reply_stream(inbound.frame, stream_id, visible_text, False)
                    last_sent = visible_text
                    last_sent_at = now

            final_text = final_text or visible_text or "本次请求没有生成有效回答，请重试。"
            await self._sdk_client.reply_stream(inbound.frame, stream_id, final_text, True)
        except asyncio.CancelledError:
            await self._sdk_client.reply_stream(inbound.frame, stream_id, "请求已取消，请重新发送。", True)
            raise
        except Exception:
            logger.exception("WeCom agent request failed for user=%s", inbound.sender_user_id)
            await self._sdk_client.reply_stream(inbound.frame, stream_id, ENGINE_ERROR_MESSAGE, True)

    async def handle_unsupported_frame(self, frame: dict[str, Any]) -> None:
        await self._reply_once(frame, UNSUPPORTED_MESSAGE)

    async def handle_enter_chat(self, frame: dict[str, Any]) -> None:
        await self._sdk_client.reply_welcome(
            frame,
            {
                "msgtype": "text",
                "text": {
                    "content": "你好，我是 SuperAssist。直接发送问题即可；使用 /rag on、/rag off 切换知识库检索。"
                },
            },
        )

    async def _reply_once(self, frame: dict[str, Any], text: str) -> None:
        await self._sdk_client.reply_stream(frame, f"superassist_{uuid4().hex}", text, True)


class WeComChannelService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.channel = WeComChannel(settings or get_settings())

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


def parse_wecom_frame(frame: dict[str, Any]) -> WeComInboundMessage:
    body = frame.get("body") if isinstance(frame.get("body"), dict) else {}
    sender = body.get("from") if isinstance(body.get("from"), dict) else {}
    sender_user_id = str(
        sender.get("userid")
        or sender.get("user_id")
        or body.get("from_userid")
        or body.get("userid")
        or ""
    )
    headers = frame.get("headers") if isinstance(frame.get("headers"), dict) else {}
    message_id = str(body.get("msgid") or body.get("msg_id") or headers.get("req_id") or uuid4().hex)
    chat_id = str(body.get("chatid") or body.get("chat_id") or sender_user_id)
    message_type = str(body.get("msgtype") or "")
    text = extract_wecom_text(body, message_type)
    return WeComInboundMessage(
        frame=frame,
        message_id=message_id,
        sender_user_id=sender_user_id,
        chat_id=chat_id,
        chat_type=str(body.get("chattype") or body.get("chat_type") or ""),
        message_type=message_type,
        text=text.strip(),
    )


def extract_wecom_text(body: dict[str, Any], message_type: str) -> str:
    if message_type == "text":
        text = body.get("text") if isinstance(body.get("text"), dict) else {}
        return str(text.get("content") or "")
    if message_type == "voice":
        voice = body.get("voice") if isinstance(body.get("voice"), dict) else {}
        return str(voice.get("content") or "")
    if message_type == "mixed":
        mixed = body.get("mixed") if isinstance(body.get("mixed"), dict) else {}
        items = mixed.get("msg_item") if isinstance(mixed.get("msg_item"), list) else []
        parts = []
        for item in items:
            if not isinstance(item, dict) or item.get("msgtype") != "text":
                continue
            text = item.get("text") if isinstance(item.get("text"), dict) else {}
            content = str(text.get("content") or "").strip()
            if content:
                parts.append(content)
        return "\n".join(parts)
    return ""


def parse_rag_command(text: str) -> bool | str | None:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"/rag on", "rag on", "开启rag", "打开rag", "开启 rag", "打开 rag"}:
        return True
    if normalized in {"/rag off", "rag off", "关闭rag", "关闭 rag"}:
        return False
    if normalized in {"/rag", "/rag status", "rag status", "rag状态", "rag 状态"}:
        return "status"
    return None


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(WeComChannelService().run_forever())
