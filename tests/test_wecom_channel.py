import asyncio
from typing import Any

import pytest

from superassist.channels.ai_engine_client import AIEngineError, iter_sse_events
from superassist.channels.store import WeComThreadStore
from superassist.channels.wecom import (
    ENGINE_ERROR_MESSAGE,
    WeComChannel,
    extract_wecom_text,
    parse_rag_command,
    parse_wecom_frame,
)
from superassist.config import Settings


def _run(coro):
    return asyncio.run(coro)


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "SUPERASSIST_DATA_DIR": tmp_path,
        "SUPERASSIST_EMBEDDING_PROVIDER": "hash",
        "SUPERASSIST_WECOM_BOT_ID": "bot_1",
        "SUPERASSIST_WECOM_BOT_SECRET": "secret_1",
        "SUPERASSIST_WECOM_STREAM_INTERVAL_MS": 100,
    }
    values.update(overrides)
    return Settings(**values)


def _frame(
    text: str = "你好",
    *,
    message_id: str = "msg_1",
    user_id: str = "zhangsan",
    chat_id: str = "chat_1",
    chat_type: str = "single",
) -> dict[str, Any]:
    return {
        "headers": {"req_id": f"req_{message_id}"},
        "body": {
            "msgid": message_id,
            "chatid": chat_id,
            "chattype": chat_type,
            "from": {"userid": user_id},
            "msgtype": "text",
            "text": {"content": text},
        },
    }


class FakeSDKClient:
    def __init__(self) -> None:
        self.handlers = {}
        self.replies = []
        self.welcome = []
        self.connected = False
        self.disconnected = False

    def on(self, name, handler):
        self.handlers[name] = handler

    async def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    async def reply_stream(self, frame, stream_id, content, finish=False, **_kwargs):
        self.replies.append({"frame": frame, "stream_id": stream_id, "content": content, "finish": finish})

    async def reply_welcome(self, frame, body):
        self.welcome.append((frame, body))


class FakeEngineClient:
    def __init__(self, events=None, error: Exception | None = None) -> None:
        self.events = events or []
        self.error = error
        self.calls = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        for event in self.events:
            yield event


def test_parse_wecom_text_voice_and_mixed_messages() -> None:
    inbound = parse_wecom_frame(_frame("介绍一下项目"))

    assert inbound.sender_user_id == "zhangsan"
    assert inbound.chat_id == "chat_1"
    assert inbound.message_id == "msg_1"
    assert inbound.text == "介绍一下项目"
    assert inbound.is_group is False
    assert extract_wecom_text({"voice": {"content": "语音转写"}}, "voice") == "语音转写"
    assert extract_wecom_text(
        {
            "mixed": {
                "msg_item": [
                    {"msgtype": "text", "text": {"content": "第一段"}},
                    {"msgtype": "image", "image": {"url": "ignored"}},
                    {"msgtype": "text", "text": {"content": "第二段"}},
                ]
            }
        },
        "mixed",
    ) == "第一段\n第二段"


@pytest.mark.parametrize(
    ("command", "expected"),
    [("/rag on", True), ("关闭 RAG", False), ("/rag status", "status"), ("rag是什么", None)],
)
def test_parse_rag_command(command, expected) -> None:
    assert parse_rag_command(command) == expected


def test_wecom_thread_store_isolates_senders_and_persists_rag_mode(tmp_path) -> None:
    path = tmp_path / "wecom_threads.json"
    store = WeComThreadStore(path)
    first, enabled = store.resolve(
        chat_id="group_1",
        sender_user_id="u1",
        user_id="wecom:bot:u1",
        rag_mode_default=False,
    )
    second, _ = store.resolve(
        chat_id="group_1",
        sender_user_id="u2",
        user_id="wecom:bot:u2",
        rag_mode_default=False,
    )
    store.set_rag_mode(chat_id="group_1", sender_user_id="u1", enabled=True)

    reloaded = WeComThreadStore(path)
    same, persisted = reloaded.resolve(
        chat_id="group_1",
        sender_user_id="u1",
        user_id="wecom:bot:u1",
        rag_mode_default=False,
    )

    assert first != second
    assert enabled is False
    assert same == first
    assert persisted is True


def test_wecom_thread_store_rotates_thread_when_identity_mapping_changes(tmp_path) -> None:
    store = WeComThreadStore(tmp_path / "wecom_threads.json")
    isolated, _ = store.resolve(
        chat_id="chat",
        sender_user_id="u1",
        user_id="wecom:bot:u1",
        rag_mode_default=False,
    )
    linked, _ = store.resolve(
        chat_id="chat",
        sender_user_id="u1",
        user_id="user_from_browser",
        rag_mode_default=False,
    )

    assert isolated != linked


def test_wecom_channel_streams_through_existing_ai_engine(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient(
        [
            {"type": "thinking", "content": "Thinking..."},
            {"type": "agent_text", "content": "处理中"},
            {"type": "done", "thread_id": "ignored", "answer": "最终回答"},
        ]
    )
    channel = WeComChannel(_settings(tmp_path), sdk_client=sdk, engine_client=engine)

    _run(channel.handle_frame(_frame("问题")))

    assert len(engine.calls) == 1
    assert engine.calls[0]["user_id"] == "wecom:bot_1:zhangsan"
    assert engine.calls[0]["thread_id"].startswith("wecom_")
    assert engine.calls[0]["rag_mode"] is False
    assert sdk.replies[0]["content"] == "正在准备上下文..."
    assert sdk.replies[-1]["content"] == "最终回答"
    assert sdk.replies[-1]["finish"] is True


def test_wecom_channel_rag_command_changes_next_request(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient([{"type": "done", "answer": "来自资料"}])
    channel = WeComChannel(_settings(tmp_path), sdk_client=sdk, engine_client=engine)

    async def scenario():
        await channel.handle_frame(_frame("/rag on", message_id="msg_1"))
        await channel.handle_frame(_frame("根据论文回答", message_id="msg_2"))

    _run(scenario())

    assert sdk.replies[0]["content"] == "知识库 RAG 模式已开启。"
    assert engine.calls[0]["rag_mode"] is True


def test_wecom_channel_can_share_browser_user_memory_and_rag(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient([{"type": "done", "answer": "共享空间回答"}])
    settings = _settings(
        tmp_path,
        SUPERASSIST_WECOM_USER_ID_MAP='{"zhangsan":"user_from_browser"}',
    )
    channel = WeComChannel(settings, sdk_client=sdk, engine_client=engine)

    _run(channel.handle_frame(_frame("查询我上传的资料")))

    assert engine.calls[0]["user_id"] == "user_from_browser"


def test_wecom_group_members_share_thread_memory_and_rag_state(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient([{"type": "done", "answer": "群回答"}])
    channel = WeComChannel(_settings(tmp_path), sdk_client=sdk, engine_client=engine)

    async def scenario():
        await channel.handle_frame(
            _frame("/rag on", message_id="group_1", user_id="u1", chat_id="group_chat", chat_type="group")
        )
        await channel.handle_frame(
            _frame("第一个问题", message_id="group_2", user_id="u1", chat_id="group_chat", chat_type="group")
        )
        await channel.handle_frame(
            _frame("接着上面的问题", message_id="group_3", user_id="u2", chat_id="group_chat", chat_type="group")
        )

    _run(scenario())

    assert len(engine.calls) == 2
    assert engine.calls[0]["user_id"] == "wecom-group:bot_1:group_chat"
    assert engine.calls[1]["user_id"] == "wecom-group:bot_1:group_chat"
    assert engine.calls[0]["thread_id"] == engine.calls[1]["thread_id"]
    assert engine.calls[0]["rag_mode"] is True
    assert engine.calls[1]["rag_mode"] is True


def test_wecom_group_can_map_to_shared_browser_knowledge_space(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient([{"type": "done", "answer": "群知识库回答"}])
    settings = _settings(
        tmp_path,
        SUPERASSIST_WECOM_USER_ID_MAP='{"chat:group_chat":"shared_browser_user"}',
    )
    channel = WeComChannel(settings, sdk_client=sdk, engine_client=engine)

    _run(
        channel.handle_frame(
            _frame("查询群知识库", user_id="u1", chat_id="group_chat", chat_type="group")
        )
    )

    assert engine.calls[0]["user_id"] == "shared_browser_user"


def test_wecom_channel_enforces_allowlist_and_deduplicates(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient([{"type": "done", "answer": "ok"}])
    settings = _settings(tmp_path, SUPERASSIST_WECOM_ALLOWED_USER_IDS="allowed")
    channel = WeComChannel(settings, sdk_client=sdk, engine_client=engine)

    async def scenario():
        await channel.handle_frame(_frame(user_id="blocked"))
        await channel.handle_frame(_frame(message_id="same", user_id="allowed"))
        await channel.handle_frame(_frame(message_id="same", user_id="allowed"))

    _run(scenario())

    assert sdk.replies[0]["content"] == "当前企业微信账号未获授权使用此助手。"
    assert len(engine.calls) == 1


def test_wecom_channel_returns_actionable_engine_error(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient(error=AIEngineError("connection refused"))
    channel = WeComChannel(_settings(tmp_path), sdk_client=sdk, engine_client=engine)

    _run(channel.handle_frame(_frame()))

    assert sdk.replies[-1]["content"] == ENGINE_ERROR_MESSAGE
    assert sdk.replies[-1]["finish"] is True


def test_wecom_channel_start_registers_sdk_handlers(tmp_path) -> None:
    sdk = FakeSDKClient()
    engine = FakeEngineClient()
    channel = WeComChannel(_settings(tmp_path), sdk_client=sdk, engine_client=engine)

    async def scenario():
        await channel.start()
        await channel.stop()

    _run(scenario())

    assert sdk.connected is True
    assert sdk.disconnected is True
    assert engine.started is True
    assert engine.closed is True
    assert "message.text" in sdk.handlers
    assert "event.enter_chat" in sdk.handlers


def test_wecom_channel_requires_credentials(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        SUPERASSIST_WECOM_BOT_ID="",
        SUPERASSIST_WECOM_BOT_SECRET="",
    )
    channel = WeComChannel(settings, sdk_client=FakeSDKClient(), engine_client=FakeEngineClient())

    with pytest.raises(RuntimeError, match="SUPERASSIST_WECOM_BOT_ID"):
        _run(channel.start())


def test_iter_sse_events_parses_engine_stream() -> None:
    async def lines():
        yield b"\n"
        yield b'data: {"type":"agent_text","content":"hello"}\n'
        yield b'data: {"type":"done","answer":"hello"}\n'

    async def collect():
        return [event async for event in iter_sse_events(lines())]

    events = _run(collect())

    assert [event["type"] for event in events] == ["agent_text", "done"]
