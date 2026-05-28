from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from superassist_plus.channels.feishu import (
    FeishuChannel,
    FeishuInboundMessage,
    build_card_content,
    clean_mention_text,
    format_subagent_card_text,
    parse_feishu_content,
    parse_feishu_event,
    should_trigger_agent,
)
from superassist_plus.channels.store import FeishuThreadStore
from superassist_plus.config import Settings
from superassist_plus.models import AgentRunEvent, AgentRunResult


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _settings(tmp_path, **overrides):
    return Settings(
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_API_KEY="",
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_PLUS_FEISHU_APP_ID="",
        SUPERASSIST_PLUS_FEISHU_APP_SECRET="",
        **overrides,
    )


def test_parse_plain_text_content() -> None:
    text, files = parse_feishu_content({"text": "hello"})

    assert text == "hello"
    assert files == []


def test_parse_rich_text_and_files() -> None:
    text, files = parse_feishu_content(
        {
            "content": [
                [
                    {"tag": "text", "text": "See"},
                    {"tag": "at", "text": "@bot"},
                    {"tag": "img", "image_key": "img_1"},
                    {"tag": "file", "file_key": "file_1"},
                ],
                [{"tag": "text", "text": "second"}],
            ]
        }
    )

    assert "See @bot [image] [file]" in text
    assert "\n\nsecond" in text
    assert files == [{"image_key": "img_1"}, {"file_key": "file_1"}]


def test_parse_feishu_event_extracts_thread_fields() -> None:
    event = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat_1",
                message_id="msg_1",
                root_id="root_1",
                chat_type="group",
                content=json.dumps({"text": "@bot do it"}),
                mentions=[{"name": "@bot"}],
            ),
            sender=SimpleNamespace(sender_id=SimpleNamespace(open_id="ou_1")),
        )
    )

    inbound = parse_feishu_event(event)

    assert inbound.chat_id == "chat_1"
    assert inbound.topic_id == "root_1"
    assert inbound.sender_open_id == "ou_1"
    assert inbound.mentions == [{"name": "@bot"}]


def test_should_trigger_private_or_mentions() -> None:
    private = FeishuInboundMessage("chat", "msg", "ou", "hello", chat_type="p2p")
    group_without_mention = FeishuInboundMessage("chat", "msg", "ou", "hello", chat_type="group")
    group_with_mention = FeishuInboundMessage(
        "chat",
        "msg",
        "ou",
        "@bot hello",
        chat_type="group",
        mentions=[{"name": "@bot"}],
    )

    assert should_trigger_agent(private, mention_only=True) is True
    assert should_trigger_agent(group_without_mention, mention_only=True) is False
    assert should_trigger_agent(group_with_mention, mention_only=True) is True


def test_clean_mention_text_removes_bot_mentions() -> None:
    assert clean_mention_text("@bot 请工作", [{"name": "@bot"}]) == "请工作"


def test_thread_store_reuses_chat_topic(tmp_path) -> None:
    store = FeishuThreadStore(tmp_path / "feishu_threads.json")

    first = store.get_or_create_thread_id(chat_id="chat", topic_id="topic", user_id="feishu:ou")
    second = store.get_or_create_thread_id(chat_id="chat", topic_id="topic", user_id="feishu:ou")
    other = store.get_or_create_thread_id(chat_id="chat", topic_id="other", user_id="feishu:ou")

    assert first == second
    assert first != other
    assert store.list_entries()[0]["user_id"] == "feishu:ou"


def test_build_card_content_uses_update_multi() -> None:
    card = json.loads(build_card_content("hello"))

    assert card["config"]["update_multi"] is True
    assert card["elements"][0]["content"] == "hello"


def test_format_subagent_card_text_uses_description() -> None:
    event = AgentRunEvent(
        type="subagent_text",
        message="checking files",
        metadata={"description": "repo scan", "subagent_type": "general-purpose"},
    )

    assert format_subagent_card_text(event) == "Subagent [repo scan]: checking files"


def test_feishu_channel_requires_credentials(tmp_path) -> None:
    channel = FeishuChannel(_settings(tmp_path))

    with pytest.raises(RuntimeError, match="SUPERASSIST_PLUS_FEISHU_APP_ID"):
        _run(channel.start())


def test_feishu_channel_uses_one_runtime_per_message(tmp_path) -> None:
    created = []

    class Runtime:
        def __init__(self, reporter):
            self.reporter = reporter
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            created.append(self)

        def run_streaming(self, message, *, user_id, thread_id):
            self.reporter(AgentRunEvent(type="agent_text", message=f"text {len(created)}", metadata={}))
            return AgentRunResult(thread_id=thread_id, answer="done", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[tuple[str, str]] = []

        async def reply(message_id, text):
            sent.append(("reply", text))
            return f"card_{len(sent)}"

        async def update(message_id, text):
            sent.append(("patch", text))

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg_1", "ou", "hello", chat_type="p2p"))
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg_2", "ou", "hello", chat_type="p2p"))
        await asyncio.sleep(0.05)

        assert len(created) == 2
        assert ("patch", "text 1") in sent
        assert ("patch", "text 2") in sent

    _run(go())


def test_feishu_channel_shows_agent_text_and_final_card(tmp_path) -> None:
    class Runtime:
        def __init__(self, reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            self.reporter = reporter

        def run_streaming(self, message, *, user_id, thread_id):
            assert message == "做个计划"
            assert user_id == "feishu:ou_1"
            assert thread_id.startswith("feishu_")
            self.reporter(AgentRunEvent(type="thinking", message="Inspecting the request...", metadata={}))
            self.reporter(
                AgentRunEvent(
                    type="subagent_text",
                    message="subagent progress",
                    metadata={"description": "plan check", "subagent_type": "general-purpose"},
                )
            )
            self.reporter(AgentRunEvent(type="agent_text", message="agent text", metadata={}))
            return AgentRunResult(thread_id=thread_id, answer="完成", metadata={})

    async def go():
        settings = _settings(
            tmp_path,
            SUPERASSIST_PLUS_FEISHU_ALLOWED_OPEN_IDS="ou_1",
        )
        channel = FeishuChannel(settings, runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[tuple[str, str]] = []

        async def reply(message_id, text):
            sent.append(("reply", text))
            return "card_1"

        async def update(message_id, text):
            sent.append(("patch", text))

        channel._reply_card = reply
        channel._update_card = update
        inbound = FeishuInboundMessage(
            chat_id="chat_1",
            message_id="msg_1",
            sender_open_id="ou_1",
            text="@bot 做个计划",
            chat_type="group",
            mentions=[{"name": "@bot"}],
        )

        await channel.handle_inbound(inbound)
        await asyncio.sleep(0.05)

        assert len([entry for entry in sent if entry[0] == "reply"]) == 1
        assert sent[0] == ("reply", "Preparing context...")
        assert ("patch", "Inspecting the request...") in sent
        assert ("patch", "Subagent [plan check]: subagent progress") in sent
        assert ("patch", "agent text") in sent
        assert sent[-1] == ("patch", "完成")

    _run(go())


def test_feishu_channel_does_not_patch_blank_text_or_blank_final(tmp_path) -> None:
    class Runtime:
        def __init__(self, reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            self.reporter = reporter

        def run_streaming(self, message, *, user_id, thread_id):
            self.reporter(AgentRunEvent(type="thinking", message="Thinking...", metadata={}))
            self.reporter(AgentRunEvent(type="agent_text", message="", metadata={}))
            self.reporter(AgentRunEvent(type="agent_text", message="visible progress", metadata={}))
            return AgentRunResult(thread_id=thread_id, answer="", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[tuple[str, str]] = []

        async def reply(message_id, text):
            sent.append(("reply", text))
            return "card_1"

        async def update(message_id, text):
            sent.append(("patch", text))

        channel._reply_card = reply
        channel._update_card = update
        inbound = FeishuInboundMessage("chat", "msg_1", "ou_1", "hello", chat_type="p2p")

        await channel.handle_inbound(inbound)
        await asyncio.sleep(0.05)

        assert ("patch", "") not in sent
        assert sent[-1] == ("patch", "visible progress")

    _run(go())


def test_feishu_channel_ignores_non_allowed_users(tmp_path) -> None:
    async def go():
        settings = _settings(tmp_path, SUPERASSIST_PLUS_FEISHU_ALLOWED_OPEN_IDS="ou_allowed")
        channel = FeishuChannel(settings, runtime_factory=lambda reporter: None)
        channel._reply_card = pytest.fail
        inbound = FeishuInboundMessage(
            chat_id="chat_1",
            message_id="msg_1",
            sender_open_id="ou_other",
            text="hello",
            chat_type="p2p",
        )

        await channel.handle_inbound(inbound)

    _run(go())
