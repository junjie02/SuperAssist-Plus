from __future__ import annotations

import asyncio
import io
import json
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from superassist.channels.feishu import (
    FeishuCardView,
    FeishuChannel,
    FeishuInboundMessage,
    ImageDownloadError,
    build_responses_image_content,
    build_card_content,
    clean_mention_text,
    attributed_feishu_text,
    feishu_memory_scope,
    format_subagent_card_text,
    format_model_error,
    image_mime_type,
    load_feishu_settings,
    normalize_image_payload,
    parse_feishu_content,
    parse_feishu_event,
    should_trigger_agent,
)
from superassist.channels.store import FeishuThreadStore
from superassist.config import Settings, get_settings
from superassist.models import AgentRunEvent, AgentRunResult


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _settings(tmp_path, **overrides):
    return Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_FEISHU_APP_ID="",
        SUPERASSIST_FEISHU_APP_SECRET="",
        **overrides,
    )


def test_feishu_settings_env_file_overrides_stale_process_environment(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "SUPERASSIST_MODEL=gpt-test\nSUPERASSIST_BASE_URL=https://gateway.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPERASSIST_MODEL", "deepseek-stale")
    monkeypatch.setenv("SUPERASSIST_BASE_URL", "https://api.deepseek.com/v1")

    settings = load_feishu_settings(env_path)

    assert settings.model == "gpt-test"
    assert settings.base_url == "https://gateway.example/v1"
    get_settings.cache_clear()


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


def test_image_mime_type_uses_content_signature() -> None:
    assert image_mime_type(b"\x89PNG\r\n\x1a\nrest", "wrong.jpg") == "image/png"
    assert image_mime_type(b"\xff\xd8\xffrest") == "image/jpeg"


def test_normalize_image_payload_reencodes_valid_image() -> None:
    source = io.BytesIO()
    Image.new("RGBA", (32, 24), (255, 0, 0, 128)).save(source, format="PNG")

    normalized, mime_type = normalize_image_payload(source.getvalue())

    assert mime_type == "image/jpeg"
    assert normalized.startswith(b"\xff\xd8\xff")
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.size == (32, 24)


def test_normalize_image_payload_rejects_non_image() -> None:
    with pytest.raises(ImageDownloadError, match="supported image"):
        normalize_image_payload(b"not an image")


def test_responses_image_content_uses_input_image_parts() -> None:
    content = build_responses_image_content("Describe it", [(b"image", "image/jpeg")])

    assert content == [
        {"type": "input_text", "text": "Describe it"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,aW1hZ2U="},
    ]


def test_describe_images_uses_responses_api(monkeypatch, tmp_path) -> None:
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="A test image.")

    class Client:
        def __init__(self, **_kwargs):
            self.responses = Responses()

    monkeypatch.setattr("superassist.channels.feishu.OpenAI", Client)
    settings = _settings(tmp_path).model_copy(
        update={"api_key": "secret", "model": "vision-model"}
    )
    channel = FeishuChannel(settings)

    description = _run(channel._describe_images("What is shown?", [(b"image", "image/jpeg")]))

    assert description == "A test image."
    assert captured["model"] == "vision-model"
    assert captured["input"][0]["content"][0]["type"] == "input_text"
    assert captured["input"][0]["content"][1]["type"] == "input_image"


def test_format_model_error_includes_bounded_provider_detail() -> None:
    text = format_model_error(
        {"model_error": "BadRequestError", "model_error_message": "unknown image_url " + ("x" * 1000)},
        has_images=True,
    )

    assert text.startswith("模型处理图片失败（`BadRequestError`）")
    assert "unknown image_url" in text
    assert len(text) < 700


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


def test_group_activation_session_resets_on_each_message_and_expires(tmp_path) -> None:
    now = [1000.0]
    channel = FeishuChannel(
        _settings(tmp_path, SUPERASSIST_FEISHU_ACTIVE_SESSION_SECONDS=180),
        monotonic_clock=lambda: now[0],
    )
    inactive = FeishuInboundMessage("chat_1", "msg_1", "ou_1", "hello", chat_type="group")
    mentioned = FeishuInboundMessage(
        "chat_1",
        "msg_2",
        "ou_1",
        "@bot hello",
        chat_type="group",
        mentions=[{"name": "@bot"}],
    )

    assert channel._should_accept_message(inactive) is False
    assert channel._should_accept_message(mentioned) is True

    now[0] += 179
    assert channel._should_accept_message(inactive) is True
    now[0] += 179
    assert channel._should_accept_message(inactive) is True
    now[0] += 180
    assert channel._should_accept_message(inactive) is False
    assert channel._should_accept_message(mentioned) is True


def test_group_activation_sessions_are_independent_and_private_is_always_accepted(tmp_path) -> None:
    now = [1000.0]
    channel = FeishuChannel(
        _settings(tmp_path, SUPERASSIST_FEISHU_ACTIVE_SESSION_SECONDS=180),
        monotonic_clock=lambda: now[0],
    )
    mentioned = FeishuInboundMessage(
        "chat_1",
        "msg_1",
        "ou_1",
        "@bot hello",
        chat_type="group",
        mentions=[{"name": "@bot"}],
    )
    other_group = FeishuInboundMessage("chat_2", "msg_2", "ou_2", "hello", chat_type="group")
    private = FeishuInboundMessage("p2p_1", "msg_3", "ou_1", "hello", chat_type="p2p")

    assert channel._should_accept_message(mentioned) is True
    assert channel._should_accept_message(other_group) is False
    now[0] += 181
    assert channel._should_accept_message(private) is True


def test_mention_only_disabled_accepts_group_messages_without_session(tmp_path) -> None:
    channel = FeishuChannel(
        _settings(tmp_path, SUPERASSIST_FEISHU_MENTION_ONLY=False),
        monotonic_clock=lambda: 1000.0,
    )
    inbound = FeishuInboundMessage("chat", "msg", "ou", "hello", chat_type="group")

    assert channel._should_accept_message(inbound) is True


def test_clean_mention_text_removes_bot_mentions() -> None:
    assert clean_mention_text("@bot 请工作", [{"name": "@bot"}]) == "请工作"


def test_group_memory_scope_is_shared_and_attributed() -> None:
    first = FeishuInboundMessage("chat_1", "msg_1", "ou_1", "hello", sender_name="Alice", chat_type="group")
    second = FeishuInboundMessage("chat_1", "msg_2", "ou_2", "hello", chat_type="group")
    private = FeishuInboundMessage("p2p_1", "msg_3", "ou_1", "hello", chat_type="p2p")

    assert feishu_memory_scope(first) == ("feishu-group:chat_1", "__group__")
    assert feishu_memory_scope(second) == ("feishu-group:chat_1", "__group__")
    assert attributed_feishu_text(first, "hello") == "[飞书群成员: ou_1] hello"
    assert attributed_feishu_text(second, "hello") == "[飞书群成员: ou_2] hello"
    assert feishu_memory_scope(private) == ("feishu:ou_1", "__private__")
    assert attributed_feishu_text(private, "hello") == "hello"


def test_thread_store_reuses_chat_topic(tmp_path) -> None:
    store = FeishuThreadStore(tmp_path / "feishu_threads.json")

    first = store.get_or_create_thread_id(chat_id="chat", topic_id="topic", user_id="feishu:ou")
    second = store.get_or_create_thread_id(chat_id="chat", topic_id="topic", user_id="feishu:ou")
    other = store.get_or_create_thread_id(chat_id="chat", topic_id="other", user_id="feishu:ou")

    assert first == second
    assert first != other
    assert store.list_entries()[0]["user_id"] == "feishu:ou"


def test_thread_store_rotates_when_scope_owner_changes(tmp_path) -> None:
    store = FeishuThreadStore(tmp_path / "feishu_threads.json")

    personal = store.get_or_create_thread_id(chat_id="chat", topic_id="scope", user_id="feishu:ou")
    group = store.get_or_create_thread_id(chat_id="chat", topic_id="scope", user_id="feishu-group:chat")

    assert personal != group
    assert store.list_entries()[0]["user_id"] == "feishu-group:chat"


def test_build_card_content_uses_update_multi() -> None:
    card = json.loads(build_card_content("hello"))

    assert card["config"]["update_multi"] is True
    assert card["elements"][0]["content"] == "hello"


def test_build_reasoning_card_expands_then_collapses() -> None:
    thinking = json.loads(
        build_card_content(FeishuCardView(reasoning="正在分析", reasoning_expanded=True))
    )
    answered = json.loads(
        build_card_content(
            FeishuCardView(answer="最终答案", reasoning="分析完成", reasoning_expanded=False)
        )
    )

    assert thinking["schema"] == "2.0"
    assert thinking["body"]["elements"][0]["tag"] == "collapsible_panel"
    assert thinking["body"]["elements"][0]["expanded"] is True
    assert answered["body"]["elements"][0]["expanded"] is False
    assert answered["body"]["elements"][1]["content"] == "最终答案"


def test_format_subagent_card_text_uses_description() -> None:
    event = AgentRunEvent(
        type="subagent_text",
        message="checking files",
        metadata={"description": "repo scan", "subagent_type": "general-purpose"},
    )

    assert format_subagent_card_text(event) == "Subagent [repo scan]: checking files"


def test_feishu_channel_requires_credentials(tmp_path) -> None:
    channel = FeishuChannel(_settings(tmp_path))

    with pytest.raises(RuntimeError, match="SUPERASSIST_FEISHU_APP_ID"):
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

def test_feishu_image_is_described_before_reaching_runtime(tmp_path) -> None:
    received = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id, message_content=None):
            received.append((message, message_content))
            return AgentRunResult(thread_id=thread_id, answer="done", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))

        async def download_payloads(images, message_id):
            assert images == [{"image_key": "img_1"}]
            assert message_id == "msg"
            return [(b"image", "image/png")]

        async def reply(_message_id, _text):
            return "card_1"

        async def update(_message_id, _text):
            return None

        channel._download_image_payloads = download_payloads

        async def describe_images(_text, _payloads):
            return "A small test image."

        channel._describe_images = describe_images
        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(
            FeishuInboundMessage(
                "chat",
                "msg",
                "ou",
                "[image]",
                chat_type="p2p",
                files=[{"image_key": "img_1"}],
            )
        )

        assert len(received) == 1
        message, content = received[0]
        assert message == "The user sent an image."
        assert content == (
            "The user sent an image.\n\n"
            "[Vision extraction for the current Feishu image]\n"
            "A small test image.\n"
            "[/Vision extraction]"
        )

    _run(go())


def test_download_image_payloads_uses_message_id_and_image_key(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))

        async def download(message_id, image_key):
            assert message_id == "msg_1"
            assert image_key == "img_1"
            return b"image", "image/png"

        channel._download_image = download
        result = await channel._download_image_payloads(
            [{"image_key": "img_1"}],
            "msg_1",
        )

        assert result == [(b"image", "image/png")]

    _run(go())


def test_feishu_channel_shows_agent_text_and_final_card(tmp_path) -> None:
    class Runtime:
        def __init__(self, reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            self.reporter = reporter

        def run_streaming(self, message, *, user_id, thread_id):
            assert message == "[飞书群成员: ou_1] 做个计划"
            assert user_id == "feishu-group:chat_1"
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
            SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS="ou_1",
        )
        channel = FeishuChannel(settings, runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[tuple[str, str]] = []

        async def create(chat_id, text):
            assert chat_id == "chat_1"
            sent.append(("create", text))
            return "card_1"

        async def update(message_id, text):
            sent.append(("patch", text))

        channel._create_card = create
        channel._update_card = update
        inbound = FeishuInboundMessage(
            chat_id="chat_1",
            message_id="msg_1",
            sender_open_id="ou_1",
            sender_name="小馨",
            text="@bot 做个计划",
            chat_type="group",
            mentions=[{"name": "@bot"}],
        )

        await channel.handle_inbound(inbound)
        await asyncio.sleep(0.05)

        assert len([entry for entry in sent if entry[0] == "create"]) == 1
        assert sent[0] == ("create", "Preparing context...")
        assert ("patch", "agent text") in sent
        assert sent[-1] == ("patch", "完成")

    _run(go())


def test_feishu_reasoning_is_expanded_until_answer_starts(tmp_path) -> None:
    class Runtime:
        def __init__(self, reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            self.reporter = reporter

        def run_streaming(self, message, *, user_id, thread_id):
            self.reporter(AgentRunEvent(type="agent_reasoning", message="第一步", metadata={}))
            time.sleep(0.05)
            self.reporter(AgentRunEvent(type="agent_reasoning", message="第一步\n第二步", metadata={}))
            time.sleep(0.05)
            self.reporter(AgentRunEvent(type="agent_text", message="正文开始", metadata={}))
            return AgentRunResult(thread_id=thread_id, answer="完整正文", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[str | FeishuCardView] = []

        async def reply(_message_id, text):
            sent.append(text)
            return "card_1"

        async def update(_message_id, text):
            sent.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(
            FeishuInboundMessage("chat", "msg", "ou", "hello", chat_type="p2p")
        )
        await asyncio.sleep(0.1)

        reasoning_views = [item for item in sent if isinstance(item, FeishuCardView)]
        assert any(item.reasoning_expanded and not item.answer for item in reasoning_views)
        assert reasoning_views[-1] == FeishuCardView(
            answer="完整正文", reasoning="第一步\n第二步", reasoning_expanded=False
        )

    _run(go())


def test_feishu_final_card_cannot_be_overwritten_by_slow_partial_update(tmp_path) -> None:
    class Runtime:
        def __init__(self, reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)
            self.reporter = reporter

        def run_streaming(self, message, *, user_id, thread_id):
            self.reporter(AgentRunEvent(type="agent_text", message="partial", metadata={}))
            self.reporter(AgentRunEvent(type="agent_text", message="partial answer", metadata={}))
            return AgentRunResult(thread_id=thread_id, answer="complete final answer", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        channel._main_loop = asyncio.get_running_loop()
        sent: list[str] = []

        async def reply(_message_id, text):
            sent.append(text)
            return "card_1"

        async def update(_message_id, text):
            if text == "partial":
                await asyncio.sleep(0.05)
            sent.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg", "ou", "hello", chat_type="p2p"))
        await asyncio.sleep(0.1)

        assert sent[-1] == "complete final answer"

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
        settings = _settings(tmp_path, SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS="ou_allowed")
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
