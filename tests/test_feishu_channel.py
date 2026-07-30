from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from superassist.channels.feishu import (
    FeishuCardView,
    FeishuChannel,
    FeishuInboundMessage,
    ImageDownloadError,
    attributed_feishu_text,
    build_card_content,
    build_image_memory_text,
    build_multimodal_image_content,
    build_multimodal_request_text,
    clean_mention_text,
    default_image_only_request,
    feishu_memory_scope,
    format_model_error,
    format_subagent_card_text,
    image_mime_type,
    load_feishu_settings,
    original_image_payload,
    parse_effort_command,
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


def test_original_image_payload_preserves_png_bytes_and_mime_type() -> None:
    source = io.BytesIO()
    Image.new("RGBA", (32, 24), (255, 0, 0, 128)).save(source, format="PNG")
    original = source.getvalue()

    payload, mime_type = original_image_payload(original)

    assert mime_type == "image/png"
    assert payload == original
    assert payload is original


def test_original_image_payload_rejects_non_image() -> None:
    with pytest.raises(ImageDownloadError, match="supported model image"):
        original_image_payload(b"not an image")


def test_multimodal_image_content_uses_langchain_image_and_text_parts() -> None:
    content = build_multimodal_image_content("Describe it", [(b"image", "image/jpeg")])

    assert content == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aW1hZ2U="},
        },
        {"type": "text", "text": "Describe it"},
    ]


def test_image_memory_text_keeps_ocr_before_current_user_request() -> None:
    text = build_image_memory_text("请讲解这道题", "[Image 1]\nx + 1 = 2")

    assert '<AuxiliaryOCR source="local" reliability="unverified">' in text
    assert "may contain errors" in text
    assert text.endswith("请讲解这道题")


def test_multimodal_request_requires_reusable_model_visual_description() -> None:
    text = build_multimodal_request_text("请讲解这道题", "OCR")

    assert "original image(s)" in text
    assert "<ImageDescription>...</ImageDescription>" in text
    assert "Do not merely repeat the OCR text" in text
    assert text.endswith("请讲解这道题")


def test_default_image_only_request_infers_intent_and_answers_visible_problem() -> None:
    single = default_image_only_request(1)
    multiple = default_image_only_request(2)

    assert "一张图片" in single
    assert "推断用户最可能希望获得的帮助" in single
    assert "题目、问题或待完成的任务，请识别并解答" in single
    assert "不要只做泛泛的图片描述" in single
    assert "2 张图片" in multiple


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


def test_thread_store_persists_reasoning_effort(tmp_path) -> None:
    path = tmp_path / "feishu_threads.json"
    store = FeishuThreadStore(path)
    store.get_or_create_thread_id(chat_id="chat", topic_id="scope", user_id="feishu:ou")

    store.set_reasoning_effort(chat_id="chat", topic_id="scope", effort="high")

    reloaded = FeishuThreadStore(path)
    assert reloaded.get_reasoning_effort(chat_id="chat", topic_id="scope", default="medium") == "high"


def test_parse_effort_command() -> None:
    assert parse_effort_command("/effort") == (True, None)
    assert parse_effort_command("/EFFORT high") == (True, "high")
    assert parse_effort_command("/effort=xhigh") == (True, "xhigh")
    assert parse_effort_command("explain effort") == (False, None)


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


def test_feishu_channel_discards_messages_received_while_scope_is_busy(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id):
            calls.append(message)
            if message == "first":
                started.set()
                release.wait(timeout=3)
            return AgentRunResult(thread_id=thread_id, answer=f"done: {message}", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        replied_message_ids: list[str] = []

        async def reply(message_id, _text):
            replied_message_ids.append(message_id)
            return f"card_{message_id}"

        async def update(_message_id, _text):
            return None

        channel._reply_card = reply
        channel._update_card = update
        first = asyncio.create_task(
            channel.handle_inbound(FeishuInboundMessage("chat", "msg_1", "ou", "first", chat_type="p2p"))
        )
        assert await asyncio.to_thread(started.wait, 2)

        await asyncio.wait_for(
            channel.handle_inbound(FeishuInboundMessage("chat", "msg_2", "ou", "second", chat_type="p2p")),
            timeout=0.2,
        )
        assert calls == ["first"]
        assert "msg_2" not in replied_message_ids

        release.set()
        await first
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg_3", "ou", "third", chat_type="p2p"))

        assert calls == ["first", "third"]
        assert replied_message_ids == ["msg_1", "msg_3"]

    _run(go())


def test_feishu_effort_command_persists_and_applies_to_next_message(monkeypatch, tmp_path) -> None:
    runtime_efforts: list[str] = []

    class Runtime:
        def __init__(self, settings, *, run_event_reporter):
            runtime_efforts.append(settings.reasoning_effort)
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id):
            return AgentRunResult(thread_id=thread_id, answer="done", metadata={})

        def close(self):
            return None

    async def go():
        monkeypatch.setattr("superassist.channels.feishu.AgentRuntime", Runtime)
        channel = FeishuChannel(
            _settings(tmp_path, SUPERASSIST_MODEL="gpt-5.6-sol", SUPERASSIST_REASONING_EFFORT="medium")
        )
        replies: list[str] = []

        async def reply(_message_id, text):
            replies.append(text)
            return "card"

        async def update(_message_id, text):
            replies.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat", "cmd_1", "ou", "/effort high", chat_type="p2p"))
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg_1", "ou", "solve it", chat_type="p2p"))
        await channel.handle_inbound(FeishuInboundMessage("chat", "cmd_2", "ou", "/effort", chat_type="p2p"))

        assert runtime_efforts == ["high"]
        assert any("`high`" in text for text in replies)
        assert channel.store.get_reasoning_effort(
            chat_id="chat", topic_id="__private__", default="medium"
        ) == "high"

    _run(go())

def test_feishu_image_and_ocr_reach_main_runtime_in_one_multimodal_call(tmp_path) -> None:
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

        async def extract_images_ocr(_payloads):
            return "A small OCR result."

        channel._extract_images_ocr = extract_images_ocr
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
        assert "A small OCR result." in message
        assert message.endswith("不要只做泛泛的图片描述。")
        assert "一张图片" in message
        assert "识别并解答" in message
        assert "data:image/png;base64,aW1hZ2U=" not in message
        assert isinstance(content, list)
        assert content[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        }
        assert content[-1]["type"] == "text"
        assert "<ImageDescription>...</ImageDescription>" in content[-1]["text"]
        assert content[-1]["text"].endswith("不要只做泛泛的图片描述。")

    _run(go())


def test_feishu_full_image_context_uses_sliding_three_minute_ttl(tmp_path) -> None:
    now = [0.0]
    received = []
    ocr_calls = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id, message_content=None):
            received.append((message, message_content))
            return AgentRunResult(
                thread_id=thread_id,
                answer="done\n\n<ImageDescription>visible details</ImageDescription>",
                metadata={},
            )

    async def go():
        channel = FeishuChannel(
            _settings(tmp_path, SUPERASSIST_FEISHU_IMAGE_CONTEXT_TTL_SECONDS=180),
            runtime_factory=lambda reporter: Runtime(reporter),
            monotonic_clock=lambda: now[0],
        )

        async def download_payloads(_images, _message_id):
            return [(b"original-image", "image/png")]

        async def extract_images_ocr(payloads):
            ocr_calls.append(payloads)
            return "OCR once"

        async def reply(_message_id, _text):
            return "card"

        async def update(_message_id, _text):
            return None

        channel._download_image_payloads = download_payloads
        channel._extract_images_ocr = extract_images_ocr
        channel._reply_card = reply
        channel._update_card = update

        await channel.handle_inbound(
            FeishuInboundMessage(
                "chat",
                "msg_image",
                "ou",
                "[image]",
                chat_type="p2p",
                files=[{"image_key": "img_1"}],
            )
        )

        now[0] = 120.0
        await channel.handle_inbound(
            FeishuInboundMessage("chat", "msg_followup", "ou", "继续分析细节", chat_type="p2p")
        )

        now[0] = 301.0
        await channel.handle_inbound(
            FeishuInboundMessage("chat", "msg_expired", "ou", "现在总结一下", chat_type="p2p")
        )

        assert len(received) == 3
        first_message, first_content = received[0]
        second_message, second_content = received[1]
        third_message, third_content = received[2]
        assert "OCR once" in first_message
        assert first_content[0]["image_url"]["url"].endswith("b3JpZ2luYWwtaW1hZ2U=")
        assert second_message == "继续分析细节"
        assert second_content[0] == first_content[0]
        assert third_message == "现在总结一下"
        assert third_content is None
        assert len(ocr_calls) == 1
        assert channel._image_contexts == {}

    _run(go())


def test_feishu_ocr_failure_is_non_fatal(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))

        def fail(_payloads):
            raise RuntimeError("OCR unavailable")

        channel._extract_images_ocr_sync = fail
        result = await channel._extract_images_ocr([(b"not-decoded", "image/jpeg")])

        assert result == ""

    _run(go())


def test_feishu_ocr_can_be_disabled(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(
            _settings(tmp_path, SUPERASSIST_FEISHU_IMAGE_OCR_ENABLED=False)
        )

        def unexpected(_payloads):
            raise AssertionError("OCR should not run")

        channel._extract_images_ocr_sync = unexpected
        result = await channel._extract_images_ocr([(b"image", "image/jpeg")])

        assert result == ""

    _run(go())


def test_feishu_local_ocr_collects_multiple_images_and_truncates(tmp_path) -> None:
    source = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(source, format="JPEG")
    channel = FeishuChannel(
        _settings(tmp_path, SUPERASSIST_FEISHU_IMAGE_OCR_MAX_CHARS=28)
    )
    calls = []

    def engine(pixels):
        calls.append(pixels.shape)
        return ([[None, "first line", 0.99], [None, "second line", 0.98]], 0.1)

    channel._ocr_engine = engine
    text = _run(
        channel._extract_images_ocr(
            [(source.getvalue(), "image/jpeg"), (source.getvalue(), "image/jpeg")]
        )
    )

    assert calls == [(4, 4, 3), (4, 4, 3)]
    assert text.startswith("[Image 1]\nfirst line")
    assert len(text) <= 28


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
