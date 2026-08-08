from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from superassist.agent.short_memory import read_jsonl
from superassist.channels.daily_brief import DailyBriefProgress, DailyBriefRunResult
from superassist.channels.daily_quiz import DailyQuizStore
from superassist.channels.feishu import (
    FeishuCardImage,
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
    format_daily_brief_progress,
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
from superassist.channels.store import FeishuMessageStore, FeishuThreadStore
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


def test_group_requires_explicit_activation_for_each_batch(tmp_path) -> None:
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


def test_feishu_message_store_is_idempotent_and_advances_cursor(tmp_path) -> None:
    store = FeishuMessageStore(tmp_path / "messages.sqlite3")
    values = {
        "message_id": "msg_1",
        "chat_id": "chat_1",
        "sender_open_id": "ou_1",
        "sender_name": "Alice",
        "text": "hello",
        "root_id": None,
        "chat_type": "group",
        "mentions": [],
        "files": [],
        "created_at": "2026-08-08T10:00:00+08:00",
    }

    inserted, seq = store.add_message(**values)
    duplicate, duplicate_seq = store.add_message(**values)

    assert inserted is True
    assert duplicate is False
    assert duplicate_seq == seq
    assert [item.message_id for item in store.list_unconsumed("chat_1")] == ["msg_1"]
    store.save_image(
        message_id="msg_1",
        image_key="img_1",
        data=b"pixels",
        mime_type="image/png",
    )
    assert store.get_image(message_id="msg_1", image_key="img_1") == (b"pixels", "image/png")
    store.commit_consumed("chat_1", seq)
    assert store.list_unconsumed("chat_1") == []
    assert store.get_image(message_id="msg_1", image_key="img_1") is None


def test_group_activation_batches_multiple_speakers_and_late_image(tmp_path) -> None:
    received = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id, message_content=None):
            received.append((message, message_content, user_id, thread_id))
            return AgentRunResult(thread_id=thread_id, answer="done", metadata={})

    async def go():
        channel = FeishuChannel(
            _settings(
                tmp_path,
                SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS="ou_owner",
                SUPERASSIST_FEISHU_ACTIVATION_DEBOUNCE_SECONDS=0.02,
                SUPERASSIST_FEISHU_ACTIVATION_MAX_WAIT_SECONDS=0.1,
            ),
            runtime_factory=lambda reporter: Runtime(reporter),
        )

        async def create(_chat_id, _text):
            return "card"

        async def update(_message_id, _text):
            return None

        async def download(message_id, image_key):
            assert (message_id, image_key) == ("msg_image", "img_1")
            return b"image", "image/png"

        async def ocr(_payloads):
            return "[Image 1]\nvisible text"

        channel._create_card = create
        channel._update_card = update
        channel._download_image = download
        channel._extract_images_ocr = ocr

        await channel.handle_inbound(
            FeishuInboundMessage("chat", "msg_a", "ou_a", "先讨论需求", sender_name="Alice", chat_type="group")
        )
        await channel.handle_inbound(
            FeishuInboundMessage("chat", "msg_b", "ou_b", "我补充一个条件", sender_name="Bob", chat_type="group")
        )
        trigger = asyncio.create_task(
            channel.handle_inbound(
                FeishuInboundMessage(
                    "chat",
                    "msg_trigger",
                    "ou_owner",
                    "@bot 请综合处理",
                    sender_name="Owner",
                    chat_type="group",
                    mentions=[{"name": "@bot"}],
                )
            )
        )
        await asyncio.sleep(0.005)
        await channel.handle_inbound(
            FeishuInboundMessage(
                "chat",
                "msg_image",
                "ou_b",
                "[image]",
                sender_name="Bob",
                chat_type="group",
                files=[{"image_key": "img_1"}],
            )
        )
        await trigger

        assert len(received) == 1
        message, content, user_id, _thread_id = received[0]
        assert user_id == "feishu-group:chat"
        assert 'sender_id="ou_a" sender_name="Alice"' in message
        assert 'sender_id="ou_b" sender_name="Bob"' in message
        assert 'sender_id="ou_owner" sender_name="Owner"' in message
        assert '<text>请综合处理</text>' in message
        assert "[Image 1]" in message
        assert "visible text" in message
        assert isinstance(content, list)
        assert any(item.get("type") == "image_url" for item in content)
        assert any("from sender ou_b" in item.get("text", "") for item in content)

    _run(go())


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


def test_thread_store_returns_latest_entry_for_chat(tmp_path) -> None:
    store = FeishuThreadStore(tmp_path / "feishu_threads.json")
    first = store.get_or_create_thread_id(chat_id="chat", topic_id="first", user_id="feishu-group:chat")
    second = store.get_or_create_thread_id(chat_id="chat", topic_id="second", user_id="feishu-group:chat")

    entry = store.get_latest_chat_entry("chat")

    assert entry is not None
    assert entry["thread_id"] in {first, second}
    assert entry["user_id"] == "feishu-group:chat"
    assert store.get_latest_chat_entry("missing") is None


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


def test_build_card_content_renders_selected_images_and_sources() -> None:
    card = json.loads(
        build_card_content(
            FeishuCardView(
                answer="这是千叶豆腐。",
                images=(
                    FeishuCardImage(
                        title="千叶豆腐成品",
                        source_url="https://example.com/recipe",
                        image_key="img_feishu_1",
                    ),
                ),
            )
        )
    )

    elements = card["body"]["elements"]
    assert elements[1] == {
        "tag": "img",
        "img_key": "img_feishu_1",
        "alt": {"tag": "plain_text", "content": "千叶豆腐成品"},
        "preview": True,
    }
    assert "https://example.com/recipe" in elements[2]["content"]


def test_prepare_outbound_images_uploads_selection(monkeypatch, tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))
        monkeypatch.setattr(
            "superassist.channels.feishu._download_outbound_candidate",
            lambda _candidate: (b"image", "image/jpeg"),
        )

        async def upload(data):
            assert data == b"image"
            return "img_uploaded"

        channel._upload_image = upload
        result = await channel._prepare_outbound_images(
            [
                {
                    "candidate_id": "img_1",
                    "title": "千叶豆腐",
                    "image_url": "https://cdn.example.com/image.jpg",
                    "source_url": "https://example.com/source",
                }
            ]
        )

        assert result == [
            FeishuCardImage(
                title="千叶豆腐",
                source_url="https://example.com/source",
                image_key="img_uploaded",
            )
        ]

    _run(go())


def test_prepare_outbound_images_degrades_to_source_link(monkeypatch, tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))

        def fail(_candidate):
            raise RuntimeError("download failed")

        monkeypatch.setattr("superassist.channels.feishu._download_outbound_candidate", fail)
        result = await channel._prepare_outbound_images(
            [{"candidate_id": "img_1", "title": "Source", "source_url": "https://example.com/source"}]
        )

        assert result == [FeishuCardImage(title="Source", source_url="https://example.com/source")]

    _run(go())


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


def test_feishu_daily_brief_command_runs_preview_for_current_chat(tmp_path) -> None:
    triggered: list[str] = []

    async def trigger(chat_id, progress_reporter):
        triggered.append(chat_id)
        progress_reporter(DailyBriefProgress(55, "正在打开官媒原文核验", "已完成 2/5 次"))
        return DailyBriefRunResult(status="sent", message="delivered")

    async def go():
        channel = FeishuChannel(_settings(tmp_path), daily_brief_trigger=trigger)
        sent: list[str] = []

        async def reply(_message_id, text):
            sent.append(text)
            return "card_1"

        async def update(_message_id, text):
            sent.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat_1", "msg_1", "ou_1", "/brief", chat_type="p2p"))

        assert triggered == ["chat_1"]
        assert any("55%" in text and "正在打开官媒原文核验" in text for text in sent)
        assert sent[-1] == "申论官媒简报已生成并发送。"

    _run(go())


def test_feishu_quiz_text_is_not_a_special_command(tmp_path) -> None:
    runtime_messages: list[str] = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, message, *, user_id, thread_id):
            runtime_messages.append(message)
            return AgentRunResult(thread_id=thread_id, answer="主 Agent 已收到普通消息。", metadata={})

    async def go():
        channel = FeishuChannel(_settings(tmp_path), runtime_factory=lambda reporter: Runtime(reporter))
        sent: list[str] = []

        async def reply(_message_id, text):
            sent.append(text)
            return "card_1"

        async def update(_message_id, text):
            sent.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat_1", "msg_1", "ou_1", "/quiz", chat_type="p2p"))

        assert runtime_messages == ["/quiz"]
        assert sent[-1] == "主 Agent 已收到普通消息。"

    _run(go())


def test_feishu_quiz_answer_reaches_main_agent_as_an_ordinary_turn(tmp_path) -> None:
    settings = _settings(tmp_path, SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT=2)
    thread_store = FeishuThreadStore(settings.feishu_thread_store_path)
    thread_id = thread_store.get_or_create_thread_id(
        chat_id="chat_1",
        topic_id="__private__",
        user_id="feishu:ou_1",
    )
    quiz_store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    quiz_store.archive_brief("chat_1", now, "今日强调提升基层治理效能。")
    quiz_store.start_session("chat_1", thread_id, now)
    questions = [
        {
            "question": "材料体现的治理要求是？",
            "option_a": "单一治理",
            "option_b": "系统治理",
            "option_c": "被动治理",
            "option_d": "封闭治理",
            "correct_option": "B",
            "explanation": "材料强调系统治理，其他选项均割裂了治理要素。",
            "source_date": now.date().isoformat(),
            "source_title": "今日简报",
            "evidence": "提升基层治理效能",
        },
        {
            "question": "下列对系统治理理解最准确的是？",
            "option_a": "只处理单一环节",
            "option_b": "完全依赖临时措施",
            "option_c": "统筹主体、资源和过程",
            "option_d": "排除公众参与",
            "correct_option": "C",
            "explanation": "系统治理强调多主体和全过程统筹。",
            "source_date": now.date().isoformat(),
            "source_title": "今日简报",
            "evidence": "提升基层治理效能",
        },
    ]
    quiz_store.save_draft(thread_id, questions)
    quiz_store.finalize(thread_id, "已逐题检查材料依据、唯一答案和干扰项质量。")
    runtime_controls: list[dict[str, object]] = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(
            self,
            message,
            *,
            user_id,
            thread_id,
            memory_query=None,
            suppress_memory_write=False,
            suppress_short_memory_write=False,
        ):
            runtime_controls.append(
                {
                    "message": message,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "memory_query": memory_query,
                    "suppress_memory_write": suppress_memory_write,
                    "suppress_short_memory_write": suppress_short_memory_write,
                }
            )
            assert message == "1B 2C"
            assert "DailyPoliticalQuizGrading" not in message
            grading_context = quiz_store.build_grading_prompt(thread_id, ["B", "C"])
            assert "<DailyPoliticalQuizGrading>" in grading_context
            saved = quiz_store.save_grading(
                thread_id,
                [
                    {
                        "number": 1,
                        "is_correct": True,
                        "feedback": "材料强调系统治理，B 项符合题意。",
                        "weakness": "",
                    },
                    {
                        "number": 2,
                        "is_correct": True,
                        "feedback": "系统治理要求统筹主体、资源和过程，C 项正确。",
                        "weakness": "",
                    },
                ],
                "两题均由主 Agent 判定正确，系统治理知识掌握较好。",
            )
            assert "Agent grading saved. Score: 2/2" in saved
            return AgentRunResult(
                thread_id=thread_id,
                answer="主 Agent 批改报告：2/2。第 1 题正确；第 2 题正确。",
                metadata={},
            )

    async def go():
        channel = FeishuChannel(
            settings,
            store=thread_store,
            daily_quiz_store=quiz_store,
            runtime_factory=lambda reporter: Runtime(reporter),
        )
        visible: list[str] = []

        async def reply(_message_id, text):
            visible.append(text)
            return "card_2"

        async def update(_message_id, text):
            visible.append(text)

        channel._reply_card = reply
        channel._update_card = update
        await channel.handle_inbound(FeishuInboundMessage("chat_1", "msg_2", "ou_1", "1B 2C", chat_type="p2p"))

        assert visible[-1] == "主 Agent 批改报告：2/2。第 1 题正确；第 2 题正确。"
        assert quiz_store.active_session(thread_id) is None
        assert runtime_controls[0]["user_id"] == "feishu:ou_1"
        assert runtime_controls[0]["thread_id"] == thread_id
        assert runtime_controls[0]["memory_query"] is None
        assert runtime_controls[0]["suppress_memory_write"] is False
        assert runtime_controls[0]["suppress_short_memory_write"] is False

    _run(go())


def test_scheduled_quiz_uses_dedicated_subagent_and_persists_only_visible_question(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT=2)
    thread_store = FeishuThreadStore(settings.feishu_thread_store_path)
    thread_id = thread_store.get_or_create_thread_id(
        chat_id="chat_1",
        topic_id="__group__",
        user_id="feishu-group:chat_1",
    )
    quiz_store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    quiz_store.archive_brief("chat_1", now, "日报内部全文：湖北推进基层治理创新。")
    runtime_controls: list[dict[str, object]] = []

    class Runtime:
        def __init__(self, _reporter):
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run(
            self,
            message,
            *,
            user_id,
            thread_id,
            memory_query=None,
            suppress_memory_write=False,
            suppress_short_memory_write=False,
        ):
            runtime_controls.append(
                {
                    "message": message,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "memory_query": memory_query,
                    "suppress_memory_write": suppress_memory_write,
                    "suppress_short_memory_write": suppress_short_memory_write,
                }
            )
            quiz_store.save_draft(
                thread_id,
                [
                    {
                        "question": "材料体现了哪一种治理理念？",
                        "option_a": "系统治理",
                        "option_b": "封闭治理",
                        "option_c": "单一治理",
                        "option_d": "被动治理",
                        "correct_option": "A",
                        "explanation": "材料体现系统治理，其他选项均与协同要求相悖。",
                        "source_date": now.date().isoformat(),
                        "source_title": "今日简报",
                        "evidence": "推进基层治理创新",
                    },
                    {
                        "question": "基层治理创新应坚持什么方法？",
                        "option_a": "单一主体包办",
                        "option_b": "多元协同参与",
                        "option_c": "只看短期指标",
                        "option_d": "排除群众参与",
                        "correct_option": "B",
                        "explanation": "基层治理创新需要多元协同，其他选项都削弱治理合力。",
                        "source_date": now.date().isoformat(),
                        "source_title": "今日简报",
                        "evidence": "推进基层治理创新",
                    },
                ],
            )
            quiz_store.finalize(thread_id, "已检查两题的材料依据、唯一答案、差异性和选项质量。")
            return AgentRunResult(thread_id=thread_id, answer="内部状态已保存。", metadata={})

    subagent_calls: list[dict[str, object]] = []

    def run_quiz_subagent(description, prompt, **kwargs):
        subagent_calls.append({"description": description, "prompt": prompt, **kwargs})
        Runtime(None).run(
            prompt,
            user_id="dedicated-subagent",
            thread_id=kwargs["parent_thread_id"],
        )
        return "Task Succeeded. Result: public questions saved"

    task_module = __import__("superassist.tools.task", fromlist=["run_task"])
    monkeypatch.setattr(task_module, "run_task", run_quiz_subagent)

    async def go():
        channel = FeishuChannel(
            settings,
            store=thread_store,
            daily_quiz_store=quiz_store,
            runtime_factory=lambda _reporter: pytest.fail("scheduled quiz must not create a main Agent runtime"),
        )
        cards: list[str] = []

        async def create(_chat_id, text):
            cards.append(text)
            return "quiz_card_1"

        channel._create_card = create
        result = await channel.start_daily_quiz("chat_1", now)

        assert result == "政治理论测验已生成并完成检查，请在新卡片中一次提交全部答案。"
        assert "材料体现了哪一种治理理念" in cards[-1]
        assert "基层治理创新应坚持什么方法" in cards[-1]
        assert subagent_calls[0]["subagent_type"] == "shenlun-quiz"
        assert subagent_calls[0]["parent_thread_id"] == thread_id
        assert runtime_controls[0]["thread_id"] == thread_id
        assert runtime_controls[0]["user_id"] == "dedicated-subagent"
        assert runtime_controls[0]["memory_query"] is None
        assert runtime_controls[0]["suppress_memory_write"] is False
        assert runtime_controls[0]["suppress_short_memory_write"] is False
        assert "日报内部全文" not in str(runtime_controls[0]["message"])

        records = read_jsonl(settings.data_dir / "threads" / thread_id / "messages.jsonl")
        assert [record["role"] for record in records] == ["assistant"]
        assert "材料体现了哪一种治理理念" in records[0]["content"]
        assert records[0]["source"] == "daily_quiz"
        assert "日报内部全文" not in str(records)
        assert "correct_option" not in str(records)
        assert "内部状态已保存" not in str(records)

    _run(go())


def test_format_daily_brief_progress_replaces_bar_and_bounds_detail() -> None:
    rendered = format_daily_brief_progress(DailyBriefProgress(68, "正在核验", "原文  读取中"))

    assert "70%" not in rendered
    assert "68%" in rendered
    assert "███████░░░" in rendered
    assert "原文 读取中" in rendered


def test_feishu_proactive_brief_is_added_to_main_short_memory_without_user_turn(tmp_path) -> None:
    async def go():
        settings = _settings(tmp_path)
        store = FeishuThreadStore(settings.feishu_thread_store_path)
        thread_id = store.get_or_create_thread_id(
            chat_id="chat_1",
            topic_id="__group__",
            user_id="feishu-group:chat_1",
        )
        channel = FeishuChannel(settings, store=store)

        async def create(_chat_id, _text):
            return "message_1"

        channel._create_card = create

        assert await channel.send_proactive_card("chat_1", "申论官媒晚报") == "message_1"
        records = read_jsonl(settings.data_dir / "threads" / thread_id / "messages.jsonl")
        assert len(records) == 1
        assert records[0]["role"] == "assistant"
        assert records[0]["content"] == "申论官媒晚报"
        assert records[0]["source"] == "daily_brief"

    _run(go())


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


def test_feishu_channel_queues_messages_received_while_scope_is_busy(tmp_path) -> None:
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

        second = asyncio.create_task(
            channel.handle_inbound(FeishuInboundMessage("chat", "msg_2", "ou", "second", chat_type="p2p"))
        )
        assert calls == ["first"]
        assert "msg_2" not in replied_message_ids

        release.set()
        await first
        await second
        await channel.handle_inbound(FeishuInboundMessage("chat", "msg_3", "ou", "third", chat_type="p2p"))

        assert calls == ["first", "second", "third"]
        assert replied_message_ids == ["msg_1", "msg_2", "msg_3"]

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
            assert '<FeishuConversationBatch format="chronological-chat"' in message
            assert 'sender_id="ou_1"' in message
            assert 'sender_name="小馨"' in message
            assert '<text>做个计划</text>' in message
            assert 'trigger="true"' in message
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
