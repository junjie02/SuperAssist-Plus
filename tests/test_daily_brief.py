from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from superassist.channels import daily_brief as module
from superassist.channels.daily_brief import (
    append_reused_source_note,
    DailyBriefProgress,
    DailyBriefScheduler,
    LatestCandidate,
    collect_latest_candidates,
    latest_scheduled_at,
    load_official_media_config,
    next_scheduled_at,
    parse_schedule_times,
    select_brief_candidates,
    validate_brief_sources,
)
from superassist.config import Settings
from superassist.channels.daily_quiz import DailyQuizStore
from superassist.models import AgentRunEvent, AgentRunResult


def _run(coro):
    return asyncio.run(coro)


def _write_config(tmp_path):
    path = tmp_path / "official_media.toml"
    path.write_text(
        """
[[sources]]
id = "gov"
name = "中国政府网"
enabled = true
domains = ["gov.cn"]
latest_pages = ["https://www.gov.cn/yaowen/"]

[[sources]]
id = "news"
name = "新华网"
enabled = true
domains = ["news.cn"]
latest_pages = ["https://www.news.cn/politics/"]
""",
        encoding="utf-8",
    )
    return path


def test_load_config_and_collect_latest_links(tmp_path, monkeypatch) -> None:
    config = load_official_media_config(_write_config(tmp_path))
    pages = {
        "https://www.gov.cn/yaowen/": "<a href='/zhengce/one.htm'>国务院发布最新政策安排</a>",
        "https://www.news.cn/politics/": "<a href='https://www.news.cn/politics/two.htm'>新华社报道今日重要会议</a>",
    }
    monkeypatch.setattr(module, "_fetch_url", lambda url, timeout=20: (pages[url], "text/html"))

    candidates = collect_latest_candidates(config, 10)

    assert len(config.sources) == 2
    assert {item.url for item in candidates} == {
        "https://www.gov.cn/zhengce/one.htm",
        "https://www.news.cn/politics/two.htm",
    }


def test_schedule_calculation_uses_local_timezone() -> None:
    timezone = ZoneInfo("Asia/Shanghai")
    schedule = parse_schedule_times("07:45,19:45")
    now = datetime(2026, 8, 1, 8, 0, tzinfo=timezone)

    assert latest_scheduled_at(now, schedule) == datetime(2026, 8, 1, 7, 45, tzinfo=timezone)
    assert next_scheduled_at(now, schedule) == datetime(2026, 8, 1, 19, 45, tzinfo=timezone)


def test_source_validation_rejects_non_official_urls() -> None:
    answer = "https://www.gov.cn/a https://example.com/repost"

    errors = validate_brief_sources(answer, ["gov.cn"], min_sources=1)

    assert any("non-official URLs" in error for error in errors)


def test_select_candidates_reuses_recent_articles_to_fill_source_gap() -> None:
    now = datetime(2026, 8, 2, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    candidates = [
        LatestCandidate("new", "新报", "新文章", "https://news.cn/new", "https://news.cn/"),
        LatestCandidate("old", "旧报", "旧文章", "https://news.cn/old", "https://news.cn/"),
    ]
    selected = select_brief_candidates(
        candidates,
        {"https://news.cn/old": "2026-08-01T01:00:00+00:00"},
        window_start=now - timedelta(days=1),
        now=now,
        min_sources=2,
    )

    assert [item.url for item in selected] == ["https://news.cn/new", "https://news.cn/old"]
    assert selected[0].reused is False
    assert selected[1].reused is True


def test_select_candidates_does_not_reuse_expired_articles() -> None:
    now = datetime(2026, 8, 8, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    candidate = LatestCandidate("old", "旧报", "旧文章", "https://news.cn/old", "https://news.cn/")
    selected = select_brief_candidates(
        [candidate],
        {"https://news.cn/old": "2026-07-31T23:00:00+00:00"},
        window_start=now - timedelta(days=1),
        now=now,
        min_sources=1,
    )

    assert selected == []


def test_reused_source_note_replaces_no_new_content_marker() -> None:
    candidate = LatestCandidate("old", "旧报", "旧文章", "https://news.cn/old", "https://news.cn/")

    rendered = append_reused_source_note(module.NO_NEW_CONTENT_MARKER, [candidate])

    assert module.NO_NEW_CONTENT_MARKER not in rendered
    assert "延续阅读" in rendered
    assert candidate.url in rendered


def test_scheduler_generates_validated_brief_and_sends_once(tmp_path, monkeypatch) -> None:
    source_path = _write_config(tmp_path)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("只整理当前时间窗口内的官媒最新内容。", encoding="utf-8")
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_DAILY_BRIEF_ENABLED=True,
        SUPERASSIST_DAILY_BRIEF_FEISHU_CHAT_IDS="chat_1",
        SUPERASSIST_DAILY_BRIEF_SOURCE_FILE=source_path,
        SUPERASSIST_DAILY_BRIEF_PROMPT_FILE=prompt_path,
        SUPERASSIST_DAILY_BRIEF_MIN_SOURCES=2,
    )
    candidates = [
        LatestCandidate("gov", "中国政府网", "国务院发布最新政策安排", "https://www.gov.cn/a", "https://www.gov.cn/"),
        LatestCandidate("news", "新华网", "新华社报道今日重要会议", "https://www.news.cn/b", "https://www.news.cn/"),
    ]
    monkeypatch.setattr(module, "collect_latest_candidates", lambda config, limit: candidates)
    runtime_calls: list[str] = []

    class Runtime:
        def __init__(self, runtime_settings):
            # The brief runtime uses the dedicated daily-brief model; the exact
            # name is deployment configuration and changes with model upgrades.
            assert runtime_settings.model == settings.daily_brief_model
            assert runtime_settings.agent_teams_enabled is False
            assert runtime_settings.subagents_enabled is False
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run(self, prompt, *, user_id, thread_id):
            runtime_calls.append(prompt)
            return AgentRunResult(
                thread_id=thread_id,
                answer="晨报\nhttps://www.gov.cn/a\nhttps://www.news.cn/b",
                metadata={},
            )

        def close(self):
            return None

    sent: list[tuple[str, str]] = []
    archived: list[tuple[str, datetime, str]] = []

    async def sender(chat_id, text):
        sent.append((chat_id, text))
        return "message_1"

    scheduler = DailyBriefScheduler(
        settings,
        sender,
        runtime_factory=Runtime,
        brief_recorder=lambda chat_id, delivered_at, content: archived.append(
            (chat_id, delivered_at, content)
        ),
    )
    scheduled_for = datetime(2026, 8, 1, 19, 45, tzinfo=ZoneInfo("Asia/Shanghai"))
    progress: list[DailyBriefProgress] = []

    first = _run(scheduler.run_once(scheduled_for=scheduled_for, progress_reporter=progress.append))
    second = _run(scheduler.run_once(scheduled_for=scheduled_for))

    assert first.status == "sent"
    assert second.status == "skipped"
    assert len(runtime_calls) == 1
    assert sent == [("chat_1", "晨报\nhttps://www.gov.cn/a\nhttps://www.news.cn/b")]
    assert archived == [
        ("chat_1", scheduled_for, "晨报\nhttps://www.gov.cn/a\nhttps://www.news.cn/b")
    ]
    assert [event.percent for event in progress] == sorted(event.percent for event in progress)
    assert progress[-1] == DailyBriefProgress(100, "简报已生成并发送", "")
    assert any(event.stage == "官媒候选收集完成" for event in progress)


def test_manual_brief_is_archived_and_available_for_quiz_generation(tmp_path, monkeypatch) -> None:
    source_path = _write_config(tmp_path)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("整理当前时间窗口内的官媒内容。", encoding="utf-8")
    now = datetime(2026, 8, 30, 7, 43, tzinfo=ZoneInfo("Asia/Shanghai"))
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_DAILY_BRIEF_SOURCE_FILE=source_path,
        SUPERASSIST_DAILY_BRIEF_PROMPT_FILE=prompt_path,
        SUPERASSIST_DAILY_BRIEF_MIN_SOURCES=1,
        SUPERASSIST_DAILY_BRIEF_GENERATION_RETRIES=0,
    )
    candidate = LatestCandidate(
        "gov",
        "中国政府网",
        "今日政策部署",
        "https://www.gov.cn/policy/today.htm",
        "https://www.gov.cn/yaowen/",
    )
    monkeypatch.setattr(module, "collect_latest_candidates", lambda _config, _limit: [candidate])

    class Runtime:
        def __init__(self, runtime_settings):
            assert runtime_settings.model_timeout_seconds == 120
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run(self, _prompt, *, user_id, thread_id):
            return AgentRunResult(
                thread_id=thread_id,
                answer="晨报\n今日政策部署\nhttps://www.gov.cn/policy/today.htm",
                metadata={},
            )

        def close(self):
            return None

    async def sender(_chat_id, _text):
        return "message_1"

    store = DailyQuizStore(settings)
    scheduler = DailyBriefScheduler(
        settings,
        sender,
        runtime_factory=Runtime,
        now_factory=lambda _timezone: now,
        brief_recorder=store.archive_brief,
    )

    result = _run(scheduler.run_now("chat_manual"))
    quiz_materials = store.prepare_generation("chat_manual", "quiz_thread", now)
    state = json.loads(settings.daily_brief_state_path.read_text(encoding="utf-8"))
    run_events = [
        json.loads(line)
        for line in settings.daily_brief_run_log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.status == "sent"
    assert "今日政策部署" in quiz_materials
    assert not quiz_materials.startswith("Error:")
    assert state["last_success_at"] == now.isoformat()
    assert any(event["event"] == "run_succeeded" and event["run_kind"] == "manual" for event in run_events)
    assert run_events[-1]["archived_count"] == 1


def test_generation_failure_is_retried_once(tmp_path, monkeypatch) -> None:
    source_path = _write_config(tmp_path)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("整理官媒内容。", encoding="utf-8")
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_DAILY_BRIEF_FEISHU_CHAT_IDS="chat_1",
        SUPERASSIST_DAILY_BRIEF_SOURCE_FILE=source_path,
        SUPERASSIST_DAILY_BRIEF_PROMPT_FILE=prompt_path,
        SUPERASSIST_DAILY_BRIEF_MIN_SOURCES=1,
        SUPERASSIST_DAILY_BRIEF_GENERATION_RETRIES=1,
        SUPERASSIST_DAILY_BRIEF_RETRY_DELAY_SECONDS=0,
    )
    candidate = LatestCandidate(
        "gov",
        "中国政府网",
        "政策部署",
        "https://www.gov.cn/a",
        "https://www.gov.cn/",
    )
    monkeypatch.setattr(module, "collect_latest_candidates", lambda _config, _limit: [candidate])
    attempts: list[int] = []

    def generate(_prompt, _config, _now, _progress, _reused, attempt):
        attempts.append(attempt)
        if attempt == 0:
            raise TimeoutError("provider timed out")
        return "晨报\nhttps://www.gov.cn/a"

    async def sender(_chat_id, _text):
        return "message_1"

    scheduler = DailyBriefScheduler(settings, sender)
    monkeypatch.setattr(scheduler, "_generate", generate)

    result = _run(
        scheduler.run_once(
            scheduled_for=datetime(2026, 8, 30, 7, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        )
    )
    events = [
        json.loads(line)["event"]
        for line in settings.daily_brief_run_log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.status == "sent"
    assert attempts == [0, 1]
    assert events.count("generation_attempt_failed") == 1
    assert events[-1] == "run_succeeded"


def test_generate_streams_real_model_and_tool_progress(tmp_path, monkeypatch) -> None:
    config = load_official_media_config(_write_config(tmp_path))
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path / "data",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_DAILY_BRIEF_MIN_SOURCES=2,
    )

    class Runtime:
        def __init__(self, _settings, *, tool_event_reporter, run_event_reporter):
            self.tool_reporter = tool_event_reporter
            self.run_reporter = run_event_reporter
            self.memory_queue = SimpleNamespace(flush=lambda: None)

        def run_streaming(self, _prompt, *, user_id, thread_id):
            self.run_reporter(AgentRunEvent(type="preparing_context", message="Preparing", metadata={}))
            self.tool_reporter({"type": "tool_start", "tool": "web_fetch"})
            self.tool_reporter({"type": "tool_result", "tool": "web_fetch", "status": "success"})
            self.run_reporter(AgentRunEvent(type="agent_reasoning", message="正在比较两篇原文", metadata={}))
            self.run_reporter(AgentRunEvent(type="agent_text", message="正文内容", metadata={}))
            return AgentRunResult(
                thread_id=thread_id,
                answer="简报\nhttps://www.gov.cn/a\nhttps://www.news.cn/b",
                metadata={},
            )

        def close(self):
            return None

    monkeypatch.setattr(module, "AgentRuntime", Runtime)
    scheduler = DailyBriefScheduler(settings, lambda _chat_id, _text: None)
    events: list[DailyBriefProgress] = []

    answer = scheduler._generate(
        "prompt",
        config,
        datetime(2026, 8, 1, 19, 45, tzinfo=ZoneInfo("Asia/Shanghai")),
        lambda percent, stage, detail="": events.append(DailyBriefProgress(percent, stage, detail)),
    )

    assert answer.startswith("简报")
    assert any(event.stage == "正在打开官媒原文核验" for event in events)
    assert any(event.stage == "DeepSeek 正在分析和归纳" for event in events)
    assert any(event.stage == "DeepSeek 正在撰写简报正文" for event in events)
