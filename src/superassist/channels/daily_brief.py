from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import threading
import tomllib
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from superassist.agent import AgentRuntime
from superassist.config import Settings
from superassist.models import AgentRunEvent
from superassist.redis_store import get_redis_store
from superassist.tools.web import _fetch_url, is_allowed_official_url, official_media_web_scope

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s<>\]\[)]+", flags=re.IGNORECASE)
NO_NEW_CONTENT_MARKER = "本时段无足够官媒新增内容"
MAX_LINKS_PER_PAGE = 30
MAX_CARD_CHARS = 12000
SEEN_URL_RETENTION_DAYS = 7


@dataclass(frozen=True)
class OfficialMediaSource:
    source_id: str
    name: str
    domains: tuple[str, ...]
    latest_pages: tuple[str, ...]


@dataclass(frozen=True)
class OfficialMediaConfig:
    sources: tuple[OfficialMediaSource, ...]

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({domain for source in self.sources for domain in source.domains}))


@dataclass(frozen=True)
class LatestCandidate:
    source_id: str
    source_name: str
    title: str
    url: str
    discovered_from: str
    published_hint: str = ""


@dataclass(frozen=True)
class DailyBriefRunResult:
    status: str
    message: str
    answer: str = ""


@dataclass(frozen=True)
class DailyBriefProgress:
    percent: int
    stage: str
    detail: str = ""


DailyBriefProgressReporter = Callable[[DailyBriefProgress], None]


def _progress_emitter(
    reporter: DailyBriefProgressReporter | None,
) -> Callable[[int, str, str], None]:
    lock = threading.Lock()
    highest_percent = 0

    def emit(percent: int, stage: str, detail: str = "") -> None:
        nonlocal highest_percent
        if reporter is None:
            return
        with lock:
            highest_percent = max(highest_percent, max(0, min(100, int(percent))))
            event = DailyBriefProgress(highest_percent, stage.strip(), detail.strip())
        try:
            reporter(event)
        except Exception:  # noqa: BLE001 - progress reporting must not abort a scheduled brief
            logger.exception("Daily brief progress reporter failed stage=%s", stage)

    return emit


def _daily_brief_tool_stage(tool_name: str) -> str:
    return {
        "web_search": "正在搜索官媒最新内容",
        "web_fetch": "正在打开官媒原文核验",
        "read_file": "正在读取研究规则和资料",
    }.get(tool_name, f"正在使用 {tool_name or '资料工具'}")


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a" or self._href is not None:
            return
        values = dict(attrs)
        href = str(values.get("href") or "").strip()
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        title = re.sub(r"\s+", " ", "".join(self._text)).strip()
        self.links.append((self._href, title))
        self._href = None
        self._text = []


def load_official_media_config(path: str | Path) -> OfficialMediaConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    sources: list[OfficialMediaSource] = []
    for item in raw.get("sources", []):
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        source_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or source_id).strip()
        domains = tuple(sorted({_normalize_domain(value) for value in item.get("domains", []) if value}))
        latest_pages = tuple(str(value).strip() for value in item.get("latest_pages", []) if str(value).strip())
        if not source_id or not name or not domains or not latest_pages:
            raise ValueError(f"Invalid official-media source in {config_path}: {item!r}")
        if any(not is_allowed_official_url(page, domains) for page in latest_pages):
            raise ValueError(f"Latest page is outside source domains for {source_id}")
        sources.append(OfficialMediaSource(source_id, name, domains, latest_pages))
    if not sources:
        raise ValueError(f"No enabled official-media sources in {config_path}")
    return OfficialMediaConfig(tuple(sources))


def collect_latest_candidates(config: OfficialMediaConfig, max_candidates: int = 80) -> list[LatestCandidate]:
    jobs = [(source, page) for source in config.sources for page in source.latest_pages]
    candidates: list[LatestCandidate] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(jobs)))) as executor:
        futures = {executor.submit(_collect_page, source, page): (source, page) for source, page in jobs}
        for future in as_completed(futures):
            source, page = futures[future]
            try:
                candidates.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one blocked official site must not abort the digest
                logger.warning(
                    "Daily brief source scan failed source=%s page=%s error_type=%s error=%s",
                    source.source_id,
                    page,
                    type(exc).__name__,
                    exc,
                )
    unique: dict[str, LatestCandidate] = {}
    for candidate in candidates:
        unique.setdefault(_canonical_url(candidate.url), candidate)
    grouped: dict[str, list[LatestCandidate]] = {source.source_id: [] for source in config.sources}
    for candidate in unique.values():
        grouped.setdefault(candidate.source_id, []).append(candidate)
    for items in grouped.values():
        items.sort(key=lambda item: item.published_hint, reverse=True)
    balanced: list[LatestCandidate] = []
    while len(balanced) < max(1, max_candidates):
        added = False
        for source in config.sources:
            items = grouped.get(source.source_id, [])
            if items:
                balanced.append(items.pop(0))
                added = True
                if len(balanced) >= max_candidates:
                    break
        if not added:
            break
    return balanced


def _collect_page(source: OfficialMediaSource, page_url: str) -> list[LatestCandidate]:
    body, _content_type = _fetch_url(page_url, timeout=20)
    parser = _LinkParser()
    parser.feed(body)
    results: list[tuple[int, LatestCandidate]] = []
    for index, (href, title) in enumerate(parser.links):
        url = urljoin(page_url, href).split("#", 1)[0]
        if len(title) < 6 or _looks_like_navigation(title) or not is_allowed_official_url(url, source.domains):
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.path in {"", "/"}:
            continue
        results.append(
            (
                index,
                LatestCandidate(
                    source.source_id,
                    source.name,
                    title[:240],
                    url,
                    page_url,
                    _date_hint_from_url(url),
                ),
            )
        )
    results.sort(key=lambda item: (item[1].published_hint, -item[0]), reverse=True)
    return [item for _index, item in results[:MAX_LINKS_PER_PAGE]]


def parse_schedule_times(value: str) -> tuple[time, ...]:
    parsed: set[time] = set()
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.add(datetime.strptime(item, "%H:%M").time())
        except ValueError as exc:
            raise ValueError(f"Invalid daily brief time {item!r}; expected HH:MM") from exc
    if not parsed:
        raise ValueError("At least one daily brief time is required")
    return tuple(sorted(parsed))


def next_scheduled_at(now: datetime, schedule: tuple[time, ...]) -> datetime:
    for item in schedule:
        candidate = now.replace(hour=item.hour, minute=item.minute, second=0, microsecond=0)
        if candidate > now:
            return candidate
    tomorrow = now + timedelta(days=1)
    first = schedule[0]
    return tomorrow.replace(hour=first.hour, minute=first.minute, second=0, microsecond=0)


def latest_scheduled_at(now: datetime, schedule: tuple[time, ...]) -> datetime:
    for item in reversed(schedule):
        candidate = now.replace(hour=item.hour, minute=item.minute, second=0, microsecond=0)
        if candidate <= now:
            return candidate
    yesterday = now - timedelta(days=1)
    last = schedule[-1]
    return yesterday.replace(hour=last.hour, minute=last.minute, second=0, microsecond=0)


def extract_urls(text: str) -> list[str]:
    return [match.rstrip(".,;:!?，。；：！？'") for match in URL_RE.findall(str(text or ""))]


def validate_brief_sources(answer: str, domains: Iterable[str], min_sources: int) -> list[str]:
    if NO_NEW_CONTENT_MARKER in answer:
        return []
    urls = list(dict.fromkeys(extract_urls(answer)))
    errors: list[str] = []
    invalid = [url for url in urls if not is_allowed_official_url(url, tuple(domains))]
    if invalid:
        errors.append("non-official URLs: " + ", ".join(invalid[:5]))
    official_count = len([url for url in urls if is_allowed_official_url(url, tuple(domains))])
    if official_count < min_sources:
        errors.append(f"only {official_count} official source URLs; require at least {min_sources}")
    return errors


class DailyBriefScheduler:
    def __init__(
        self,
        settings: Settings,
        sender: Callable[[str, str], Awaitable[str | None]],
        *,
        runtime_factory: Callable[[Settings], Any] | None = None,
        now_factory: Callable[[ZoneInfo], datetime] | None = None,
        brief_recorder: Callable[[str, datetime, str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.sender = sender
        self.runtime_factory = runtime_factory
        self.timezone = ZoneInfo(settings.daily_brief_timezone)
        self.schedule = parse_schedule_times(settings.daily_brief_times)
        self.now_factory = now_factory or (lambda timezone: datetime.now(timezone))
        self.brief_recorder = brief_recorder
        self._redis = get_redis_store(settings)
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.settings.daily_brief_enabled:
            logger.info("Daily brief scheduler is disabled")
            return
        if not self.settings.daily_brief_feishu_chat_id_list:
            logger.warning("Daily brief scheduler has no target Feishu chat IDs")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop(), name="shenlun-daily-brief")
            logger.info(
                "Daily brief scheduler started times=%s timezone=%s targets=%d",
                self.settings.daily_brief_times,
                self.settings.daily_brief_timezone,
                len(self.settings.daily_brief_feishu_chat_id_list),
            )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_now(
        self,
        chat_id: str,
        progress_reporter: DailyBriefProgressReporter | None = None,
    ) -> DailyBriefRunResult:
        return await self._run_safely(
            scheduled_for=self.now_factory(self.timezone),
            target_chat_ids=[chat_id],
            force=True,
            persist=False,
            progress_reporter=progress_reporter,
        )

    async def run_once(
        self,
        *,
        scheduled_for: datetime | None = None,
        target_chat_ids: list[str] | None = None,
        force: bool = False,
        persist: bool = True,
        progress_reporter: DailyBriefProgressReporter | None = None,
    ) -> DailyBriefRunResult:
        async with self._lock:
            progress = _progress_emitter(progress_reporter)
            progress(5, "正在初始化简报任务")
            now = scheduled_for or self.now_factory(self.timezone)
            if now.tzinfo is None:
                now = now.replace(tzinfo=self.timezone)
            slot_id = now.strftime("%Y-%m-%dT%H:%M")
            if not force and not self._redis.claim_once(
                "schedule-daily-brief",
                slot_id,
                ttl_seconds=max(3600, self.settings.daily_brief_catch_up_minutes * 60 + 3600),
            ):
                return DailyBriefRunResult("skipped", f"slot {slot_id} is already running or completed")
            state = self._load_state()
            if not force and slot_id in state.get("completed_slots", []):
                return DailyBriefRunResult("skipped", f"slot {slot_id} already completed")

            targets = target_chat_ids or self.settings.daily_brief_feishu_chat_id_list
            if not targets:
                return DailyBriefRunResult("skipped", "no target Feishu chat IDs configured")

            progress(12, "正在读取官媒来源配置")
            config = load_official_media_config(self.settings.resolved_daily_brief_source_file)
            prompt_template = self.settings.resolved_daily_brief_prompt_file.read_text(encoding="utf-8").strip()
            window_start = self._window_start(state, now)
            progress(20, "正在扫描官媒最新栏目", f"共 {len(config.sources)} 个来源")
            candidates = await asyncio.to_thread(
                collect_latest_candidates,
                config,
                self.settings.daily_brief_max_candidates,
            )
            seen_urls = set(state.get("seen_urls", {}))
            earliest_date = (window_start - timedelta(days=1)).date().isoformat()
            candidates = [
                item
                for item in candidates
                if _canonical_url(item.url) not in seen_urls
                and (not item.published_hint or item.published_hint >= earliest_date)
            ]
            progress(
                38,
                "官媒候选收集完成",
                f"获得 {len(candidates)} 条候选，覆盖 {len({item.source_id for item in candidates})} 个来源",
            )
            prompt = build_daily_brief_prompt(prompt_template, config, candidates, window_start, now)

            try:
                progress(45, "DeepSeek 正在核验候选和原文")
                model_progress = progress if progress_reporter is not None else None
                answer = await asyncio.to_thread(self._generate, prompt, config, now, model_progress)
            except Exception as exc:  # noqa: BLE001 - scheduled job must stay alive after one model failure
                logger.exception("Daily brief generation failed slot=%s", slot_id)
                return DailyBriefRunResult("failed", f"generation failed: {type(exc).__name__}: {exc}")

            progress(88, "正在校验官媒来源和引用", f"初稿约 {len(answer)} 字")
            validation_errors = validate_brief_sources(
                answer,
                config.domains,
                self.settings.daily_brief_min_sources,
            )
            if validation_errors:
                logger.warning("Daily brief source validation failed slot=%s errors=%s", slot_id, validation_errors)
                return DailyBriefRunResult("failed", "; ".join(validation_errors), answer)

            chunks = split_brief_cards(answer)
            progress(95, "正在发送飞书简报", f"共 {len(chunks)} 张卡片，发送至 {len(targets)} 个会话")
            for chat_id in targets:
                for chunk in chunks:
                    message_id = await self.sender(chat_id, chunk)
                    if not message_id:
                        return DailyBriefRunResult("failed", f"Feishu send failed for chat {chat_id}", answer)
                if persist and self.brief_recorder is not None:
                    try:
                        self.brief_recorder(chat_id, now, answer)
                    except Exception:  # noqa: BLE001 - notebook failure must not discard a delivered brief
                        logger.exception(
                            "Failed to archive delivered daily brief chat_suffix=%s",
                            chat_id[-8:] if chat_id else "unknown",
                        )

            if persist:
                self._record_success(state, slot_id, now, answer)
            progress(100, "简报已生成并发送")
            logger.info(
                "Daily brief delivered slot=%s candidates=%d sources=%d targets=%d",
                slot_id,
                len(candidates),
                len(set(extract_urls(answer))),
                len(targets),
            )
            return DailyBriefRunResult("sent", f"delivered to {len(targets)} chat(s)", answer)

    async def _run_loop(self) -> None:
        recent = latest_scheduled_at(self.now_factory(self.timezone), self.schedule)
        now = self.now_factory(self.timezone)
        if now - recent <= timedelta(minutes=self.settings.daily_brief_catch_up_minutes):
            await self._run_safely(scheduled_for=recent)
        while True:
            now = self.now_factory(self.timezone)
            target = next_scheduled_at(now, self.schedule)
            await asyncio.sleep(max(0.1, (target - now).total_seconds()))
            await self._run_safely(scheduled_for=target)

    async def _run_safely(self, **kwargs: Any) -> DailyBriefRunResult:
        try:
            return await self.run_once(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve future scheduled runs after one failure
            logger.exception("Daily brief run failed before completion")
            return DailyBriefRunResult("failed", f"{type(exc).__name__}: {exc}")

    def _generate(
        self,
        prompt: str,
        config: OfficialMediaConfig,
        now: datetime,
        progress: Callable[[int, str, str], None] | None = None,
    ) -> str:
        runtime_settings = self.settings.model_copy(
            update={
                "enable_tools": True,
                "tool_network_enabled": True,
                "tool_shell_enabled": False,
                "subagents_enabled": False,
                "agent_teams_enabled": False,
                "max_tool_calls": max(16, self.settings.max_tool_calls),
                "memory_llm_writer_enabled": False,
                "daily_quiz_enabled": False,
                "model": self.settings.daily_brief_model,
                "api_key": self.settings.resolved_daily_brief_api_key,
                "base_url": self.settings.resolved_daily_brief_base_url,
            }
        )
        tool_started = 0
        tool_completed = 0

        def report_tool(event: dict[str, Any]) -> None:
            nonlocal tool_started, tool_completed
            if progress is None:
                return
            event_type = str(event.get("type") or "")
            tool_name = str(event.get("tool") or "")
            if event_type == "tool_start":
                tool_started += 1
                progress(
                    min(74, 48 + tool_started * 3),
                    _daily_brief_tool_stage(tool_name),
                    f"正在执行第 {tool_started} 次资料操作",
                )
            elif event_type == "tool_result":
                tool_completed += 1
                status = "成功" if event.get("status") != "error" else "失败，正在调整"
                progress(
                    min(78, 50 + tool_completed * 3),
                    "DeepSeek 正在核验资料",
                    f"已完成 {tool_completed}/{max(tool_started, tool_completed)} 次，最近一次{status}",
                )

        def report_run(event: AgentRunEvent) -> None:
            if progress is None:
                return
            if event.type == "preparing_context":
                progress(47, "正在整理简报上下文")
            elif event.type == "agent_reasoning":
                detail = re.sub(r"\s+", " ", event.message).strip()
                progress(79, "DeepSeek 正在分析和归纳", detail[-220:])
            elif event.type == "agent_text":
                progress(83, "DeepSeek 正在撰写简报正文", f"已生成约 {len(event.message)} 字")

        runtime = (
            self.runtime_factory(runtime_settings)
            if self.runtime_factory
            else AgentRuntime(
                runtime_settings,
                tool_event_reporter=report_tool,
                run_event_reporter=report_run,
            )
        )
        thread_id = f"daily_brief_{now.strftime('%Y%m%d_%H%M')}"
        try:
            with official_media_web_scope(config.domains):
                invoke = getattr(runtime, "run_streaming", None) if progress is not None else None
                invoke = invoke or runtime.run
                result = invoke(prompt, user_id="system:shenlun-daily", thread_id=thread_id)
                if result.metadata.get("model_error"):
                    raise RuntimeError(result.metadata.get("model_error_message") or result.metadata["model_error"])
                answer = result.answer.strip()
                validation_errors = validate_brief_sources(
                    answer,
                    config.domains,
                    self.settings.daily_brief_min_sources,
                )
                if validation_errors:
                    if progress is not None:
                        progress(85, "引用校验未通过，DeepSeek 正在修订", "；".join(validation_errors[:3]))
                    correction = _source_correction_prompt(config, validation_errors)
                    result = invoke(correction, user_id="system:shenlun-daily", thread_id=thread_id)
                    if result.metadata.get("model_error"):
                        raise RuntimeError(result.metadata.get("model_error_message") or result.metadata["model_error"])
                    answer = result.answer.strip()
            if not answer:
                raise RuntimeError("model returned an empty daily brief")
            return answer
        finally:
            flush = getattr(getattr(runtime, "memory_queue", None), "flush", None)
            if callable(flush):
                flush()
            close = getattr(runtime, "close", None)
            if callable(close):
                close()

    def _window_start(self, state: dict[str, Any], now: datetime) -> datetime:
        raw = state.get("last_success_at")
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is not None and parsed < now:
                    return max(parsed.astimezone(self.timezone), now - timedelta(hours=self.settings.daily_brief_lookback_hours))
            except ValueError:
                pass
        return now - timedelta(hours=self.settings.daily_brief_lookback_hours)

    def _load_state(self) -> dict[str, Any]:
        path = self.settings.daily_brief_state_path
        if not path.exists():
            return {"completed_slots": [], "seen_urls": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"completed_slots": [], "seen_urls": {}}
        return value if isinstance(value, dict) else {"completed_slots": [], "seen_urls": {}}

    def _record_success(self, state: dict[str, Any], slot_id: str, now: datetime, answer: str) -> None:
        cutoff = now.astimezone(UTC) - timedelta(days=SEEN_URL_RETENTION_DAYS)
        seen = state.get("seen_urls", {})
        if not isinstance(seen, dict):
            seen = {}
        retained: dict[str, str] = {}
        for url, timestamp in seen.items():
            try:
                if datetime.fromisoformat(str(timestamp)).astimezone(UTC) >= cutoff:
                    retained[str(url)] = str(timestamp)
            except ValueError:
                continue
        for url in extract_urls(answer):
            retained[_canonical_url(url)] = now.astimezone(UTC).isoformat()
        slots = [str(item) for item in state.get("completed_slots", []) if str(item) >= cutoff.strftime("%Y-%m-%dT%H:%M")]
        slots.append(slot_id)
        state.update(
            {
                "last_success_at": now.isoformat(),
                "completed_slots": list(dict.fromkeys(slots)),
                "seen_urls": retained,
            }
        )
        _write_json_atomic(self.settings.daily_brief_state_path, state)


def build_daily_brief_prompt(
    template: str,
    config: OfficialMediaConfig,
    candidates: list[LatestCandidate],
    window_start: datetime,
    window_end: datetime,
) -> str:
    edition = "晨报" if window_end.hour < 13 else "晚报"
    sources = "\n".join(
        f"- {source.name}: {', '.join(source.domains)}" for source in config.sources
    )
    candidate_text = "\n".join(
        f"{index}. [{item.source_name}] {item.title}"
        f"{f'（URL 日期线索: {item.published_hint}）' if item.published_hint else ''}\n   {item.url}"
        for index, item in enumerate(candidates, start=1)
    ) or "（最新栏目暂未解析出候选链接，请仅按当前日期在允许的官媒域名内补充检索。）"
    return f"""{template}

<DailyBriefContext>
- edition: {edition}
- timezone: {window_end.tzinfo}
- window_start: {window_start.isoformat()}
- window_end: {window_end.isoformat()}
- current_date: {window_end.date().isoformat()}
</DailyBriefContext>

<OfficialMediaAllowlist>
{sources}
</OfficialMediaAllowlist>

<LatestPageCandidates>
以下条目按官媒最新栏目页发现，仅是候选，不代表发布时间已经通过核验：
{candidate_text}
</LatestPageCandidates>
"""


def split_brief_cards(text: str, max_chars: int = MAX_CARD_CHARS) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        addition = paragraph if not current else f"\n\n{paragraph}"
        if current and len(current) + len(addition) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current += addition
    if current:
        chunks.append(current)
    return chunks


def _source_correction_prompt(config: OfficialMediaConfig, errors: list[str]) -> str:
    domains = ", ".join(config.domains)
    details = "\n".join(f"- {error}" for error in errors)
    return f"""上一版简报未通过官媒来源校验：
{details}

请直接重写完整简报。只能保留已经核验且属于以下域名的内容和链接：{domains}。
不得新增未经工具核验的事实；每个采用的事实都保留官媒名称、标题、发布日期和原文链接。
如果无法满足最低来源数量，请改为明确输出“{NO_NEW_CONTENT_MARKER}”，不要用非官媒来源补足。
"""


def _normalize_domain(value: str) -> str:
    value = str(value or "").strip().lower().rstrip(".")
    if "://" in value:
        value = (urlparse(value).hostname or "").lower().rstrip(".")
    return value.removeprefix("www.")


def _canonical_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    return parsed._replace(fragment="").geturl().rstrip("/")


def _date_hint_from_url(url: str) -> str:
    value = str(url)
    patterns = (
        r"(?<!\d)(20\d{2})[/-]?(0[1-9]|1[0-2])[/-]?([0-2]\d|3[01])(?!\d)",
        r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])/([0-2]\d|3[01])(?!\d)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return "-".join(match.groups())
    return ""


def _looks_like_navigation(title: str) -> bool:
    normalized = re.sub(r"\s+", "", title).lower()
    exact = {
        "网站首页",
        "网站地图",
        "地方政府网站",
        "国务院部门网站",
        "驻港澳机构网站",
        "中国政府网微博、微信",
        "国务院客户端",
        "国务院客户端小程序",
    }
    return normalized in exact or any(
        marker in normalized
        for marker in ("版权所有", "备案号", "违法和不良信息举报", "客户端下载", "englishversion")
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temp_name = handle.name
    Path(temp_name).replace(path)


__all__ = [
    "DailyBriefProgress",
    "DailyBriefProgressReporter",
    "DailyBriefRunResult",
    "DailyBriefScheduler",
    "LatestCandidate",
    "OfficialMediaConfig",
    "OfficialMediaSource",
    "build_daily_brief_prompt",
    "collect_latest_candidates",
    "extract_urls",
    "latest_scheduled_at",
    "load_official_media_config",
    "next_scheduled_at",
    "parse_schedule_times",
    "split_brief_cards",
    "validate_brief_sources",
]
