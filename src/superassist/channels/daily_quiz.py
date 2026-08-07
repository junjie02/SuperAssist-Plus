from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool

from superassist.config import Settings
from superassist.teams.context import current_team_thread_id

logger = logging.getLogger(__name__)

QUIZ_OPTIONS = ("A", "B", "C", "D")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temp_name = handle.name
    Path(temp_name).replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(value)
        temp_name = handle.name
    Path(temp_name).replace(path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return value if isinstance(value, dict) else dict(default)


class DailyQuizStore:
    """Persistent three-day notebook, quiz sessions, and reviewable wrong answers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.daily_quiz_data_dir
        self.timezone = ZoneInfo(settings.daily_brief_timezone)
        self._lock = threading.RLock()

    def archive_brief(self, chat_id: str, delivered_at: datetime, content: str) -> None:
        when = self._local_datetime(delivered_at)
        chat_dir = self._chat_dir(chat_id)
        path = chat_dir / "briefs.json"
        with self._lock:
            data = _read_json(path, {"chat_id": chat_id, "briefs": []})
            cutoff = when.date() - timedelta(days=self.settings.daily_quiz_notebook_days - 1)
            records = [item for item in data.get("briefs", []) if self._entry_date(item) >= cutoff]
            entry_id = when.strftime("%Y%m%d_%H%M")
            edition = "晨报" if when.hour < 13 else "晚报"
            records = [item for item in records if str(item.get("id")) != entry_id]
            records.append(
                {
                    "id": entry_id,
                    "date": when.date().isoformat(),
                    "edition": edition,
                    "delivered_at": when.isoformat(),
                    "content": str(content or "").strip(),
                }
            )
            records.sort(key=lambda item: str(item.get("delivered_at") or ""))
            data = {"chat_id": chat_id, "updated_at": when.isoformat(), "briefs": records}
            _atomic_json(path, data)
            _atomic_text(chat_dir / "日报笔记本.md", self._render_notebook(records, when.date()))

    def has_notebook(self, chat_id: str) -> bool:
        return bool(self._notebook_records(chat_id))

    def start_session(
        self,
        chat_id: str,
        thread_id: str,
        now: datetime,
        *,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            existing = self._load_session(thread_id)
            if (
                not replace_existing
                and existing.get("version") == 2
                and existing.get("status") in {"generating", "active"}
                and existing.get("chat_id") == chat_id
            ):
                return existing
            when = self._local_datetime(now)
            session = {
                "version": 2,
                "session_id": when.strftime("quiz_%Y%m%d_%H%M%S"),
                "chat_id": chat_id,
                "thread_id": thread_id,
                "status": "generating",
                "started_at": when.isoformat(),
                "updated_at": when.isoformat(),
                "question_count": self.settings.daily_quiz_question_count,
                "questions": [],
                "review_summary": "",
                "answered_count": 0,
                "correct_count": 0,
                "results": [],
                "question_message_id": "",
            }
            self._save_session(session)
            return session

    def active_session(self, thread_id: str) -> dict[str, Any] | None:
        session = self._load_session(thread_id)
        return session if session.get("status") == "active" else None

    def is_active_reply(self, chat_id: str, root_id: str | None) -> bool:
        if not root_id:
            return False
        for path in self._sessions_dir().glob("*.json"):
            session = _read_json(path, {})
            if (
                session.get("status") == "active"
                and session.get("chat_id") == chat_id
                and session.get("question_message_id") == root_id
            ):
                return True
        return False

    def set_question_message_id(self, thread_id: str, message_id: str | None) -> None:
        if not message_id:
            return
        with self._lock:
            session = self._load_session(thread_id)
            if session.get("status") != "active":
                return
            session["question_message_id"] = message_id
            session["updated_at"] = datetime.now(UTC).isoformat()
            self._save_session(session)

    def current_quiz_text(self, thread_id: str) -> str:
        session = self.active_session(thread_id)
        questions = session.get("questions") if session else None
        return self._render_question_set(questions) if isinstance(questions, list) and questions else ""

    def build_start_prompt(self, chat_id: str, thread_id: str, now: datetime) -> str:
        session = self.start_session(chat_id, thread_id, now, replace_existing=True)
        return self._build_agent_context(session)

    def save_draft(self, thread_id: str, questions: list[dict[str, Any]]) -> str:
        with self._lock:
            session = self._load_session(thread_id)
            if session.get("status") != "generating":
                return "Error: no quiz set is being generated"
            try:
                normalized = self._validate_question_set(questions, int(session.get("question_count") or 0))
            except ValueError as exc:
                return f"Error: quiz draft validation failed: {exc}"
            session["questions"] = normalized
            session["updated_at"] = datetime.now(UTC).isoformat()
            self._save_session(session)
            return (
                "Draft saved and structurally validated. Now review every item against its evidence, confirm exactly one "
                "best answer, remove duplicates and misleading wording, then call action=finalize with a review summary."
            )

    def finalize(self, thread_id: str, review_summary: str) -> str:
        with self._lock:
            session = self._load_session(thread_id)
            if session.get("status") != "generating" or not session.get("questions"):
                return "Error: save a valid quiz draft before finalizing"
            summary = str(review_summary or "").strip()
            if len(summary) < 12:
                return "Error: review_summary must describe the completed factual and single-answer checks"
            try:
                session["questions"] = self._validate_question_set(
                    list(session["questions"]), int(session.get("question_count") or 0)
                )
            except ValueError as exc:
                return f"Error: final quiz validation failed: {exc}"
            session["review_summary"] = summary
            session["status"] = "active"
            session["finalized_at"] = datetime.now(UTC).isoformat()
            session["updated_at"] = session["finalized_at"]
            self._save_session(session)
            self._archive_quiz(session)
            return f"Quiz set finalized: {len(session['questions'])} questions and answer explanations saved."

    def build_grading_prompt(self, thread_id: str, answers: list[str]) -> str:
        with self._lock:
            session = self._load_session(thread_id)
            questions = list(session.get("questions") or [])
            if session.get("status") != "active" or not questions:
                return ""
            if len(answers) != len(questions) or any(answer not in QUIZ_OPTIONS for answer in answers):
                return ""
            session["submitted_answers"] = list(answers)
            session["updated_at"] = datetime.now(UTC).isoformat()
            self._save_session(session)
            grading_items = [
                {**question, "selected_option": selected}
                for question, selected in zip(questions, answers, strict=True)
            ]
            return f"""<DailyPoliticalQuizGrading>
你是当前主 Agent，必须亲自批改用户一次提交的整套政治理论测验。程序不会比较答案，也不会替你判分。

题目、内部标准答案、解析、材料依据和用户答案如下：
{json.dumps(grading_items, ensure_ascii=False, indent=2)}

批改流程：
1. 逐题查看 selected_option、correct_option、explanation 和 evidence，由你判断用户答案是否正确。
2. 为每题形成结果对象：number、is_correct、feedback、weakness。feedback 要解释正确依据及主要干扰项；答错时 weakness 要指出具体薄弱点，答对时可为空。
3. 调用 `daily_quiz_update`，action=`grade`，一次提交全部 results，并填写 overall_feedback。工具只保存你的判断并据此更新错题本，不会重新比较答案。
4. 工具保存成功后，由你输出完整批改报告：总分、逐题用户答案、标准答案、正误、解析，最后总结薄弱点和复习建议。
5. 不得跳过工具调用，不得只给答案表，也不得生成下一组题。
</DailyPoliticalQuizGrading>"""

    def save_grading(self, thread_id: str, results: list[dict[str, Any]], overall_feedback: str) -> str:
        with self._lock:
            session = self._load_session(thread_id)
            questions = list(session.get("questions") or [])
            answers = list(session.get("submitted_answers") or [])
            if session.get("status") != "active" or not questions or len(answers) != len(questions):
                return "Error: no complete answer sheet is awaiting Agent grading"
            if len(results) != len(questions):
                return f"Error: expected exactly {len(questions)} grading results, received {len(results)}"
            by_number: dict[int, dict[str, Any]] = {}
            for item in results:
                if not isinstance(item, dict):
                    return "Error: every grading result must be an object"
                try:
                    number = int(item.get("number"))
                except (TypeError, ValueError):
                    return "Error: every grading result requires a valid question number"
                if number in by_number or number < 1 or number > len(questions):
                    return f"Error: invalid or duplicate grading result number {number}"
                if not isinstance(item.get("is_correct"), bool):
                    return f"Error: result {number} requires Agent judgement is_correct=true/false"
                feedback = str(item.get("feedback") or "").strip()
                if not feedback:
                    return f"Error: result {number} requires explanatory feedback"
                by_number[number] = {
                    "is_correct": item["is_correct"],
                    "feedback": feedback,
                    "weakness": str(item.get("weakness") or "").strip(),
                }
            summary = str(overall_feedback or "").strip()
            if len(summary) < 8:
                return "Error: overall_feedback must summarize the Agent's grading assessment"

            now = datetime.now(UTC).isoformat()
            saved_results: list[dict[str, Any]] = []
            for question, selected in zip(questions, answers, strict=True):
                number = int(question.get("number") or 0)
                judgement = by_number[number]
                result = {
                    "number": number,
                    "question": question.get("question"),
                    "selected_option": selected,
                    "correct_option": question.get("correct_option"),
                    "is_correct": judgement["is_correct"],
                    "feedback": judgement["feedback"],
                    "weakness": judgement["weakness"],
                    "answered_at": now,
                    "graded_by": "main_agent",
                }
                saved_results.append(result)
                self._update_wrongbook(
                    str(session.get("chat_id") or ""),
                    question,
                    judgement["is_correct"],
                    selected,
                    judgement["feedback"],
                    judgement["weakness"],
                )

            correct_count = sum(1 for item in saved_results if item["is_correct"])
            session["results"] = saved_results
            session["grading_summary"] = summary
            session["answered_count"] = len(questions)
            session["correct_count"] = correct_count
            session["status"] = "completed"
            session["completed_at"] = now
            session["updated_at"] = now
            session["question_message_id"] = ""
            self._save_session(session)
            self._archive_quiz(session)
            return f"Agent grading saved. Score: {correct_count}/{len(questions)}. Now present the full grading report."

    def _build_agent_context(self, session: dict[str, Any]) -> str:
        count = int(session.get("question_count") or self.settings.daily_quiz_question_count)
        return f"""<DailyPoliticalQuiz>
你是当前主 Agent。请基于近几日日报和错题本，一次性生成 {count} 道 A、B、C、D 四选一政治理论题，并保存答案。

执行流程：
1. 先完整生成 {count} 题草稿，每题必须包含 question、option_a、option_b、option_c、option_d、correct_option、explanation、source_date、source_title、evidence、wrong_question_id。
2. 调用 `daily_quiz_update`，action=`draft`，一次提交全部 questions。工具会检查题量、字段、重复题、选项和答案分布；若失败，修正后重新提交草稿。
3. 草稿保存成功后，逐题对照 evidence 做第二遍检查：事实和答案是否一致、是否只有一个最佳答案、干扰项是否同层级、题目是否重复、解析是否足以证明答案。发现问题时修正整套题并重新调用 action=`draft`。
4. 检查完成后调用 action=`finalize`，填写具体 review_summary。只有 finalize 后题目才会发送给用户。
5. 最终回复只说明整套题已生成并检查，不展示题面、答案或解析；飞书通道会从保存状态渲染题目。

出题要求：
- 重点测查党的创新理论、方针政策和材料体现的治理逻辑，不考日期、人名、地名等新闻细枝末节。
- 四个选项必须完整、同层级、有区分度，且只能有一个最佳答案。
- 题目之间不得重复或仅替换措辞；正确选项应合理分布。
- 优先针对 active 错题生成变式题，填写 wrong_question_id，但同一道错题在本组最多复习一次。
- explanation 必须说明正确依据及主要干扰项为什么错误；evidence 必须摘自给定材料，禁止编造来源。

{self._study_material(session)}
</DailyPoliticalQuiz>"""

    def _study_material(self, session: dict[str, Any]) -> str:
        chat_id = str(session.get("chat_id") or "")
        records = self._notebook_records(chat_id)
        active_wrong = self._wrong_items(chat_id, active_only=True)
        notebook = self._render_notebook(records, self._local_datetime(None).date())
        wrongbook = self._render_wrongbook(self._wrong_items(chat_id), include_heading=False)
        return f"""总题量：{session.get('question_count')} 题。覆盖近几日日报的主要政策理论，并优先复习未掌握错题。
当前 active 错题数：{len(active_wrong)}。

<ThreeDayNotebook>
{notebook}
</ThreeDayNotebook>

<WrongQuestionNotebook>
{wrongbook}
</WrongQuestionNotebook>"""

    def _validate_question_set(self, questions: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
        if len(questions) != expected_count:
            raise ValueError(f"expected exactly {expected_count} questions, received {len(questions)}")
        normalized: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        seen_wrong_ids: set[str] = set()
        for number, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"question {number} must be an object")
            question = str(item.get("question") or "").strip()
            raw_options = item.get("options") if isinstance(item.get("options"), dict) else {}
            options = {
                key: str(raw_options.get(key) or item.get(f"option_{key.lower()}") or "").strip()
                for key in QUIZ_OPTIONS
            }
            correct = str(item.get("correct_option") or "").strip().upper()
            explanation = str(item.get("explanation") or "").strip()
            source_date = str(item.get("source_date") or "").strip()
            source_title = str(item.get("source_title") or "").strip()
            evidence = str(item.get("evidence") or "").strip()[:4000]
            wrong_id = str(item.get("wrong_question_id") or "").strip()
            if not question or not explanation or not source_date or not source_title or not evidence:
                raise ValueError(f"question {number} is missing question, explanation, source, or evidence")
            if correct not in QUIZ_OPTIONS or any(not value for value in options.values()):
                raise ValueError(f"question {number} must have four options and one A/B/C/D answer")
            if len({value.casefold() for value in options.values()}) != 4:
                raise ValueError(f"question {number} contains duplicate options")
            key = re.sub(r"\W+", "", question).casefold()
            if key in seen_questions:
                raise ValueError(f"question {number} duplicates another question")
            seen_questions.add(key)
            if wrong_id:
                if wrong_id in seen_wrong_ids:
                    raise ValueError(f"wrong question {wrong_id} is reviewed more than once")
                seen_wrong_ids.add(wrong_id)
            normalized.append(
                {
                    "number": number,
                    "question": question,
                    "options": options,
                    "correct_option": correct,
                    "explanation": explanation,
                    "source_date": source_date,
                    "source_title": source_title,
                    "evidence": evidence,
                    "wrong_question_id": wrong_id,
                }
            )
        counts = Counter(item["correct_option"] for item in normalized)
        if expected_count >= 6 and (len(counts) < 3 or max(counts.values()) > (expected_count + 1) // 2):
            raise ValueError("correct answers must use at least three option letters without excessive concentration")
        return normalized

    def _update_wrongbook(
        self,
        chat_id: str,
        question: dict[str, Any],
        is_correct: bool,
        selected: str,
        feedback: str,
        weakness: str,
    ) -> None:
        chat_dir = self._chat_dir(chat_id)
        path = chat_dir / "wrong_questions.json"
        data = _read_json(path, {"chat_id": chat_id, "items": []})
        items = list(data.get("items") or [])
        linked_id = str(question.get("wrong_question_id") or "")
        linked = next((item for item in items if str(item.get("id")) == linked_id), None)
        now = datetime.now(UTC).isoformat()
        if linked is not None:
            linked["last_question"] = question.get("question")
            linked["last_selected_option"] = selected
            linked["last_review_at"] = now
            if is_correct:
                linked["status"] = "mastered"
                linked["mastered_at"] = now
            else:
                linked["status"] = "active"
                linked["wrong_count"] = int(linked.get("wrong_count") or 0) + 1
                linked["last_wrong_at"] = now
                linked["weakness"] = str(weakness or linked.get("weakness") or "").strip()
        elif not is_correct:
            raw = f"{question.get('question')}|{question.get('source_date')}|{now}"
            item_id = "W-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10].upper()
            items.append(
                {
                    "id": item_id,
                    "status": "active",
                    "question": question.get("question"),
                    "options": question.get("options"),
                    "correct_option": question.get("correct_option"),
                    "explanation": question.get("explanation"),
                    "source_date": question.get("source_date"),
                    "source_title": question.get("source_title"),
                    "evidence": question.get("evidence"),
                    "wrong_count": 1,
                    "last_selected_option": selected,
                    "feedback": str(feedback).strip(),
                    "weakness": str(weakness).strip(),
                    "created_at": now,
                    "last_wrong_at": now,
                }
            )
        data = {"chat_id": chat_id, "updated_at": now, "items": items}
        _atomic_json(path, data)
        _atomic_text(chat_dir / "错题本.md", self._render_wrongbook(items))

    def _notebook_records(self, chat_id: str) -> list[dict[str, Any]]:
        data = _read_json(self._chat_dir(chat_id) / "briefs.json", {"briefs": []})
        cutoff = self._local_datetime(None).date() - timedelta(days=self.settings.daily_quiz_notebook_days - 1)
        return [item for item in data.get("briefs", []) if self._entry_date(item) >= cutoff]

    def _wrong_items(self, chat_id: str, *, active_only: bool = False) -> list[dict[str, Any]]:
        data = _read_json(self._chat_dir(chat_id) / "wrong_questions.json", {"items": []})
        items = [item for item in data.get("items", []) if isinstance(item, dict)]
        return [item for item in items if item.get("status") == "active"] if active_only else items

    @staticmethod
    def _render_question_set(questions: list[dict[str, Any]]) -> str:
        sections = []
        for question in questions:
            options = question.get("options") or {}
            sections.append(
                f"**第 {question.get('number')} 题（政治理论·单项选择）**\n\n"
                f"{question.get('question')}\n\n"
                + "\n".join(f"{key}. {options.get(key, '')}" for key in QUIZ_OPTIONS)
            )
        total = len(questions)
        return (
            "# 每日政治理论测验\n\n"
            + "\n\n---\n\n".join(sections)
            + f"\n\n---\n\n请一次性提交全部 {total} 个答案，例如：`1A 2B 3C ... {total}D`，也可以直接回复连续字母。"
        )

    def _archive_quiz(self, session: dict[str, Any]) -> None:
        chat_dir = self._chat_dir(str(session.get("chat_id") or ""))
        archive_dir = chat_dir / "quizzes"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_id = str(session.get("session_id") or "quiz")
        _atomic_json(archive_dir / f"{archive_id}.json", session)
        _atomic_text(archive_dir / f"{archive_id}.md", self._render_quiz_archive(session))

    @staticmethod
    def _render_quiz_archive(session: dict[str, Any]) -> str:
        lines = [
            "# 每日政治理论测验与答案",
            "",
            f"- 测验编号：{session.get('session_id')}",
            f"- 状态：{session.get('status')}",
            f"- 复核记录：{session.get('review_summary') or '未填写'}",
            f"- 主 Agent 评分总结：{session.get('grading_summary') or '尚未作答'}",
            "",
            "## 题目与答案",
            "",
        ]
        result_by_number = {int(item.get("number") or 0): item for item in session.get("results") or []}
        for question in session.get("questions") or []:
            number = int(question.get("number") or 0)
            options = question.get("options") or {}
            lines.extend(
                [
                    f"### 第 {number} 题",
                    "",
                    str(question.get("question") or ""),
                    "",
                    *(f"{key}. {options.get(key, '')}" for key in QUIZ_OPTIONS),
                    "",
                    f"- 正确答案：{question.get('correct_option')}",
                    f"- 解析：{question.get('explanation')}",
                    f"- 来源：{question.get('source_date')}｜{question.get('source_title')}",
                    f"- 依据：{question.get('evidence')}",
                ]
            )
            result = result_by_number.get(number)
            if result:
                lines.append(f"- 用户答案：{result.get('selected_option')}（{'正确' if result.get('is_correct') else '错误'}）")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_notebook(records: list[dict[str, Any]], today: date) -> str:
        lines = ["# 近三日申论日报笔记本", "", f"> 滑动窗口更新日期：{today.isoformat()}", ""]
        if not records:
            return "\n".join(lines + ["暂无日报记录。"])
        for item in sorted(records, key=lambda value: str(value.get("delivered_at") or ""), reverse=True):
            lines.extend(
                [
                    f"## {item.get('date')} {item.get('edition')}",
                    "",
                    str(item.get("content") or "").strip(),
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_wrongbook(items: list[dict[str, Any]], *, include_heading: bool = True) -> str:
        lines = ["# 政治理论错题本", ""] if include_heading else []
        if not items:
            return "\n".join(lines + ["暂无错题。"])
        for item in sorted(items, key=lambda value: str(value.get("created_at") or ""), reverse=True):
            mastered = item.get("status") == "mastered"
            title = f"{item.get('id')}｜错 {item.get('wrong_count', 1)} 次"
            lines.append(f"## ~~{title}~~（已掌握）" if mastered else f"## {title}（待复习）")
            lines.extend(
                [
                    "",
                    f"- 来源：{item.get('source_date') or '未知'}｜{item.get('source_title') or '未标注'}",
                    f"- 题目：{item.get('question') or ''}",
                    f"- 正确选项：{item.get('correct_option') or ''}",
                    f"- 薄弱点：{item.get('weakness') or '待归纳'}",
                    f"- 解析：{item.get('explanation') or ''}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _entry_date(item: dict[str, Any]) -> date:
        try:
            return date.fromisoformat(str(item.get("date") or ""))
        except ValueError:
            return date.min

    def _local_datetime(self, value: datetime | None) -> datetime:
        current = value or datetime.now(self.timezone)
        if current.tzinfo is None:
            return current.replace(tzinfo=self.timezone)
        return current.astimezone(self.timezone)

    def _chat_dir(self, chat_id: str) -> Path:
        key = hashlib.sha256(str(chat_id).encode("utf-8")).hexdigest()[:16]
        return self.root / "chats" / key

    def _sessions_dir(self) -> Path:
        path = self.root / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _session_path(self, thread_id: str) -> Path:
        key = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()[:20]
        return self._sessions_dir() / f"{key}.json"

    def _load_session(self, thread_id: str) -> dict[str, Any]:
        return _read_json(self._session_path(thread_id), {})

    def _save_session(self, session: dict[str, Any]) -> None:
        _atomic_json(self._session_path(str(session.get("thread_id") or "")), session)


_STORE_LOCK = threading.Lock()
_STORES: dict[str, DailyQuizStore] = {}


def get_daily_quiz_store(settings: Settings) -> DailyQuizStore:
    key = str(settings.daily_quiz_data_dir.resolve())
    with _STORE_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = DailyQuizStore(settings)
            _STORES[key] = store
        return store


def make_daily_quiz_tool(settings: Settings) -> BaseTool:
    store = get_daily_quiz_store(settings)

    @tool("daily_quiz_update")
    def daily_quiz_update(
        action: Literal["draft", "finalize", "grading_context", "grade"],
        questions: list[dict[str, Any]] | None = None,
        review_summary: str = "",
        answers: list[str] | None = None,
        results: list[dict[str, Any]] | None = None,
        overall_feedback: str = "",
    ) -> str:
        """Save and review a quiz set, or persist the main Agent's grading decisions.

        Use action=draft with every question in one call. Each question must contain
        question, option_a through option_d, correct_option, explanation, source_date,
        source_title, evidence, and optional wrong_question_id. After reviewing the saved
        draft, use action=finalize with a concrete review_summary. When the user answers
        in ordinary conversation, use action=grading_context with the complete ordered
        answers to load the saved questions, answer key, explanations, and evidence. Then
        use action=grade with one result per question containing number, is_correct,
        feedback, and weakness, plus overall_feedback. The tool persists the Agent's
        judgement without re-grading it.
        """

        thread_id = current_team_thread_id()
        if not thread_id:
            return "Error: quiz tool requires an active conversation thread"
        if action == "grading_context":
            context = store.build_grading_prompt(thread_id, list(answers or []))
            return context or "Error: no active quiz matches that complete answer sheet"
        if action == "grade":
            return store.save_grading(thread_id, list(results or []), overall_feedback)
        if action == "finalize":
            return store.finalize(thread_id, review_summary)
        return store.save_draft(thread_id, list(questions or []))

    return daily_quiz_update


class DailyQuizScheduler:
    def __init__(
        self,
        settings: Settings,
        starter: Callable[[str, datetime], Awaitable[str]],
        *,
        now_factory: Callable[[ZoneInfo], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.starter = starter
        self.timezone = ZoneInfo(settings.daily_brief_timezone)
        self.quiz_time = _parse_quiz_time(settings.daily_quiz_time)
        self.now_factory = now_factory or (lambda timezone: datetime.now(timezone))
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.settings.daily_quiz_enabled or not self.settings.daily_brief_feishu_chat_id_list:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop(), name="shenlun-daily-quiz")
            logger.info("Daily quiz scheduler started time=%s", self.settings.daily_quiz_time)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_now(self, chat_id: str) -> str:
        return await self._run_chat(chat_id, self.now_factory(self.timezone))

    async def run_once(self, scheduled_for: datetime | None = None, *, force: bool = False) -> list[str]:
        async with self._lock:
            now = self._local(scheduled_for or self.now_factory(self.timezone))
            slot = now.strftime("%Y-%m-%dT%H:%M")
            state = _read_json(self.settings.daily_quiz_scheduler_state_path, {"completed_slots": []})
            if not force and slot in state.get("completed_slots", []):
                return []
            results = [await self._run_chat(chat_id, now) for chat_id in self.settings.daily_brief_feishu_chat_id_list]
            state["completed_slots"] = [slot]
            state["last_run_at"] = now.isoformat()
            _atomic_json(self.settings.daily_quiz_scheduler_state_path, state)
            return results

    async def _run_chat(self, chat_id: str, now: datetime) -> str:
        try:
            return await self.starter(chat_id, now)
        except Exception as exc:  # noqa: BLE001 - one chat must not stop future schedules
            logger.exception("Daily quiz start failed chat_suffix=%s", chat_id[-8:])
            return f"failed: {type(exc).__name__}: {exc}"

    async def _run_loop(self) -> None:
        now = self.now_factory(self.timezone)
        recent = datetime.combine(now.astimezone(self.timezone).date(), self.quiz_time, self.timezone)
        if timedelta(0) <= now - recent <= timedelta(minutes=self.settings.daily_brief_catch_up_minutes):
            await self.run_once(recent)
        while True:
            now = self.now_factory(self.timezone)
            target = _next_quiz_at(now, self.quiz_time, self.timezone)
            await asyncio.sleep(max(0.1, (target - now).total_seconds()))
            await self.run_once(target)

    def _local(self, value: datetime) -> datetime:
        return value.replace(tzinfo=self.timezone) if value.tzinfo is None else value.astimezone(self.timezone)


def _parse_quiz_time(value: str) -> time:
    try:
        return time.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError("Daily quiz time must use HH:MM format") from exc


def _next_quiz_at(now: datetime, quiz_time: time, timezone: ZoneInfo) -> datetime:
    local = now.astimezone(timezone)
    candidate = datetime.combine(local.date(), quiz_time, timezone)
    return candidate if candidate > local else candidate + timedelta(days=1)


__all__ = [
    "DailyQuizScheduler",
    "DailyQuizStore",
    "get_daily_quiz_store",
    "make_daily_quiz_tool",
]
