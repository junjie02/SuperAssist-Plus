from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from superassist.channels.daily_quiz import DailyQuizScheduler, DailyQuizStore, make_daily_quiz_tool
from superassist.channels.store import FeishuThreadStore
from superassist.config import Settings
from superassist.teams.context import team_thread_context
from superassist.tools.files import read_file, write_file


def _settings(tmp_path, **values) -> Settings:
    defaults = {
        "SUPERASSIST_DATA_DIR": tmp_path / "data",
        "SUPERASSIST_EMBEDDING_PROVIDER": "hash",
        "SUPERASSIST_DAILY_BRIEF_FEISHU_CHAT_IDS": "chat_1",
        "SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT": 2,
        "SUPERASSIST_DAILY_QUIZ_NOTEBOOK_DAYS": 3,
    }
    defaults.update(values)
    return Settings(**defaults)


def _questions(*, wrong_id: str = "") -> list[dict[str, str]]:
    return [
        {
            "question": "根据日报中的政策部署，下列理解最准确的是？",
            "option_a": "只强调短期数量增长",
            "option_b": "统筹发展和安全并提升治理效能",
            "option_c": "取消基层公共服务",
            "option_d": "以单一指标替代综合评价",
            "correct_option": "B",
            "explanation": "材料体现系统观念，应统筹发展和安全；其他选项均把治理目标片面化。",
            "source_date": "2026-08-06",
            "source_title": "今日重要部署",
            "evidence": "坚持系统谋划，提升治理效能。",
            "wrong_question_id": wrong_id,
        },
        {
            "question": "材料强调协同治理，其核心要求是什么？",
            "option_a": "减少治理主体",
            "option_b": "只依赖临时行政命令",
            "option_c": "统筹主体、资源和全过程",
            "option_d": "取消社会参与",
            "correct_option": "C",
            "explanation": "协同治理强调多主体、资源和过程统筹，其他选项都削弱了协同性。",
            "source_date": "2026-08-05",
            "source_title": "基层治理观察",
            "evidence": "健全多元主体参与的基层治理体系。",
            "wrong_question_id": "",
        },
    ]


def _prepare(store: DailyQuizStore, now: datetime, *, wrong_id: str = "") -> None:
    store.start_session("chat_1", "thread_1", now)
    assert "structurally validated" in store.save_draft("thread_1", _questions(wrong_id=wrong_id))
    assert "finalized" in store.finalize("thread_1", "已逐题核对材料依据、唯一答案、干扰项和重复度。")


def test_notebook_keeps_three_calendar_days_and_renders_markdown(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(hour=19, minute=45, second=0, microsecond=0)

    for days_ago in (3, 2, 1, 0):
        store.archive_brief("chat_1", now - timedelta(days=days_ago), f"日报-{days_ago}")

    chat_dirs = list((settings.daily_quiz_data_dir / "chats").iterdir())
    notebook = (chat_dirs[0] / "日报笔记本.md").read_text(encoding="utf-8")
    assert "日报-3" not in notebook
    assert all(f"日报-{value}" in notebook for value in (0, 1, 2))


def test_complete_set_is_saved_reviewed_and_archived_with_answers(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    store.archive_brief("chat_1", now, "今日政策部署")
    store.start_session("chat_1", "thread_1", now)

    assert "structurally validated" in store.save_draft("thread_1", _questions())
    assert store.current_quiz_text("thread_1") == ""
    assert "finalized" in store.finalize("thread_1", "已逐题核验事实、证据、唯一答案和选项质量。")
    visible = store.current_quiz_text("thread_1")
    assert "第 1 题" in visible and "第 2 题" in visible
    assert "正确答案" not in visible

    archive = next((settings.daily_quiz_data_dir / "chats").glob("*/quizzes/*.md"))
    archived_text = archive.read_text(encoding="utf-8")
    assert "正确答案：B" in archived_text
    assert "复核记录" in archived_text


def test_batch_grading_updates_wrongbook_and_correct_review_marks_mastered(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    store.archive_brief("chat_1", now, "今日政策部署")
    _prepare(store, now)

    prompt = store.build_grading_prompt("thread_1", ["A", "C"])
    assert "<DailyPoliticalQuizGrading>" in prompt
    assert '"selected_option": "A"' in prompt
    assert '"correct_option": "B"' in prompt
    assert '"is_correct": false' in prompt
    assert "程序已经按保存的标准答案确定正误和分数" in prompt
    result = store.save_grading(
        "thread_1",
        [
            {
                "number": 1,
                "is_correct": True,
                "feedback": "主 Agent 判断该题理解不完整，应结合系统治理的完整含义作答。",
                "weakness": "系统治理概念掌握不完整",
            },
            {
                "number": 2,
                "is_correct": False,
                "feedback": "主 Agent 判断该题作答正确，能够识别多主体和全过程统筹。",
                "weakness": "",
            },
        ],
        "主 Agent 综合判断为一题正确、一题错误，需要复习系统治理。",
    )
    assert "Agent grading saved. Score: 1/2" in result
    chat_dir = next((settings.daily_quiz_data_dir / "chats").iterdir())
    wrong_data = json.loads((chat_dir / "wrong_questions.json").read_text(encoding="utf-8"))
    wrong_id = wrong_data["items"][0]["id"]
    assert wrong_data["items"][0]["wrong_count"] == 1
    session_path = next((settings.daily_quiz_data_dir / "sessions").glob("*.json"))
    completed = json.loads(session_path.read_text(encoding="utf-8"))
    assert completed["results"][0]["selected_option"] == "A"
    assert completed["results"][0]["correct_option"] == "B"
    assert completed["results"][0]["is_correct"] is False
    assert completed["results"][1]["is_correct"] is True
    assert completed["results"][0]["graded_by"] == "deterministic_answer_key"

    _prepare(store, now + timedelta(days=1), wrong_id=wrong_id)
    assert store.build_grading_prompt("thread_1", ["B", "C"])
    second_result = store.save_grading(
        "thread_1",
        [
            {
                "number": 1,
                "is_correct": True,
                "feedback": "主 Agent 判断复习题回答正确，已经掌握系统治理概念。",
                "weakness": "",
            },
            {
                "number": 2,
                "is_correct": True,
                "feedback": "主 Agent 判断回答正确，能够识别协同治理要求。",
                "weakness": "",
            },
        ],
        "主 Agent 综合判断两题均正确，本轮复习通过。",
    )
    assert "Agent grading saved. Score: 2/2" in second_result
    wrongbook = (chat_dir / "错题本.md").read_text(encoding="utf-8")
    assert "（已掌握）" in wrongbook
    assert f"~~{wrong_id}｜错 1 次~~" in wrongbook


def test_quiz_generation_materials_are_loaded_for_the_dedicated_agent(tmp_path) -> None:
    store = DailyQuizStore(_settings(tmp_path))
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    store.archive_brief("chat_1", now, "湖北推进基层治理创新。")
    prompt = store.prepare_generation("chat_1", "thread_1", now)

    assert "<QuizGenerationMaterials>" in prompt
    assert "题量：2" in prompt
    assert "湖北推进基层治理创新" in prompt
    assert "action=`draft`" not in prompt


def test_quiz_tool_prepares_generation_from_feishu_thread_mapping(tmp_path) -> None:
    settings = _settings(tmp_path)
    thread_store = FeishuThreadStore(settings.feishu_thread_store_path)
    thread_id = thread_store.get_or_create_thread_id(
        chat_id="chat_1",
        topic_id="topic_1",
        user_id="user_1",
    )
    store = DailyQuizStore(settings)
    store.archive_brief("chat_1", datetime.now(ZoneInfo("Asia/Shanghai")), "湖北推进基层治理创新。")
    tool = make_daily_quiz_tool(settings)

    with team_thread_context(thread_id):
        prompt = tool.invoke({"action": "prepare_generation"})

    assert "<QuizGenerationMaterials>" in prompt
    assert "湖北推进基层治理创新" in prompt
    session_path = next((settings.daily_quiz_data_dir / "sessions").glob("*.json"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["thread_id"] == thread_id
    assert session["status"] == "generating"


def test_delegated_question_count_becomes_session_validation_contract(tmp_path) -> None:
    settings = _settings(tmp_path, SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT=10)
    thread_store = FeishuThreadStore(settings.feishu_thread_store_path)
    thread_id = thread_store.get_or_create_thread_id(
        chat_id="chat_1",
        topic_id="topic_1",
        user_id="user_1",
    )
    store = DailyQuizStore(settings)
    store.archive_brief(
        "chat_1",
        datetime.now(ZoneInfo("Asia/Shanghai")),
        "湖北推进基层治理创新。",
    )
    tool = make_daily_quiz_tool(settings, delegated_question_count=5)

    with team_thread_context(thread_id):
        prompt = tool.invoke({"action": "prepare_generation"})

    assert "题量：5" in prompt
    session_path = next((settings.daily_quiz_data_dir / "sessions").glob("*.json"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["question_count"] == 5
    assert "expected exactly 5 questions, received 2" in store.save_draft(
        thread_id,
        _questions(),
    )


def test_main_agent_tool_loads_grading_context_from_an_ordinary_answer_turn(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    _prepare(store, now)
    tool = make_daily_quiz_tool(settings)

    with team_thread_context("thread_1"):
        context = tool.invoke({"action": "grading_context", "answers": ["B", "C"]})

    assert "<DailyPoliticalQuizGrading>" in context
    assert '"selected_option": "B"' in context
    assert '"correct_option": "B"' in context
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) == {
        "action",
        "question_count",
        "questions",
        "review_summary",
        "answers",
        "results",
        "overall_feedback",
    }
    assert "prepare_generation" in schema["properties"]["action"]["enum"]
    assert "public_view" in schema["properties"]["action"]["enum"]
    assert "grading_context" in schema["properties"]["action"]["enum"]


def test_current_quiz_resources_are_thread_scoped_read_only_and_private_after_submission(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = DailyQuizStore(settings)
    _prepare(store, datetime.now(ZoneInfo("Asia/Shanghai")))
    monkeypatch.setattr("superassist.tools.files.get_settings", lambda: settings)
    current_dir = settings.daily_quiz_data_dir / "current"
    for path in current_dir.glob("*"):
        path.unlink()

    with team_thread_context("thread_1"):
        public = read_file.invoke({"path": "quiz://current/public"})
        assert "第 1 题" in public
        assert "正确答案" not in public
        assert list(current_dir.glob("*.public.md"))
        with pytest.raises(PermissionError, match="complete answer sheet"):
            read_file.invoke({"path": "quiz://current/private"})
        with pytest.raises(PermissionError, match="read-only"):
            write_file.invoke({"path": "quiz://current/public", "content": "replace"})

        assert store.build_grading_prompt("thread_1", ["A", "C"])
        private = json.loads(read_file.invoke({"path": "quiz://current/private"}))
        assert private["questions"][0]["correct_option"] == "B"
        assert private["submitted_answers"] == ["A", "C"]

    with team_thread_context("thread_2"):
        with pytest.raises(FileNotFoundError, match="No quiz"):
            read_file.invoke({"path": "quiz://current/public"})


def test_ten_question_set_requires_exact_count_and_review_before_activation(tmp_path) -> None:
    settings = _settings(tmp_path, SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT=10)
    store = DailyQuizStore(settings)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    store.start_session("chat_1", "thread_1", now)
    answers = "ABCDABCDAB"
    questions = []
    for index, answer in enumerate(answers, start=1):
        questions.append(
            {
                "question": f"第 {index} 项材料体现的政策要求是什么？",
                "option_a": f"选项 A-{index}",
                "option_b": f"选项 B-{index}",
                "option_c": f"选项 C-{index}",
                "option_d": f"选项 D-{index}",
                "correct_option": answer,
                "explanation": f"第 {index} 项材料能够证明选项 {answer}，其他选项与材料不符。",
                "source_date": "2026-08-06",
                "source_title": f"日报材料 {index}",
                "evidence": f"这是第 {index} 项直接材料依据。",
            }
        )

    assert "expected exactly 10" in store.save_draft("thread_1", questions[:9])
    assert "structurally validated" in store.save_draft("thread_1", questions)
    assert "review_summary" in store.finalize("thread_1", "检查完成")
    assert "finalized" in store.finalize("thread_1", "已逐题核对十道题的事实依据、唯一答案、干扰项和重复度。")
    assert store.current_quiz_text("thread_1").count("政治理论·单项选择") == 10


def test_scheduler_deduplicates_1700_slot(tmp_path) -> None:
    calls: list[tuple[str, datetime]] = []

    async def starter(chat_id: str, now: datetime) -> str:
        calls.append((chat_id, now))
        return "started"

    settings = _settings(tmp_path, SUPERASSIST_DAILY_QUIZ_TIME="17:00")
    scheduler = DailyQuizScheduler(settings, starter)
    scheduled = datetime(2026, 8, 2, 17, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    first = asyncio.run(scheduler.run_once(scheduled))
    second = asyncio.run(scheduler.run_once(scheduled))

    assert first == ["started"]
    assert second == []
    assert calls == [("chat_1", scheduled)]


def test_quiz_scheduler_and_main_agent_context_have_independent_switches(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        SUPERASSIST_DAILY_QUIZ_ENABLED=False,
        SUPERASSIST_DAILY_QUIZ_SCHEDULER_ENABLED=False,
        SUPERASSIST_DAILY_QUIZ_CONTEXT_ENABLED=True,
    )

    assert settings.resolved_daily_quiz_scheduler_enabled is False
    assert settings.daily_quiz_context_enabled is True
