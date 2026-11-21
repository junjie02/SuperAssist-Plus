from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from superassist.config import Settings
from superassist.shenlun_grading import ShenlunGradingService, _validate_grader_result


def _ready_batch(service: ShenlunGradingService, *, thread_id: str = "thread-1") -> dict[str, object]:
    reservation = service.reserve(user_scope="user-1", thread_id=thread_id, question_count=5)
    workspace = service.workspace_root
    for number, relative in enumerate(reservation["question_paths"], start=1):
        (workspace / str(relative)).write_text(
            f"# 第{number}题\n\n## 基本信息\n\n- 原试卷题号：第{number + 1}题\n- 题目信息状态：完整\n\n"
            f"## 题干\n\n测试题干{number}。\n\n## 相关材料\n\n测试材料。\n\n"
            "## 考生答案逐字转写\n\n- 状态：已提供\n\n测试答案。\n",
            encoding="utf-8",
        )
    types = ["summary", "analysis", "countermeasure", "document", "essay"]
    manifest = {
        "schema_version": 1,
        "batch_id": reservation["batch_id"],
        "status": "ready",
        "user_scope": reservation["user_scope"],
        "thread_id": reservation["thread_id"],
        "question_count": 5,
        "created_at": reservation["created_at"],
        "questions": [
            {
                "number": number,
                "type": question_type,
                "max_score": 20,
                "file": f"questions/question-{number:02d}.md",
            }
            for number, question_type in enumerate(types, start=1)
        ],
    }
    (workspace / str(reservation["manifest_path"])).write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return reservation


def test_grading_service_runs_double_review_third_review_and_arbitration(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    scores = {
        1: {"a": 14, "b": 16},
        2: {"a": 10, "b": 16, "c": 15},
        3: {"a": 5, "b": 15, "c": 10},
        4: {"a": 12, "b": 12.5},
        5: {"a": 11, "b": 13},
    }

    def grader(agent_name, mode, question_path, context, parent_thread_id):
        nonlocal active, max_active
        assert agent_name.startswith("shenlun-")
        assert question_path.is_file()
        assert parent_thread_id == "thread-1"
        assert context["candidate_answer_word_count"] == 5
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            if mode == "arbitrate":
                return {
                    "final_score": 8.5,
                    "reason": "A、C 的材料核验更具体",
                    "evidence": ["逐点复核"],
                    "disputed_points": ["第三采分点"],
                }
            grader_id = context["grader_id"]
            return {
                "score": scores[context["question_number"]][grader_id],
                "derived_scoring_points": [
                    {
                        "point": "材料核心要点",
                        "max_score": 4,
                        "source_evidence": "材料第1段",
                        "answer_evidence": "测试答案",
                        "awarded_score": 4,
                        "match_level": "full",
                    }
                ],
                "score_components": {"content": scores[context["question_number"]][grader_id]},
                "deductions": [],
                "calculation": "分项合计",
                "word_count": 100,
                "typo_types": [],
                "hit_points": [f"{grader_id} 命中点"],
                "missed_points": [f"{grader_id} 遗漏点"],
                "comments": f"{grader_id} 简评",
                "suggestions": [f"{grader_id} 建议"],
                "confidence": "high",
            }
        finally:
            with lock:
                active -= 1

    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=grader)
    reservation = _ready_batch(service)
    events = []

    started = service.start(
        str(reservation["manifest_path"]),
        parent_thread_id="thread-1",
        reporter=events.append,
    )
    results = [service.next_result(started["job_id"], wait_seconds=5)["question_result"] for _ in range(5)]
    status = service.status(started["job_id"])

    assert max_active == 2
    assert results[0]["final_score"] == 15
    assert results[0]["third_review_triggered"] is False
    assert results[1]["adopted_grader_ids"] == ["b", "c"]
    assert results[1]["final_score"] == 15.5
    assert results[1]["third_review_triggered"] is True
    assert results[1]["arbitration_triggered"] is False
    assert results[2]["adopted_grader_ids"] == []
    assert results[2]["arbitration_triggered"] is True
    assert results[2]["decision_method"] == "arbitration"
    assert results[2]["final_score"] == 8.5
    assert results[2]["arbitration"]["final_score"] == 8.5
    assert results[2]["calculation"] == "专家仲裁终裁分 = 8.5"
    assert results[3]["final_score"] == 12.5
    assert results[0]["threshold"] == 4
    assert results[4]["threshold"] == 3
    assert results[0]["graders"]["a"]["derived_scoring_points"][0]["point"] == "材料核心要点"
    assert results[0]["graders"]["a"]["word_count"] == 5
    assert results[0]["question"] == {"display_number": "第2题", "stem": "测试题干1。"}
    assert status["status"] == "completed"
    assert status["summary"]["total_score"] == sum(result["final_score"] for result in results)
    assert [event.type for event in events].count("grading_question_completed") == 5
    assert "grading_third_review_started" in [event.type for event in events]
    assert "grading_arbitration_started" in [event.type for event in events]
    assert events[-1].type == "grading_batch_completed"

    batch_dir = service.workspace_root / str(reservation["batch_dir"])
    assert (batch_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == len(events)
    assert (batch_dir / "results/question-03/grader-c.json").is_file()
    assert (batch_dir / "results/question-03/arbitration.json").is_file()
    assert (batch_dir / "summary.json").is_file()
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["questions"][4]["question"] == {
        "display_number": "第6题",
        "stem": "测试题干5。",
    }

    recovered = ShenlunGradingService(settings, grader=grader).status(started["job_id"])
    assert recovered["status"] == "completed"
    assert recovered["completed"] == 5


def test_grader_result_clamps_score_and_deduction_total() -> None:
    result = _validate_grader_result(
        {
            "score": 11,
            "derived_scoring_points": [{"point": "要点一", "awarded_score": 2}],
            "score_components": {"content": 17},
            "deductions": [
                {"type": "typo", "basis": "9个不同错别字", "points": 3},
                {"type": "word_count", "basis": "超出90字", "points": 3},
            ],
            "calculation": "17-6=11",
        },
        20,
        "a",
        question_type="summary",
    )

    assert result["score"] == 13
    assert result["deduction_total"] == 4
    assert result["derived_scoring_points"][0]["point"] == "要点一"
    assert any("20%" in item for item in result["validation_adjustments"])


def test_program_word_count_overrides_grader_report() -> None:
    result = _validate_grader_result(
        {
            "score": 10,
            "score_components": {"content": 10},
            "word_count": 999,
        },
        20,
        "a",
        question_type="summary",
        expected_word_count=123,
    )

    assert result["word_count"] == 123


def test_essay_score_is_recomputed_inside_selected_tier() -> None:
    result = _validate_grader_result(
        {
            "score": 39,
            "essay_tier": {
                "tier": 1,
                "range": [32, 40],
                "base_score": 36,
                "mandatory_conditions": [],
                "forced_downgrade": False,
            },
            "score_components": {
                "tier_base": 36,
                "in_tier_bonus": 3,
                "in_tier_penalty": 1,
            },
            "deductions": [{"type": "typo", "basis": "6个不同错别字", "points": 2}],
            "calculation": "36+3-1-2=36",
        },
        40,
        "a",
        question_type="essay",
    )

    assert result["score"] == 36
    assert result["calculation"] == "36 + 3 - 1 - 2 = 36"
    assert result["essay_tier"]["range"] == [32, 40]


def test_grading_reservations_never_reuse_a_batch_directory(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})

    first = service.reserve(user_scope="same-user", thread_id="same-thread", question_count=5)
    second = service.reserve(user_scope="same-user", thread_id="same-thread", question_count=5)

    assert first["batch_id"] != second["batch_id"]
    assert first["batch_dir"] != second["batch_dir"]
    assert (service.workspace_root / str(first["batch_dir"])).is_dir()
    assert (service.workspace_root / str(second["batch_dir"])).is_dir()


def test_current_returns_latest_unfinished_question_batch(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})

    first = service.reserve(user_scope="same-user", thread_id="same-thread", question_count=2)
    second = service.reserve(user_scope="same-user", thread_id="same-thread", question_count=4)
    second_manifest_path = service.workspace_root / str(second["manifest_path"])
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    second_manifest["status"] = "collecting_questions"
    second_manifest["questions"] = [
        {"number": 1, "type": "summary", "max_score": 20, "file": "questions/第1题.md"},
        {"number": 2, "type": "analysis", "max_score": 20, "file": "questions/第2题.md"},
    ]
    second_manifest_path.write_text(json.dumps(second_manifest, ensure_ascii=False), encoding="utf-8")

    current = service.current(user_scope="same-user", thread_id="same-thread")

    assert current["batch_id"] == second["batch_id"]
    assert current["status"] == "collecting_questions"
    assert current["question_paths"] == [
        f"{second['batch_dir']}/questions/第1题.md",
        f"{second['batch_dir']}/questions/第2题.md",
    ]
    assert current["question_count"] == 4
    assert current["questions_dir"].endswith("/questions")
    assert current["batch_id"] != first["batch_id"]


def test_current_does_not_fall_back_to_an_older_unfinished_batch(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})

    service.reserve(user_scope="same-user", thread_id="same-thread", question_count=2)
    latest = service.reserve(user_scope="same-user", thread_id="same-thread", question_count=4)
    latest_manifest_path = service.workspace_root / str(latest["manifest_path"])
    latest_manifest = json.loads(latest_manifest_path.read_text(encoding="utf-8"))
    latest_manifest["status"] = "completed"
    latest_manifest_path.write_text(json.dumps(latest_manifest, ensure_ascii=False), encoding="utf-8")

    try:
        service.current(user_scope="same-user", thread_id="same-thread")
    except ValueError as exc:
        assert "latest Shenlun batch is 'completed'" in str(exc)
    else:
        raise AssertionError("current should not return an older unfinished batch")


def test_start_rejects_missing_candidate_answer_before_spawning_graders(tmp_path: Path) -> None:
    grader_calls = 0

    def grader(*_):
        nonlocal grader_calls
        grader_calls += 1
        return {}

    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=grader)
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    question_path = service.workspace_root / str(reservation["question_paths"][0])
    question_path.write_text(
        "# 第1题\n\n## 基本信息\n\n- 题目信息状态：完整\n\n"
        "## 题干\n\n题干。\n\n## 相关材料\n\n材料。\n\n"
        "## 考生答案逐字转写\n\n- 状态：待上传\n\n待用户上传答案。\n",
        encoding="utf-8",
    )
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="ready",
        questions=[{"number": 1, "type": "summary", "max_score": 20, "file": "questions/question-01.md"}],
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    try:
        service.start(str(reservation["manifest_path"]), parent_thread_id="thread-1")
    except ValueError as exc:
        assert "candidate answer status must be '已提供'" in str(exc)
    else:
        raise AssertionError("start should reject a question whose answer is still pending")

    assert grader_calls == 0


def test_start_rejects_incomplete_question_information(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    question_path = service.workspace_root / str(reservation["question_paths"][0])
    question_path.write_text(
        "# 第1题\n\n## 基本信息\n\n- 题目信息状态：未完\n- 缺失位置：材料第3段中断\n\n"
        "## 题干\n\n题干。\n\n## 相关材料\n\n已有半页材料。\n\n"
        "## 考生答案逐字转写\n\n- 状态：已提供\n\n测试答案。\n",
        encoding="utf-8",
    )
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="ready",
        questions=[{"number": 1, "type": "summary", "max_score": 20, "file": "questions/question-01.md"}],
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    try:
        service.start(str(reservation["manifest_path"]), parent_thread_id="thread-1")
    except ValueError as exc:
        assert "information status must be '完整'" in str(exc)
    else:
        raise AssertionError("start should reject an incomplete question file")


def test_check_returns_collection_state_without_starting_graders(tmp_path: Path) -> None:
    grader_calls = 0

    def grader(*_):
        nonlocal grader_calls
        grader_calls += 1
        return {}

    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=grader)
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    result = service.check(str(reservation["manifest_path"]), parent_thread_id="thread-1")

    assert result["ready"] is False
    assert result["status"] == "reserved"
    assert result["next_action"] == "continue_collection"
    assert grader_calls == 0
    assert not (manifest_path.parent / "summary.json").exists()
    events = manifest_path.parent.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert '"type":"grading_preflight_failed"' in events


def test_start_tool_preflight_returns_not_ready_without_spawning_graders(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    monkeypatch.setattr("superassist.tools.shenlun_grading.current_team_thread_id", lambda: "thread-1")
    monkeypatch.setattr("superassist.tools.shenlun_grading.get_shenlun_grading_service", lambda _: service)
    from superassist.tools.shenlun_grading import make_shenlun_grading_tool

    result = json.loads(make_shenlun_grading_tool(settings).invoke({
        "action": "start",
        "manifest_path": reservation["manifest_path"],
    }))

    assert result["ready"] is False
    assert result["next_action"] == "continue_collection"
    assert result["status"] == "reserved"


def test_start_rejects_noncanonical_question_path(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    question_path = service.workspace_root / str(reservation["question_paths"][0])
    question_path.write_text(
        "# 第1题\n\n## 基本信息\n\n- 题目信息状态：完整\n\n"
        "## 题干\n\n题干。\n\n## 相关材料\n\n材料。\n\n"
        "## 考生答案逐字转写\n\n- 状态：已提供\n\n答案。\n",
        encoding="utf-8",
    )
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="ready", questions=[{
        "number": 1, "type": "summary", "max_score": 20, "file": "questions/第1题.md",
    }])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    try:
        service.start(str(reservation["manifest_path"]), parent_thread_id="thread-1")
    except ValueError as exc:
        assert "ASCII canonical path" in str(exc)
    else:
        raise AssertionError("start should reject a noncanonical question path")


def test_check_requires_ocr_confirmation(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=1)
    question_path = service.workspace_root / str(reservation["question_paths"][0])
    question_path.write_text(
        "# 第1题\n\n## 基本信息\n\n- 题目信息状态：完整\n\n"
        "## 题干\n\n题干。\n\n## 相关材料\n\n材料。\n\n"
        "## 考生答案逐字转写\n\n- 状态：已提供\n\n答案。\n\n"
        "## OCR疑点\n\n- 第2行字迹不清，暂记为‘治理’。\n",
        encoding="utf-8",
    )
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="ready", questions=[{
        "number": 1, "type": "summary", "max_score": 20, "file": "questions/question-01.md",
    }])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = service.check(str(reservation["manifest_path"]), parent_thread_id="thread-1")
    assert result["ready"] is False
    assert "OCR uncertainty" in result["message"]
    assert result["next_action"] == "confirm_ocr"

    manifest["ocr_confirmed"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    confirmed = service.check(str(reservation["manifest_path"]), parent_thread_id="thread-1")
    assert confirmed["ready"] is True


def test_scope_reduces_unstarted_batch_to_user_confirmed_prefix(tmp_path: Path) -> None:
    settings = Settings(SUPERASSIST_DATA_DIR=tmp_path, SUPERASSIST_EMBEDDING_PROVIDER="hash")
    service = ShenlunGradingService(settings, grader=lambda *_: {})
    reservation = service.reserve(user_scope="user-1", thread_id="thread-1", question_count=5)
    for number, relative in enumerate(reservation["question_paths"], start=1):
        answer_status = "已提供" if number <= 4 else "待上传"
        answer = "测试答案。" if number <= 4 else "待用户上传答案。"
        (service.workspace_root / str(relative)).write_text(
            f"# 第{number}题\n\n## 基本信息\n\n- 题目信息状态：完整\n\n"
            f"## 题干\n\n题干{number}。\n\n## 相关材料\n\n材料{number}。\n\n"
            f"## 考生答案逐字转写\n\n- 状态：{answer_status}\n\n{answer}\n",
            encoding="utf-8",
        )
    manifest_path = service.workspace_root / str(reservation["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="awaiting_answers",
        questions=[
            {"number": number, "type": "summary", "max_score": 20, "file": f"questions/question-{number:02d}.md"}
            for number in range(1, 6)
        ],
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    scoped = service.scope(str(reservation["manifest_path"]), parent_thread_id="thread-1", question_count=4)
    assert scoped["question_count"] == 4
    assert scoped["next_action"] == "check"
    assert service.check(str(reservation["manifest_path"]), parent_thread_id="thread-1")["ready"] is True
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["scope"]["original_question_count"] == 5
    assert persisted["scope"]["confirmed_by_user"] is True
    assert '"type":"grading_scope_updated"' in manifest_path.parent.joinpath("events.jsonl").read_text(encoding="utf-8")
