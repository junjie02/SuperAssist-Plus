from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from superassist.config import Settings
from superassist.models import AgentRunEvent

logger = logging.getLogger(__name__)

QuestionType = Literal["summary", "analysis", "countermeasure", "document", "essay"]
Grader = Callable[[str, str, Path, dict[str, Any], str], dict[str, Any]]

QUESTION_TYPES: dict[str, tuple[str, str]] = {
    "summary": ("归纳概括", "shenlun-summary-grader"),
    "analysis": ("综合分析", "shenlun-analysis-grader"),
    "countermeasure": ("提出对策", "shenlun-countermeasure-grader"),
    "document": ("贯彻执行", "shenlun-document-grader"),
    "essay": ("申发论述", "shenlun-essay-grader"),
}
QUESTION_TYPE_ALIASES = {
    "归纳概括": "summary",
    "概括": "summary",
    "综合分析": "analysis",
    "分析": "analysis",
    "提出对策": "countermeasure",
    "对策": "countermeasure",
    "贯彻执行": "document",
    "应用文": "document",
    "公文": "document",
    "申发论述": "essay",
    "文章写作": "essay",
    "大作文": "essay",
}

# Coordinator threads never run a model directly. All grading model calls share
# this pool, so multiple batches still cannot exceed two concurrent graders.
_GRADER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shenlun-grader")
_JSON_WRITE_LOCK = threading.Lock()


class ShenlunGradingError(ValueError):
    pass


@dataclass
class GradingJob:
    job_id: str
    batch_dir: Path
    manifest: dict[str, Any]
    parent_thread_id: str
    reporter: Callable[[AgentRunEvent], None] | None = None
    status: str = "queued"
    completed: int = 0
    current_question: int | None = None
    error: str = ""
    result_queue: deque[dict[str, Any]] = field(default_factory=deque)
    condition: threading.Condition = field(default_factory=threading.Condition)


class ShenlunGradingService:
    def __init__(self, settings: Settings, *, grader: Grader | None = None) -> None:
        self.settings = settings
        self.workspace_root = settings.resolved_tool_workspace_dir.resolve()
        self.root = (self.workspace_root / "shenlun-grading").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._grader = grader or self._run_grader_agent
        self._jobs: dict[str, GradingJob] = {}
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()

    def reserve(self, *, user_scope: str, thread_id: str, question_count: int = 5) -> dict[str, Any]:
        if question_count <= 0 or question_count > 20:
            raise ShenlunGradingError("question_count must be between 1 and 20")
        safe_scope = _safe_segment(user_scope, "local")
        safe_thread = _safe_segment(thread_id, "thread")
        created_at = datetime.now(UTC)
        batch_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex}"
        batch_dir = self.root / safe_scope / safe_thread / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        questions_dir = batch_dir / "questions"
        results_dir = batch_dir / "results"
        questions_dir.mkdir()
        results_dir.mkdir()
        for number in range(1, question_count + 1):
            (results_dir / f"question-{number:02d}").mkdir()
        (batch_dir / "events.jsonl").touch()
        manifest = {
            "schema_version": 1,
            "batch_id": batch_id,
            "status": "reserved",
            "user_scope": user_scope,
            "thread_id": thread_id,
            "question_count": question_count,
            "created_at": created_at.isoformat(),
            "ocr_confirmed": False,
            "questions": [],
        }
        _write_json(batch_dir / "manifest.json", manifest)
        relative_batch = batch_dir.relative_to(self.workspace_root).as_posix()
        return {
            "batch_id": batch_id,
            "status": "reserved",
            "user_scope": user_scope,
            "thread_id": thread_id,
            "created_at": created_at.isoformat(),
            "batch_dir": relative_batch,
            "questions_dir": f"{relative_batch}/questions",
            "manifest_path": f"{relative_batch}/manifest.json",
            "question_paths": [
                f"{relative_batch}/questions/question-{number:02d}.md"
                for number in range(1, question_count + 1)
            ],
            "result_dirs": [
                f"{relative_batch}/results/question-{number:02d}"
                for number in range(1, question_count + 1)
            ],
        }

    def current(self, *, user_scope: str, thread_id: str) -> dict[str, Any]:
        """Return the conversation's latest batch only when it is still unfinished."""

        batches_root = self.root / _safe_segment(user_scope, "local") / _safe_segment(thread_id, "thread")
        if batches_root.is_dir():
            for batch_dir in sorted(
                (path for path in batches_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            ):
                manifest_path = batch_dir / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = _read_json(manifest_path)
                if str(manifest.get("user_scope") or "") != user_scope:
                    continue
                if str(manifest.get("thread_id") or "") != thread_id:
                    continue
                status = str(manifest.get("status") or "")
                if status not in {"reserved", "collecting_questions", "awaiting_answers", "ready"}:
                    raise ShenlunGradingError(
                        f"latest Shenlun batch is '{status or 'unknown'}'; upload question information to create a new batch"
                    )
                question_count = int(manifest.get("question_count") or 0)
                relative_batch = batch_dir.relative_to(self.workspace_root).as_posix()
                questions = manifest.get("questions")
                saved_paths = [
                    f"{relative_batch}/{str(question.get('file') or '').strip()}"
                    for question in questions
                    if isinstance(question, dict) and str(question.get("file") or "").strip()
                ] if isinstance(questions, list) else []
                return {
                    "batch_id": str(manifest.get("batch_id") or batch_dir.name),
                    "status": status,
                    "user_scope": user_scope,
                    "thread_id": thread_id,
                    "batch_dir": relative_batch,
                    "questions_dir": f"{relative_batch}/questions",
                    "manifest_path": f"{relative_batch}/manifest.json",
                    "question_count": question_count,
                    "question_paths": saved_paths or [
                        f"{relative_batch}/questions/question-{number:02d}.md"
                        for number in range(1, question_count + 1)
                    ],
                    "questions": questions if isinstance(questions, list) else [],
                }
        raise ShenlunGradingError("no Shenlun question batch exists for this conversation")

    def start(
        self,
        manifest_path: str,
        *,
        parent_thread_id: str,
        reporter: Callable[[AgentRunEvent], None] | None = None,
    ) -> dict[str, Any]:
        path = self._resolve_manifest_path(manifest_path)
        manifest = _read_json(path)
        batch_dir = path.parent
        manifest = self._validate_manifest(manifest, batch_dir, parent_thread_id)
        job_id = str(manifest["batch_id"])

        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return self._job_status(existing)
            summary_path = batch_dir / "summary.json"
            if summary_path.is_file():
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "completed": int(manifest["question_count"]),
                    "total": int(manifest["question_count"]),
                    "summary": _read_json(summary_path),
                }
            job = GradingJob(
                job_id=job_id,
                batch_dir=batch_dir,
                manifest=manifest,
                parent_thread_id=parent_thread_id,
                reporter=reporter,
            )
            self._jobs[job_id] = job

        manifest["status"] = "running"
        manifest["job_id"] = job_id
        manifest["started_at"] = datetime.now(UTC).isoformat()
        _write_json(path, manifest)
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            name=f"shenlun-job-{job_id[-8:]}",
            daemon=True,
        )
        thread.start()
        return self._job_status(job)

    def check(
        self,
        manifest_path: str,
        *,
        parent_thread_id: str,
    ) -> dict[str, Any]:
        """Validate a batch without starting graders.

        This is intentionally safe to call while images or answers are still
        being collected.  The tool layer uses the result to keep the lead agent
        in collection mode instead of turning an expected intermediate state
        into an error/retry loop.
        """

        path = self._resolve_manifest_path(manifest_path)
        manifest = _read_json(path)
        status = str(manifest.get("status") or "unknown")
        try:
            self._validate_manifest(manifest, path.parent, parent_thread_id)
        except ShenlunGradingError as exc:
            ready_states = {"reserved", "collecting_questions", "awaiting_answers"}
            reason = str(exc)
            if "OCR uncertainty" in reason:
                next_action = "confirm_ocr"
            elif status in ready_states:
                next_action = "continue_collection"
            else:
                next_action = "fix_manifest"
            result: dict[str, Any] = {
                "ready": False,
                "status": status,
                "manifest_path": path.relative_to(self.workspace_root).as_posix(),
                "message": reason,
                "next_action": next_action,
            }
            self._record_batch_event(
                path.parent,
                "grading_preflight_failed",
                str(exc),
                {"batch_id": manifest.get("batch_id"), "status": status, "next_action": next_action},
            )
            logger.warning(
                "Shenlun grading preflight failed batch_id=%s status=%s reason=%s",
                manifest.get("batch_id"),
                status,
                exc,
            )
            return result
        logger.info("Shenlun grading preflight passed batch_id=%s", manifest.get("batch_id"))
        self._record_batch_event(
            path.parent,
            "grading_preflight_passed",
            "申论批改预检通过",
            {"batch_id": manifest.get("batch_id"), "status": "ready"},
        )
        return {
            "ready": True,
            "status": "ready",
            "manifest_path": path.relative_to(self.workspace_root).as_posix(),
            "next_action": "start",
        }

    def scope(
        self,
        manifest_path: str,
        *,
        parent_thread_id: str,
        question_count: int,
    ) -> dict[str, Any]:
        """Restrict an unstarted batch to its first N questions."""

        if question_count <= 0 or question_count > 20:
            raise ShenlunGradingError("question_count must be between 1 and 20")
        path = self._resolve_manifest_path(manifest_path)
        manifest = _read_json(path)
        if str(manifest.get("thread_id") or "") != parent_thread_id:
            raise ShenlunGradingError("manifest thread_id does not match the active conversation")
        status = str(manifest.get("status") or "")
        if status not in {"reserved", "collecting_questions", "awaiting_answers", "ready"}:
            raise ShenlunGradingError(f"cannot change question scope while manifest.status is '{status}'")
        questions = manifest.get("questions")
        if not isinstance(questions, list) or question_count > len(questions):
            raise ShenlunGradingError("question_count cannot exceed the questions already collected")
        original_count = int(manifest.get("question_count") or len(questions))
        selected = questions[:question_count]
        manifest["questions"] = selected
        manifest["question_count"] = question_count
        manifest["status"] = _infer_manifest_status(manifest, path.parent)
        manifest["scope"] = {
            "original_question_count": original_count,
            "selected_question_count": question_count,
            "confirmed_by_user": True,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_json(path, manifest)
        self._record_batch_event(
            path.parent,
            "grading_scope_updated",
            f"评分范围已按用户确认收缩为前 {question_count} 题",
            {
                "batch_id": manifest.get("batch_id"),
                "original_question_count": original_count,
                "question_count": question_count,
            },
        )
        return {
            "status": manifest.get("status"),
            "question_count": question_count,
            "questions": selected,
            "manifest_path": path.relative_to(self.workspace_root).as_posix(),
            "next_action": "check",
        }

    def next_result(self, job_id: str, *, wait_seconds: float = 120.0) -> dict[str, Any]:
        job = self._get_job(job_id)
        if job is None:
            return self.status(job_id)
        timeout = max(0.0, min(float(wait_seconds), 300.0))
        with job.condition:
            if not job.result_queue and job.status not in {"completed", "failed"}:
                job.condition.wait(timeout=timeout)
            if job.result_queue:
                result = job.result_queue.popleft()
                response = {
                    "job_id": job.job_id,
                    "status": job.status,
                    "completed": job.completed,
                    "total": len(job.manifest["questions"]),
                    "question_result": result,
                }
                if job.status == "completed":
                    response["summary"] = _read_json(job.batch_dir / "summary.json")
                return response
            response = self._job_status(job)
            if job.status == "completed":
                response["summary"] = _read_json(job.batch_dir / "summary.json")
            elif job.status not in {"failed", "completed"}:
                response["timed_out"] = True
            return response

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._get_job(job_id)
        if job is not None:
            response = self._job_status(job)
            if job.status == "completed":
                response["summary"] = _read_json(job.batch_dir / "summary.json")
            return response
        batch_dir = self._find_batch(job_id)
        if batch_dir is None:
            raise ShenlunGradingError(f"unknown job_id: {job_id}")
        manifest = _read_json(batch_dir / "manifest.json")
        total = int(manifest.get("question_count") or 0)
        completed = sum(
            1 for number in range(1, total + 1)
            if (batch_dir / "results" / f"question-{number:02d}" / "final.json").is_file()
        )
        response: dict[str, Any] = {
            "job_id": job_id,
            "status": str(manifest.get("status") or "unknown"),
            "completed": completed,
            "total": total,
            "manifest_path": (batch_dir / "manifest.json").relative_to(self.workspace_root).as_posix(),
        }
        summary_path = batch_dir / "summary.json"
        if summary_path.is_file():
            response["status"] = "completed"
            response["summary"] = _read_json(summary_path)
        if manifest.get("error"):
            response["error"] = str(manifest["error"])
        return response

    def _run_job(self, job: GradingJob) -> None:
        job.status = "running"
        last_result: dict[str, Any] | None = None
        try:
            questions = job.manifest["questions"]
            for index, question in enumerate(questions):
                job.current_question = int(question["number"])
                final = self._grade_question(job, question)
                with job.condition:
                    job.completed += 1
                    if index == len(questions) - 1:
                        last_result = final
                    else:
                        job.result_queue.append(final)
                        job.condition.notify_all()
            summary = self._build_summary(job)
            _write_json(job.batch_dir / "summary.json", summary)
            job.status = "completed"
            job.current_question = None
            self._update_manifest(job, status="completed", completed_at=datetime.now(UTC).isoformat())
            self._emit(
                job,
                "grading_batch_completed",
                f"申论评分完成，共 {job.completed} 题",
                {"completed": job.completed, "total": len(job.manifest["questions"]), "summary": summary},
            )
            with job.condition:
                if last_result is not None:
                    job.result_queue.append(last_result)
                job.condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - failures must be persisted for status recovery
            logger.exception("Shenlun grading job failed job_id=%s", job.job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self._update_manifest(job, status="failed", error=job.error, failed_at=datetime.now(UTC).isoformat())
        finally:
            with job.condition:
                job.condition.notify_all()

    def _grade_question(self, job: GradingJob, question: dict[str, Any]) -> dict[str, Any]:
        number = int(question["number"])
        question_type = str(question["type"])
        label, agent_name = QUESTION_TYPES[question_type]
        max_score = float(question["max_score"])
        threshold = _score_threshold(max_score, question_type)
        question_path = self._question_path(job.batch_dir, question)
        report_question = _question_report_content(question_path, number)
        candidate_answer_word_count = _candidate_answer_word_count(question_path)
        result_dir = job.batch_dir / "results" / f"question-{number:02d}"
        result_dir.mkdir(parents=True, exist_ok=True)
        common = {
            "job_id": job.job_id,
            "question_number": number,
            "total": len(job.manifest["questions"]),
            "question_type": question_type,
            "question_type_label": label,
            "max_score": max_score,
            "threshold": threshold,
            "question_path": question_path.relative_to(self.workspace_root).as_posix(),
            "question": report_question,
            "candidate_answer_word_count": candidate_answer_word_count,
        }
        self._emit(
            job,
            "grading_question_started",
            f"第{number}题（{label}）开始评分",
            common,
        )

        graders: dict[str, dict[str, Any]] = {}
        futures = {
            _GRADER_POOL.submit(
                self._grade_once,
                agent_name,
                grader_id,
                question_path,
                common,
                job.parent_thread_id,
            ): grader_id
            for grader_id in ("a", "b")
        }
        for future in as_completed(futures):
            grader_id = futures[future]
            result = future.result()
            graders[grader_id] = result
            _write_json(result_dir / f"grader-{grader_id}.json", result)
            self._emit_grader_completed(job, common, grader_id, result)

        third_review = abs(graders["a"]["score"] - graders["b"]["score"]) > threshold
        arbitration = False
        arbitration_result: dict[str, Any] | None = None
        if not third_review:
            adopted = ["a", "b"]
        else:
            self._emit(
                job,
                "grading_third_review_started",
                f"第{number}题分差超过 {threshold:g} 分，启动三评",
                {**common, "scores": {key: graders[key]["score"] for key in ("a", "b")}},
            )
            graders["c"] = _GRADER_POOL.submit(
                self._grade_once,
                agent_name,
                "c",
                question_path,
                common,
                job.parent_thread_id,
            ).result()
            _write_json(result_dir / "grader-c.json", graders["c"])
            self._emit_grader_completed(job, common, "c", graders["c"])
            adopted = _unique_closest_pair(graders, threshold)
            if adopted is None:
                arbitration = True
                self._emit(
                    job,
                    "grading_arbitration_started",
                    f"第{number}题三评分歧仍无法自动收敛，启动仲裁",
                    {**common, "scores": {key: value["score"] for key, value in graders.items()}},
                )
                arbitration_result = _GRADER_POOL.submit(
                    self._arbitrate,
                    agent_name,
                    question_path,
                    common,
                    graders,
                    job.parent_thread_id,
                ).result()
                _write_json(result_dir / "arbitration.json", arbitration_result)
                adopted = []

        if arbitration_result is not None:
            final_score = float(arbitration_result["final_score"])
            calculation = f"专家仲裁终裁分 = {final_score:g}"
            decision_method = "arbitration"
            advice_grader_ids = sorted(graders)
        else:
            first, second = adopted
            first_score = float(graders[first]["score"])
            second_score = float(graders[second]["score"])
            final_score = _round_half((first_score + second_score) / 2)
            calculation = f"({first_score:g} + {second_score:g}) / 2 = {final_score:g}"
            decision_method = "third_review_average" if third_review else "double_review_average"
            advice_grader_ids = adopted
        final = {
            **common,
            "scores": {key: value["score"] for key, value in sorted(graders.items())},
            "graders": {key: graders[key] for key in sorted(graders)},
            "score_difference": abs(float(graders["a"]["score"]) - float(graders["b"]["score"])),
            "third_review_triggered": third_review,
            "arbitration_triggered": arbitration,
            "arbitration": arbitration_result,
            "adopted_grader_ids": adopted,
            "decision_method": decision_method,
            "calculation": calculation,
            "final_score": final_score,
            "revision_advice": _merge_unique(
                suggestion
                for grader_id in advice_grader_ids
                for suggestion in graders[grader_id].get("suggestions", [])
            ),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(result_dir / "final.json", final)
        self._emit(
            job,
            "grading_question_completed",
            f"第{number}题评分完成：{final_score:g}/{max_score:g}",
            final,
        )
        return final

    def _grade_once(
        self,
        agent_name: str,
        grader_id: str,
        question_path: Path,
        context: dict[str, Any],
        parent_thread_id: str,
    ) -> dict[str, Any]:
        raw = self._grader(agent_name, "grade", question_path, {**context, "grader_id": grader_id}, parent_thread_id)
        return _validate_grader_result(
            raw,
            float(context["max_score"]),
            grader_id,
            question_type=str(context["question_type"]),
            expected_word_count=int(context["candidate_answer_word_count"]),
        )

    def _arbitrate(
        self,
        agent_name: str,
        question_path: Path,
        context: dict[str, Any],
        graders: dict[str, dict[str, Any]],
        parent_thread_id: str,
    ) -> dict[str, Any]:
        raw = self._grader(
            agent_name,
            "arbitrate",
            question_path,
            {**context, "grader_results": graders},
            parent_thread_id,
        )
        try:
            reported_final_score = float(raw.get("final_score"))
        except (TypeError, ValueError) as exc:
            raise ShenlunGradingError("arbitration returned an invalid final_score") from exc
        final_score = _round_half(min(max(reported_final_score, 0.0), float(context["max_score"])))
        adjustments = []
        if final_score != reported_final_score:
            adjustments.append(
                f"仲裁分 {reported_final_score:g} 已修正到合法范围并取到 0.5 分：{final_score:g}"
            )
        return {
            "final_score": final_score,
            "reported_final_score": reported_final_score,
            "calculation": f"专家仲裁终裁分 = {final_score:g}",
            "reason": str(raw.get("reason") or "").strip(),
            "evidence": _string_list(raw.get("evidence")),
            "disputed_points": _string_list(raw.get("disputed_points")),
            "validation_adjustments": adjustments,
        }

    def _run_grader_agent(
        self,
        agent_name: str,
        mode: str,
        question_path: Path,
        context: dict[str, Any],
        parent_thread_id: str,
    ) -> dict[str, Any]:
        from superassist.tools.task import run_task

        relative_path = question_path.relative_to(self.workspace_root).as_posix()
        if mode == "arbitrate":
            instruction = (
                "执行严格、保守的专家仲裁。读取题目文件和 A/B/C 完整结果，逐项核对官方答案或评分参考；"
                "只认可考生答案中明确写出的内容，不得因方向正确而补齐未写出的要点或操作环节。"
                "独立复核采分依据、分项计算、字数和证据，确保终裁理由与终裁分一致；"
                "直接给出终裁分，只输出 system.md 规定的仲裁 JSON 对象。"
            )
        else:
            instruction = (
                "执行严格、保守的独立评分。读取题目文件并逐项核对官方答案或评分参考；"
                "只按考生答案中明确写出的内容给分，不得因方向正确而补齐未写出的操作环节。"
                "输出前必须检查采分点、分项分数、扣分项、评分理由和固定公式彼此一致，"
                "不得一边指出关键要点缺失一边将对应维度给满分。"
                "使用参数中的 candidate_answer_word_count 作为唯一字数，不得自行估算，"
                "只输出一个 JSON 对象，不要输出 Markdown 代码块或额外说明。"
            )
        result = run_task(
            description=f"申论第{context['question_number']}题{mode}",
            prompt=instruction,
            subagent_type=agent_name,
            parameters={**context, "mode": mode, "question_path": relative_path},
            parent_thread_id=parent_thread_id,
            settings=self.settings,
        )
        return _extract_json_object(result)

    def _build_summary(self, job: GradingJob) -> dict[str, Any]:
        question_results = [
            _read_json(job.batch_dir / "results" / f"question-{int(question['number']):02d}" / "final.json")
            for question in job.manifest["questions"]
        ]
        total_score = _round_half(sum(float(item["final_score"]) for item in question_results))
        max_score = sum(float(item["max_score"]) for item in question_results)
        return {
            "schema_version": 1,
            "batch_id": job.job_id,
            "status": "completed",
            "question_count": len(question_results),
            "total_score": total_score,
            "max_score": max_score,
            "questions": question_results,
            "overall_suggestions": _merge_unique(
                suggestion
                for result in question_results
                for suggestion in result.get("revision_advice", [])
            ),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    def _emit_grader_completed(
        self,
        job: GradingJob,
        common: dict[str, Any],
        grader_id: str,
        result: dict[str, Any],
    ) -> None:
        self._emit(
            job,
            "grading_grader_completed",
            f"第{common['question_number']}题评分员 {grader_id.upper()}：{result['score']:g} 分",
            {**common, "grader_id": grader_id, "grader": result},
        )

    def _emit(self, job: GradingJob, event_type: str, message: str, metadata: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            "message": message,
            "metadata": metadata,
        }
        with self._event_lock:
            with (job.batch_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if job.reporter is not None:
            try:
                job.reporter(AgentRunEvent(type=event_type, message=message, metadata=metadata))
            except Exception:  # noqa: BLE001 - a channel failure must not abort grading
                logger.exception("Failed to report grading event type=%s job_id=%s", event_type, job.job_id)

    def _record_batch_event(
        self,
        batch_dir: Path,
        event_type: str,
        message: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist preflight events before a grading job exists."""

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            "message": message,
            "metadata": metadata,
        }
        with self._event_lock:
            with (batch_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _validate_manifest(
        self,
        manifest: dict[str, Any],
        batch_dir: Path,
        parent_thread_id: str,
    ) -> dict[str, Any]:
        if manifest.get("status") != "ready":
            raise ShenlunGradingError("manifest.status must be 'ready' before start")
        if str(manifest.get("thread_id") or "") != parent_thread_id:
            raise ShenlunGradingError("manifest thread_id does not match the active conversation")
        questions = manifest.get("questions")
        question_count = int(manifest.get("question_count") or 0)
        if not isinstance(questions, list) or len(questions) != question_count:
            raise ShenlunGradingError("manifest questions must match question_count")
        expected_numbers = list(range(1, question_count + 1))
        numbers = [int(item.get("number") or 0) for item in questions if isinstance(item, dict)]
        if numbers != expected_numbers:
            raise ShenlunGradingError("manifest questions must be ordered and numbered from 1")
        for question in questions:
            question["type"] = _normalize_question_type(str(question.get("type") or ""))
            max_score = float(question.get("max_score") or 0)
            if max_score <= 0:
                raise ShenlunGradingError(f"question {question['number']} max_score must be positive")
            question["max_score"] = max_score
            question_path = self._question_path(batch_dir, question)
            _validate_answer_section(question_path, int(question["number"]))
            _validate_ocr_confirmation(manifest, question, question_path)
        if str(manifest.get("batch_id") or "") != batch_dir.name:
            raise ShenlunGradingError("manifest batch_id does not match its directory")
        return manifest

    def _question_path(self, batch_dir: Path, question: dict[str, Any]) -> Path:
        configured = str(question.get("file") or f"questions/question-{int(question['number']):02d}.md")
        expected = f"questions/question-{int(question['number']):02d}.md"
        if configured != expected:
            raise ShenlunGradingError(
                f"question {question['number']} file must be '{expected}' (ASCII canonical path)"
            )
        path = (batch_dir / configured).resolve()
        questions_root = (batch_dir / "questions").resolve()
        if not path.is_relative_to(questions_root):
            raise ShenlunGradingError("question file must stay inside the batch questions directory")
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise ShenlunGradingError(f"question file is missing or empty: {configured}")
        return path

    def _resolve_manifest_path(self, manifest_path: str) -> Path:
        raw = Path(manifest_path)
        path = (raw if raw.is_absolute() else self.workspace_root / raw).resolve()
        if not path.is_relative_to(self.root) or path.name != "manifest.json":
            raise ShenlunGradingError("manifest_path must point inside the shenlun-grading workspace")
        if not path.is_file():
            raise ShenlunGradingError(f"manifest not found: {manifest_path}")
        return path

    def _find_batch(self, job_id: str) -> Path | None:
        safe_job_id = _safe_segment(job_id, "")
        if not safe_job_id or safe_job_id != job_id:
            return None
        matches = list(self.root.glob(f"*/*/{safe_job_id}"))
        return matches[0] if len(matches) == 1 else None

    def _get_job(self, job_id: str) -> GradingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _job_status(self, job: GradingJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "completed": job.completed,
            "total": len(job.manifest["questions"]),
            "current_question": job.current_question,
            "error": job.error or None,
            "manifest_path": (job.batch_dir / "manifest.json").relative_to(self.workspace_root).as_posix(),
        }

    def _update_manifest(self, job: GradingJob, **updates: Any) -> None:
        path = job.batch_dir / "manifest.json"
        manifest = _read_json(path)
        manifest.update(updates)
        _write_json(path, manifest)
        job.manifest.update(updates)


def _normalize_question_type(value: str) -> QuestionType:
    normalized = value.strip().lower()
    normalized = QUESTION_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in QUESTION_TYPES:
        supported = ", ".join(QUESTION_TYPES)
        raise ShenlunGradingError(f"unsupported question type '{value}'; expected one of {supported}")
    return normalized  # type: ignore[return-value]


def _validate_answer_section(path: Path, question_number: int) -> None:
    content = path.read_text(encoding="utf-8")
    basic_info = _markdown_section(content, "基本信息")
    info_statuses = re.findall(
        r"(?m)^\s*-\s*题目信息状态\s*[：:]\s*(\S.*?)\s*$",
        basic_info or "",
    )
    if not info_statuses or info_statuses[-1].strip() != "完整":
        raise ShenlunGradingError(f"question {question_number} information status must be '完整'")
    for heading, label in (("题干", "question stem"), ("相关材料", "source material")):
        body = _markdown_section(content, heading)
        if body is None or not body.strip() or body.strip() in {"未提供", "待上传"}:
            raise ShenlunGradingError(f"question {question_number} {label} is missing or empty")
    body = _markdown_section(content, "考生答案逐字转写")
    if body is None:
        raise ShenlunGradingError(f"question {question_number} is missing the candidate answer section")
    body = body.strip()
    status = re.search(r"(?m)^\s*-\s*状态\s*[：:]\s*(?P<status>\S.*?)\s*$", body)
    if status is None or status.group("status").strip() != "已提供":
        raise ShenlunGradingError(f"question {question_number} candidate answer status must be '已提供'")
    answer = body[status.end() :].strip()
    if not answer or answer in {"未提供", "待上传", "待用户上传答案"}:
        raise ShenlunGradingError(f"question {question_number} candidate answer is empty")


def _validate_ocr_confirmation(
    manifest: dict[str, Any],
    question: dict[str, Any],
    path: Path,
) -> None:
    """Require explicit confirmation before grading uncertain OCR text."""

    section = _markdown_section(path.read_text(encoding="utf-8"), "OCR疑点")
    if section is None:
        return
    lines = [line.strip(" -*\t") for line in section.splitlines() if line.strip()]
    uncertain = [line for line in lines if line not in {"无", "无疑点", "未提供", "无 OCR 疑点"}]
    confirmed = bool(manifest.get("ocr_confirmed") or question.get("ocr_confirmed"))
    if uncertain and not confirmed:
        raise ShenlunGradingError(
            f"question {question['number']} has unconfirmed OCR uncertainty; "
            "ask the user to confirm the transcription, then set ocr_confirmed=true"
        )


def _infer_manifest_status(manifest: dict[str, Any], batch_dir: Path) -> str:
    """Recompute collection status after changing the selected question set."""

    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        return "collecting_questions"
    has_pending_answer = False
    for question in questions:
        try:
            path = (batch_dir / str(question.get("file") or "")).resolve()
            _validate_answer_section(path, int(question["number"]))
        except ShenlunGradingError as exc:
            if "candidate answer" in str(exc) or "answer section" in str(exc) or "answer is empty" in str(exc):
                has_pending_answer = True
                continue
            return "collecting_questions"
        except (KeyError, OSError, ValueError):
            return "collecting_questions"
    return "awaiting_answers" if has_pending_answer else "ready"


def _markdown_section(content: str, heading: str) -> str | None:
    match = re.search(
        rf"(?ms)^##\s*{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        content,
    )
    return match.group("body") if match is not None else None


def _question_report_content(path: Path, fallback_number: int) -> dict[str, str]:
    """Extract the original number and stem needed in user-facing reports."""

    content = path.read_text(encoding="utf-8")
    basic_info = _markdown_section(content, "基本信息") or ""
    original_number = re.search(
        r"(?m)^\s*-\s*原试卷题号\s*[：:]\s*(?P<number>\S.*?)\s*$",
        basic_info,
    )
    heading = re.search(r"(?m)^#\s+(?P<title>\S.*?)\s*$", content)
    display_number = (
        original_number.group("number").strip()
        if original_number is not None
        else heading.group("title").strip()
        if heading is not None
        else f"第{fallback_number}题"
    )
    stem = (_markdown_section(content, "题干") or "").strip()
    return {"display_number": display_number, "stem": stem}


def _score_threshold(max_score: float, question_type: str) -> float:
    rate = 0.15 if question_type == "essay" else 0.2
    return _round_half(max_score * rate)


def _round_half(value: float) -> float:
    return math.floor(float(value) * 2 + 0.5) / 2


def _unique_closest_pair(
    graders: dict[str, dict[str, Any]], threshold: float
) -> list[str] | None:
    distances = [
        (abs(float(graders[a]["score"]) - float(graders[b]["score"])), [a, b])
        for a, b in combinations(sorted(graders), 2)
    ]
    distances.sort(key=lambda item: item[0])
    if not distances or distances[0][0] > threshold:
        return None
    if len(distances) > 1 and distances[0][0] == distances[1][0]:
        return None
    return distances[0][1]


def _validate_grader_result(
    raw: dict[str, Any],
    max_score: float,
    grader_id: str,
    *,
    question_type: str,
    expected_word_count: int | None = None,
) -> dict[str, Any]:
    try:
        reported_score = float(raw.get("score"))
    except (TypeError, ValueError) as exc:
        raise ShenlunGradingError(f"grader {grader_id} returned an invalid score") from exc
    adjustments: list[str] = []
    score_components = _numeric_dict(raw.get("score_components"))
    essay_tier = _plain_dict(raw.get("essay_tier"))
    deductions, deduction_relief = _normalize_deductions(raw.get("deductions"), max_score, adjustments)
    calculated = _score_from_components(
        question_type,
        score_components,
        deductions,
        essay_tier,
        max_score,
    )
    if calculated is None:
        adjusted_reported_score = reported_score + deduction_relief
        score = _round_half(min(max(adjusted_reported_score, 0.0), max_score))
        program_calculation = str(raw.get("calculation") or "").strip()
    else:
        score, program_calculation = calculated
    if score != reported_score:
        adjustments.append(
            f"评分员报告分 {reported_score:g} 已按程序公式修正为 {score:g}"
        )
    reported_word_count = _non_negative_int(raw.get("word_count"))
    if expected_word_count is not None and reported_word_count != expected_word_count:
        adjustments.append(
            f"评分员报告字数 {reported_word_count if reported_word_count is not None else '缺失'} "
            f"已按程序计算值 {expected_word_count} 修正"
        )
    return {
        "grader_id": grader_id,
        "score": score,
        "reported_score": reported_score,
        "question_type": question_type,
        "derived_scoring_points": _dict_list(raw.get("derived_scoring_points"), limit=20),
        "score_components": score_components,
        "essay_tier": essay_tier,
        "deductions": deductions,
        "deduction_total": _round_half(sum(float(item["points"]) for item in deductions)),
        "calculation": program_calculation,
        "reported_calculation": str(raw.get("calculation") or "").strip(),
        "word_count": (
            max(0, int(expected_word_count))
            if expected_word_count is not None
            else reported_word_count
        ),
        "typo_types": _string_list(raw.get("typo_types")),
        "hit_points": _string_list(raw.get("hit_points")),
        "missed_points": _string_list(raw.get("missed_points")),
        "comments": str(raw.get("comments") or "").strip(),
        "suggestions": _string_list(raw.get("suggestions")),
        "confidence": str(raw.get("confidence") or "medium").strip().lower(),
        "validation_adjustments": adjustments,
    }


def _normalize_deductions(
    value: Any,
    max_score: float,
    adjustments: list[str],
) -> tuple[list[dict[str, Any]], float]:
    if not isinstance(value, list):
        return [], 0.0
    deductions: list[dict[str, Any]] = []
    for item in value[:10]:
        if not isinstance(item, dict):
            continue
        try:
            points = max(0.0, _round_half(float(item.get("points") or 0)))
        except (TypeError, ValueError):
            points = 0.0
        deductions.append(
            {
                "type": str(item.get("type") or "other").strip(),
                "basis": str(item.get("basis") or "").strip(),
                "points": points,
            }
        )
    cap = _round_half(max_score * 0.2)
    total = sum(float(item["points"]) for item in deductions)
    deduction_relief = 0.0
    if total > cap:
        deduction_relief = _round_half(total - cap)
        excess = deduction_relief
        for item in reversed(deductions):
            reduction = min(float(item["points"]), excess)
            item["points"] = _round_half(float(item["points"]) - reduction)
            excess -= reduction
            if excess <= 0:
                break
        adjustments.append(f"扣分合计超过题目满分的20%，已按上限 {cap:g} 分修正")
    return deductions, deduction_relief


def _score_from_components(
    question_type: str,
    components: dict[str, float],
    deductions: list[dict[str, Any]],
    essay_tier: dict[str, Any] | None,
    max_score: float,
) -> tuple[float, str] | None:
    if not components:
        return None
    deduction_total = sum(float(item["points"]) for item in deductions)
    if question_type == "essay" and "tier_base" in components:
        base = components.get("tier_base", 0.0)
        bonus = components.get("in_tier_bonus", 0.0)
        penalty = components.get("in_tier_penalty", 0.0)
        raw_score = base + bonus - penalty - deduction_total
        lower, upper = 0.0, max_score
        if isinstance(essay_tier, dict):
            tier_range = essay_tier.get("range")
            if isinstance(tier_range, list) and len(tier_range) == 2:
                try:
                    lower = max(0.0, float(tier_range[0]))
                    upper = min(max_score, float(tier_range[1]))
                except (TypeError, ValueError):
                    lower, upper = 0.0, max_score
        score = _round_half(min(max(raw_score, lower), upper))
        calculation = (
            f"{base:g} + {bonus:g} - {penalty:g} - {deduction_total:g} = {score:g}"
        )
        return score, calculation
    component_total = sum(components.values())
    score = _round_half(min(max(component_total - deduction_total, 0.0), max_score))
    calculation = f"{component_total:g} - {deduction_total:g} = {score:g}"
    return score, calculation


def _dict_list(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_plain_dict(item) for item in value[:limit] if isinstance(item, dict)]


def _plain_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items() if isinstance(item, (str, int, float, bool, list, dict)) or item is None}


def _numeric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = _round_half(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _non_negative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _candidate_answer_word_count(path: Path) -> int:
    """Count non-whitespace characters in the persisted candidate answer."""

    content = path.read_text(encoding="utf-8")
    body = _markdown_section(content, "考生答案逐字转写") or ""
    status = re.search(r"(?m)^\s*-\s*状态\s*[：:]\s*(?P<status>\S.*?)\s*$", body)
    if status is None:
        raise ShenlunGradingError("candidate answer section is missing its status")
    answer = body[status.end() :]
    return sum(1 for character in answer if not character.isspace())


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ShenlunGradingError("grader did not return a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ShenlunGradingError("grader returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ShenlunGradingError("grader result must be a JSON object")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _merge_unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _safe_segment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip(".-")
    return normalized[:120] or fallback


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with _JSON_WRITE_LOCK:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShenlunGradingError(f"cannot read valid JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise ShenlunGradingError(f"{path.name} must contain a JSON object")
    return value


__all__ = [
    "QUESTION_TYPES",
    "ShenlunGradingError",
    "ShenlunGradingService",
]
