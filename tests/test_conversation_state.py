from __future__ import annotations

import json

import pytest

from superassist.config import Settings
from superassist.conversation_state import clear_short_state, parse_clear_command


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )


@pytest.mark.parametrize("command", ["/clear", " /clear ", "/CLEAR", "\n/Clear\t"])
def test_parse_clear_command_matches_only_standalone_command(command) -> None:
    assert parse_clear_command(command) is True


@pytest.mark.parametrize("text", ["", "/clear now", "please /clear", "/clear/", "clear"])
def test_parse_clear_command_rejects_non_commands(text) -> None:
    assert parse_clear_command(text) is False


def test_clear_short_state_preserves_owned_durable_data_and_unknown_metadata(tmp_path) -> None:
    settings = _settings(tmp_path)
    thread_dir = tmp_path / "threads" / "t1"
    thread_dir.mkdir(parents=True)
    (thread_dir / "messages.jsonl").write_text(
        '{"role":"user","content":"hello"}\n{"role":"assistant","content":"hi"}\n',
        encoding="utf-8",
    )
    (thread_dir / "thread_meta.json").write_text(
        json.dumps(
            {
                "user_id": "u1",
                "summary": "recent summary",
                "summary_version": 8,
                "short_memory_compacted_records": 2,
                "short_memory_compressed": True,
                "loaded_skills": ["deep-research"],
                "skill_activations": {"deep-research": 123.0},
                "unrelated_setting": {"rag_mode": True},
            }
        ),
        encoding="utf-8",
    )
    durable_paths = [
        tmp_path / "superassist.sqlite3",
        tmp_path / "faiss" / "u1.index",
        tmp_path / "rag" / "u1" / "documents.json",
        tmp_path / "study" / "shenlun" / "sessions" / "t1.json",
        tmp_path / "workspace" / "shenlun-grading" / "t1" / "question.md",
    ]
    for path in durable_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("important", encoding="utf-8")

    result = clear_short_state(settings, thread_id="t1", user_id="u1")

    metadata = json.loads((thread_dir / "thread_meta.json").read_text(encoding="utf-8"))
    assert result.cleared_message_records == 2
    assert result.cleared_skills == ("deep-research",)
    assert result.generation == 1
    assert (thread_dir / "messages.jsonl").read_text(encoding="utf-8") == ""
    assert metadata["user_id"] == "u1"
    assert metadata["summary"] == ""
    assert metadata["summary_version"] == 0
    assert metadata["short_memory_compacted_records"] == 0
    assert metadata["loaded_skills"] == []
    assert metadata["skill_activations"] == {}
    assert metadata["conversation_generation"] == 1
    assert metadata["unrelated_setting"] == {"rag_mode": True}
    assert all(path.read_text(encoding="utf-8") == "important" for path in durable_paths)


def test_clear_short_state_is_idempotent_and_advances_generation(tmp_path) -> None:
    settings = _settings(tmp_path)

    first = clear_short_state(settings, thread_id="t1", user_id="u1")
    second = clear_short_state(settings, thread_id="t1", user_id="u1")

    assert first.generation == 1
    assert second.generation == 2
    assert second.cleared_message_records == 0


def test_clear_short_state_rejects_wrong_owner(tmp_path) -> None:
    settings = _settings(tmp_path)
    thread_dir = tmp_path / "threads" / "t1"
    thread_dir.mkdir(parents=True)
    (thread_dir / "thread_meta.json").write_text('{"user_id":"owner"}', encoding="utf-8")

    with pytest.raises(PermissionError):
        clear_short_state(settings, thread_id="t1", user_id="other")

    assert json.loads((thread_dir / "thread_meta.json").read_text(encoding="utf-8"))["user_id"] == "owner"


def test_clear_short_state_does_not_overwrite_corrupt_metadata(tmp_path) -> None:
    settings = _settings(tmp_path)
    thread_dir = tmp_path / "threads" / "t1"
    thread_dir.mkdir(parents=True)
    metadata_path = thread_dir / "thread_meta.json"
    metadata_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        clear_short_state(settings, thread_id="t1", user_id="u1")

    assert metadata_path.read_text(encoding="utf-8") == "{broken"
