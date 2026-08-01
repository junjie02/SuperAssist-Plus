from __future__ import annotations

import base64
import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from superassist.config import Settings
from superassist.agent import AgentRuntime
from superassist.llm import FallbackChatModel, create_chat_model
from superassist.tools import images as image_module
from superassist.tools.images import image_search, inspect_image, present_images, validate_image_bytes


def _candidate(candidate_id: str = "img_test") -> dict:
    return {
        "candidate_id": candidate_id,
        "query": "千叶豆腐",
        "title": "千叶豆腐成品",
        "image_url": "https://cdn.example.com/original.jpg",
        "thumbnail_url": "https://cdn.example.com/thumb.jpg",
        "source_url": "https://example.com/recipe",
        "source": "example",
        "width": 1200,
        "height": 800,
    }


def _tool_message(command: Command) -> ToolMessage:
    assert isinstance(command.update, dict)
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    return message


def test_image_search_returns_temporary_multimodal_candidates(monkeypatch, tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_TOOL_NETWORK_ENABLED=True,
    )
    monkeypatch.setattr(image_module, "get_settings", lambda: settings)

    class Search:
        def images(self, query, **kwargs):
            assert query == "千叶豆腐"
            assert kwargs == {"max_results": 2, "safesearch": "moderate"}
            return [
                {
                    "title": "千叶豆腐成品",
                    "image": "https://cdn.example.com/original.jpg",
                    "thumbnail": "https://cdn.example.com/thumb.jpg",
                    "url": "https://example.com/recipe",
                    "source": "example",
                    "width": 1200,
                    "height": 800,
                }
            ]

    monkeypatch.setattr("ddgs.DDGS", Search)
    monkeypatch.setattr(
        image_module,
        "download_image_url",
        lambda *_args, **_kwargs: (b"\xff\xd8\xffimage", "image/jpeg"),
    )

    result = image_search.func(
        "千叶豆腐",
        2,
        state={"image_search_results": {}, "outbound_images": []},
        tool_call_id="call_search",
    )

    assert isinstance(result, Command)
    assert isinstance(result.update, dict)
    candidates = result.update["image_search_results"]
    assert len(candidates) == 1
    candidate_id = next(iter(candidates))
    assert candidate_id.startswith("img_")
    assert result.update.get("outbound_images") is None
    content = _tool_message(result).content
    assert isinstance(content, list)
    assert any(item.get("type") == "input_image" and item.get("detail") == "low" for item in content)
    assert any(candidate_id in str(item.get("text")) for item in content if item.get("type") == "input_text")


def test_inspect_image_returns_high_detail_original(monkeypatch) -> None:
    candidate = _candidate()
    monkeypatch.setattr(
        image_module,
        "download_image_url",
        lambda *_args, **_kwargs: (b"\x89PNG\r\n\x1a\nimage", "image/png"),
    )

    result = inspect_image.func(
        [candidate["candidate_id"]],
        state={"image_search_results": {candidate["candidate_id"]: candidate}},
        tool_call_id="call_inspect",
    )

    content = _tool_message(result).content
    assert isinstance(content, list)
    image = next(item for item in content if item.get("type") == "input_image")
    assert image["detail"] == "high"
    assert image["image_url"].startswith("data:image/png;base64,")


def test_present_images_is_the_only_tool_that_creates_outbound_selection() -> None:
    candidate = _candidate()
    result = present_images.func(
        [candidate["candidate_id"]],
        state={"image_search_results": {candidate["candidate_id"]: candidate}},
        tool_call_id="call_present",
    )

    assert isinstance(result.update, dict)
    assert result.update["outbound_images"] == [
        {
            "candidate_id": "img_test",
            "title": "千叶豆腐成品",
            "image_url": "https://cdn.example.com/original.jpg",
            "thumbnail_url": "https://cdn.example.com/thumb.jpg",
            "source_url": "https://example.com/recipe",
        }
    ]


def test_present_images_rejects_unknown_or_excessive_ids() -> None:
    unknown = present_images.func(
        ["img_unknown"],
        state={"image_search_results": {}},
        tool_call_id="call_unknown",
    )
    excessive = present_images.func(
        ["a", "b", "c", "d"],
        state={"image_search_results": {}},
        tool_call_id="call_many",
    )

    assert _tool_message(unknown).status == "error"
    assert _tool_message(excessive).status == "error"
    assert isinstance(unknown.update, dict) and "outbound_images" not in unknown.update


def test_image_tool_schemas_hide_injected_state() -> None:
    for image_tool in (image_search, inspect_image, present_images):
        properties = image_tool.tool_call_schema.model_json_schema()["properties"]
        assert "state" not in properties
        assert "tool_call_id" not in properties


def test_image_validation_rejects_header_only_corrupt_png() -> None:
    corrupt = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nS8AAAAASUVORK5CYII="
    )

    with pytest.raises(ValueError, match="corrupt or incomplete"):
        validate_image_bytes(corrupt)


def test_multimodal_tool_result_becomes_responses_function_output() -> None:
    model = create_chat_model(
        Settings(
            SUPERASSIST_MODEL="gpt-5.6-sol",
            SUPERASSIST_API_KEY="test-key",
            SUPERASSIST_BASE_URL="https://gateway.example/v1",
            SUPERASSIST_EMBEDDING_PROVIDER="hash",
        )
    )
    payload = model._get_request_payload(
        [
            HumanMessage(content="找一张千叶豆腐图片"),
            AIMessage(
                content="我先搜索图片。",
                tool_calls=[{"name": "image_search", "args": {"query": "千叶豆腐"}, "id": "call_1"}],
            ),
            ToolMessage(
                content=[
                    {"type": "input_text", "text": "candidate_id=img_1"},
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP8z4AdMOEQH6QSAM1BAQ/oQeJvAAAAAElFTkSuQmCC"
                        ),
                        "detail": "low",
                    },
                ],
                tool_call_id="call_1",
            ),
        ]
    )

    tool_output = next(item for item in payload["input"] if item.get("type") == "function_call_output")
    assert tool_output["call_id"] == "call_1"
    assert tool_output["output"][1] == {
        "type": "input_image",
        "image_url": (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP8z4AdMOEQH6QSAM1BAQ/oQeJvAAAAAElFTkSuQmCC"
        ),
        "detail": "low",
    }


def test_agent_searches_then_explicitly_selects_image_without_persisting_tool_payload(
    monkeypatch,
    tmp_path,
) -> None:
    factory_module = importlib.import_module("superassist.agent.factory")

    class Search:
        def images(self, _query, **_kwargs):
            return [
                {
                    "title": "千叶豆腐成品",
                    "image": "https://cdn.example.com/original.jpg",
                    "thumbnail": "https://cdn.example.com/thumb.jpg",
                    "url": "https://example.com/recipe",
                    "source": "example",
                    "width": 1200,
                    "height": 800,
                }
            ]

    class LeadImageModel(FallbackChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            if not tool_messages:
                message = AIMessage(
                    content="我先搜索图片。",
                    tool_calls=[
                        {"name": "image_search", "args": {"query": "千叶豆腐"}, "id": "call_search"}
                    ],
                )
            elif tool_messages[-1].name == "image_search":
                content = tool_messages[-1].content
                assert isinstance(content, list)
                assert any(item.get("type") == "input_image" for item in content)
                metadata = next(
                    json.loads(item["text"])
                    for item in content
                    if item.get("type") == "input_text" and item.get("text", "").startswith("{")
                )
                message = AIMessage(
                    content="这张图符合语境，我选择展示。",
                    tool_calls=[
                        {
                            "name": "present_images",
                            "args": {"candidate_ids": [metadata["candidate_id"]]},
                            "id": "call_present",
                        }
                    ],
                )
            else:
                message = AIMessage(content="这是千叶豆腐的成品图片。")
            return ChatResult(generations=[ChatGeneration(message=message)])

    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_ENABLE_TOOLS=True,
        SUPERASSIST_SUBAGENTS_ENABLED=False,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_MEMORY_LLM_WRITER_ENABLED=False,
    )
    monkeypatch.setattr(image_module, "get_settings", lambda: settings)
    monkeypatch.setattr("ddgs.DDGS", Search)
    monkeypatch.setattr(
        image_module,
        "download_image_url",
        lambda *_args, **_kwargs: (b"\xff\xd8\xffimage", "image/jpeg"),
    )
    monkeypatch.setattr(factory_module, "create_chat_model", lambda _settings: LeadImageModel())

    runtime = AgentRuntime(settings)
    result = runtime.run("千叶豆腐长什么样？", user_id="u", thread_id="image-loop")
    runtime.memory_queue.flush()

    assert result.answer == "这是千叶豆腐的成品图片。"
    assert result.metadata["outbound_images"][0]["title"] == "千叶豆腐成品"
    records = [
        json.loads(line)
        for line in (tmp_path / "threads" / "image-loop" / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["role"] for record in records] == ["user", "assistant"]
    assert "data:image" not in json.dumps(records, ensure_ascii=False)
    assert "img_" not in json.dumps(records, ensure_ascii=False)
